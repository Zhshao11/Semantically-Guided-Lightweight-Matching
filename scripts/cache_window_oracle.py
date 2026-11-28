"""Run original frozen T05K256 once; save immutable model packets and actual window supports."""
import argparse,hashlib,json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np,torch
from sglm.t05 import UniformB0
from sglm.t02.data import AffinePairs256
from sglm.model import coarse_xy
from sglm.geometry import transform_points
from sglm.diagnostics import InferenceStateAudit
from sglm.evaluation import estimate_from_correspondences,fixed_synthetic_probes,evaluate_transform
from sglm.window_oracle import sample_actual_support,mask_valid_at,support_hits,frozen_fit_mask,array_digest,inside_convex_support


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    p=json.loads(Path('configs/window_oracle_v1.json').read_text());out=Path(p['output']);out.mkdir(parents=True,exist_ok=True);cache=Path(p['cache']);cache.mkdir(parents=True,exist_ok=True)
    if (out/'cache_freeze.json').exists():raise FileExistsError('Frozen oracle candidate cache already exists')
    assert sha(p['checkpoint'])==p['checkpoint_sha256']
    saved=torch.load(p['checkpoint'],map_location='cpu',weights_only=False);cfg=saved['config']
    paths=['configs/window_oracle_v1.json',p['checkpoint'],p['frozen']+'/validation_plan.json',p['frozen']+'/subsets.json',
        'sglm/t05.py','sglm/t02/model.py','sglm/model.py','sglm/evaluation.py','sglm/geometry.py','sglm/window_oracle.py','scripts/cache_window_oracle.py']
    input_hashes={path:sha(path) for path in paths}
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    ds=AffinePairs256(a.root,p['frozen']+'/validation_plan.json',split='val',preload=True);assert len(ds)==952
    reference={r['pair_id']:r for r in json.loads(Path('reports/t05_results_v1/K256_validation.json').read_text())['pairs']}
    model=UniformB0(cfg,k=256).cuda().eval();model.load_state_dict(saved['model'],strict=True);audit=InferenceStateAudit(model);audit.check();records=[];max_bf16_readout_delta=0.
    with torch.inference_mode():
        for index,item in enumerate(ds):
            args=[item[key][None].cuda() for key in ['optical','sar_aug','optical_valid_mask','sar_aug_valid_mask']]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(*args);source,centers,confidence=model.coarse_matches(output,0)
                local=model.local(output['optical_fine'],output['sar_fine'],source,centers,args[2],args[3])
                native=model.refine(output,args,0)
            physical,tokens,source_valid=sample_actual_support(output['optical_fine'],output['sar_fine'],source,centers,args[2],args[3],model.local.radius)
            local_valid=local.get('valid',torch.ones(len(source),dtype=torch.bool,device='cuda'));assert torch.equal(local_valid,tokens.any(-1)&source_valid)
            assert all(torch.equal(native[key],value[local_valid]) for key,value in [('source',source),('target',local['target']),('score',confidence),('coarse_target',centers)])
            # Coarse IDs reconstructed from original coordinates, then checked against full query IDs.
            qid=torch.round((source[:,1]-3.5)/8).long()*32+torch.round((source[:,0]-3.5)/8).long()
            tid=torch.round((centers[:,1]-3.5)/8).long()*32+torch.round((centers[:,0]-3.5)/8).long()
            assert torch.equal(coarse_xy(qid,32),source) and torch.equal(coarse_xy(tid,32),centers)
            assert torch.isin(qid,output['query_indices']).all() and len(torch.unique(qid))==len(qid)
            src,dst,score,center,pos,tokenmask,sourcevalid,localmask,ids=[x.detach().cpu().numpy() for x in [source,local['target'],confidence,centers,physical,tokens,source_valid,local_valid,qid*1024+tid]]
            estimator_mask=frozen_fit_mask(src,dst,score,cfg['evaluation']['minimum_match_score']);fit_mask=localmask&estimator_mask
            fit_src,fit_dst,fit_score=src[fit_mask],dst[fit_mask],score[fit_mask]
            fit=estimate_from_correspondences(fit_src,fit_dst,fit_score,cfg['evaluation'],seed=cfg['seed'],include_inlier_indices=True)
            orig=estimate_from_correspondences(native['source'].float().cpu().numpy(),native['target'].float().cpu().numpy(),native['score'].float().cpu().numpy(),cfg['evaluation'],seed=cfg['seed'],include_inlier_indices=True)
            assert fit==orig,'Prefit freeze altered original fit'
            # GT is first accessed after native prediction/filter/fit have been saved.
            A=item['A_forward'].numpy();truth=transform_points(src,A)
            target_valid=mask_valid_at(truth,args[3][0]);actual_source_valid=mask_valid_at(src,args[2][0])
            hit=support_hits(truth,pos,tokenmask,sourcevalid&actual_source_valid,target_valid,p['numeric_tolerance_px'])
            legacy_box=target_valid&actual_source_valid&((np.abs(truth-center)<=6+1e-6).all(1))
            probes,probe_truth=fixed_synthetic_probes(A,grid=cfg['evaluation']['probe_grid'],optical_mask=item['optical_valid_mask'],sar_aug_mask=item['sar_aug_valid_mask'])
            ev=evaluate_transform(fit['matrix'],probes,probe_truth,cfg['evaluation']['success_thresholds_px'],status=fit['status'])
            ref=reference[item['pair_id']];assert fit['matrix']==ref['estimation']['matrix'] and ev==ref['evaluation'],('Baseline regression',item['pair_id'])
            # Mathematical continuous readout vs finite BF16 arithmetic is reported, never used to enlarge hit support.
            if len(src):
                mathematical=(local['heatmap'].double()[:,:,None]*physical.double()).sum(1).cpu().numpy()
                max_bf16_readout_delta=max(max_bf16_readout_delta,float(np.max(np.abs(mathematical-dst))))
            file=cache/f'{index:04d}.npz'
            np.savez_compressed(file,pair_id=np.array(item['pair_id']),candidate_id=ids,query_indices=output['query_indices'].cpu().numpy(),source=src,target=dst,confidence=score,
                coarse_center=center,window_positions=pos,window_token_valid=tokenmask,source_token_valid=sourcevalid,local_valid=localmask,original_estimator_mask=estimator_mask,
                frozen_fit_mask=fit_mask,nominal_truth=truth,nominal_target_valid=target_valid,actual_source_valid=actual_source_valid,hit=hit,legacy_box_hit=legacy_box,
                A_forward=A,probes=probes,probe_truth=probe_truth,local_heatmap=local['heatmap'].float().cpu().numpy())
            records.append(dict(index=index,pair_id=item['pair_id'],path=str(file),sha256=sha(file),refine_candidates=len(src),fit_candidates=int(fit_mask.sum()),
                refine_hits=int(hit.sum()),fit_hits=int(hit[fit_mask].sum()),fit_packet_sha256=array_digest(ids[fit_mask],fit_src,fit_score,fit_mask),
                original_target_sha256=array_digest(fit_dst),baseline_exact=True,source_unchanged_by_refiner=True,
                box_hits_fit=int(legacy_box[fit_mask].sum()),box_only_hits_fit=int((legacy_box&~hit&fit_mask).sum()),actual_support_only_fit=int((hit&~legacy_box&fit_mask).sum()),
                probes_and_fit_source_disjoint=not any(np.array_equal(x,y) for x in fit_src for y in probes)))
            if (index+1)%100==0:print('cached',index+1,'/',952,flush=True)
    state=audit.finish();assert all(r['probes_and_fit_source_disjoint'] for r in records)
    (out/'cache_freeze.json').write_text(json.dumps(dict(passed=True,protocol=p,config=cfg,input_sha256=input_hashes,pairs=records,state_audit=state,
        original_T05_exact_pairs=952,probe_lattice_disjoint_from_fit_sources=True,max_finite_BF16_readout_vs_continuous_heatmap_expectation_px=max_bf16_readout_delta,
        note='Closed convex hull is continuous heatmap idealization; finite BF16 rounding is recorded separately and never expands hit support.',GT_first_access_after_original_fit=True),indent=2))
    print('Cache and originalT05 regression passed952',flush=True)


if __name__=='__main__':main()
