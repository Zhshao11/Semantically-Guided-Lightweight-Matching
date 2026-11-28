"""All952 fixed validation pairs. Native dense and common affine are distinct protocols."""
import argparse,hashlib,json,os,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np,torch
from sglm.sar_references import SARReference
from sglm.sar_reference_geometry import evaluate_dense_probes
from sglm.external_reference import cpu_input
from sglm.t02.data import AffinePairs256
from sglm.evaluation import estimate_from_correspondences,fixed_synthetic_probes,evaluate_transform,summarize_evaluations
from sglm.diagnostics import bn_snapshot,bn_unchanged
from sglm.experiment_io import write_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--method',required=True);a=ap.parse_args()
    p=json.loads(Path('configs/sar_references_v1.json').read_text());out=Path(p['output']);freeze=json.loads((out/'freeze.json').read_text())
    for path,digest in freeze['input_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    for name,path in [(a.method,out/f'{a.method}_smoke.json')]:assert json.loads(path.read_text())['passed']
    cfg=json.loads(Path(p['frozen']+'/candidate_config.json').read_text());torch.set_num_threads(p['timing']['threads']);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    ds=AffinePairs256(a.root,p['frozen']+'/validation_plan.json',split='val',preload=True);assert len(ds)==952
    model=SARReference(a.method,p,cfg);before=bn_snapshot(model.model);records=[];start=time.monotonic();native={key:[] for key in ['raw_dense','native32_LSR']}
    original={}
    if a.method.startswith('T05'):
        K=1024 if a.method=='T05_Full' else 256
        original={r['pair_id']:r for r in json.loads(Path(f'reports/t05_results_v1/K{K}_validation.json').read_text())['pairs']}
    save_indices=set(json.loads(Path(p['frozen']+'/subsets.json').read_text())['timing_indices']);private=Path(p['source_root'])/'predictions'/a.method;private.mkdir(parents=True,exist_ok=True)
    for index,item in enumerate(ds):
        pred=model.predict(cpu_input(item),include_native_dense=a.method=='GDROS_WHU')
        fit=estimate_from_correspondences(pred['source'],pred['target'],pred['score'],cfg['evaluation'],seed=cfg['seed'])
        # NoGT until all model prediction, filtering and fitting have finished.
        probes,truth=fixed_synthetic_probes(item['A_forward'].numpy(),grid=cfg['evaluation']['probe_grid'],optical_mask=item['optical_valid_mask'],sar_aug_mask=item['sar_aug_valid_mask'])
        ev=evaluate_transform(fit['matrix'],probes,truth,cfg['evaluation']['success_thresholds_px'],status=fit['status'])
        row=dict(pair_id=item['pair_id'],method=a.method,raw_matches=pred['raw_matches'],valid_matches=pred['valid_matches'],removed_invalid=pred['removed_invalid'],fit_status=fit['status'],
            inliers=fit['inliers'],probe_rmse_px=ev['rmse_px'],success3=ev['success']['3'],success5=ev['success']['5'],evaluation=ev,estimation=fit)
        if original:
            ref=original[item['pair_id']];assert fit['matrix']==ref['estimation']['matrix'] and ev==ref['evaluation'],(a.method,item['pair_id'],'T05reproduction')
        if a.method=='GDROS_WHU':
            for label,key in [('raw_dense','raw_flow'),('native32_LSR','native_lsr_flow')]:
                evaluation=evaluate_dense_probes(pred[key],probes,truth,cfg['evaluation']['success_thresholds_px'])
                native[label].append(dict(pair_id=item['pair_id'],evaluation=evaluation));row[label+'_rmse_px']=evaluation['rmse_px']
        records.append(row)
        if index in save_indices:np.savez_compressed(private/f'{index:04d}.npz',source=pred['source'],target=pred['target'],score=pred['score'],probes=probes,truth=truth)
        if (index+1)%25==0:
            print(a.method,index+1,'/',952,'seconds',round(time.monotonic()-start,1),flush=True)
            (out/f'{a.method}_status.json').write_text(json.dumps(dict(phase='accuracy',done=index+1,total=952)))
    assert bn_unchanged(before,model.model) and not any(m.training for m in model.model.modules())
    errors=[r['probe_rmse_px'] for r in records if r['evaluation']['solved']]
    summary=dict(method=a.method,**summarize_evaluations([r['evaluation'] for r in records]),solved_p90_rmse_px=float(np.percentile(errors,90)) if errors else None,
        registration_failures5=sum(not r['success5'] for r in records),mean_raw_matches=float(np.mean([r['raw_matches'] for r in records])),mean_valid_matches=float(np.mean([r['valid_matches'] for r in records])),
        parameters=model.parameters,all_modules_eval=True,BN_unchanged=True,accuracy_wall_seconds=time.monotonic()-start,formal_timing=False,original_T05_exact_pairs=952 if original else None,
        protocol='predicted correspondences -> original full affineRANSAC -> original independent fixed probes',no_training=True,test_used=False)
    (out/f'{a.method}_accuracy.json').write_text(json.dumps(summary,indent=2));(Path(p['source_root'])/f'{a.method}_validation.json').write_text(json.dumps(dict(summary=summary,pairs=records),indent=2))
    write_csv(out/f'{a.method}_per_pair.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in records])
    for label,values in native.items():
        if not values:continue
        errors=[r['evaluation']['rmse_px'] for r in values if r['evaluation']['solved']]
        secondary=dict(method='GDROS_WHU',native_protocol=label,**summarize_evaluations([r['evaluation'] for r in values]),solved_p90_rmse_px=float(np.percentile(errors,90)) if errors else None,
            fixed_independent_probes=True,common_affine_RANSAC=False,not_official_repository_variance_metric=True,main_frontier=False)
        (out/f'GDROS_WHU_{label}_accuracy.json').write_text(json.dumps(secondary,indent=2));write_csv(out/f'GDROS_WHU_{label}_per_pair.csv',[dict(pair_id=r['pair_id'],rmse_px=r['evaluation']['rmse_px'],success3=r['evaluation']['success']['3'],success5=r['evaluation']['success']['5']) for r in values])
    (out/f'{a.method}_status.json').write_text(json.dumps(dict(phase='accuracy_completed',pairs=952)))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
