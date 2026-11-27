"""Frozen 952-pair evaluation. GT is used only by independent post-hoc diagnostics."""
import argparse,hashlib,json,os,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from sglm.local_interaction import build_local_group
from sglm.t02.data import AffinePairs256
from sglm.diagnostics import diagnose_pair,aggregate_diagnostics,InferenceStateAudit,errors_summary
from sglm.evaluation import summarize_evaluations
from sglm.experiment_io import write_csv,flatten_diagnostic


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    protocol=json.loads(Path('configs/local_interaction_v1.json').read_text());out=Path('reports/local_interaction_v1');run=Path('runs/local_interaction_v1')
    assert json.loads((run/'training_completed.json').read_text())['passed']
    cfg=torch.load(protocol['checkpoint'],map_location='cpu',weights_only=False)['config']
    diag=json.loads(Path('reports/t05_frozen_v1/freeze.json').read_text())['protocol']
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    ds=AffinePairs256(a.root,'reports/t05_frozen_v1/validation_plan.json',split='val',preload=True);assert len(ds)==952
    reference={r['pair_id']:r for r in json.loads(Path('reports/t05_results_v1/K256_validation.json').read_text())['pairs']}
    candidate_hashes={};summaries=[];reproduction=[];results={};flat=[];checkpoint_shas={}
    for group in ['untrained_T05',*protocol['groups']]:
        path=Path(protocol['checkpoint']) if group=='untrained_T05' else run/f'{group}_last.pt'
        checkpoint_shas[group]=sha(path);saved=torch.load(path,map_location='cpu',weights_only=False)
        if group!='untrained_T05':assert saved['updates']==7912 and saved['epoch']==4 and saved['group']==group
        model=build_local_group(cfg,saved['model'],'original_refiner' if group=='untrained_T05' else group)
        audit=InferenceStateAudit(model);audit.check();records=[];pooled={key:[] for key in ['before_inside','after_inside','before_outside','after_outside']}
        for index,item in enumerate(ds):
            d=diagnose_pair(model,item,cfg,diag)
            pred=d['predictions'];h=hashlib.sha256()
            for key in ['source','coarse_target','scores']:h.update(np.asarray(pred[key],np.float32).tobytes())
            signature=(h.hexdigest(),d['coarse']['predictions_before_local_mask'],d['coarse']['predicted_correspondences'])
            if group=='untrained_T05':candidate_hashes[item['pair_id']]=signature
            else:assert candidate_hashes[item['pair_id']]==signature,(group,item['pair_id'])
            if group=='untrained_T05':
                r=reference[item['pair_id']]
                reproduction.append(dict(pair_id=item['pair_id'],matrix_exact=d['geometry']['estimation']['matrix']==r['estimation']['matrix'],
                    evaluation_exact=d['evaluation']==r['evaluation']))
            # Reconstruct per-correspondence membership solely for pooled post-hoc statistics.
            src=np.asarray(pred['source']).reshape(-1,2);coarse=np.asarray(pred['coarse_target']).reshape(-1,2);fine=np.asarray(pred['target']).reshape(-1,2)
            from sglm.geometry import transform_points
            from sglm.evaluation import _mask_at
            truth=transform_points(src,item['A_forward'].numpy());visible=((truth>=0)&(truth<=255)).all(1)&_mask_at(truth,item['sar_aug_valid_mask'])
            inside=visible&(np.abs(truth-coarse)<=6+1e-6).all(1)
            before=np.linalg.norm(coarse-truth,axis=1);after=np.linalg.norm(fine-truth,axis=1)
            for key,v in [('before_inside',before[inside]),('after_inside',after[inside]),('before_outside',before[~inside]),('after_outside',after[~inside])]:pooled[key].extend(v.tolist())
            # Private correspondence arrays are saved separately, never published.
            private=run/f'{group}_predictions';private.mkdir(exist_ok=True)
            np.savez_compressed(private/f'{index:04d}.npz',source=src,coarse=coarse,target=fine,inside=inside)
            del d['predictions'];records.append(d);flat.append(dict(group=group,**flatten_diagnostic(d)))
            if (index+1)%100==0:print(group,index+1,flush=True)
        state=audit.finish();errs=[r['evaluation']['rmse_px'] for r in records if r['evaluation']['solved']]
        summary=dict(group=group,**summarize_evaluations([r['evaluation'] for r in records]),solved_p90_rmse_px=float(np.percentile(errs,90)),
            parameters=sum(p.numel() for p in model.parameters()),refiner_parameters=sum(p.numel() for p in model.local.parameters()),
            mean_refinements=float(np.mean([r['coarse']['predictions_before_local_mask'] for r in records])),
            mean_final_correspondences=float(np.mean([r['coarse']['predicted_correspondences'] for r in records])),
            **{key:errors_summary(value) for key,value in pooled.items()},pair_macro=aggregate_diagnostics(records))
        summaries.append(summary);results[group]=records
        (out/f'{group}_validation.json').write_text(json.dumps(dict(summary=summary,state_audit=state,pairs=records),indent=2));del model;torch.cuda.empty_cache()
    assert all(r['matrix_exact'] and r['evaluation_exact'] for r in reproduction),'Untrained T05 regression failed'
    changes=[]
    for threshold in ['3','5']:
        control=[r['evaluation']['success'][threshold] for r in results['original_refiner']];experiment=[r['evaluation']['success'][threshold] for r in results['local_cross_attention']]
        changes.append(dict(threshold_px=int(threshold),failure_to_success=sum(not x and y for x,y in zip(control,experiment)),success_to_failure=sum(x and not y for x,y in zip(control,experiment)),both_success=sum(x and y for x,y in zip(control,experiment)),both_failure=sum(not x and not y for x,y in zip(control,experiment))))
    write_csv(out/'accuracy_summary.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in summaries]);write_csv(out/'layered_diagnostics.csv',flat);write_csv(out/'paired_outcomes.csv',changes)
    (out/'evaluation_acceptance.json').write_text(json.dumps(dict(passed=True,checkpoint_sha256=checkpoint_shas,untrained_T05_exact_reproduction=all(r['matrix_exact'] and r['evaluation_exact'] for r in reproduction),
        same_coarse_packets_and_refinement_counts_all_952=True,all_eval_and_BN_unchanged=True,GT_posthoc_only=True,test_used=False,
        validation_plan_sha256=sha('reports/t05_frozen_v1/validation_plan.json'),source_sha256={p:sha(p) for p in ['scripts/evaluate_local_refinement.py','sglm/local_interaction.py','sglm/diagnostics.py','sglm/evaluation.py']},summaries=summaries,paired_changes=changes),indent=2))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
