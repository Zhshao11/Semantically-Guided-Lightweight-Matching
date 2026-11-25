"""Full fixed validation plus preselected layered diagnostics, no training."""
import argparse
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import random
import time
import numpy as np
import torch
from sglm.t02.data import AffinePairs256
from sglm.t02.model import B0Full
from sglm.t05 import UniformB0
from sglm.diagnostics import InferenceStateAudit,diagnose_pair,aggregate_diagnostics
from sglm.evaluation import estimate_from_correspondences,fixed_synthetic_probes,evaluate_transform,summarize_evaluations
from sglm.experiment_io import flatten_diagnostic,write_csv


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--frozen',default='reports/t05_frozen_v1');ap.add_argument('--output',default='reports/t05_results_v1')
    a=ap.parse_args();frozen=Path(a.frozen);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    freeze=json.loads((frozen/'freeze.json').read_text());protocol=freeze['protocol'];subsets=json.loads((frozen/'subsets.json').read_text())
    assert sha(freeze['frozen_checkpoint'])==freeze['checkpoint_sha256']
    assert sha(frozen/'validation_plan.json')==freeze['validation_plan_sha256'] and sha(frozen/'subsets.json')==freeze['subsets_sha256']
    saved=torch.load(freeze['frozen_checkpoint'],map_location='cpu',weights_only=False);cfg=saved['config']
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    ds=AffinePairs256(a.root,frozen/'validation_plan.json',preload=True,split='val');assert len(ds)==952
    original={r['pair_id']:r for r in json.loads(Path('runs/t03_main_lr1e3_v1/validation_epoch_018.json').read_text())['pairs']}
    model=UniformB0(cfg).cuda().eval();model.load_state_dict(saved['model'],strict=True)
    base=B0Full(cfg).cuda().eval();base.load_state_dict(saved['model'],strict=True)
    summaries=[];all_layered=[];reproduction=[]
    with torch.inference_mode():
        for k in protocol['budgets']:
            model.k=k;records=[];diagnostics=[];audit=InferenceStateAudit(model);audit.check();begun=time.monotonic()
            for index,item in enumerate(ds):
                args=[item[key][None].cuda() for key in ('optical','sar_aug','optical_valid_mask','sar_aug_valid_mask')]
                with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model.predict(*args)[0]
                arrays=[prediction[key].float().cpu().numpy() for key in ('source','target','score')]
                fit=estimate_from_correspondences(*arrays,cfg['evaluation'],seed=cfg['seed'])
                probes,truth=fixed_synthetic_probes(item['A_forward'].numpy(),grid=cfg['evaluation']['probe_grid'],
                    optical_mask=item['optical_valid_mask'],sar_aug_mask=item['sar_aug_valid_mask'])
                evaluation=evaluate_transform(fit['matrix'],probes,truth,cfg['evaluation']['success_thresholds_px'],status=fit['status'])
                record=dict(pair_id=item['pair_id'],K=k,attempted_refinements=prediction['attempted_refinements'],
                    final_correspondences=len(arrays[0]),estimation=fit,evaluation=evaluation)
                records.append(record)
                if k==1024:
                    reference=original[item['pair_id']]
                    with torch.autocast('cuda',dtype=torch.bfloat16):base_prediction=base.predict(*args)[0]
                    equal_predictions=all(torch.equal(prediction[key],base_prediction[key]) for key in ('source','target','score','coarse_target'))
                    new,old=evaluation['rmse_px'],reference['evaluation']['rmse_px']
                    delta=abs(new-old) if new is not None and old is not None else 0 if new==old else None
                    reproduction.append(dict(pair_id=item['pair_id'],original_B0_predictions_bitwise_equal=equal_predictions,
                        matrix_exact=fit['matrix']==reference['estimation']['matrix'],RMSE_delta_px=delta,
                        success_flags_equal=evaluation['success']==reference['evaluation']['success']))
                if index in subsets['diagnostic_indices']:
                    d=diagnose_pair(model,item,cfg,protocol)
                    d.update(K=k,available_optical_queries=d['coarse']['valid_optical_queries'])
                    d['coarse']['valid_optical_queries']=k
                    d['coarse']['query_budget']=k
                    assert d['evaluation']==evaluation
                    # Post-hoc compare geometric fits of identical coarse/fine source sets.
                    p=d['predictions'];coarse_fit=estimate_from_correspondences(p['source'],p['coarse_target'],p['scores'],cfg['evaluation'],seed=cfg['seed'])
                    d['coarse_transform_evaluation']=evaluate_transform(coarse_fit['matrix'],probes,truth,cfg['evaluation']['success_thresholds_px'],status=coarse_fit['status'])
                    diagnostics.append(d);all_layered.append(dict(K=k,**flatten_diagnostic(d)))
                if (index+1)%100==0:
                    (out/'status.json').write_text(json.dumps(dict(phase='accuracy',K=k,pairs_done=index+1,total=952)))
                    print('evaluated',k,index+1,flush=True)
            state=audit.finish()
            errors=[r['evaluation']['rmse_px'] for r in records if r['evaluation']['solved']]
            summary=dict(K=k,**summarize_evaluations([r['evaluation'] for r in records]),
                solved_p90_rmse_px=float(np.percentile(errors,90)) if errors else None,
                mean_attempted_refinements=float(np.mean([r['attempted_refinements'] for r in records])),
                median_attempted_refinements=float(np.median([r['attempted_refinements'] for r in records])),
                mean_final_correspondences=float(np.mean([r['final_correspondences'] for r in records])),
                accuracy_pass_wall_seconds=time.monotonic()-begun,formal_timing=False)
            groups={}
            for label,flag in [('success@5px',True),('failure@5px',False)]:
                group=[d for d in diagnostics if d['evaluation']['success']['5']==flag]
                if not group:groups[label]=dict(pairs=0);continue
                agg=aggregate_diagnostics(group)
                agg.update(pairs=len(group),median_inliers=float(np.median([d['geometry']['estimation']['inliers'] for d in group])),
                    median_inlier_hull_fraction=float(np.median([d['geometry']['inlier_source_coverage']['hull_area_fraction'] for d in group])),
                    coarse_transform_SR5=float(np.mean([d['coarse_transform_evaluation']['success']['5'] for d in group])))
                groups[label]=agg
            (out/f'K{k}_validation.json').write_text(json.dumps(dict(summary=summary,state_audit=state,pairs=records),indent=2))
            (out/f'K{k}_diagnostics.json').write_text(json.dumps(dict(summary=aggregate_diagnostics(diagnostics),success_failure=groups,pairs=diagnostics),indent=2))
            summaries.append(summary)
            if k==1024:
                passed=all(r['original_B0_predictions_bitwise_equal'] and r['matrix_exact'] and r['success_flags_equal'] and r['RMSE_delta_px']==0 for r in reproduction)
                (out/'candidate_reproduction.json').write_text(json.dumps(dict(passed=passed,reference_epoch=18,pairs=reproduction),indent=2))
                if not passed:raise AssertionError('Candidate or dense-wrapper reproduction failed; stop before sparse comparison')
    write_csv(out/'accuracy_summary.csv',summaries);write_csv(out/'layered_diagnostics.csv',all_layered)
    (out/'accuracy_summary.json').write_text(json.dumps(dict(checkpoint_sha256=freeze['checkpoint_sha256'],summaries=summaries,
        runtime_source_sha256={p:sha(p) for p in ['sglm/t05.py','sglm/t02/model.py','sglm/model.py','sglm/evaluation.py','sglm/diagnostics.py','scripts/evaluate_t05.py']},
        subsets_sha256=freeze['subsets_sha256'],test_used=False,training_performed=False),indent=2))
    (out/'status.json').write_text(json.dumps(dict(phase='accuracy_completed',budgets=protocol['budgets'])))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
