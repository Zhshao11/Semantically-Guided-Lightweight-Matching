"""Frozen952 synthetic-affine evaluation of unmodified official zero-shot matchers."""
import argparse,hashlib,json,os,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from sglm.external_reference import OfficialReference,cpu_input
from sglm.t02.data import AffinePairs256
from sglm.evaluation import estimate_from_correspondences,fixed_synthetic_probes,evaluate_transform,summarize_evaluations
from sglm.diagnostics import bn_snapshot,bn_unchanged
from sglm.experiment_io import write_csv


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--method',required=True);a=ap.parse_args()
    p=json.loads(Path('configs/external_references_v1.json').read_text());out=Path(p['output']);freeze=json.loads((out/'freeze.json').read_text())
    for path,digest in freeze['input_sha256'].items():assert sha(path)==digest,path
    assert sha(freeze['checkpoint'])==freeze['checkpoint_sha256']
    assert json.loads((out/f'{a.method}_smoke.json').read_text())['passed']
    cfg=json.loads(Path(p['frozen']+'/candidate_config.json').read_text())
    torch.set_num_threads(p['timing']['threads']);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    ds=AffinePairs256(a.root,p['frozen']+'/validation_plan.json',split='val',preload=True);assert len(ds)==952
    adapter=OfficialReference(a.method,p,cfg);before=bn_snapshot(adapter.model)
    original={r['pair_id']:r for r in json.loads(Path('reports/t05_results_v1/K1024_validation.json').read_text())['pairs']}
    private=Path(p['source_root'])/'predictions'/a.method;private.mkdir(parents=True,exist_ok=True)
    records=[];begun=time.monotonic();exact_original=0
    with torch.inference_mode():
        for i,item in enumerate(ds):
            pred=adapter.predict(cpu_input(item))
            fit=estimate_from_correspondences(pred['source'],pred['target'],pred['score'],cfg['evaluation'],seed=cfg['seed'],include_inlier_indices=True)
            # GT is accessed only after prediction and fit, through independent fixed probes.
            probes,truth=fixed_synthetic_probes(item['A_forward'].numpy(),grid=cfg['evaluation']['probe_grid'],
                optical_mask=item['optical_valid_mask'],sar_aug_mask=item['sar_aug_valid_mask'])
            ev=evaluate_transform(fit['matrix'],probes,truth,cfg['evaluation']['success_thresholds_px'],status=fit['status'])
            record=dict(pair_id=item['pair_id'],method=a.method,raw_matches=pred['raw_matches'],valid_matches=pred['valid_matches'],
                removed_invalid=pred['removed_invalid'],fit_success=fit['success'],fit_status=fit['status'],inliers=fit['inliers'],
                probe_rmse_px=ev['rmse_px'],success3=ev['success']['3'],success5=ev['success']['5'],evaluation=ev,estimation=fit)
            if a.method=='T05':
                ref=original[item['pair_id']];same=fit['matrix']==ref['estimation']['matrix'] and ev==ref['evaluation']
                record['original_T05_exact']=same;exact_original+=int(same)
            records.append(record)
            np.savez_compressed(private/f'{i:04d}.npz',source=pred['source'],target=pred['target'],score=pred['score'],probes=probes,truth=truth)
            if (i+1)%20==0:
                print(a.method,i+1,952,'elapsed',round(time.monotonic()-begun,1),flush=True)
                (out/f'{a.method}_status.json').write_text(json.dumps(dict(phase='accuracy',done=i+1,total=952)))
        assert bn_unchanged(before,adapter.model)
    errors=[r['probe_rmse_px'] for r in records if r['evaluation']['solved']]
    summary=dict(method=a.method,**summarize_evaluations([r['evaluation'] for r in records]),
        solved_p90_rmse_px=float(np.percentile(errors,90)) if errors else None,
        registration_failures5=sum(not r['success5'] for r in records),mean_raw_matches=float(np.mean([r['raw_matches'] for r in records])),
        mean_valid_matches=float(np.mean([r['valid_matches'] for r in records])),removed_invalid_total=sum(r['removed_invalid'] for r in records),
        parameters_registered=adapter.model_parameters,parameters_dino_hidden=adapter.hidden_dino_parameters,
        actual_parameters_total=adapter.model_parameters+adapter.hidden_dino_parameters,BN_unchanged=True,
        accuracy_wall_seconds=time.monotonic()-begun,formal_timing=False,
        original_T05_exact_pairs=exact_original if a.method=='T05' else None)
    if a.method=='T05':assert exact_original==952,'Common boundary/filter changed T05; diagnose before comparison'
    (Path(p['source_root'])/f'{a.method}_validation.json').write_text(json.dumps(dict(summary=summary,pairs=records),indent=2))
    (out/f'{a.method}_accuracy.json').write_text(json.dumps(summary,indent=2))
    write_csv(out/f'{a.method}_per_pair.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in records])
    (out/f'{a.method}_status.json').write_text(json.dumps(dict(phase='accuracy_completed',pairs=952)))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
