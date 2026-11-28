"""Post-hoc sampling diagnostic. Primary seed remains fixed; never choose a best seed."""
import hashlib,json
from pathlib import Path
import numpy as np
from sglm.window_oracle import replace_targets
from sglm.geometry import transform_points
from sglm.evaluation import estimate_from_correspondences,evaluate_transform,summarize_evaluations
from sglm.experiment_io import write_csv


def evaluate(c,cfg,seed):
    mask=c['frozen_fit_mask'];src=c['source'][mask];dst=c['target'][mask];truth=c['nominal_truth'][mask];hit=c['hit'][mask];score=c['confidence'][mask]
    outputs={}
    for group,target in [('base',dst),('oracle',replace_targets(dst,truth,hit))]:
        fit=estimate_from_correspondences(src,target,score,cfg['evaluation'],seed=seed,include_inlier_indices=True)
        ev=evaluate_transform(fit['matrix'],c['probes'],c['probe_truth'],cfg['evaluation']['success_thresholds_px'],status=fit['status'])
        outputs[group]=dict(fit=fit,evaluation=ev,nominal_exact_transform_consensus_under_original_threshold=int((np.linalg.norm(target-truth,axis=1)<=cfg['evaluation']['ransac_threshold_px']).sum()))
    return outputs


def main():
    p=json.loads(Path('configs/window_oracle_v1.json').read_text());out=Path(p['output']);freeze=json.loads((out/'cache_freeze.json').read_text());cfg=freeze['config'];primary=json.loads((out/'results.json').read_text())
    regressed=[r for r in primary['pairs'] if r['branches']['original_prediction']['evaluation']['success']['5'] and not r['branches']['inside_actual_support_nominalGT_target_only']['evaluation']['success']['5']]
    # Inspect all regressed primary5px pairs, no choosing successful probe seeds.
    probe_seeds=[cfg['seed'],cfg['seed']+1,cfg['seed']+2];probe=[]
    for r in regressed:
        c=np.load(Path(p['cache'])/f"{r['index']:04d}.npz")
        for seed in probe_seeds:probe.append(dict(pair_id=r['pair_id'],seed=seed,branches=evaluate(c,cfg,seed)))
    sensitive=any(len({round(x['branches']['oracle']['evaluation']['rmse_px'],6) if x['branches']['oracle']['evaluation']['rmse_px'] is not None else None for x in probe if x['pair_id']==r['pair_id']})>1 for r in regressed)
    record=dict(trigger='Primary5px success->failure despite37nominal-correct replacements and broad hit-source support; probe original seed and nexttwo fixed seeds for all primary regressed pairs.',
        primary_seed=cfg['seed'],predeclared_probe_seeds=probe_seeds,probe_results=probe,probe_numerical_seed_sensitivity=sensitive,primary_result_unchanged=True,no_seed_selection=True,no_network_or_training=True)
    if sensitive:
        runs=[];summaries=[];changes=[]
        for seed in probe_seeds:
            evs={'base':[],'oracle':[]}
            for row in freeze['pairs']:
                assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'];c=np.load(row['path']);r=evaluate(c,cfg,seed)
                for group in evs:evs[group].append(r[group]['evaluation'])
                runs.append(dict(seed=seed,pair_id=row['pair_id'],base_rmse_px=r['base']['evaluation']['rmse_px'],oracle_rmse_px=r['oracle']['evaluation']['rmse_px'],
                    base_SR3=r['base']['evaluation']['success']['3'],oracle_SR3=r['oracle']['evaluation']['success']['3'],base_SR5=r['base']['evaluation']['success']['5'],oracle_SR5=r['oracle']['evaluation']['success']['5']))
            for group,values in evs.items():
                errors=[v['rmse_px'] for v in values if v['solved']];summaries.append(dict(seed=seed,branch=group,**summarize_evaluations(values),P90_RMSE_px=float(np.percentile(errors,90)) if errors else None))
            for threshold in ['3','5']:
                a=np.array([e['success'][threshold] for e in evs['base']]);b=np.array([e['success'][threshold] for e in evs['oracle']]);saved=int((~a&b).sum());lost=int((a&~b).sum())
                changes.append(dict(seed=seed,threshold_px=int(threshold),failure_to_success=saved,success_to_failure=lost,original_failures=int((~a).sum()),failure_rescue_rate=saved/int((~a).sum()),delta_SR=(saved-lost)/952))
        write_csv(out/'ransac_seed_per_pair.csv',runs);write_csv(out/'ransac_seed_summary.csv',summaries);write_csv(out/'ransac_seed_changes.csv',changes)
        record.update(all952_seeds=probe_seeds,full_summaries=summaries,full_paired_changes=changes)
    (out/'ransac_sensitivity.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
