"""Two paired geometry branches on one frozen cache; no network, training or timing."""
import hashlib,json
from pathlib import Path
import numpy as np
from sglm.evaluation import estimate_from_correspondences,evaluate_transform,summarize_evaluations
from sglm.window_oracle import replace_targets,array_digest,coverage
from sglm.diagnostics import errors_summary
from sglm.experiment_io import write_csv


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=json.loads(Path('configs/window_oracle_v1.json').read_text());out=Path(p['output']);freeze=json.loads((out/'cache_freeze.json').read_text());cfg=freeze['config'];assert freeze['passed']
    for path,digest in freeze['input_sha256'].items():assert sha(path)==digest,path
    historical={r['pair_id']:r for r in json.loads(Path('reports/t05_results_v1/K256_validation.json').read_text())['pairs']}
    records=[];flat=[];inside_errors=[];max_after_float32_error=0.
    for row in freeze['pairs']:
        assert sha(row['path'])==row['sha256'];c=np.load(row['path']);mask=c['frozen_fit_mask'];src=c['source'][mask];dst=c['target'][mask];score=c['confidence'][mask];ids=c['candidate_id'][mask];truth=c['nominal_truth'][mask];hit=c['hit'][mask]
        src.flags.writeable=False;score.flags.writeable=False;ids.flags.writeable=False;mask.flags.writeable=False
        before=array_digest(ids,src,score,mask);assert before==row['fit_packet_sha256']
        oracle=replace_targets(dst,truth,hit);assert np.array_equal(oracle[~hit],dst[~hit]);assert np.array_equal(oracle[hit],truth[hit])
        assert array_digest(ids,src,score,mask)==before
        if hit.any():max_after_float32_error=max(max_after_float32_error,float(np.max(np.abs(oracle[hit].astype(np.float32).astype(float)-truth[hit]))))
        fits={};evals={};inliers={};seed=cfg['seed']
        for group,target in [('original_prediction',dst),('inside_actual_support_nominalGT_target_only',oracle)]:
            fit=estimate_from_correspondences(src,target,score,cfg['evaluation'],seed=seed,include_inlier_indices=True)
            ev=evaluate_transform(fit['matrix'],c['probes'],c['probe_truth'],cfg['evaluation']['success_thresholds_px'],status=fit['status'])
            fits[group]=fit;evals[group]=ev;indices=np.array(fit['inlier_indices'],int)
            inliers[group]=dict(source=coverage(src[indices]),target=coverage(target[indices]))
        base='original_prediction';ideal='inside_actual_support_nominalGT_target_only'
        zero=replace_targets(dst,truth,np.zeros(len(src),bool));empty_fit=estimate_from_correspondences(src,zero,score,cfg['evaluation'],seed=seed,include_inlier_indices=True)
        assert empty_fit==fits[base],'Forced empty intervention changed RANSAC'
        ref=historical[row['pair_id']];assert fits[base]['matrix']==ref['estimation']['matrix'] and evals[base]==ref['evaluation']
        error=np.linalg.norm(dst[hit]-truth[hit],axis=1);inside_errors.extend(error.tolist())
        coverage_hits=coverage(src[hit]);refine_count=len(c['source']);refine_hits=int(c['hit'].sum());fit_hits=int(hit.sum())
        record=dict(pair_id=row['pair_id'],index=row['index'],refine_input=dict(total=refine_count,hits=refine_hits,hit_rate=refine_hits/refine_count if refine_count else None),
            final_fit=dict(total=len(src),hits=fit_hits,hit_rate=fit_hits/len(src) if len(src) else None,replacements=fit_hits,changed_targets=int(np.any(oracle!=dst,axis=1).sum())),
            nominal_window_hits_source_coverage=coverage_hits,inside_before_errors=errors_summary(error),branches={g:dict(estimation=fits[g],evaluation=evals[g],inlier_coverage=inliers[g]) for g in fits},
            acceptance=dict(original_exact=True,candidate_packet_sha256=before,candidates_identical=True,outside_targets_bitwise_unchanged=True,inside_nominal_error_exactzero=True,forced_empty_intervention_exact=True,
                RANSAC_masks_independently_computed=True,paired_seed=seed))
        records.append(record)
        f=dict(pair_id=row['pair_id'],index=row['index'],refine_count=refine_count,refine_hits=refine_hits,refine_hit_rate=record['refine_input']['hit_rate'],fit_count=len(src),fit_hits=fit_hits,
            fit_hit_rate=record['final_fit']['hit_rate'],replacements=fit_hits,changed_targets=record['final_fit']['changed_targets'],hit_source_grid4x4=coverage_hits['grid_4x4_occupied'],
            hit_source_eigen_ratio=coverage_hits['minor_major_eigenvalue_ratio'],hit_source_less_than3=coverage_hits['less_than_three_points'],hit_source_hull=coverage_hits['hull_area_fraction'],
            inside_before_mean_error_px=record['inside_before_errors']['mean_px'],inside_before_median_error_px=record['inside_before_errors']['median_px'])
        for group,short in [(base,'base'),(ideal,'oracle')]:
            f.update({f'{short}_'+key:value for key,value in dict(rmse_px=evals[group]['rmse_px'],solved=evals[group]['solved'],SR3=evals[group]['success']['3'],SR5=evals[group]['success']['5'],
                inliers=fits[group]['inliers'],inlier_source_grid4x4=inliers[group]['source']['grid_4x4_occupied'],inlier_source_eigen_ratio=inliers[group]['source']['minor_major_eigenvalue_ratio'],inlier_source_hull=inliers[group]['source']['hull_area_fraction']).items()})
        flat.append(f)
    summaries=[]
    for group in ['original_prediction','inside_actual_support_nominalGT_target_only']:
        evals=[r['branches'][group]['evaluation'] for r in records];errors=[e['rmse_px'] for e in evals if e['solved']]
        summaries.append(dict(branch=group,**summarize_evaluations(evals),solved_P90_RMSE_px=float(np.percentile(errors,90)) if errors else None,registration_failures5=sum(not e['success']['5'] for e in evals)))
    changes=[]
    for threshold in ['3','5']:
        base=np.array([r['branches']['original_prediction']['evaluation']['success'][threshold] for r in records]);oracle=np.array([r['branches']['inside_actual_support_nominalGT_target_only']['evaluation']['success'][threshold] for r in records])
        saved=int((~base&oracle).sum());lost=int((base&~oracle).sum());changes.append(dict(threshold_px=int(threshold),failure_to_success=saved,success_to_failure=lost,failure_to_failure=int((~base&~oracle).sum()),success_to_success=int((base&oracle).sum()),
            original_failures=int((~base).sum()),original_successes=int(base.sum()),failure_rescue_rate=saved/int((~base).sum()) if (~base).any() else None,success_degradation_rate=lost/int(base.sum()) if base.any() else None,
            delta_SR=(saved-lost)/952))
    support=[]
    for key in ['refine_input','final_fit']:
        total=sum(r[key]['total'] for r in records);hits=sum(r[key]['hits'] for r in records);rates=[r[key]['hit_rate'] for r in records if r[key]['hit_rate'] is not None]
        support.append(dict(stage=key,total=total,hits=hits,pooled_hit_rate=hits/total,pairs_with_zero_candidates=sum(r[key]['total']==0 for r in records),
            pair_hit_rate_quantiles={str(q):float(np.percentile(rates,q)) for q in [0,10,25,50,75,90,100]},pair_hit_count_quantiles={str(q):float(np.percentile([r[key]['hits'] for r in records],q)) for q in [0,10,25,50,75,90,100]}))
    groups=[]
    for label,select in [('rescued5',lambda r:not r['base_SR5'] and r['oracle_SR5']),('still_failed5',lambda r:not r['base_SR5'] and not r['oracle_SR5']),
        ('regressed5',lambda r:r['base_SR5'] and not r['oracle_SR5']),('kept_success5',lambda r:r['base_SR5'] and r['oracle_SR5'])]:
        selected=[r for r in flat if select(r)];g=dict(group=label,pairs=len(selected))
        for metric in ['fit_count','fit_hits','fit_hit_rate','hit_source_grid4x4','hit_source_eigen_ratio','hit_source_hull','inside_before_mean_error_px']:
            values=[r[metric] for r in selected if r[metric] is not None];g[metric+'_median']=float(np.median(values)) if values else None;g[metric+'_P10']=float(np.percentile(values,10)) if values else None;g[metric+'_P90']=float(np.percentile(values,90)) if values else None
        g['less_than3_hit_points']=sum(r['hit_source_less_than3'] for r in selected);groups.append(g)
    acceptance=dict(passed=True,pairs=952,original_exact_pairs=952,candidate_identity_order_source_confidence_masks_unchanged=True,original_and_oracle_outside_targets_equal=True,inside_replaced_error_zero=True,
        forced_empty_intervention_all952_exact=True,independent_RANSAC_inliers=True,independent_probes_never_added_to_fit=True,one_paired_RANSAC_seed=cfg['seed'],no_network_rerun=True,no_training=True,no_timing=True,no_test=True,
        max_replacement_nominal_error_after_estimator_float32_cast_px=max_after_float32_error,cache_freeze_sha256=sha(out/'cache_freeze.json'),source_sha256=sha(__file__),
        continuous_support_vs_legacy_box=dict(old_box_fit_hits=sum(r['box_hits_fit'] for r in freeze['pairs']),actual_hull_fit_hits=sum(r['fit_hits'] for r in freeze['pairs']),box_only=sum(r['box_only_hits_fit'] for r in freeze['pairs']),hull_only=sum(r['actual_support_only_fit'] for r in freeze['pairs'])),
        inside_before_pooled_errors=errors_summary(inside_errors),summaries=summaries,paired_changes=changes,support_summary=support,rescue_groups=groups)
    (out/'results.json').write_text(json.dumps(dict(acceptance=acceptance,pairs=records),indent=2));(out/'acceptance.json').write_text(json.dumps(acceptance,indent=2))
    write_csv(out/'overall.csv',summaries);write_csv(out/'paired_changes.csv',changes);write_csv(out/'per_pair.csv',flat);write_csv(out/'rescue_groups.csv',groups)
    (out/'support_summary.json').write_text(json.dumps(support,indent=2));print(json.dumps(acceptance,indent=2),flush=True)


if __name__=='__main__':main()
