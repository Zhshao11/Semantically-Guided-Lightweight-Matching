"""Frozen heatmaps: A(original), B(max), C(peak3x3), everyfitcandidate; no network."""
import hashlib,json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np,torch
from sglm.heatmap_readout import read_frozen_heatmap,describe_frozen_heatmap
from sglm.window_oracle import array_digest,coverage
from sglm.evaluation import estimate_from_correspondences,evaluate_transform,summarize_evaluations
from sglm.diagnostics import errors_summary
from sglm.experiment_io import write_csv


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    values=np.asarray(values,float)
    return {str(q):float(np.percentile(values,q)) for q in [0,10,25,50,75,90,95,99,100]} if len(values) else {}


def main():
    p=json.loads(Path('configs/heatmap_readout_v1.json').read_text());out=Path(p['output']);out.mkdir(parents=True,exist_ok=True);private=Path(p['private']);private.mkdir(parents=True,exist_ok=True)
    if (out/'acceptance.json').exists():raise FileExistsError('Finished readout run exists; preserve fixed protocol')
    source=json.loads(Path(p['source_freeze']).read_text());assert source['passed']
    for file,digest in source['input_sha256'].items():assert sha(file)==digest,file
    cfg=source['config'];checkpoint=torch.load(source['protocol']['checkpoint'],map_location='cpu',weights_only=False)
    temperature=checkpoint['model']['local.log_temperature'].float();temperature_record=dict(log_temperature=float(temperature),exp=float(temperature.exp()),used_clamped=float(temperature.exp().clamp(.02,.5)),recomputed_probabilities=False);del checkpoint
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    original={r['pair_id']:r for r in json.loads(Path('reports/t05_results_v1/K256_validation.json').read_text())['pairs']}
    source_files=['configs/heatmap_readout_v1.json','sglm/heatmap_readout.py','scripts/evaluate_heatmap_readout.py','sglm/evaluation.py','sglm/geometry.py','sglm/t02/model.py','sglm/t05.py',p['source_freeze']]
    freeze=dict(protocol=p,input_sha256={f:sha(f) for f in source_files},temperature=temperature_record,torch=torch.__version__,cuda=torch.version.cuda,cache_files=len(source['pairs']),original_weights_unchanged=True)
    (out/'freeze.json').write_text(json.dumps(freeze,indent=2))
    pairs=[];flat=[];hit_errors={g:[] for g in ['A','B','C']};all_errors={g:[] for g in hit_errors};diag_rows=[];all_count=0;hit_count=0;max_C_sum_error=0.
    for row in source['pairs']:
        assert sha(row['path'])==row['sha256'];c=np.load(row['path']);mask=c['frozen_fit_mask'];N=len(c['source']);fit_N=int(mask.sum());all_count+=fit_N
        packet_sha=array_digest(c['candidate_id'][mask],c['source'][mask],c['confidence'][mask],mask);assert packet_sha==row['fit_packet_sha256']
        before_heat=array_digest(c['local_heatmap'],c['window_token_valid']);args=[torch.from_numpy(c[key]).cuda() for key in ['local_heatmap','window_token_valid','coarse_center','window_positions']]
        # Apply every branch to the original refinepacket. NO access to nominalGT/hit here.
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):r=read_frozen_heatmap(*args)
        assert torch.equal(r['A'],torch.from_numpy(c['target']).cuda()),(row['pair_id'],'OriginalBF16reconstructionfailed')
        targets={g:r[g].float().cpu().numpy()[mask] for g in ['A','B','C']};peak=r['peak_index'].cpu().numpy()[mask];local_mask=r['local_mask'].cpu().numpy()[mask];local_mass=r['local_mass'].float().cpu().numpy()[mask]
        pnorm=c['local_heatmap'][mask]*local_mask/local_mass[:,None];max_C_sum_error=max(max_C_sum_error,float(np.max(np.abs(pnorm.sum(-1)-1))) if len(pnorm) else 0.)
        src=c['source'][mask];score=c['confidence'][mask];fits={};evaluations={};inlier_coverage={}
        for g in targets:
            assert len(targets[g])==fit_N and np.isfinite(targets[g]).all()
            fits[g]=estimate_from_correspondences(src,targets[g],score,cfg['evaluation'],seed=cfg['seed'],include_inlier_indices=True)
        assert packet_sha==array_digest(c['candidate_id'][mask],src,score,mask) and before_heat==array_digest(c['local_heatmap'],c['window_token_valid'])
        # Only now use GT and independent probes for reporting/interpretation.
        truth=c['nominal_truth'][mask];hit=c['hit'][mask];hit_count+=int(hit.sum());errors={g:np.linalg.norm(targets[g]-truth,axis=1) for g in targets}
        diagnostics=describe_frozen_heatmap(c['local_heatmap'][mask],c['window_token_valid'][mask],c['window_positions'][mask],truth,peak)
        for g in targets:
            evaluations[g]=evaluate_transform(fits[g]['matrix'],c['probes'],c['probe_truth'],cfg['evaluation']['success_thresholds_px'],status=fits[g]['status'])
            indices=np.array(fits[g]['inlier_indices'],int);inlier_coverage[g]=coverage(src[indices]);hit_errors[g].extend(errors[g][hit].tolist());all_errors[g].extend(errors[g].tolist())
        ref=original[row['pair_id']];assert fits['A']['matrix']==ref['estimation']['matrix'] and evaluations['A']==ref['evaluation']
        record=dict(pair_id=row['pair_id'],index=row['index'],fit_candidates=fit_N,inside_window_count=int(hit.sum()),packet_sha256=packet_sha,heatmap_and_mask_sha256=before_heat,
            inside_source_coverage=coverage(src[hit]),branches={g:dict(estimation=fits[g],evaluation=evaluations[g],inlier_source_coverage=inlier_coverage[g],
                inside_errors=errors_summary(errors[g][hit]),outside_errors=errors_summary(errors[g][~hit]),all_correspondence_errors=errors_summary(errors[g])) for g in targets},
            inside_diagnostics={key:float(np.mean(value[hit])) if hit.any() else None for key,value in diagnostics.items() if key not in ['secondary_index','truth_nearest_index']})
        pairs.append(record)
        f=dict(pair_id=row['pair_id'],index=row['index'],fit_candidates=fit_N,inside_window_count=int(hit.sum()),hit_fraction=float(hit.mean()) if fit_N else None)
        for g in targets:
            ev=evaluations[g];f.update({g+'_'+key:val for key,val in dict(probe_RMSE_px=ev['rmse_px'],success3=ev['success']['3'],success5=ev['success']['5'],solved=ev['solved'],inliers=fits[g]['inliers'],
                inlier_hull=inlier_coverage[g]['hull_area_fraction'],inside_error_mean=record['branches'][g]['inside_errors']['mean_px'],inside_error_median=record['branches'][g]['inside_errors']['median_px'],inside_error_P90=record['branches'][g]['inside_errors']['p90_px']).items()})
        f.update(record['inside_diagnostics']);flat.append(f)
        np.savez_compressed(private/f"{row['index']:04d}.npz",candidate_id=c['candidate_id'][mask],source=src,confidence=score,original_fit_mask=mask,hit=hit,
            A=targets['A'],B=targets['B'],C=targets['C'],peak_index=peak,C_local_mask=local_mask,C_original_mass=local_mass,
            nominal_truth=truth,**{g+'_error':errors[g] for g in targets},**diagnostics)
        for k in np.flatnonzero(hit):
            d=dict(pair_id=row['pair_id'],candidate_id=int(c['candidate_id'][mask][k]),A_error_px=float(errors['A'][k]),B_error_px=float(errors['B'][k]),C_error_px=float(errors['C'][k]))
            d.update({key:(bool(value[k]) if value.dtype==bool else int(value[k]) if np.issubdtype(value.dtype,np.integer) else float(value[k])) for key,value in diagnostics.items()});diag_rows.append(d)
        if (row['index']+1)%100==0:print('readout evaluated',row['index']+1,952,flush=True)
    assert len(pairs)==p['expected_pairs'] and all_count==p['expected_fit_candidates'] and hit_count==p['expected_window_hits'] and len(diag_rows)==hit_count
    summary=[]
    for g in ['A','B','C']:
        evs=[r['branches'][g]['evaluation'] for r in pairs];values=[v['rmse_px'] for v in evs if v['solved']]
        summary.append(dict(branch=g,**summarize_evaluations(evs),solved_P90_RMSE_px=float(np.percentile(values,90)) if values else None,registration_failures5=sum(not e['success']['5'] for e in evs)))
    changes=[];mover_groups=[]
    for g in ['B','C']:
        for threshold in ['3','5']:
            a=np.array([r['branches']['A']['evaluation']['success'][threshold] for r in pairs]);b=np.array([r['branches'][g]['evaluation']['success'][threshold] for r in pairs]);rescued=(~a&b);lost=(a&~b)
            changes.append(dict(comparison=g+'_vs_A',threshold_px=int(threshold),failure_to_success=int(rescued.sum()),success_to_failure=int(lost.sum()),failure_to_failure=int((~a&~b).sum()),success_to_success=int((a&b).sum()),
                rescue_rate=float(rescued.sum()/(~a).sum()),degradation_rate=float(lost.sum()/a.sum()),delta_SR=float((b.sum()-a.sum())/len(a))))
        for label,selection in [('rescued5',rescued),('lost5',lost),('stable_failure5',~a&~b),('stable_success5',a&b)]:
            selected=[r for r,take in zip(pairs,selection) if take];group=dict(comparison=g+'_vs_A',group=label,pairs=len(selected))
            for key,extract in [('inside_count',lambda r:r['inside_window_count']),('inside_A_error',lambda r:r['branches']['A']['inside_errors']['mean_px']),('inside_new_error',lambda r:r['branches'][g]['inside_errors']['mean_px']),
                ('peak_GT_distance',lambda r:r['inside_diagnostics']['peak_GT_distance']),('truth_mass',lambda r:r['inside_diagnostics']['truth_mass']),('competing_fraction',lambda r:r['inside_diagnostics']['competing'])]:
                v=[extract(r) for r in selected if extract(r) is not None];group[key+'_median']=float(np.median(v)) if len(v) else None
            mover_groups.append(group)
    write_csv(out/'overall.csv',summary);write_csv(out/'paired_changes.csv',changes);write_csv(out/'per_pair.csv',flat);write_csv(out/'mover_groups.csv',mover_groups)
    write_csv(private/'all45974_inside_correspondences.csv',diag_rows)
    distributions=[]
    for subset,errors_by_branch in [('inside_actual_support',hit_errors),('all_fit_candidates',all_errors)]:
        for g,values in errors_by_branch.items():
            v=np.asarray(values);distributions.append(dict(subset=subset,branch=g,**errors_summary(v),quantiles=quantiles(v),fraction_below={str(t):float((v<t).mean()) for t in [1,3,5]}))
    diagnostic_summary={}
    for key in ['peak_GT_distance','truth_mass','peak_mass','peak_probability','secondary_ratio','normalized_entropy','max_plateau_size']:
        values=[r[key] for r in diag_rows];diagnostic_summary[key]=dict(mean=float(np.mean(values)),quantiles=quantiles(values))
    Ae=np.array([r['A_error_px'] for r in diag_rows]);Be=np.array([r['B_error_px'] for r in diag_rows]);Ce=np.array([r['C_error_px'] for r in diag_rows]);accurate_peak=Be<3
    diagnostic_summary.update(inside_count=hit_count,obvious_competing_fraction=float(np.mean([r['competing'] for r in diag_rows])),peak_inside_truth3x3_fraction=float(np.mean([r['peak_inside_truth3x3'] for r in diag_rows])),
        C_vs_A_improved_fraction=float((Ce<Ae-1e-6).mean()),C_vs_A_worsened_fraction=float((Ce>Ae+1e-6).mean()),C_vs_B_improved_fraction=float((Ce<Be-1e-6).mean()),C_vs_B_worsened_fraction=float((Ce>Be+1e-6).mean()),
        peak_within3px_count=int(accurate_peak.sum()),C_minus_B_when_peak_within3px_mean=float((Ce-Be)[accurate_peak].mean()) if accurate_peak.any() else None,
        peak_within3px_but_C_not=int((accurate_peak&(Ce>=3)).sum()),peak_outside3px_but_C_inside=int((~accurate_peak&(Ce<3)).sum()))
    conditional=[]
    competing=np.array([r['competing'] for r in diag_rows],bool)
    for label,take in [('competitive_secondary_peak',competing),('no_competitive_secondary_peak',~competing),('peak_within3px',accurate_peak),('peak_outside3px',~accurate_peak)]:
        conditional.append(dict(group=label,count=int(take.sum()),A_mean_error_px=float(Ae[take].mean()) if take.any() else None,B_mean_error_px=float(Be[take].mean()) if take.any() else None,C_mean_error_px=float(Ce[take].mean()) if take.any() else None,
            truth_mass_mean=float(np.mean([r['truth_mass'] for r,t in zip(diag_rows,take) if t])) if take.any() else None))
    write_csv(out/'conditional_diagnostics.csv',conditional)
    (out/'correspondence_distributions.json').write_text(json.dumps(distributions,indent=2));(out/'heatmap_diagnostics.json').write_text(json.dumps(diagnostic_summary,indent=2))
    (out/'pair_results.json').write_text(json.dumps(pairs,indent=2))
    result=dict(passed=True,pairs=952,fit_candidates=all_count,inside_hit_count=hit_count,A_reconstructed_BF16_target_bitwise_exact=True,A_all952_historical_matrix_and_evaluation_exact=True,
        all_three_applied_to_all_fit_candidates=True,candidateIDs_order_source_confidence_masks_frozen=True,original_heatmap_temperature_unchanged=True,
        C_neighborhood_only_originalvalidtokens_no_expansion=True,C_conditional_probability_sum_max_error=max_C_sum_error,GT_only_posthoc_after_allthreefits=True,
        independent_RANSAC_inliers_and_probes=True,paired_seed=cfg['seed'],no_network=True,no_training=True,no_test=True,no_temperature_or_threshold_scan=True,
        summaries=summary,paired_changes=changes,freeze_sha256=sha(out/'freeze.json'))
    (out/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
