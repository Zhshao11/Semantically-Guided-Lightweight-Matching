"""Fixed random30, no-network coordinates and independent952-denominator ledger."""
import hashlib,json
from pathlib import Path
import numpy as np
from sglm.fhreg_coordinate_audit import known_cases,roundtrip_case
from sglm.evaluation import estimate_from_correspondences,fixed_synthetic_probes,evaluate_transform
from sglm.experiment_io import write_csv


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=json.loads(Path('configs/fhreg_result_audit_v1.json').read_text());out=Path(p['output']);out.mkdir(exist_ok=True,parents=True)
    if (out/'audit_freeze.json').exists():raise FileExistsError('Audit already frozen')
    source=json.loads(Path(p['source_protocol']).read_text());private=Path(source['source_root'])
    records={name:json.loads((private/f'{name}_validation.json').read_text()) for name in ['FHReg_GUSO','T05_Full']}
    fh={r['pair_id']:r for r in records['FHReg_GUSO']['pairs']};t05={r['pair_id']:r for r in records['T05_Full']['pairs']};assert set(fh)==set(t05) and len(fh)==952
    ledger=[]
    for uid in sorted(fh):
        r=fh[uid];fit=r['estimation'];err=r['evaluation']['rmse_px'];fit_success=bool(fit['success'])
        assert (err is not None)==fit_success
        success3=fit_success and np.isfinite(err) and err<3;success5=fit_success and np.isfinite(err) and err<5
        assert success3==r['evaluation']['success']['3'] and success5==r['evaluation']['success']['5']
        reason=fit['status'] if not fit_success else ('probe_RMSE_ge_5px' if not success5 else '')
        ledger.append(dict(pair_id=uid,fit_success=fit_success,num_matches=r['valid_matches'],num_inliers=fit['inliers'],probe_error=err,success_at_3px=success3,success_at_5px=success5,
            failure_reason=reason,num_raw_matches=r['raw_matches'],num_common_filter_removed=r['removed_invalid'],fit_status=fit['status']))
    total=len(ledger);success3=sum(r['success_at_3px'] for r in ledger);success5=sum(r['success_at_5px'] for r in ledger);failed=sum(not r['fit_success'] for r in ledger);solved=[r['probe_error'] for r in ledger if r['fit_success']]
    original=json.loads(Path('reports/sar_references_v1/FHReg_GUSO_accuracy.json').read_text())
    assert total==952 and failed==21 and success5/total==original['success_rate@5px'] and success3/total==original['success_rate@3px']
    assert float(np.median(solved))==original['solved_median_rmse_px'] and float(np.percentile(solved,90))==original['solved_p90_rmse_px']
    from collections import Counter
    denominator=dict(passed=True,pairs=total,fit_successes=len(solved),fit_failures=failed,success3=success3,success5=success5,SR3=success3/total,SR5=success5/total,
        registration_failures5=total-success5,failure_reasons=dict(Counter(r['failure_reason'] for r in ledger if r['failure_reason'])),
        SR_denominator=952,RMSE_scope='all931 fit-success pairs, including308 solvedbutSR5fail pairs; exclude only21undefinederrors, explicitly reported',
        RMSE_denominator=len(solved),solved_mean_RMSE_px=float(np.mean(solved)),solved_median_RMSE_px=float(np.median(solved)),solved_P90_RMSE_px=float(np.percentile(solved,90)),
        SR5_success_only_count=success5,SR5_success_only_median_RMSE_px=float(np.median([r['probe_error'] for r in ledger if r['success_at_5px']])),
        no_network_rerun=True,source_validation_sha256=sha(private/'FHReg_GUSO_validation.json'))
    write_csv(out/'FHReg_all952_fixed_ledger.csv',ledger);(out/'denominator_audit.json').write_text(json.dumps(denominator,indent=2))
    rng=np.random.default_rng(p['sample_selection']['seed']);sample=[];classes={}
    for category in p['sample_selection']['categories']:
        pool=[]
        for uid in sorted(fh):
            a=fh[uid]['evaluation']['success']['5'];b=t05[uid]['evaluation']['success']['5']
            if ((category=='FHReg_success_T05_failure' and a and not b) or (category=='T05_success_FHReg_failure' and b and not a) or (category=='both_success' and a and b)):pool.append(uid)
        assert len(pool)>=10;classes[category]=len(pool)
        chosen=rng.choice(len(pool),p['sample_selection']['per_category'],replace=False)
        for order,k in enumerate(chosen):
            uid=pool[int(k)];sample.append(dict(category=category,draw_order=order,pair_id=uid,FHReg_rmse_px=fh[uid]['evaluation']['rmse_px'],T05_rmse_px=t05[uid]['evaluation']['rmse_px'],
                FHReg_valid_matches=fh[uid]['valid_matches'],FHReg_inliers=fh[uid]['inliers']))
    write_csv(out/'random30_selection.csv',sample);selection=dict(seed=p['sample_selection']['seed'],threshold_px=5,eligible_pools=classes,pairs=sample,no_manual_replacement=True)
    (out/'random30_selection.json').write_text(json.dumps(selection,indent=2))
    cfg=json.loads(Path(source['frozen']+'/candidate_config.json').read_text());coordinate=[]
    for case in known_cases():
        result,(src,dst)=roundtrip_case(case);assert all(x['max_px']<1e-4 for x in result['errors'].values())
        fit=estimate_from_correspondences(src,dst,np.ones(len(src)),cfg['evaluation'],seed=cfg['seed'])
        probes,truth=fixed_synthetic_probes(case['A'],width=case['original_hw'][1],height=case['original_hw'][0])
        ev=evaluate_transform(fit['matrix'],probes,truth);assert ev['rmse_px']<1e-4
        result['known_affine_independent_probe_rmse_px']=ev['rmse_px'];result['passed']=True;coordinate.append(result)
    (out/'coordinate_roundtrip.json').write_text(json.dumps(dict(passed=True,tolerance_px=1e-4,no_network=True,production_restore='sglm.sar_reference_geometry.restore_resized_coordinates exactlyused byFHReg',
        production_geometry='crop/pad zero; syntheticextensions do not implyactualcrop/padding inpriorFH table',cases=coordinate,
        max_error_px=max(e['max_px'] for c in coordinate for e in c['errors'].values()),mean_error_px_mean_across_equal_size_cases=float(np.mean([e['mean_px'] for c in coordinate for e in c['errors'].values()]))),indent=2))
    files=['configs/fhreg_result_audit_v1.json',p['source_protocol'],'sglm/fhreg_coordinate_audit.py','sglm/sar_references.py','sglm/sar_reference_geometry.py',
        'sglm/external_reference.py','sglm/evaluation.py','sglm/geometry.py','scripts/prepare_fhreg_result_audit.py','reports/t05_frozen_v1/validation_plan.json','reports/t05_frozen_v1/subsets.json',
        'reports/sar_references_v1/freeze.json','reports/sar_references_v1/FHReg_GUSO_accuracy.json',str(private/'FHReg_GUSO_validation.json'),str(private/'T05_Full_validation.json'),str(out/'random30_selection.json')]
    (out/'audit_freeze.json').write_text(json.dumps(dict(protocol=p,inputs_sha256={f:sha(f) for f in files},no_training=True,test_used=False),indent=2))
    print('LedgerSR5:',success5,'/',total,'fitfailures',failed,'coordinate_max',max(e['max_px'] for c in coordinate for e in c['errors'].values()),flush=True)


if __name__=='__main__':main()
