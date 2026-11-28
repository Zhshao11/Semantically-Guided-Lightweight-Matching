"""Independent final audit: frozen candidates, target-only oracle and all952 paired results."""
import hashlib,json
from pathlib import Path
import numpy as np
from sglm.window_oracle import replace_targets,array_digest,support_hits
from sglm.evaluation import estimate_from_correspondences,evaluate_transform


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=json.loads(Path('configs/window_oracle_v1.json').read_text());out=Path(p['output']);freeze=json.loads((out/'cache_freeze.json').read_text());result=json.loads((out/'results.json').read_text());cfg=freeze['config'];assert result['acceptance']['passed']
    assert len(freeze['pairs'])==len(result['pairs'])==952
    for path,digest in freeze['input_sha256'].items():assert sha(path)==digest,path
    totals=dict(refine_candidates=0,fit_candidates=0,refine_hits=0,fit_hits=0,baseline_exact_pairs=0,empty_intervention_exact_pairs=0,oracle_fit_exact_pairs=0,mask_local_removed=0,mask_estimator_removed=0)
    for original,r in zip(freeze['pairs'],result['pairs']):
        assert original['pair_id']==r['pair_id'] and sha(original['path'])==original['sha256'];c=np.load(original['path']);mask=c['frozen_fit_mask'];hit=c['hit'][mask]
        assert np.array_equal(mask,c['local_valid']&c['original_estimator_mask'])
        truth=c['nominal_truth'];positions=c['window_positions'];valid=c['window_token_valid'];hits=support_hits(truth,positions,valid,c['source_token_valid']&c['actual_source_valid'],c['nominal_target_valid'],p['numeric_tolerance_px']);assert np.array_equal(hits,c['hit'])
        src=c['source'][mask];dst=c['target'][mask];score=c['confidence'][mask];ids=c['candidate_id'][mask]
        assert len(np.unique(ids))==len(ids) and array_digest(ids,src,score,mask)==original['fit_packet_sha256']
        oracle=replace_targets(dst,truth[mask],hit);assert np.array_equal(oracle[~hit],dst[~hit]) and np.array_equal(oracle[hit],truth[mask][hit])
        probe=c['probes'];assert not np.any(np.all(src[:,None,:]==probe[None,:,:],axis=-1))
        for group,target in [('original_prediction',dst),('inside_actual_support_nominalGT_target_only',oracle)]:
            fit=estimate_from_correspondences(src,target,score,cfg['evaluation'],seed=cfg['seed'],include_inlier_indices=True)
            ev=evaluate_transform(fit['matrix'],probe,c['probe_truth'],cfg['evaluation']['success_thresholds_px'],status=fit['status'])
            assert fit==r['branches'][group]['estimation'] and ev==r['branches'][group]['evaluation']
        zero=replace_targets(dst,truth[mask],np.zeros(len(src),bool));fit=estimate_from_correspondences(src,zero,score,cfg['evaluation'],seed=cfg['seed'],include_inlier_indices=True)
        assert fit==r['branches']['original_prediction']['estimation']
        totals['refine_candidates']+=len(c['source']);totals['fit_candidates']+=len(src);totals['refine_hits']+=int(c['hit'].sum());totals['fit_hits']+=int(hit.sum())
        totals['baseline_exact_pairs']+=1;totals['empty_intervention_exact_pairs']+=1;totals['oracle_fit_exact_pairs']+=1
        totals['mask_local_removed']+=int((~c['local_valid']).sum());totals['mask_estimator_removed']+=int((c['local_valid']&~c['original_estimator_mask']).sum())
    assert totals['fit_hits']==result['acceptance']['support_summary'][1]['hits']
    assert '145 passed' in (out/'regression_tests.txt').read_text()
    sensitivity=json.loads((out/'ransac_sensitivity.json').read_text());assert not sensitivity['probe_numerical_seed_sensitivity'] and 'all952_seeds' not in sensitivity
    regression=sensitivity['probe_results'];assert all(r['branches']['oracle']['fit']['matrix']==regression[0]['branches']['oracle']['fit']['matrix'] for r in regression)
    record=dict(passed=True,totals=totals,real_sampler_support_recomputed=True,all952_oracle_and_original_fits_replayed=True,all952_probes_independent=True,
        original_model_and_estimator_sources_unchanged=True,one_main_paired_seed=cfg['seed'],posthoc_single_regression_pair_seed_check_only=True,no_best_seed_selection=True,
        no_model_training=True,no_timing_or_memory_ranking=True,no_test=True,source_sha256={str(file):sha(file) for file in [Path('sglm/window_oracle.py'),*sorted(Path('scripts').glob('*window_oracle*.py')),Path('tests/test_window_oracle.py')]})
    (out/'final_verification.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


if __name__=='__main__':main()
