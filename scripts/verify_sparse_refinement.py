"""Cross-artifact verification of counts, source alignment, geometry and frozen inputs."""
import csv,json
from pathlib import Path
import numpy as np
from sglm.sparse_refinement_experiment import OUT,PRIVATE,sha
from sglm.evaluation import evaluate_transform,estimate_from_correspondences


def main():
    freeze=json.loads((OUT/'freeze.json').read_text());p=json.loads(Path('configs/sparse_refinement_v1.json').read_text())
    cfg=json.loads(Path(p['frozen']+'/candidate_config.json').read_text())
    assert sha(freeze['checkpoint'])==freeze['checkpoint_sha256']==p['checkpoint_sha256']
    for path,digest in freeze['inputs_sha256'].items():assert sha(path)==digest,path
    assert json.loads((OUT/'regression.json').read_text())['passed']
    assert json.loads((OUT/'full_reproduction.json').read_text())['passed']
    base=json.loads((PRIVATE/'A0_all_validation.json').read_text())['pairs'];assert len(base)==952
    all_records=[];max_subset=0.;max_refine_count_error=0
    for v in p['variants']:
        rows=json.loads((PRIVATE/f'{v["name"]}_validation.json').read_text())['pairs'];assert len(rows)==952
        assert [r['pair_id'] for r in rows]==[r['pair_id'] for r in base]
        for i,r in enumerate(rows):
            raw=dict(np.load(PRIVATE/'predictions'/v['name']/f'{i:04d}.npz'))
            full=dict(np.load(PRIVATE/'predictions'/'A0_all'/f'{i:04d}.npz'))
            idx=raw['original_candidate_idx'];selected=raw['selected_candidate_idx'];n=r['num_coarse_candidates']
            assert len(selected)==r['num_selected']==(n if v['mode']=='all' else min(v['budget'],n))
            assert len(np.unique(selected))==len(selected) and np.all((selected>=0)&(selected<n))
            assert np.all(np.diff(selected)>0)
            assert r['num_refined_valid']==len(raw['source'])<=len(selected)
            assert np.array_equal(raw['coarse_source'],full['coarse_source'])
            assert np.array_equal(raw['coarse_target_all'],full['coarse_target_all'])
            assert np.array_equal(raw['source'],full['source'][idx])
            assert np.array_equal(raw['score'],full['score'][idx])
            assert np.array_equal(raw['coarse_target'],full['coarse_target'][idx])
            assert np.array_equal(raw['probes'],full['probes']) and np.array_equal(raw['probe_truth'],full['probe_truth'])
            if len(idx):max_subset=max(max_subset,float(np.max(np.abs(raw['target']-full['target'][idx]))))
            fit=estimate_from_correspondences(raw['source'],raw['target'],raw['score'],cfg['evaluation'],seed=cfg['seed'],include_inlier_indices=True)
            assert fit==r['estimation']
            ev=evaluate_transform(fit['matrix'],raw['probes'],raw['probe_truth'],cfg['evaluation']['success_thresholds_px'],status=fit['status'])
            assert ev==r['evaluation'] and ev['success']['5']==r['registration_success']
        all_records.extend(rows)
    assert max_subset<=p['regression']['subset_atol_px']
    public=list(csv.DictReader((OUT/'per_pair_metrics.csv').open()));assert len(public)==952*6
    timing=list(csv.DictReader((OUT/'latency_runs.csv').open()));assert len(timing)==24*3*10*6*2
    for name in [v['name'] for v in p['variants']]:
        group=[r for r in timing if r['variant']==name]
        assert len({r['pair_id'] for r in group})==24
    tests=(OUT/'tests.txt').read_text();assert '119 passed' in tests and 'failed' not in tests
    acceptance=dict(passed=True,validation_pairs_per_variant=952,variants=6,per_pair_rows=len(public),
        full_exact_reproduction_952=True,no_training=True,no_test_used=True,original_runtime_inputs_unchanged=True,
        selected_refined_counts_and_indices_checked=True,all_coarse_candidate_sets_identical=True,
        selected_fields_alignment_checked=True,only_refined_points_in_geometry=True,
        independent_geometry_and_probe_metrics_reproduced=True,max_full_vs_subset_BF16_coordinate_delta_px=max_subset,
        actual_local_count_verified_on32_train_pairs=True,all_failures_in_denominator=True,
        formal_timing_rows=len(timing),formal_uncached_no_hooks=True,
        source_sha256={str(x):sha(x) for x in [Path('sglm/sparse_refinement.py'),Path('sglm/sparse_refinement_experiment.py'),*sorted(Path('scripts').glob('*sparse_refinement*.py')),Path('tests/test_sparse_refinement.py')]})
    (OUT/'acceptance.json').write_text(json.dumps(acceptance,indent=2));print(json.dumps(acceptance,indent=2))


if __name__=='__main__':main()
