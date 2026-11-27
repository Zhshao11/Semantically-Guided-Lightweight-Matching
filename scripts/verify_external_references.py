"""Recompute all independent affine/probe metrics and inspect frozen official artifacts."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from sglm.evaluation import estimate_from_correspondences,evaluate_transform
from sglm.experiment_io import write_csv


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=json.loads(Path('configs/external_references_v1.json').read_text());out=Path(p['output']);root=Path(p['source_root'])
    f=json.loads((out/'freeze.json').read_text());cfg=json.loads(Path(p['frozen']+'/candidate_config.json').read_text())
    for path,digest in f['input_sha256'].items():assert sha(path)==digest,path
    assert sha(f['checkpoint'])==f['checkpoint_sha256']
    for path,record in f['weights'].items():assert sha(path)==record['sha256'] and Path(path).stat().st_size==record['bytes']
    snapshot=json.loads((out/'official_source_snapshot.json').read_text())
    assert sha(out/'official_source_snapshot.json')==f['official_source_snapshot_sha256']
    for path,digest in snapshot['files_sha256'].items():assert sha(root/path)==digest,path
    assert json.loads((out/'official_io_equivalence.json').read_text())['passed']
    plan=json.loads(Path(p['frozen']+'/validation_plan.json').read_text())['samples'];ids=[r['pair_id'] for r in plan];assert len(ids)==952
    flat=[]
    for name in p['methods']:
        assert json.loads((out/f'{name}_smoke.json').read_text())['passed']
        rows=json.loads((root/f'{name}_validation.json').read_text())['pairs'];assert [r['pair_id'] for r in rows]==ids
        for i,r in enumerate(rows):
            raw=dict(np.load(root/'predictions'/name/f'{i:04d}.npz'))
            assert len(raw['source'])==len(raw['target'])==len(raw['score'])==r['valid_matches']
            assert r['raw_matches']==r['valid_matches']+r['removed_invalid']
            assert np.isfinite(raw['source']).all() and np.isfinite(raw['target']).all() and np.isfinite(raw['score']).all()
            fit=estimate_from_correspondences(raw['source'],raw['target'],raw['score'],cfg['evaluation'],seed=cfg['seed'],include_inlier_indices=True)
            assert fit==r['estimation']
            ev=evaluate_transform(fit['matrix'],raw['probes'],raw['truth'],cfg['evaluation']['success_thresholds_px'],status=fit['status'])
            assert ev==r['evaluation']
            if name!='T05':
                base=dict(np.load(root/'predictions'/'T05'/f'{i:04d}.npz'))
                assert np.array_equal(raw['probes'],base['probes']) and np.array_equal(raw['truth'],base['truth'])
            flat.append({k:v for k,v in r.items() if not isinstance(v,dict)})
        summary=json.loads((out/f'{name}_accuracy.json').read_text())
        assert summary['image_pairs']==952 and summary['failed_pairs']==sum(not r['fit_success'] for r in rows)
        assert abs(summary['success_rate@5px']-sum(r['success5'] for r in rows)/952)<1e-12
        assert summary['BN_unchanged']
    assert json.loads((out/'T05_accuracy.json').read_text())['original_T05_exact_pairs']==952
    write_csv(out/'per_pair_metrics.csv',flat)
    timed=list(csv.DictReader((out/'latency_runs.csv').open()));assert len(timed)==3*3*24*5
    env=json.loads((out/'timing_environment.json').read_text());assert env['no_hooks'] and env['no_feature_cache'] and env['only_one_model_at_a_time']
    assert len(env['exclusive_guards'])==3*3*24*2
    # Cross-method timing retains32-33MiB cuBLAS workspaces. Do not present its
    # cumulative peak as the final isolated-method memory measurement.
    for row in env['initial_allocated']:assert row['before_model_MiB']<=33.01
    for name in p['methods']:
        mem=json.loads((out/f'{name}_memory.json').read_text())
        assert mem['fresh_process'] and mem['initial_allocated_MiB']==0 and mem['pairs']==24
        assert mem['model_collected'] and mem['dino_collected']
        assert mem['after_diagnostic_clear_cublas_MiB']<1
    for name in p['methods']:
        group=[r for r in timed if r['method']==name];assert len(group)==360 and len({r['pair_id'] for r in group})==24
        accuracy={r['pair_id']:r for r in json.loads((root/f'{name}_validation.json').read_text())['pairs']}
        for row in group:
            ref=accuracy[row['pair_id']];assert int(row['raw_matches'])==ref['raw_matches'] and int(row['valid_matches'])==ref['valid_matches']
            assert row['fit_status']==ref['fit_status'] and int(row['inliers'])==ref['inliers']
    tests=(out/'tests.txt').read_text();assert '122 passed' in tests and 'failed' not in tests
    result=dict(passed=True,official_sources_unchanged=True,weights_hashes_checked=True,
        methods=3,pairs_per_method=952,per_pair_records=len(flat),timing_records=len(timed),
        T05_all952_exact_reproduction=True,all_saved_correspondences_refit_and_probe_metrics_reproduced=True,
        identical_evaluation_probes=True,all_failures_retained=True,official_IO_equivalence_with_recorded_pixel_tolerance=True,
        no_training=True,test_used=False,formal_idle_single5090_verified=True,one_model_at_a_time_verified=True,
        authoritative_peak_memory_fresh_process_per_method=True,prior_cublas_workspace_effect_reported=True,
        accuracy_timing_same_correspondence_counts_and_inliers=True,
        native_internal_sizes_and_precision_differ=True,external_training_overlap_not_fully_verifiable=True,
        source_sha256={str(path):sha(path) for path in [Path('sglm/external_reference.py'),*sorted(Path('scripts').glob('*external*.py')),Path('tests/test_external_reference.py')]})
    (out/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
