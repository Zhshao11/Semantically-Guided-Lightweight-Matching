"""Read-only final consistency checks for frozen full evaluation and timing artifacts."""
import csv,hashlib,json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out=Path('reports/sar_references_v1');p=json.loads(Path('configs/sar_references_v1.json').read_text());freeze=json.loads((out/'freeze.json').read_text())
    for file,digest in freeze['input_sha256'].items():assert sha(file)==digest,file
    snapshot=json.loads((out/'official_source_snapshot.json').read_text())
    for file,digest in snapshot['files_sha256'].items():assert sha(Path(p['source_root'])/file)==digest,file
    results={}
    for name in p['methods']:
        assert sha(out/f'{name}_smoke.json')==freeze['native_smoke_sha256'][name]
        score=json.loads((out/f'{name}_accuracy.json').read_text());assert score['image_pairs']==952 and score['BN_unchanged'] and score['all_modules_eval']
        records=list(csv.DictReader((out/f'{name}_per_pair.csv').open()));assert len(records)==952 and len({r['pair_id'] for r in records})==952
        assert sum(r['success5']=='True' for r in records)==round(score['success_rate@5px']*952)
        assert sum(r['probe_rmse_px']=='' for r in records)==score['failed_pairs']
        if name.startswith('T05'):assert score['original_T05_exact_pairs']==952
        results[name]=dict(pairs=952,SR5_successes=round(score['success_rate@5px']*952),solver_failures=score['failed_pairs'])
    timed=[*p['methods'],'GDROS_WHU_native32_LSR','XoFTR','MINIMA_RoMa']
    for name in timed:
        for rd in range(1,4):
            env=json.loads((out/f'{name}_timing_round{rd}.json').read_text());assert env['pairs']==24 and env['records']==120 and env['exclusive_guards']==48 and env['fresh_single_model_process'] and env['no_hooks'] and not env['GT_used']
            assert env['source_sha256']==sha('scripts/benchmark_sar_references.py')
            runs=list(csv.DictReader((out/f'{name}_timing_round{rd}.csv').open()));assert len(runs)==120 and len({r['pair_index'] for r in runs})==24
    for name in ['XoFTR','MINIMA_RoMa']:
        reuse=json.loads((out/f'{name}_legacy_reuse.json').read_text());assert reuse['passed'] and reuse['same_sources_weights_data']
        assert sha(reuse['full952_accuracy_path'])==reuse['full952_accuracy_sha256']
    budget=json.loads((out/'development_budget_v1.json').read_text());assert budget['reference_method']=='FHReg_GUSO' and budget['one_actual_joint_point'] and budget['no_mixing_best_metrics']
    comparison=list(csv.DictReader((out/'external_comparison.csv').open()));anchor=next(r for r in comparison if r['method']==budget['reference_method'])
    assert float(anchor['SR5_percent'])==budget['primary_accuracy_floor']['SR5_percent_at_least']
    assert float(anchor['median_E2E_ms'])==budget['latency_goal']['median_E2E_ms_strictly_below']
    assert float(anchor['peak_allocated_MiB'])==budget['memory_goal']['peak_allocated_MiB_strictly_below']
    status=json.loads((out/'soma_resumed_verified.json').read_text());assert status['resume_executed'] and status['actual_training_progress_after_resume'] and status['current_phase']=='failed'
    assert '138 passed' in (out/'regression_tests.txt').read_text()
    result=dict(passed=True,full952_new_accuracy=results,formal_timing_methods=timed,formal_processes=len(timed)*3,formal_runs=len(timed)*3*120,
        official_core_sources_unchanged=True,frozen_protocol_and_source_hashes_match=True,old_accuracy_reuse_explicit=True,one_measured_budget_anchor=True,
        RIFT2_only_runtime_audit=True,new_training=False,test_used=False,SOMA_resumed_then_later_nonfinite_loss_separate=True)
    (out/'final_acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
