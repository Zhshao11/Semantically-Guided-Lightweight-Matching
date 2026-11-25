"""Final cross-artifact acceptance: frozen identity, full denominators and timing workload."""
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    frozen=Path('reports/t05_frozen_v1');results=Path('reports/t05_results_v1');out=Path('reports/t05_delivery_v1')
    f=read(frozen/'freeze.json');s=read(frozen/'subsets.json');plan=read(frozen/'validation_plan.json')
    assert sha(f['frozen_checkpoint'])==sha(f['source_checkpoint'])==f['checkpoint_sha256']
    assert sha(frozen/'validation_plan.json')==f['validation_plan_sha256'] and sha(frozen/'subsets.json')==f['subsets_sha256']
    assert sha(frozen/'candidate_config.json')==f['candidate_config_sha256']
    assert plan['config']==read('runs/t03_main_lr1e3_v1/config.json')
    ids={r['pair_id'] for r in plan['samples']};assert len(ids)==952
    expected_diag={s['records'][i]['pair_id'] for i in s['diagnostic_indices']}
    expected_timing={s['records'][i]['pair_id'] for i in s['timing_indices']}
    assert len(expected_diag)==96 and len(expected_timing)==24 and s['timing_distinct_verified_512_parents']==24
    reproduction=read(results/'candidate_reproduction.json');assert reproduction['passed'] and len(reproduction['pairs'])==952
    for r in reproduction['pairs']:
        assert r['original_B0_predictions_bitwise_equal'] and r['matrix_exact'] and r['success_flags_equal'] and r['RMSE_delta_px']==0
    for k in f['protocol']['budgets']:
        v=read(results/f'K{k}_validation.json');d=read(results/f'K{k}_diagnostics.json')
        assert len(v['pairs'])==952 and {r['pair_id'] for r in v['pairs']}==ids
        assert len(d['pairs'])==96 and {r['pair_id'] for r in d['pairs']}==expected_diag
        assert v['state_audit']['all_modules_eval'] and v['state_audit']['BN_buffers_unchanged']
        assert v['summary']['failed_pairs']==sum(not r['evaluation']['solved'] for r in v['pairs'])
        for t in [3,5]:assert abs(v['summary'][f'success_rate@{t}px']-sum(r['evaluation']['success'][str(t)] for r in v['pairs'])/952)<1e-15
    t=read(results/'timing/timing.json');assert t['no_hooks_confirmed'] and t['checkpoint_sha256']==f['checkpoint_sha256']
    assert len(t['records'])==1920 and len(t['stage_records'])==1920 and len(t['other_process_guards'])==48
    for k in f['protocol']['budgets']:
        rows=[r for r in t['records'] if r['K']==k]
        assert {r['pair_id'] for r in rows}==expected_timing and len(rows)==480
        for uid in expected_timing:assert {r['repeat'] for r in rows if r['pair_id']==uid}==set(range(20))
        expected={r['pair_id']:r for r in read(results/f'K{k}_validation.json')['pairs']}
        for row in rows+[r for r in t['stage_records'] if r['K']==k]:
            reference=expected[row['pair_id']]
            assert row['attempted_refinements']==reference['attempted_refinements']
            assert row['final_correspondences']==reference['final_correspondences']
            assert row['inliers']==reference['estimation']['inliers'] and row['solver_status']==reference['estimation']['status']
    for p,expected in read(results/'accuracy_summary.json')['runtime_source_sha256'].items():assert sha(p)==expected
    for p,expected in t['source_sha256'].items():assert sha(p)==expected
    teacher=read(out/'teacher_source_control.json');assert len(teacher['source_manifest']['samples'])==8 and len(teacher['pairs'])==16
    assert {r['mode'] for r in teacher['pairs']}=={'RGB','gray_RGB'}
    pp='runs/teacher_preflight_v1/teacher_candidate/preprocessor_config.json'
    model_cfg='runs/teacher_preflight_v1/teacher_candidate/config.json'
    assert sha(pp)=='e5a25d8fc054bf780be930e161b47d188bec1b0bf0a44fd5fdfb3f3312c0be0b'
    assert sha(model_cfg)=='7bad6bc16075fe0b5af4bc0493cf6240873938b7a859d7401dce18faba44acf3'
    report=dict(passed=True,checkpoint_sha256=f['checkpoint_sha256'],unchanged_source_best=True,
        full_reproduction_pairs=952,all_validation_pairs=952,budgets=4,diagnostic_pairs_per_budget=96,timing_pairs=24,
        timing_repetitions_per_budget=480,exclusive_GPU_pair_boundary_checks=48,
        source_validation_images=8,source_input_modes=2,source_GT_observed_ids=teacher['class_mapping_verified']['native_GT_values'],
        preprocessor_matches_T04_sha256=sha(pp),model_config_matches_T04_sha256=sha(model_cfg),
        matching_source_hashes_unchanged=True,timing_stage_workloads_equal_validation=True,
        model_RANSAC_config_unchanged=True,training_performed=False,test_used=False)
    (out/'acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
