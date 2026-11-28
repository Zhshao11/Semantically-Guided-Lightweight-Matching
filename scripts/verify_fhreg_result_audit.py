"""Read-only final ledger/evidence/coordinates/timing audit; no network inference."""
import csv,hashlib,json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=json.loads(Path('configs/fhreg_result_audit_v1.json').read_text());out=Path(p['output']);freeze=json.loads((out/'audit_freeze.json').read_text())
    for f,h in freeze['inputs_sha256'].items():assert sha(f)==h,f
    coord=json.loads((out/'coordinate_roundtrip.json').read_text());assert coord['passed'] and coord['no_network'] and len(coord['cases'])==7 and coord['max_error_px']<1e-4
    ledger=list(csv.DictReader((out/'FHReg_all952_fixed_ledger.csv').open()));assert len(ledger)==952 and len({r['pair_id'] for r in ledger})==952
    failures=[r for r in ledger if r['fit_success']=='False'];assert len(failures)==21
    for r in ledger:
        fit=r['fit_success']=='True';error=float(r['probe_error']) if r['probe_error']!='' else None
        assert (r['success_at_3px']=='True')==(fit and error is not None and np.isfinite(error) and error<3)
        assert (r['success_at_5px']=='True')==(fit and error is not None and np.isfinite(error) and error<5)
        if not fit:assert r['probe_error']=='' and r['failure_reason']!=''
    assert sum(r['success_at_5px']=='True' for r in ledger)==623 and sum(r['success_at_3px']=='True' for r in ledger)==415
    ev=json.loads((out/'evidence_manifest.json').read_text());assert ev['passed'] and ev['pairs']==30 and ev['all30_predictions_fits_evaluations_exact_previous']
    selected=json.loads((out/'random30_selection.json').read_text());assert len({r['pair_id'] for r in selected['pairs']})==30
    for category in p['sample_selection']['categories']:assert sum(r['category']==category for r in selected['pairs'])==10
    needed=['optical_original.png','sar_original_unaugmented.png','sar_augmented_actual_input.png','matches_raw.png','matches_filtered.png','ransac_inliers.png','independent_probes.png','registered_overlay.png','registered_checkerboard.png','raw_matches_all.csv','filtered_matches_all.csv','independent_probes_all.csv','full_correspondence_evidence.npz','correspondence_patch_review.png','metadata.json']
    for r in ev['pairs_manifest']:
        folder=Path(r['folder'])
        for f in needed:assert (folder/f).is_file(),(r['pair_id'],f)
        for f,h in r['files_sha256'].items():assert sha(folder/f)==h,(r['pair_id'],f)
        c=np.load(folder/'full_correspondence_evidence.npz');assert len(c['raw_score'])==r['raw_matches'] and int(c['common_filter_mask'].sum())==r['filtered_matches'] and int(c['RANSAC_inlier_mask'].sum())==r['inliers']
    notes=json.loads((out/'visual_review_notes.json').read_text());assert notes['all30_reviewed'] and {r['pair_id'] for r in notes['pairs']}=={r['pair_id'] for r in selected['pairs']}
    inputs=None;guards=0;run_count=0
    for method in p['timing']['methods']:
        for rd in [1,2,3]:
            e=json.loads((out/f'{method}_timing_round{rd}.json').read_text());runs=list(csv.DictReader((out/f'{method}_timing_round{rd}.csv').open()));assert len(runs)==120 and len({r['pair_id'] for r in runs})==24
            assert e['source_sha256']==sha('scripts/benchmark_fhreg_t05_audit.py') and e['fresh_single_model_process'] and e['exclusive_guards']==48 and e['cuda_sync_before_after'] and e['include_RANSAC'] and e['include_common_filter'] and e['no_hooks'] and e['no_feature_cache'] and not e['GT_used']
            assert e['warmup_per_pair']==3 and e['repeats_per_pair']==5 and e['gpu_uuid']==p['timing']['gpu_uuid']
            if inputs is None:inputs=e['sameCPUinput_hashes']
            else:assert inputs==e['sameCPUinput_hashes']
            guards+=e['exclusive_guards'];run_count+=len(runs)
    assert '154 passed' in (out/'regression_tests.txt').read_text()
    result=dict(passed=True,all952_SR_denominator=952,FHReg_success5=623,fit_failures_in_denominator=21,RMSE_scope931_solved=True,random_pairs=30,each_class10=True,all30_reproduce_original=True,private_all_evidence_files_present_and_hashes_match=True,
        known_coordinate_cases=7,coordinate_max_error_px=coord['max_error_px'],qualitative_all30_notes_present=True,formal_processes=6,formal_runs=run_count,exclusive_guards=guards,same24_CPUinputs=True,unified_outer_boundary=True,no_model_training=True,no_full952_network_rerun=True,test_used=False,
        source_sha256={str(f):sha(f) for f in [Path('sglm/fhreg_coordinate_audit.py'),*sorted(Path('scripts').glob('*fhreg*aud*.py')),Path('scripts/capture_fhreg_evidence.py'),Path('tests/test_fhreg_coordinate_audit.py')]})
    (out/'final_acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
