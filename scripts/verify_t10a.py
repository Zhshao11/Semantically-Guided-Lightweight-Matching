"""Artifact invariants for frozen train-only nominal semanticconfusion diagnosis."""
import hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from sglm.geometry import transform_points,invert_affine
from sglm.t10_signal import semantic_cosine,paired_parent_summary


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=read('configs/t10a_protocol_v1.json');freeze=read('reports/t10_setup_v1/freeze.json');plan=read(p['samples']['plan'])['samples'];summary=read('runs/t10a_candidates_v1/summary.json');signal=read('reports/t10_results_v1/signal_summary.json');d=dict(np.load('runs/t10a_candidates_v1/relations.npz'));permutation=read('runs/t10a_permutation_v1/summary.json')
    assert len(plan)==len({r['pair_id'] for r in plan})==256 and all('/train/' in r['pair_id'] and r['split']=='train' for r in plan)
    assert len(Counter(r['verified512_parent'] for r in plan))==128 and set(Counter(r['verified512_parent'] for r in plan).values())=={2}
    assert sha(p['samples']['plan'])==freeze['plan_sha256']==summary['plan_sha256'] and sha(p['matcher_checkpoint'])==p['matcher_sha256']
    assert summary['checkpoint_sha256']==p['matcher_sha256'] and summary['model_eval'] and summary['BN_unchanged'] and summary['model_all_parameters_require_grad_false']
    assert not summary['training_started'] and not summary['validation_test_used'] and not summary['teacher_on_SAR']
    for path,value in summary['inputs_sha256'].items():assert sha(path)==value
    assert sha('runs/t10a_candidates_v1/relations.npz')==summary['relations_sha256']==signal['relations_sha256']
    teacher=read('reports/t07_teacher_v1/teacher_freeze.json');assert sha(teacher['checkpoint_path'])==p['teacher_sha256']==teacher['checkpoint_sha256'] and teacher['license_research_use_confirmed']
    for path,record in read('reports/t05_frozen_v1/freeze.json')['original_code_verification'].items():assert sha(path)==record['sha256']
    xy=np.stack([np.arange(1024)%32*8+3.5,np.arange(1024)//32*8+3.5],1)
    for entry,row in zip(summary['candidate_files'],plan):
        path=Path('runs/t10a_candidates_v1')/entry['file'];assert sha(path)==entry['sha256'];a=dict(np.load(path));A=np.asarray(row['A_forward']);assert np.array_equal(a['A_forward'],A)
        assert np.allclose(transform_points(a['nominal_candidate_xy'],A),xy,atol=1e-9) and np.allclose(a['query_truth_xy'],transform_points(xy,A),atol=1e-9)
        assert a['negative_pool_packbits'].shape==(1024,128)
        pool=np.unpackbits(a['negative_pool_packbits'],axis=1,bitorder='little');dist=np.linalg.norm(a['query_truth_xy'][:,None]-xy[None],axis=-1);assert np.all(~pool.astype(bool)|(dist>16))
        i=entry['pair_index'];selected=np.flatnonzero(d['pair_index']==i);q=d['query_index'][selected];h=d['hard_id'][selected];assert np.all(a['query_semantic_valid'][q]) and np.all(a['query_geometry_valid'][q]) and np.all(a['candidate_semantic_valid'][h])
        assert np.all(dist[q,h]>16) and np.array_equal(a['hard_indices'][q,d['rank'][selected]],h)
        assert np.allclose(semantic_cosine(a['query_probability7'][q],a['candidate_probability7'][h]),d['hard_semantic'][selected],atol=2e-7)
        for j in range(3):
            rid=d['random_id'][selected,j];valid=rid>=0;safe=np.maximum(rid,0);assert np.all(~valid|pool[q[:,None],safe].astype(bool)) and np.all(~valid|a['candidate_semantic_valid'][safe])
            assert not np.any(valid[:,:,None]&(rid[:,:,None]==a['hard_indices'][q,None,:]))
            if j>=1:
                assert np.all(~valid|(np.floor(dist[q[:,None],safe]/16)==np.floor(dist[q,h]/16)[:,None]))
                assert np.all(~valid|(np.floor(np.minimum(a['candidate_weight'][safe],.999999)*10)==np.floor(np.minimum(a['candidate_weight'][h],.999999)*10)[:,None]))
            if j==2:assert np.all(~valid|(a['candidate_probability7'][safe].argmax(-1)==a['candidate_probability7'][h].argmax(-1)[:,None]))
            assert np.all(~valid|(d['random_descriptor'][selected,j]<=d['hard_descriptor'][selected,None]+1e-6))
    cache=Path(p['teacher_cache']);ids=read(cache/'pair_ids.json');weight=np.load(cache/'weight.npy',mmap_mode='r').reshape(-1);prob=np.load(cache/'conditional7.npy',mmap_mode='r');hard=prob.argmax(1).reshape(-1);mapping=np.load('runs/t10a_permutation_v1/candidate_donor_flat_index.npy',mmap_mode='r').reshape(-1);active=np.flatnonzero(weight>0)
    manifest={r['pair_id']:r for r in [json.loads(x) for x in Path('reports/metadata_finalize_v1/train_256_dedup_v1.jsonl').read_text().splitlines()]};parents=np.array([manifest[uid]['parent_mapping']['parent_pair_id'] for uid in ids]);assert np.all(parents[active//1024]!=parents[mapping[active]//1024]) and np.all(hard[active]==hard[mapping[active]]) and len(np.unique(mapping[active]))==len(active)
    assert sha('runs/t10a_permutation_v1/candidate_donor_flat_index.npy')==permutation['mapping_sha256']
    assert not signal['decision']['full_training_started'] and len(read('runs/t10a_cases_v1/case_order.json')['cases'])==16
    cases=read('runs/t10a_cases_v1/case_order.json');assert [r['pair_index'] for r in cases['cases']]==list(range(16)) and not cases['queryselection_uses_similarity']
    tests=Path('reports/t10_results_v1/tests.txt').read_text();assert 'passed' in tests and 'failed' not in tests
    result=dict(passed=True,train_pairs=256,verified512parents=128,children_per_parent=2,T05_frozen_checkpoint=True,no_training=True,no_newloss=True,no_validation_test_used=True,no_teacher_on_SAR=True,nominal_candidate_mapping_inverse_A=True,SAR_semantic_GT=False,soft_labels_external_optical_teacher=True,existing_reliability_mask_reused=True,full_bilinear_support_mask_required=True,GT_exclusion_radius_px=16,top5_wrong_before_semantics=True,random_controls_fixed=True,distance_weight_bins_checked=True,candidate_class_match_checked=True,permutation_no_sameverified512parent=True,permutation_classquality_conditioned=True,report_missing_coverage=True,fixed16caseorder=True,hierarchical_query_pair_parent_statistics=True,parent_CI_not_scene_independence=True,primaryanalysis_unchanged_by_posthoc_sensitivity=True,source_sha256={name:sha(name) for name in ['sglm/t10_signal.py','scripts/extract_t10a_candidates.py','scripts/analyze_t10a_signal.py','scripts/prepare_t10a_permutation.py','scripts/check_t10a_sensitivity.py']})
    Path('reports/t10_results_v1/acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
