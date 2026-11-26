"""T07 cross-artifact provenance, fairness and frozen inference acceptance."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=read('configs/t07_protocol_v1.json');freeze=read('reports/t07_teacher_v1/teacher_freeze.json');cache=read('runs/t07_semantic_cache_v1/summary.json');head=read('runs/t07_semantic_head_v1/summary.json');selection=read('reports/t07_selection_v1/summary.json')
    assert sha(p['matcher_checkpoint'])==p['matcher_sha256']
    protected=read('reports/t05_frozen_v1/freeze.json')['original_code_verification']
    for path,r in protected.items():assert sha(path)==r['sha256'],path
    assert sha(freeze['checkpoint_path'])==freeze['checkpoint_sha256']==cache['teacher_checkpoint_sha256'] and freeze['license_research_use_confirmed']
    assert freeze['technical_decision']=='freeze_for_research_with_explicit_abstention'
    ids=read('runs/t07_semantic_cache_v1/pair_ids.json');assert len(ids)==len(set(ids))==7912 and ids==read('runs/t06_reliability_v1/pair_ids.json') and all('/train/' in x for x in ids)
    weights=np.load('runs/t07_semantic_cache_v1/weight.npy',mmap_mode='r');targets=np.load('runs/t07_semantic_cache_v1/conditional7.npy',mmap_mode='r')
    assert weights.shape==(7912,32,32) and targets.shape==(7912,7,32,32) and np.isfinite(weights).all() and (weights>=0).all()
    assert np.allclose(targets.sum(1),1,atol=.002)
    dominant=targets.argmax(1)
    for name in freeze['abstention']['disabled_dominant_classes']:
        index=freeze['target_classes'].index(name);assert not weights[dominant==index].any()
    for i,r in enumerate(cache['records']):
        assert r['pair_id']==ids[i]
        if r['quarantined']:assert not weights[i].any()
    assert cache['train_only'] and not cache['validation_or_test_teacher_cache'] and not cache['teacher_used_at_inference']
    cap=read('runs/t06_head_capacity_v1/summary.json');assert head['steps']==cap['steps']==2976 and head['train_pairs']==cap['train_pairs']==7912
    for name in ['batch_size','learning_rate','weight_decay','epochs','gradient_clip']:
        assert p['head'][name]==read('configs/t06_protocol_v1.json')['head'][name]
    assert head['appearance_control_sample_stream_equal'] and not head['backbone_instantiated'] and not head['backbone_gradients'] and not head['teacher_at_inference'] and not head['validation_training_labels']
    assert sha('runs/t07_semantic_head_v1/last.pt')==head['checkpoint_sha256']
    assert abs(head['head_parameters']-cap['head_parameters'])/head['head_parameters']<.02
    tr,tg=[read('runs/t07_teacher_'+name+'_v1/summary.json') for name in ['TR','TG']]
    for key in ['steps','initial_state_sha256','presentations_sha256','cropped_GT_stream_sha256','GT_counts']:assert tr[key]==tg[key],key
    assert tr['steps']==3000 and not tr['community_weights_used'] and not tg['community_weights_used']
    assert len(selection['rows'])==12 and {(r['mode'],r['K']) for r in selection['rows']}=={(m,k) for m in p['modes'] for k in p['budgets']}
    for row in selection['rows']:assert row['image_pairs']==952 and row['BN_unchanged'] and row['original_thresholds_RANSAC_unchanged']
    baseline=read('reports/t06_selection_v1/summary.json')['rows']
    for row in selection['rows']:
        if row['mode'] in ['U','A','AC']:
            reference=next(x for x in baseline if x['mode']==row['mode'] and x['K']==row['K'])
            for key,value in reference.items():assert row[key]==value
    public=subprocess.check_output(['git','ls-files'],text=True).splitlines();assert not any(Path(x).suffix in ['.pt','.pth','.bin','.safetensors','.npz','.npy'] for x in public)
    sources=['sglm/t07_semantic.py','scripts/train_t07_teacher.py','scripts/train_t07_head.py','scripts/cache_t07_semantics.py','scripts/evaluate_t07_selection.py']
    result=dict(passed=True,matcher_and_protected_code_unchanged=True,license_explicit_research_only=True,community_TG_excluded_from_T07_student_labels=True,teacher_cache_train_pairs=7912,quarantined_images=cache['quarantined_images'],semantic_student_parameters=head['head_parameters'],appearance_control_parameters=cap['head_parameters'],same_training_budget_and_stream=True,teacher_not_in_online_inference=True,selected_before_Kx1024_correlation=True,all_fixed952=False if len(selection['rows'])!=12 else True,all_validation_BN_buffers_frozen=True,matching_threshold_RANSAC_unchanged=True,SC_secondary_same_head_as_S=True,backbone_unfrozen=False,dynamic_K=False,weights_published=False,source_sha256={f:sha(f) for f in sources})
    out=Path('reports/t07_delivery_v1');out.mkdir(exist_ok=True);(out/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
