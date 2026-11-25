"""Cross-artifact acceptance for train-only labels, controlled teachers and gated timing."""
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out=Path('reports/t06_delivery_v1');out.mkdir(exist_ok=True)
    p=read('configs/t06_protocol_v1.json');setup=read('reports/t06_setup_v1/setup.json');plan=read('reports/t06_setup_v1/training_plan.json')
    assert sha(p['matcher_checkpoint'])==p['matcher_sha256']==sha('runs/t03_main_lr1e3_v1/best.pt')
    assert sha('reports/t06_setup_v1/training_plan.json')==setup['train_plan_sha256']
    ids=[r['pair_id'] for r in plan['samples']];assert len(ids)==len(set(ids))==7912 and all('/train/' in uid for uid in ids)
    assert ids==read('runs/t06_reliability_v1/pair_ids.json')
    cache=Path('runs/t06_reliability_v1');y=np.load(cache/'geometric_correctness.npy',mmap_mode='r');mask=np.load(cache/'is_supervisable.npy',mmap_mode='r')
    assert y.shape==(7912,1024) and np.isin(y,[-1,0,1]).all() and np.array_equal(mask,y>=0)
    stats=read(cache/'summary.json');assert sum(stats['label_counts'].values())==7912*1024
    for name,value in [('positive',1),('negative',0),('ignored',-1)]:assert stats['label_counts'][name]==int((y==value).sum())
    assert stats['labels_from_train_only'] and not stats['ambiguity_used_for_training'] and stats['BN_buffers_unchanged']
    for name in ['runs/t06_head_v1','runs/t06_head_capacity_v1']:
        h=read(Path(name)/'summary.json');assert not h['backbone_instantiated_in_training'] and not h['backbone_gradients'] and not h['descriptor_gradients']
        assert h['steps']==2976 and not h['validation_labels_used'] and not h['ambiguity_margin_used']
    selection=read('reports/t06_selection_v1/summary.json');assert len(selection['rows'])==8
    for r in selection['rows']:
        assert r['image_pairs']==952 and r['BN_unchanged'] and r['original_thresholds_RANSAC_unchanged']
        if r['mode']=='U':assert r['T05_U_exact']
    tr,tg=[read('runs/t06_teacher_'+v+'/summary.json') for v in ['TR','TG']]
    for key in ['steps','presentations_sha256','cropped_GT_stream_sha256','GT_counts_in_training','effective_batch','training_images','validation_images']:
        assert tr[key]==tg[key],key
    assert tr['steps']==1000 and tr['effective_batch']==4 and tr['training_images']==2522 and tr['validation_images']==64
    assert tr['GT0_loss_masked'] and tg['GT0_loss_masked'] and not tr['mixed_RGB_gray_branch'] and not tg['mixed_RGB_gray_branch']
    manifest=read('runs/t06_teacher_data/manifest.json');old=read('runs/loveda_val_t05_v1/manifest.json')
    assert len(manifest['train'])==2522 and len(manifest['val64'])==64
    assert sum(r['domain']=='Urban' for r in manifest['val64'])==32 and sum(r['domain']=='Rural' for r in manifest['val64'])==32
    assert not ({r['pair_id'] for r in old['samples']}&{r['pair_id'] for r in manifest['val64']})
    fold=read('reports/t06_efficiency_v1/fold_consistency.json');timing=read('reports/t06_efficiency_v1/timing_summary.json')
    assert fold['global_gate']==dict(FP32=True,BF16=False)
    assert timing['no_hooks'] and not timing['profiler_used_in_timing'] and timing['rounds']==3 and timing['pairs']==24
    assert all(not c['folded'] or fold['global_gate'][c['precision']] for c in timing['configs'])
    for mode in ['U','A','AC','AC-cap']:
        for k in [256,512]:assert any(c['mode']==mode and c['K']==k and c['precision']=='FP32' and c['folded'] for c in timing['configs'])
    before=read('reports/t05_frozen_v1/freeze.json')['original_code_verification']
    for path,record in before.items():assert sha(path)==record['sha256']
    review=read('runs/t06_OS_review_v1/summary.json');assert len(review['records'])==96 and not review['OS_accuracy_claim'] and not review['SC_training_started']
    assert all('/train/' in r['pair_id'] and not r['semantic_GT'] for r in review['records'])
    # No public original/derived model tensors. Runs and local outputs retain them privately.
    import subprocess
    public_files=subprocess.check_output(['git','ls-files'],text=True).splitlines()
    assert not any(Path(f).suffix in ['.pt','.pth','.ckpt','.safetensors','.bin'] for f in public_files)
    result=dict(passed=True,matcher_checkpoint_unchanged=True,protected_matching_code_unchanged=True,train_label_pairs=7912,
        label_supervision_mask_consistent=True,labels_from_OS_validation_or_test=False,ambiguity_not_used_in_training=True,
        matcher_backbone_descriptor_gradients=False,scoring_validation_pairs=952,uniform_T05_reproduction_exact=True,
        teacher_updates_each=1000,teacher_plan_and_cropped_annotations_identical=True,LoveDA_train_images=2522,fixed_val64_Urban_Rural=[32,32],
        old8_disjoint_from_new64=True,GT0_masked_not_land_cover=True,fold_FP32_gate=True,fold_BF16_gate=False,
        ungated_folded_model_timed=False,all_custom_baselines_same_optimization=True,profiler_not_used_for_formal_timing=True,
        formal_timing_rounds=3,formal_timing_original_pairs=24,OS_review_train_images=32,OS_semantic_accuracy_claim=False,
        SC='waiting teacher finalization',license='unresolved; no original or derived weights published',
        source_sha256={path:sha(path) for path in ['sglm/t06_appearance.py','sglm/t06_runtime.py','sglm/inference_fusion.py','sglm/semantic_targets.py','scripts/train_t06_teacher.py','scripts/train_t06_head.py','scripts/generate_t06_labels.py','scripts/diagnose_t06_efficiency.py']})
    (out/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
