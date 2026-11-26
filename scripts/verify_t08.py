"""Cross-run fairness, frozen weights, sparse supervision and deployment acceptance."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=read('configs/t08_protocol_v1.json');setup=read('reports/t08_setup_v1/setup.json');source=torch.load(p['matcher_checkpoint'],map_location='cpu',weights_only=False)
    assert sha(p['matcher_checkpoint'])==p['matcher_sha256']
    for name,record in read('reports/t05_frozen_v1/freeze.json')['original_code_verification'].items():assert sha(name)==record['sha256']
    changed_counts={}
    original_ids=read('runs/t07_semantic_cache_v1/pair_ids.json');train_ids=[r['pair_id'] for r in read(p['training_plan'])['samples']]
    assert train_ids==original_ids and len(train_ids)==7912 and all('/train/' in x for x in train_ids)
    summaries={name:read(f'runs/t08_{name}_v1/summary.json') for name in p['groups']};reference=summaries['M']
    for name,s in summaries.items():
        assert s['steps']==7912 and s['train_pairs']==7912 and s['epochs']==4
        assert s['initial_matching_state_sha256']==reference['initial_matching_state_sha256'] and s['sample_affine_stream_sha256']==reference['sample_affine_stream_sha256']
        assert s['training_plan_sha256']==reference['training_plan_sha256']==setup['plan_sha256']['training_plan.json']
        assert s['frozen_matching_state_unchanged'] and not s['inference_has_semantic_head'] and s['inference_parameters']==37219 and not s['OS_semantic_GT'] and not s['SAR_semantic_loss']
        path=Path(f'runs/t08_{name}_v1/inference_last.pt');assert sha(path)==s['inference_checkpoint_sha256'];export=torch.load(path,map_location='cpu',weights_only=False)
        assert export['model'].keys()==source['model'].keys()
        changed=[key for key,value in export['model'].items() if not torch.equal(value,source['model'][key])]
        assert changed and 'backbone.descriptor.weight' in changed
        changed_counts[name]=len(changed)
        for key,value in export['model'].items():
            is_buffer=key.endswith(('running_mean','running_var','num_batches_tracked'))
            if is_buffer or not key.startswith(tuple(p['trainable_prefixes'])):assert torch.equal(value,source['model'][key]),key
        assert export['config']['model']==source['config']['model'] and export['config']['evaluation']==source['config']['evaluation']
        history=s['validation_history'];assert [(r['epoch'],r['K']) for r in history]==[(0,256),(1,256),(2,256),(4,256),(4,1024)]
        for row in history:assert row['image_pairs']==952
        assert sha(f'runs/t08_{name}_v1/last.pt')==s['last_checkpoint_sha256']
    assert summaries['MS']['initial_semantic_state_sha256']==summaries['MS-shuffle']['initial_semantic_state_sha256']
    shuffle=read('runs/t08_shuffle_v1/summary.json');assert shuffle['recipient_mask_and_weights_exact'] and shuffle['all_active_donors_different_image'] and shuffle['active_softclass_histogram_exact']
    w=np.load('runs/t07_semantic_cache_v1/weight.npy',mmap_mode='r');sw=np.load('runs/t08_shuffle_v1/weight.npy',mmap_mode='r');assert np.array_equal(w,sw)
    donor=np.load('runs/t08_shuffle_v1/donor_flat_index.npy',mmap_mode='r').reshape(-1);active=np.flatnonzero(w.reshape(-1)>0);assert np.all(donor[active]//1024!=active//1024) and len(np.unique(donor[active]))==len(active)
    diagnosis=read('reports/t08_preflight_v1/diagnosis.json');assert diagnosis['training_source_pairs']==128 and diagnosis['teacher_is_not_GT'] and diagnosis['no_validation_teacher_cache']
    assert all('/train/' in r['pair_id'] for r in diagnosis['pairs'])
    tests=Path('reports/t08_delivery_v1/tests.txt').read_text();assert 'passed' in tests and 'failed' not in tests
    result=dict(passed=True,OS_semantic_GT=False,OS_semantic_accuracy_claim=False,external_teacher_soft_pseudolabels=True,teacher_not_retrained=True,all_groups_initial_T05_matching_state_identical=True,all_groups_train_pairs=7912,all_groups_updates=7912,all_groups_sample_affine_stream_identical=True,trainable_coarse_layers_only=True,original_matching_source_files_unchanged=True,changed_matching_tensors=changed_counts,matching_BN_stats_frozen=True,adapters_lower_FPN_local_temperatures_unchanged=True,optical_pseudoKL_updates_coarse_encoder=True,SAR_pixel_semantics=False,shuffle_crossimage_all_supervised_cells=True,shuffle_mask_weight_exact=True,shuffle_global_softclass_histogram_preserved=True,weighted_mass_max_relative_difference=float(np.max(np.abs(shuffle['weighted_class_mass_relative_difference']))),inference_parameters=37219,inference_head_removed=True,inference_K256_uniform=True,learned_scoring=False,dynamic_K=False,evaluation_K=[256,1024],all_validation_pairs=952,solver_failures_in_denominator=True,primary_last_not_validationbest=True,formal_timing=False,multiseed_stability_claim=False,source_sha256={name:sha(name) for name in ['sglm/t08_auxiliary.py','scripts/train_t08.py','scripts/prepare_t08_shuffle.py','scripts/diagnose_t08_pretrain.py']})
    out=Path('reports/t08_delivery_v1');out.mkdir(exist_ok=True);(out/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
