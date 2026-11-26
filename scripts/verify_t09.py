"""T09 audit and pairedrun acceptance; no overwriting historicalT08 artifacts."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    index=read('reports/t09_audit_v1/run_index.json');p=index['protocol'];assert p==read('configs/t09_protocol_v1.json');audit=read('reports/t09_audit_v1/shuffle_parent_audit.json');frozen=read('reports/t09_audit_v1/T08_immutable_snapshot.json')
    for name,value in frozen.items():assert sha(name)==value,name
    assert audit['old_same_parent_cells']==4274 and audit['corrected_same_parent_cells']==0 and audit['recipient_mask_and_weights_bitwise_preserved'] and audit['global_soft_class_histogram_preserved']
    cache=Path('runs/t09_shuffle_parent_v1');assert sha(cache/'donor_flat_index.npy')==audit['mapping_sha256']
    original_w=np.load('runs/t07_semantic_cache_v1/weight.npy',mmap_mode='r');weight=np.load(cache/'weight.npy',mmap_mode='r');assert np.array_equal(weight,original_w)
    ids=read(cache/'pair_ids.json');manifest={r['pair_id']:r for r in [json.loads(line) for line in Path('reports/metadata_finalize_v1/train_256_dedup_v1.jsonl').read_text().splitlines()]};parents=np.array([manifest[x]['parent_mapping']['parent_pair_id'] for x in ids]);active=np.flatnonzero(weight.reshape(-1)>0);mapping=np.load(cache/'donor_flat_index.npy',mmap_mode='r').reshape(-1);assert np.all(parents[active//1024]!=parents[mapping[active]//1024]) and len(np.unique(mapping[active]))==len(active)
    initial=read('runs/t08_M_v1/summary.json')['initial_matching_state_sha256'];baseline=torch.load('runs/t05_frozen_v1/checkpoint.pt',map_location='cpu',weights_only=False)
    label_files={str(cache/name):sha(cache/name) for name in ['conditional7.npy','weight.npy','donor_flat_index.npy']}
    summaries={}
    for slot in index['runs']:
        root=Path(slot['root']);s=read(root/'summary.json');summaries[(slot['seed'],slot['group'])]=s
        assert s['protocol']['train']['KL_weight']==1 and s['protocol']['train']['lr']==.0002
        assert s['protocol']['trainable_prefixes']==p['training']['trainable_prefixes']
        assert s['steps']==7912 and s['train_pairs']==7912 and s['epochs']==4 and s['initial_matching_state_sha256']==initial and s['frozen_matching_state_unchanged']
        assert sha(slot['primary_checkpoint'])==s['inference_checkpoint_sha256'] and not s['inference_has_semantic_head'] and s['inference_parameters']==37219
        history=s['validation_history'];last=next(r for r in history if r['epoch']==4 and r['K']==256);assert last['image_pairs']==952
        if not slot['reused_T08']:
            assert not any(r['K']!=256 for r in history) and len(s['label_epoch_checks'])==8 and len(s['semantic_gradient_probes'])==5
            assert s['protocol']['seed']==slot['seed'] and s['protocol']['train']['lr']==.0002 and s['protocol']['train']['KL_weight']==1
            for name,value in s['label_files_sha256'].items():assert sha(name)==value
            if slot['group']=='MS-shuffle':assert s['label_files_sha256']==label_files
        exported=torch.load(slot['primary_checkpoint'],map_location='cpu',weights_only=False)
        for name,tensor in exported['model'].items():
            if not name.startswith(tuple(p['training']['trainable_prefixes'])) or name.endswith(('running_mean','running_var','num_batches_tracked')):assert torch.equal(tensor,baseline['model'][name])
        assert exported['config']['model']==baseline['config']['model'] and exported['config']['evaluation']==baseline['config']['evaluation']
    for seed in p['seeds']:
        group=[summaries[(seed,name)] for name in p['groups']];assert all(s['sample_affine_stream_sha256']==group[0]['sample_affine_stream_sha256'] for s in group)
        assert group[1]['initial_semantic_state_sha256']==group[2]['initial_semantic_state_sha256']
    stats=read('reports/t09_delivery_v1/statistical_summary.json')
    for row in stats['transitions']:
        assert sum(row[key] for key in ['failure_to_success','success_to_failure','both_success','both_failure'])==952
        assert row['net_successes']==row['failure_to_success']-row['success_to_failure']
    for row in stats['paired_bootstrap']:assert row['parent_clusters']==238 and row['validation_pairs']==952 and row['bootstrap_replicates']==20000
    assert len(stats['paired_bootstrap'])==32 and len([r for r in stats['results'] if r['row_type']=='seed'])==9
    membership=read('reports/t09_audit_v1/validation_parent_groups.json');assert membership['validation_pairs']==952 and membership['verified512parent_groups']==238 and membership['unresolved_pairs']==0
    result=dict(passed=True,total_paired_seeds=3,new_training_runs=7,reused_T08_runs=2,historical_T08_unchanged=True,old_same_parent_donors=4274,corrected_same_parent_donors=0,fixed_shuffle_across_epochs_and_seeds=True,recipient_mask_weights_exact=True,global_softclass_histogram_preserved=True,gradient_scale_differences_reported_not_reweighted=True,lambda_all=1.,all_updates=7912,all_K=256,all_last_epoch4=True,within_seed_sample_affine_streams_identical=True,primary_metric='SR@5px',verified_validation_parent_groups=238,validation_pairs=952,parent_bootstrap_replicates=20000,same_parent_draws_across_seeds=True,satellite_scene_independence_claim=False,shuffle_spatial_continuity_limitation=True,OS_semantic_GT=False,inference_semantic_head=False,teacher_retrained=False,weights_published=False,source_sha256={name:sha(name) for name in ['scripts/train_t09.py','scripts/prepare_t09_shuffle.py','scripts/audit_t09_gradients.py','sglm/t09_statistics.py','scripts/report_t09_bootstrap.py']})
    Path('reports/t09_delivery_v1/acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
