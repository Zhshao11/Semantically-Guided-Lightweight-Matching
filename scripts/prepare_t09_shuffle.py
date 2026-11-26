"""Audit T08 and minimally repair same512parent donors within fixed weight strata."""
import hashlib,json
from pathlib import Path
import numpy as np
from sglm.experiment_io import write_csv


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out=Path('runs/t09_shuffle_parent_v1');out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Do not overwrite corrected shuffle')
    audit=Path('reports/t09_audit_v1');audit.mkdir(exist_ok=True)
    source=Path('runs/t08_shuffle_v1');cache=Path('runs/t07_semantic_cache_v1');ids=json.loads((cache/'pair_ids.json').read_text());manifest={r['pair_id']:r for r in [json.loads(x) for x in Path('reports/metadata_finalize_v1/train_256_dedup_v1.jsonl').read_text().splitlines()]}
    assert all(manifest[x]['parent_mapping']['status']=='verified' for x in ids)
    names=sorted({manifest[x]['parent_mapping']['parent_pair_id'] for x in ids});lookup={name:i for i,name in enumerate(names)};parents=np.array([lookup[manifest[x]['parent_mapping']['parent_pair_id']] for x in ids])
    original=np.load(source/'donor_flat_index.npy').reshape(-1);mapping=original.copy();w=np.load(cache/'weight.npy',mmap_mode='r');weight=w.reshape(-1);active=np.flatnonzero(weight>0);old_same=parents[active//1024]==parents[mapping[active]//1024];assert sha(source/'donor_flat_index.npy')==json.loads((source/'summary.json').read_text())['mapping_sha256']
    # Preserve original32strata and all valid assignments except paired swaps needed for exclusions.
    ranked=np.argsort(weight[active],kind='stable');rng=np.random.default_rng(20261126);swaps=[]
    for bin_id,selected in enumerate(np.array_split(ranked,32)):
        recipient=active[selected];bad=recipient[parents[recipient//1024]==parents[mapping[recipient]//1024]]
        for i in bad:
            if parents[i//1024]!=parents[mapping[i]//1024]:continue
            for attempt in range(100000):
                j=int(recipient[rng.integers(len(recipient))])
                if i!=j and parents[i//1024]!=parents[mapping[j]//1024] and parents[j//1024]!=parents[mapping[i]//1024]:break
            else:raise RuntimeError('Could not repair parent exclusion')
            left,right=int(mapping[i]),int(mapping[j]);mapping[i],mapping[j]=right,left
            swaps.append(dict(weight_stratum=bin_id,recipient_flat=int(i),partner_flat=j,old_donor_left=left,old_donor_right=right))
    assert np.all(parents[active//1024]!=parents[mapping[active]//1024]) and np.array_equal(np.sort(mapping[active]),np.sort(original[active]))
    p=np.array(np.load(cache/'conditional7.npy',mmap_mode='r'));flat=p.transpose(0,2,3,1).reshape(-1,7);shuffled=flat[mapping]
    before=flat[active].astype(np.float64).sum(0);after=shuffled[active].astype(np.float64).sum(0);assert np.array_equal(before,after)
    weighted_before=(flat.astype(np.float64)*weight[:,None]).sum(0);weighted_after=(shuffled.astype(np.float64)*weight[:,None]).sum(0)
    np.save(out/'donor_flat_index.npy',mapping.reshape(7912,1024));np.save(out/'conditional7.npy',shuffled.reshape(7912,32,32,7).transpose(0,3,1,2));np.save(out/'weight.npy',w)
    def entropy(prob):
        prob=prob.astype(np.float64);prob/=np.maximum(prob.sum(-1,keepdims=True),1e-12);return -(prob*np.log(np.maximum(prob,1e-12))).sum(-1)
    e=entropy(flat[active]);er=entropy(shuffled[active]);assert abs(e.sum()-er.sum())<1e-7
    result=dict(corrected_version='T09-parent-exclusion-v1',pairs=7912,verified512parents=len(names),active_cells=len(active),old_same_parent_cells=int(old_same.sum()),old_same_parent_fraction=float(old_same.mean()),old_recipient_images_with_same_parent=len(np.unique(active[old_same]//1024)),corrected_same_parent_cells=0,swaps=len(swaps),changed_recipient_cells=int((mapping[active]!=original[active]).sum()),unchanged_recipient_fraction=float((mapping[active]==original[active]).mean()),recipient_mask_and_weights_bitwise_preserved=True,global_soft_class_histogram_preserved=True,global_entropy_histogram_preserved=True,active_class_mass=before.tolist(),original_weighted_class_mass=weighted_before.tolist(),corrected_weighted_class_mass=weighted_after.tolist(),weighted_class_mass_relative_difference=((weighted_after-weighted_before)/np.maximum(weighted_before,1e-12)).tolist(),accepted_label_entropy_mean_nats=float(e.mean()),accepted_label_entropy_P10_P50_P90=np.percentile(e,[10,50,90]).tolist(),mapping_sha256=sha(out/'donor_flat_index.npy'),old_mapping_sha256=sha(source/'donor_flat_index.npy'),fixed_for_all_epochs_and_all_training_seeds=True,T08_files_overwritten=False,limitations='cellwise shuffle also destroys spatial continuity; known512parent exclusion does not establish independent satellite scenes')
    (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'swap_records.json').write_text(json.dumps(swaps,separators=(',',':')));(out/'pair_ids.json').write_text(json.dumps(ids))
    (audit/'shuffle_parent_audit.json').write_text(json.dumps(result,indent=2));write_csv(audit/'donor_repairs.csv',swaps)
    fixed_evidence=dict(T08_loader_path='runs/t08_shuffle_v1/conditional7.npy and weight.npy loaded as read-only memmap once per TrainingPairs initialization',T08_mapping_generator='one frozen mapping outside training loop; no epoch-dependent regeneration',source_files_sha256={name:sha(name) for name in ['scripts/train_t08.py','scripts/prepare_t08_shuffle.py']},old_mapping_file_matches_saved_summary=True,T08_epochlogs_did_not_store_mappinghash_each_epoch=True,claim_scope='source inspection and current artifact hash establish fixed mapping by implementation; no invented historic per-epoch hash',T09_enforcement='every epoch hashchecks mapping/probability/weight files before and after; same cache across3seeds')
    (audit/'mapping_fixedness.json').write_text(json.dumps(fixed_evidence,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
