"""Class/qualityconditioned candidate semantic null, excluding sameverified512parent."""
import hashlib,json
from pathlib import Path
import numpy as np
from prepare_t08_shuffle import cross_image_permutation


def main():
    out=Path('runs/t10a_permutation_v1');out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Do not overwrite fixed permutation')
    root=Path('runs/t07_semantic_cache_v1');ids=json.loads((root/'pair_ids.json').read_text());manifest={r['pair_id']:r for r in [json.loads(x) for x in Path('reports/metadata_finalize_v1/train_256_dedup_v1.jsonl').read_text().splitlines()]};names=sorted({manifest[uid]['parent_mapping']['parent_pair_id'] for uid in ids});lookup={name:i for i,name in enumerate(names)};parents=np.array([lookup[manifest[uid]['parent_mapping']['parent_pair_id']] for uid in ids]);assert all(manifest[uid]['parent_mapping']['status']=='verified' for uid in ids)
    raw=np.load(root/'conditional7.npy',mmap_mode='r');flat=np.array(raw).transpose(0,2,3,1).reshape(-1,7);weight=np.load(root/'weight.npy',mmap_mode='r').reshape(-1);active=np.flatnonzero(weight>0);hard=flat.argmax(1);mapping=np.arange(len(weight),dtype=np.int64);rng=np.random.default_rng(20261126);strata=[]
    for c in range(7):
        cells=active[hard[active]==c]
        if not len(cells):continue
        ranked=cells[np.argsort(weight[cells],kind='stable')]
        for n in [8,4,2,1]:
            bins=np.array_split(ranked,n)
            if all(np.unique(parents[b//1024],return_counts=True)[1].max()*2<=len(b) for b in bins):break
        else:raise RuntimeError('Class cannot be crossparent deranged without changing composition')
        for j,b in enumerate(bins):
            permutation=cross_image_permutation(parents[b//1024],rng);mapping[b]=b[permutation];strata.append(dict(dominant_class=c,quality_bins=n,bin=j,cells=len(b),parents=len(np.unique(parents[b//1024])),recipient_weight_mean=float(weight[b].mean()),donor_weight_mean=float(weight[mapping[b]].mean()),mean_abs_weight_difference=float(np.abs(weight[b]-weight[mapping[b]]).mean())))
    assert np.all(parents[active//1024]!=parents[mapping[active]//1024]) and np.array_equal(hard[active],hard[mapping[active]]) and len(np.unique(mapping[active]))==len(active)
    np.save(out/'candidate_donor_flat_index.npy',mapping.reshape(7912,1024))
    plan=json.loads(Path('reports/t10_setup_v1/plan.json').read_text())['samples'];subset_indices=np.array([r['original_cache_index'] for r in plan]);selected=mapping.reshape(7912,1024)[subset_indices];prob=flat[selected].reshape(256,32,32,7).transpose(0,3,1,2);np.save(out/'selected_candidate_probability7.npy',prob)
    summary=dict(pairs=7912,active_cells=len(active),selected_diagnostic_pairs=256,no_same_verified512parent=True,dominant_class_preserved_at_all_active_grid_nodes=True,global_active_probability_vectors_bijective=True,recipient_mask_weights_unchanged=True,quality_strata=strata,query_probabilities_unpermuted=True,candidate_side_only=True,all_donors_train=True,validation_test_used=False,nominal_semantics_not_GT=True,bilinear_sample_note='class is preserved atgridnodes; sampledclassmaychange nearboundaries; strict sampledclassstable comparison separately reported',mapping_sha256=hashlib.sha256((out/'candidate_donor_flat_index.npy').read_bytes()).hexdigest(),selected_probability_sha256=hashlib.sha256((out/'selected_candidate_probability7.npy').read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(summary,indent=2));Path('reports/t10_setup_v1/permutation.json').write_text(json.dumps(summary,indent=2));print('Fixedclassconditionedcrossparentsemanticnull',len(strata),'strata')


if __name__=='__main__':main()
