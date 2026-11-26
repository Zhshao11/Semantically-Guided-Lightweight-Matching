"""Fixed cross-image active-label derangement with exact recipient masks/weights."""
import hashlib,json
from pathlib import Path
import numpy as np


def cross_image_permutation(owners,rng):
    unique,counts=np.unique(owners,return_counts=True)
    if counts.max()*2>len(owners):raise ValueError('Cannot derange across images in this stratum')
    order=[]
    for owner in rng.permutation(unique):
        indices=np.flatnonzero(owners==owner);order.extend(rng.permutation(indices).tolist())
    order=np.asarray(order);donor=np.empty(len(owners),np.int64);donor[order]=np.roll(order,-int(counts.max()))
    if np.any(owners==owners[donor]):raise AssertionError('Same-image donor remains')
    return donor


def main():
    out=Path('runs/t08_shuffle_v1');out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('No overwrite shuffle')
    setup=json.loads(Path('reports/t08_setup_v1/setup.json').read_text());ids=np.asarray(setup['train_original_indices']);cache=Path('runs/t07_semantic_cache_v1');p=np.array(np.load(cache/'conditional7.npy',mmap_mode='r')[ids],copy=True);w=np.array(np.load(cache/'weight.npy',mmap_mode='r')[ids],copy=True)
    flat=p.transpose(0,2,3,1).reshape(-1,7);weight=w.reshape(-1);active=np.flatnonzero(weight>0);owners=active//1024;rng=np.random.default_rng(20261126)
    #32 equal-count weight strata minimize weighted-histogram drift without moving any weights.
    order=np.argsort(weight[active],kind='stable');mapping=np.arange(len(weight),dtype=np.int64);bins=[]
    for selected in np.array_split(order,32):
        local=cross_image_permutation(owners[selected],rng);recipient=active[selected];mapping[recipient]=active[selected[local]]
        bins.append(dict(cells=len(selected),weight_min=float(weight[recipient].min()),weight_max=float(weight[recipient].max())))
    assert np.unique(mapping[active]).size==len(active) and np.all(mapping[active]//1024!=active//1024)
    shuffled=flat[mapping];original_hist=flat[active].astype(np.float64).sum(0);shuffled_hist=shuffled[active].astype(np.float64).sum(0)
    assert np.allclose(original_hist,shuffled_hist,rtol=0,atol=1e-8)
    before=(flat.astype(np.float64)*weight[:,None]).sum(0);after=(shuffled.astype(np.float64)*weight[:,None]).sum(0)
    np.save(out/'donor_flat_index.npy',mapping.reshape(len(ids),1024));np.save(out/'conditional7.npy',shuffled.reshape(len(ids),32,32,7).transpose(0,3,1,2));np.save(out/'weight.npy',w)
    result=dict(train_pairs=len(ids),active_cells=len(active),all_active_donors_different_image=True,recipient_mask_and_weights_exact=True,active_softclass_histogram_exact=True,original_active_class_mass=original_hist.tolist(),shuffled_active_class_mass=shuffled_hist.tolist(),original_weighted_class_mass=before.tolist(),shuffled_weighted_class_mass=after.tolist(),weighted_class_mass_relative_difference=((after-before)/np.maximum(before,1e-12)).tolist(),weight_strata=bins,limitations='preserves global active softclass mass and recipient mask/weights, not eachimage class histogram or exact weighted class mass; donor probabilities are crossimage at celllevel',training_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),mapping_sha256=hashlib.sha256((out/'donor_flat_index.npy').read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='weight_strata'},indent=2))


if __name__=='__main__':main()
