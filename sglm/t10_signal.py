"""Offline nominal semantic sampling and candidate controls, never a training loss."""
import numpy as np
import torch
import torch.nn.functional as F
from .geometry import invert_affine,transform_points,valid_pixel_centers


def sar_candidate_to_nominal_optical(y,A_forward):
    return transform_points(y,invert_affine(A_forward))


@torch.no_grad()
def sample_nominal_semantics(probabilities7,weight,points_xy):
    """Sample coarse32 optical probabilities at input256 pixel coordinates.

    Require every nonzero bilinear support corner to pass the existing mask.
    Returned semantic probabilities remain externalteacher nominal pseudo distributions.
    """
    p=probabilities7.float();w=weight.float();xy=torch.as_tensor(points_xy,device=p.device,dtype=torch.float32)
    if p.shape!=(7,32,32) or w.shape!=(32,32) or xy.ndim!=2 or xy.shape[1]!=2:raise ValueError('Expected7x32x32,32x32,Nx2')
    p=p/p.sum(0,keepdim=True).clamp_min(1e-12)
    grid=(2*(xy+.5)/256-1)[None,None]
    sampled=F.grid_sample(p[None],grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0,:,0].T
    sampled_w=F.grid_sample(w[None,None],grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0,0,0]
    feature=(xy+.5)/8-.5;low=feature.floor().long();fraction=feature-low.float();valid=((xy>=3.5)&(xy<=251.5)).all(1)
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        corner=low+low.new_tensor([dx,dy]);coefficient=(fraction[:,0] if dx else 1-fraction[:,0])*(fraction[:,1] if dy else 1-fraction[:,1]);needed=coefficient>1e-7
        inside=((corner>=0)&(corner<32)).all(1);safe=corner.clamp(0,31);good=inside&(w[safe[:,1],safe[:,0]]>0);valid&=(~needed)|good
    valid&=torch.isfinite(sampled).all(1)&(sampled.sum(1)>0)&(sampled_w>0)
    sampled=sampled/sampled.sum(1,keepdim=True).clamp_min(1e-12)
    return sampled,sampled_w,valid


def semantic_cosine(query,candidates,epsilon=1e-8):
    q=np.asarray(query,float);c=np.asarray(candidates,float)
    return np.sum(q*c,axis=-1)/(np.linalg.norm(q,axis=-1)*np.linalg.norm(c,axis=-1)+epsilon)


def negative_pool(query_truth,candidate_xy,candidate_geometric_valid,exclude_radius=16.):
    distances=np.linalg.norm(np.asarray(candidate_xy)[None]-np.asarray(query_truth)[:,None],axis=-1)
    return (distances>exclude_radius)&np.asarray(candidate_geometric_valid,bool)[None],distances


def choose_controls(valid_pool,hard_indices,distances,candidate_weights,candidate_classes,hard_index,rng,repeats=5):
    available=np.asarray(valid_pool,bool).copy();available[np.asarray(hard_indices,dtype=int)]=False
    db=np.floor(np.asarray(distances)/16).astype(int);wb=np.floor(np.clip(candidate_weights,0,.999999)*10).astype(int)
    same_distance_weight=available&(db==db[hard_index])&(wb==wb[hard_index])
    pools=[available,same_distance_weight,same_distance_weight&(candidate_classes==candidate_classes[hard_index])]
    choices=[]
    for pool in pools:
        indices=np.flatnonzero(pool);result=np.full(repeats,-1,np.int16)
        if len(indices):
            selected=rng.choice(indices,min(repeats,len(indices)),replace=False);result[:len(selected)]=selected
        choices.append(result)
    return choices


def paired_parent_summary(pair_indices,query_indices,values,all_pair_parent_codes,replicates=20000,seed=20261126):
    """Equalweight slots withinquery, queries withinimage, images within knownparent."""
    pair_indices=np.asarray(pair_indices,int);query_indices=np.asarray(query_indices,int);v=np.asarray(values,float);parent_codes=np.asarray(all_pair_parent_codes,int)
    finite=np.isfinite(v);pairs=pair_indices[finite];queries=query_indices[finite];v=v[finite]
    pair_means=np.full(len(parent_codes),np.nan);query_count=np.zeros(len(parent_codes),int)
    if len(v):
        keys=pairs*1024+queries;unique,inverse=np.unique(keys,return_inverse=True);sums=np.bincount(inverse,weights=v);counts=np.bincount(inverse);qmean=sums/counts;query_pairs=unique//1024
        for pair in np.unique(query_pairs):pair_means[pair]=qmean[query_pairs==pair].mean();query_count[pair]=int((query_pairs==pair).sum())
    parents=np.unique(parent_codes);parent_means=np.full(len(parents),np.nan)
    for i,parent in enumerate(parents):
        children=pair_means[parent_codes==parent];defined=children[np.isfinite(children)]
        if len(defined):parent_means[i]=defined.mean()
    defined=np.isfinite(parent_means)
    if not defined.any():return dict(mean=None,CI95_low=None,CI95_high=None,valid_pairs=0,valid_parents=0,total_pairs=len(parent_codes),total_parents=len(parents)),pair_means,parent_means,query_count
    rng=np.random.default_rng(seed);draw=parent_means[rng.integers(0,len(parents),size=(replicates,len(parents)))];n=np.isfinite(draw).sum(1);averages=np.nansum(draw,1)/np.maximum(n,1);averages=averages[n>0]
    result=dict(mean=float(parent_means[defined].mean()),CI95_low=float(np.percentile(averages,2.5)),CI95_high=float(np.percentile(averages,97.5)),valid_pairs=int(np.isfinite(pair_means).sum()),valid_parents=int(defined.sum()),total_pairs=len(parent_codes),total_parents=len(parents),defined_parent_bootstraps=len(averages),relations=int(len(v)),queries=int(query_count.sum()))
    return result,pair_means,parent_means,query_count


def hierarchical_relation_weights(pair_indices,query_indices,all_pair_parent_codes):
    pair=np.asarray(pair_indices,int);query=np.asarray(query_indices,int);parents=np.asarray(all_pair_parent_codes,int)
    if not len(pair):return np.array([],float)
    keys=pair*1024+query;unique,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True);qp=unique//1024;queries_per_pair=np.bincount(qp,minlength=len(parents));valid_pairs=np.flatnonzero(queries_per_pair>0);active_parents=np.unique(parents[valid_pairs]);pairs_per_parent=np.bincount(parents[valid_pairs],minlength=parents.max()+1)
    weight=1/(counts[inverse]*queries_per_pair[pair]*pairs_per_parent[parents[pair]]*len(active_parents))
    return weight
