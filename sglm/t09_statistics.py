"""Paired resampling of verified512parent clusters, with trainingseeds held fixed."""
import numpy as np


def metric_values(rmse):
    rmse=np.asarray(rmse,float)
    if rmse.ndim!=2:raise ValueError('Expected seed x image RMSE; NaN means solver failure')
    result=np.stack([(rmse<5).mean(1),(rmse<3).mean(1),np.nanmedian(rmse,axis=1),np.nanpercentile(rmse,90,axis=1)],1)
    return result


def parent_sample_indices(parent_ids,draws):
    parents=np.asarray(parent_ids);unique=np.unique(parents);draws=np.asarray(draws,dtype=int)
    if len(draws)!=len(unique) or (draws<0).any() or (draws>=len(unique)).any():raise ValueError('One parent draw per original parent required')
    groups=[np.flatnonzero(parents==uid) for uid in unique]
    return np.concatenate([groups[i] for i in draws])


def paired_cluster_bootstrap(reference,variant,parent_ids,replicates=20000,seed=20261126):
    reference,variant=np.asarray(reference,float),np.asarray(variant,float);parents=np.asarray(parent_ids)
    if reference.shape!=variant.shape or reference.ndim!=2 or reference.shape[1]!=len(parents):raise ValueError('Paired seeds/images must align')
    if len(np.unique(parents))<2:raise ValueError('At least2parentclusters required')
    groups=[np.flatnonzero(parents==uid) for uid in np.unique(parents)];rng=np.random.default_rng(seed)
    delta=metric_values(variant)-metric_values(reference);samples=np.empty((replicates,reference.shape[0]+1,4))
    for r in range(replicates):
        draws=rng.integers(0,len(groups),size=len(groups));idx=np.concatenate([groups[i] for i in draws]);d=metric_values(variant[:,idx])-metric_values(reference[:,idx]);samples[r,:-1]=d;samples[r,-1]=np.mean(d,axis=0)
    return dict(point=np.concatenate([delta,delta.mean(0,keepdims=True)],0),low=np.nanpercentile(samples,2.5,axis=0),high=np.nanpercentile(samples,97.5,axis=0),samples=samples,undefined_replicates=np.isnan(samples).sum(0))
