"""Frozen probabilities to coordinates. Ground truth is absent from readout APIs."""
import numpy as np
import torch


def grid_delta(side,device):
    if side<1 or side%2!=1:raise ValueError('Odd square local window required')
    radius=side//2;y,x=torch.meshgrid(torch.arange(-radius,radius+1,device=device),torch.arange(-radius,radius+1,device=device),indexing='ij')
    return torch.stack([x,y],-1).reshape(-1,2).float()


def peak_neighborhood(peak_indices,side,valid):
    locations=torch.arange(side*side,device=valid.device)
    return valid&((locations[None]//side-peak_indices[:,None]//side).abs()<=1)&((locations[None]%side-peak_indices[:,None]%side).abs()<=1)


def read_frozen_heatmap(heatmap,valid,centers,sample_positions):
    """Return A(full),B(argmax),C(local3x3) for every supplied candidate.

    Caller controls precision context. A/C share the original center+2*(p@delta)
    arithmetic. Temperature and probabilities are never recomputed or modified.
    """
    if heatmap.ndim!=2 or valid.shape!=heatmap.shape:raise ValueError('Expected NxW heatmap and mask')
    side=int(heatmap.shape[1]**.5)
    if side*side!=heatmap.shape[1] or side%2!=1:raise ValueError('Odd square local window required')
    if centers.shape!=(len(heatmap),2) or sample_positions.shape!=(len(heatmap),side*side,2):raise ValueError('Misaligned coordinate packet')
    if not len(heatmap):
        return dict(A=centers.clone(),B=centers.clone(),C=centers.clone(),peak_index=torch.empty(0,dtype=torch.long,device=centers.device),local_mask=valid.clone(),local_mass=heatmap.new_empty(0))
    if not bool(valid.any(-1).all()):raise ValueError('All-invalid candidates must remain excluded by frozen original filter')
    if not bool(torch.isfinite(heatmap).all() and (heatmap>=0).all()):raise ValueError('Invalid probability')
    if not bool((heatmap[~valid]==0).all()):raise ValueError('Original invalid token probability must be zero')
    delta=grid_delta(side,heatmap.device)
    if not torch.equal(sample_positions,centers[:,None]+2*delta[None]):raise ValueError('Sampling grid differs from original readout')
    peak=heatmap.masked_fill(~valid,-torch.inf).argmax(-1)
    neighborhood=peak_neighborhood(peak,side,valid);mass=(heatmap*neighborhood).sum(-1)
    if not bool((mass>0).all()):raise ValueError('Peak neighborhood has zero mass')
    local=heatmap*neighborhood/mass[:,None]
    A=centers+2*(heatmap@delta);B=sample_positions[torch.arange(len(heatmap),device=heatmap.device),peak];C=centers+2*(local@delta)
    return dict(A=A,B=B,C=C,peak_index=peak,local_mask=neighborhood,local_mass=mass)


def describe_frozen_heatmap(heatmap,valid,positions,nominal_truth,peak_indices):
    """GT-only post-hoc interpretation; never called by a prediction decision."""
    p=np.asarray(heatmap,np.float64);valid=np.asarray(valid,bool);positions=np.asarray(positions,float);truth=np.asarray(nominal_truth,float);peak=np.asarray(peak_indices,int)
    N,W=p.shape;side=int(W**.5);locations=np.arange(W);row=locations//side;col=locations%side
    if not N:return {key:np.empty(0) for key in ['truth_mass','peak_mass','peak_probability','peak_GT_distance','secondary_probability','secondary_ratio','competing','secondary_index','truth_nearest_index','peak_inside_truth3x3','normalized_entropy','max_plateau_size']}
    nearest=np.where(valid,np.linalg.norm(positions-truth[:,None],axis=-1),np.inf).argmin(-1)
    around_truth=valid&(np.abs(row[None]-row[nearest,None])<=1)&(np.abs(col[None]-col[nearest,None])<=1)
    around_peak=valid&(np.abs(row[None]-row[peak,None])<=1)&(np.abs(col[None]-col[peak,None])<=1)
    local_max=np.zeros_like(valid)
    for k in range(W):
        neighbors=(np.abs(row-row[k])<=1)&(np.abs(col-col[k])<=1)&(locations!=k)
        values=np.where(valid[:,neighbors],p[:,neighbors],-np.inf)
        ge=(p[:,k,None]>=values).all(-1)
        strict=((p[:,k,None]>values)&valid[:,neighbors]).any(-1)
        local_max[:,k]=valid[:,k]&ge&strict
    alternatives=np.where(local_max&~around_peak,p,-np.inf);second_index=alternatives.argmax(-1);second=alternatives[np.arange(N),second_index];exists=np.isfinite(second)
    second=np.where(exists,second,0);second_index=np.where(exists,second_index,-1);main=p[np.arange(N),peak];ratio=second/main
    entropy=-(np.where(p>0,p*np.log(np.clip(p,1e-300,None)),0)).sum(-1);normalizer=np.log(np.maximum(valid.sum(-1),2))
    return dict(truth_mass=(p*around_truth).sum(-1),peak_mass=(p*around_peak).sum(-1),peak_probability=main,
        peak_GT_distance=np.linalg.norm(positions[np.arange(N),peak]-truth,axis=-1),secondary_probability=second,secondary_ratio=ratio,
        competing=exists&(ratio>=.5),secondary_index=second_index,truth_nearest_index=nearest,peak_inside_truth3x3=around_truth[np.arange(N),peak],
        normalized_entropy=entropy/normalizer,max_plateau_size=((p==main[:,None])&valid).sum(-1))
