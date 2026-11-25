"""Train-only reliability targets and pre-correlation appearance query selection."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .model import coarse_xy
from .t05 import UniformB0,uniform_queries


def sar_global(features):
    mask=features['sar_coarse_mask'][:,None].float()
    return (features['sar_coarse'].float()*mask).sum((2,3))/mask.sum((2,3)).clamp_min(1)


class AppearanceHead(nn.Module):
    def __init__(self,width=16):
        super().__init__();self.width=width
        self.optical=nn.Conv2d(64,width,1);self.sar=nn.Linear(64,width)
        self.score=nn.Sequential(nn.Conv2d(2*width,2*width,1),nn.ReLU(),
            nn.Conv2d(2*width,2*width,3,padding=1,groups=2*width),nn.ReLU(),nn.Conv2d(2*width,1,1))
    def forward(self,optical,sar_mean):
        # Head training and prediction both FP32; frozen BF16 encoder outputs are promoted.
        with torch.autocast(optical.device.type,enabled=False):
            optical,sar_mean=optical.detach().float(),sar_mean.detach().float()
            x=F.relu(self.optical(optical));z=F.relu(self.sar(sar_mean))[:,:,None,None].expand_as(x)
            return self.score(torch.cat([x,z],1))[:,0]


def capped_quotas(capacities,weights,k):
    if sum(capacities)<k:raise ValueError('Insufficient valid coverage capacity')
    quota=[min(2,c) for c in capacities]
    if sum(quota)>k:raise ValueError('Budget below base quota')
    while sum(quota)<k:
        remaining=k-sum(quota);active=[i for i,c in enumerate(capacities) if quota[i]<c]
        total=sum(weights[i] for i in active);shares={i:remaining*weights[i]/total for i in active};added=0
        for i in active:
            n=min(capacities[i]-quota[i],int(shares[i]));quota[i]+=n;added+=n
        if not added:
            for i in sorted(active,key=lambda j:(-(shares[j]%1),j))[:remaining]:quota[i]+=1
    return quota


@torch.no_grad()
def coverage_queries(score,valid,k):
    cells=score.reshape(4,8,4,8).permute(0,2,1,3).reshape(16,64)
    good=valid.reshape(4,8,4,8).permute(0,2,1,3).reshape(16,64)
    ordered=torch.argsort(cells.masked_fill(~good,-torch.inf),dim=1,descending=True,stable=True)
    top=cells.masked_fill(~good,-torch.inf).gather(1,ordered[:,:8]);finite=torch.isfinite(top)
    utility=torch.where(finite,top,0).sum(1)/finite.sum(1).clamp_min(1)
    utility=utility.masked_fill(~good.any(1),-1e4)
    capacities=good.sum(1).clamp_max(32).cpu().tolist();weights=utility.softmax(0).cpu().tolist()
    quotas=torch.tensor(capped_quotas(capacities,weights,k),device=score.device)
    ids=torch.arange(1024,device=score.device).reshape(4,8,4,8).permute(0,2,1,3).reshape(16,64)
    chosen=ids.gather(1,ordered)[torch.arange(64,device=score.device)[None]<quotas[:,None]]
    return chosen.sort().values


class AppearanceB0(UniformB0):
    def __init__(self,config,k=256,selection='U',head=None):
        super().__init__(config,k);self.selection=selection;self.appearance_head=head
    def correlate(self,features):
        if self.selection=='U':return super().correlate(features)
        if len(features['optical_coarse'])!=1:raise ValueError('Selection inference uses batch1')
        logits=self.appearance_head(features['optical_coarse'],sar_global(features))[0]
        valid=features['optical_coarse_mask'][0]
        if self.selection=='A':qi=torch.argsort(logits.flatten().masked_fill(~valid.flatten(),-torch.inf),descending=True,stable=True)[:self.k].sort().values
        elif self.selection=='AC':qi=coverage_queries(logits.sigmoid(),valid,self.k)
        else:raise ValueError('Unknown selection mode')
        do,ds=features['optical_descriptors'],features['sar_descriptors']
        queries=do.index_select(1,qi)  # Scoring/selection complete BEFORE the only cross-image bmm.
        correlation=torch.bmm(queries,ds.transpose(1,2))
        row=correlation.float()/self.log_temperature.exp().clamp(.02,.5)
        row=row.masked_fill(~features['sar_coarse_mask'].flatten(1)[:,None],-torch.inf)
        row=torch.cat([row,self.no_match.float().expand(1,self.k,1)],-1)
        om=features['optical_coarse_mask'].flatten(1).index_select(1,qi)
        reverse=correlation.transpose(1,2).float()/self.log_temperature.exp().clamp(.02,.5)
        reverse=reverse.masked_fill(~om[:,None],-torch.inf)
        reverse=torch.cat([reverse,self.no_match.float().expand(1,1024,1)],-1)
        return dict(**features,query_indices=qi,query_scores=logits,coarse_correlation=correlation,row_logits=row,reverse_logits=reverse)


@torch.no_grad()
def reliability_targets(output,A_forward,sar_pixel_mask,config):
    """Dense forward predictions; geometric label separated from diagnostic logit margin."""
    device=A_forward.device;b=len(A_forward);xy=coarse_xy(torch.arange(1024,device=device),32)
    homogeneous=torch.cat([xy,torch.ones(1024,1,device=device)],1)
    truth=torch.bmm(homogeneous[None].expand(b,-1,-1),A_forward.float().transpose(1,2))
    distance2=(truth[:,:,None]-xy[None,None]).square().sum(-1)
    logits=output['row_logits'].float();scores=logits[:,:,:1024]
    valid_target=output['sar_coarse_mask'].flatten(1)[:,None]
    near=(distance2<=config['positive_tolerance_px']**2)&valid_target
    far=(distance2>=config['negative_tolerance_px']**2)&valid_target
    correct_exists=near.any(-1);wrong_exists=far.any(-1)
    correct_score=scores.masked_fill(~near,-torch.inf).max(-1).values
    wrong_score=scores.masked_fill(~far,-torch.inf).max(-1).values
    raw_sar=scores.argmax(-1);pred=logits.argmax(-1);has_match=pred<1024
    error=distance2.gather(-1,raw_sar[:,:,None])[:,:,0].sqrt()
    margin=float(config['GT_valid_boundary_margin_px']);radius=int(math.ceil(margin))
    eroded=1-F.max_pool2d((~sar_pixel_mask.bool())[:,None].float(),2*radius+1,1,radius)
    grid=2*(truth+.5)/256-1
    reliable=F.grid_sample(eroded,grid[:,None],align_corners=False,mode='bilinear')[:,0,0]>.999
    reliable&=((truth>=margin)&(truth<=255-margin)).all(-1)
    valid=output['optical_coarse_mask'].flatten(1)
    core=valid&reliable&correct_exists
    decisive=(~has_match)|(error<=config['positive_tolerance_px'])|(error>=config['negative_tolerance_px'])
    supervisable=core&decisive
    positive=has_match&(error<=config['positive_tolerance_px'])
    labels=torch.where(supervisable,positive.to(torch.int8),-torch.ones_like(pred,dtype=torch.int8))
    margin_valid=valid&reliable&correct_exists&wrong_exists
    nan=torch.full_like(correct_score,float('nan'))
    return dict(geometric_correctness=labels,ambiguity_margin=torch.where(margin_valid,correct_score-wrong_score,nan),
        best_correct_score=torch.where(correct_exists,correct_score,nan),best_wrong_score=torch.where(wrong_exists,wrong_score,nan),
        predicted_candidate=pred.to(torch.int16),raw_SAR_candidate=raw_sar.to(torch.int16),prediction_error_px=error,
        optical_valid=valid,is_supervisable=supervisable,GT_reliable=reliable,
        positive_candidate_exists=correct_exists,ambiguity_margin_valid=margin_valid)
