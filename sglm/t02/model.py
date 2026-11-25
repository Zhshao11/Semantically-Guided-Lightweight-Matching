"""B0-Full: existing backbone/FPN, full coarse correlation, simple local correlation.

No semantic/score/budget/attention modules are instantiated. Prediction has no GT input.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from ..model import Backbone,coarse_xy,sample_windows


class LocalCorrelation(nn.Module):
    def __init__(self,radius=3):
        super().__init__()
        self.project=nn.Linear(16,32,bias=False)
        self.log_temperature=nn.Parameter(torch.tensor(math.log(.1)))
        self.radius=radius

    def forward(self,optical_fine,sar_fine,source_xy,target_centers,optical_mask=None,sar_mask=None):
        if not len(source_xy):
            return dict(target=target_centers,heatmap=target_centers.new_empty((0,(2*self.radius+1)**2)))
        source,_,source_valid=sample_windows(optical_fine.float(),source_xy.float(),0,optical_mask)
        candidates,delta,valid=sample_windows(sar_fine.float(),target_centers.float(),self.radius,sar_mask)
        q=F.normalize(self.project(source[:,0]).float(),dim=-1)
        k=F.normalize(self.project(candidates).float(),dim=-1)
        logits=(q[:,None]*k).sum(-1)/self.log_temperature.exp().clamp(.02,.5)
        safe=valid.any(-1)&source_valid[:,0]
        # Empty/invalid windows are explicitly marked; never let all -inf softmax create NaNs.
        fallback=valid.clone()
        fallback[~safe,(2*self.radius+1)**2//2]=True
        heatmap=logits.masked_fill(~fallback,-torch.inf).softmax(-1)
        target=target_centers+2*(heatmap@delta)
        return dict(target=target,heatmap=heatmap,valid=safe)


class B0Full(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.backbone=Backbone()
        self.local=LocalCorrelation(config['model']['local_radius_fine_cells'])
        self.log_temperature=nn.Parameter(torch.tensor(math.log(.1)))
        self.no_match=nn.Parameter(torch.tensor(0.))
        self.min_probability=config['model']['coarse_min_probability']
        self.mutual=config['model']['mutual_nearest']

    def forward(self,optical,sar_aug,optical_valid_mask,sar_aug_valid_mask):
        if optical.shape[1:]!=(3,256,256) or sar_aug.shape[1:]!=(1,256,256):
            raise ValueError('B0-Full expects optical Bx3x256x256 and SAR Bx1x256x256')
        fo,lo,do=self.backbone(optical,0)
        fs,ls,ds=self.backbone(sar_aug,1)
        om=F.avg_pool2d(optical_valid_mask[:,None].float(),8,8)[:,0]>.999
        sm=F.avg_pool2d(sar_aug_valid_mask[:,None].float(),8,8)[:,0]>.999
        # This is the full 1024x1024 correlation, before no-match column extension.
        correlation=torch.bmm(do,ds.transpose(1,2))
        logits=correlation.float()/self.log_temperature.exp().clamp(.02,.5)
        logits=logits.masked_fill(~sm.flatten(1)[:,None],-torch.inf)
        row_logits=torch.cat([logits,self.no_match.float().expand(len(optical),1024,1)],-1)
        reverse=correlation.transpose(1,2).float()/self.log_temperature.exp().clamp(.02,.5)
        reverse=reverse.masked_fill(~om.flatten(1)[:,None],-torch.inf)
        reverse_logits=torch.cat([reverse,self.no_match.float().expand(len(optical),1024,1)],-1)
        return dict(optical_coarse=fo,sar_coarse=fs,optical_fine=lo,sar_fine=ls,
            optical_descriptors=do,sar_descriptors=ds,coarse_correlation=correlation,
            row_logits=row_logits,reverse_logits=reverse_logits,optical_coarse_mask=om,sar_coarse_mask=sm)

    @torch.no_grad()
    def coarse_matches(self,output,index):
        probabilities=output['row_logits'][index].float().softmax(-1)
        confidence,target=probabilities.max(-1)
        source=torch.arange(1024,device=target.device)
        valid=output['optical_coarse_mask'][index].flatten()&(target<1024)&(confidence>=self.min_probability)
        if self.mutual:
            reverse=output['reverse_logits'][index].argmax(-1)
            valid&=reverse[target.clamp_max(1023)]==source
        source,target,confidence=source[valid],target[valid],confidence[valid]
        return coarse_xy(source,32),coarse_xy(target,32),confidence

    @torch.no_grad()
    def predict(self,optical,sar_aug,optical_valid_mask,sar_aug_valid_mask):
        output=self(optical,sar_aug,optical_valid_mask,sar_aug_valid_mask)
        results=[]
        for index in range(len(optical)):
            source,target,confidence=self.coarse_matches(output,index)
            refined=self.local(output['optical_fine'][index:index+1],output['sar_fine'][index:index+1],source,target,
                optical_valid_mask[index:index+1],sar_aug_valid_mask[index:index+1])
            valid=refined.get('valid',torch.ones(len(source),device=source.device,dtype=torch.bool))
            results.append(dict(source=source[valid],target=refined['target'][valid],score=confidence[valid],
                coarse_target=target[valid],local_window_source='predicted_coarse_only'))
        return results
