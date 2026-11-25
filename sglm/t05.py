"""Frozen-weight, content-independent query-budget screening of B0.

Only the queried descriptor rows change. Reverse mutual competition is among
selected queries, since evaluating omitted queries would recreate dense cost.
"""
import torch
import torch.nn.functional as F
from .model import coarse_xy
from .t02.model import B0Full


UNIFORM_GRIDS={1024:(32,32),512:(16,32),256:(16,16),128:(8,16)}


def uniform_queries(k,device=None):
    if k not in UNIFORM_GRIDS:raise ValueError('Expected K in 1024,512,256,128')
    rows,cols=UNIFORM_GRIDS[k]
    y=((torch.arange(rows,device=device).float()+.5)*32/rows).long()
    x=((torch.arange(cols,device=device).float()+.5)*32/cols).long()
    return (y[:,None]*32+x[None]).flatten()


class UniformB0(B0Full):
    def __init__(self,config,k=1024):
        super().__init__(config)
        self.k=k

    def encode(self,optical,sar_aug,optical_valid_mask,sar_aug_valid_mask):
        if optical.shape[1:]!=(3,256,256) or sar_aug.shape[1:]!=(1,256,256):
            raise ValueError('Expected 256-pixel paired inputs')
        fo,lo,do=self.backbone(optical,0);fs,ls,ds=self.backbone(sar_aug,1)
        om=F.avg_pool2d(optical_valid_mask[:,None].float(),8,8)[:,0]>.999
        sm=F.avg_pool2d(sar_aug_valid_mask[:,None].float(),8,8)[:,0]>.999
        return dict(optical_coarse=fo,sar_coarse=fs,optical_fine=lo,sar_fine=ls,
            optical_descriptors=do,sar_descriptors=ds,optical_coarse_mask=om,sar_coarse_mask=sm)

    def correlate(self,features):
        do,ds=features['optical_descriptors'],features['sar_descriptors']
        qi=uniform_queries(self.k,do.device)
        # Full budget uses the original view/layout exactly; sparse budgets gather
        # before the ONLY cross-image bmm, with no dense reverse-correlation path.
        queries=do if self.k==1024 else do.index_select(1,qi)
        correlation=torch.bmm(queries,ds.transpose(1,2))
        logits=correlation.float()/self.log_temperature.exp().clamp(.02,.5)
        logits=logits.masked_fill(~features['sar_coarse_mask'].flatten(1)[:,None],-torch.inf)
        row=torch.cat([logits,self.no_match.float().expand(len(do),self.k,1)],-1)
        selected_mask=features['optical_coarse_mask'].flatten(1).index_select(1,qi)
        reverse=correlation.transpose(1,2).float()/self.log_temperature.exp().clamp(.02,.5)
        reverse=reverse.masked_fill(~selected_mask[:,None],-torch.inf)
        reverse=torch.cat([reverse,self.no_match.float().expand(len(do),1024,1)],-1)
        return dict(**features,query_indices=qi,coarse_correlation=correlation,row_logits=row,reverse_logits=reverse)

    def forward(self,optical,sar_aug,optical_valid_mask,sar_aug_valid_mask):
        return self.correlate(self.encode(optical,sar_aug,optical_valid_mask,sar_aug_valid_mask))

    @torch.no_grad()
    def coarse_matches(self,output,index):
        confidence,target=output['row_logits'][index].float().softmax(-1).max(-1)
        qi=output['query_indices'];row=torch.arange(len(qi),device=qi.device)
        valid=output['optical_coarse_mask'][index].flatten()[qi]&(target<1024)&(confidence>=self.min_probability)
        if self.mutual:
            reverse=output['reverse_logits'][index].argmax(-1)
            valid&=reverse[target.clamp_max(1023)]==row
        return coarse_xy(qi[valid],32),coarse_xy(target[valid],32),confidence[valid]

    def refine(self,output,args,index=0):
        source,target,score=self.coarse_matches(output,index)
        local=self.local(output['optical_fine'][index:index+1],output['sar_fine'][index:index+1],source,target,
            args[2][index:index+1],args[3][index:index+1])
        valid=local.get('valid',torch.ones(len(source),device=source.device,dtype=torch.bool))
        return dict(source=source[valid],target=local['target'][valid],score=score[valid],coarse_target=target[valid],
            attempted_refinements=len(source),local_window_source='predicted_coarse_only')

    @torch.no_grad()
    def predict(self,optical,sar_aug,optical_valid_mask,sar_aug_valid_mask):
        args=(optical,sar_aug,optical_valid_mask,sar_aug_valid_mask);output=self(*args)
        return [self.refine(output,args,i) for i in range(len(optical))]
