"""One small local optical-query/SAR-window interaction; frozen B0 matching path."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .model import sample_windows
from .t05 import UniformB0
from .t02.model import LocalCorrelation


class LocalCrossCorrelation(LocalCorrelation):
    def __init__(self,radius=3,attention_dim=16):
        super().__init__(radius)
        self.cross_q=nn.Linear(32,attention_dim,bias=False)
        self.cross_k=nn.Linear(32,attention_dim,bias=False)
        self.cross_v=nn.Linear(32,attention_dim,bias=False)
        self.cross_out=nn.Linear(attention_dim,32,bias=False)
        for m in [self.cross_q,self.cross_k,self.cross_v]:nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(self.cross_out.weight)
        self.attention_dim=attention_dim

    def forward(self,optical_fine,sar_fine,source_xy,target_centers,optical_mask=None,sar_mask=None):
        if not len(source_xy):return dict(target=target_centers,heatmap=target_centers.new_empty((0,(2*self.radius+1)**2)))
        source,_,source_valid=sample_windows(optical_fine.float(),source_xy.float(),0,optical_mask)
        candidates,delta,valid=sample_windows(sar_fine.float(),target_centers.float(),self.radius,sar_mask)
        qraw=self.project(source[:,0]).float();kraw=self.project(candidates).float()
        safe=valid.any(-1)&source_valid[:,0];fallback=valid.clone()
        fallback[~safe,(2*self.radius+1)**2//2]=True
        aq=self.cross_q(qraw);ak=self.cross_k(kraw);av=self.cross_v(kraw)
        weights=((aq[:,None].float()*ak.float()).sum(-1)/math.sqrt(self.attention_dim)).masked_fill(~fallback,-torch.inf).softmax(-1)
        context=(weights[:,:,None]*av.float()).sum(1)
        q=F.normalize(qraw+self.cross_out(context).float(),dim=-1);k=F.normalize(kraw,dim=-1)
        logits=(q[:,None]*k).sum(-1)/self.log_temperature.exp().clamp(.02,.5)
        heatmap=logits.masked_fill(~fallback,-torch.inf).softmax(-1)
        target=target_centers+2*(heatmap@delta)
        return dict(target=target,heatmap=heatmap,valid=safe)


def build_local_group(cfg,checkpoint,group,device='cuda'):
    model=UniformB0(cfg,k=256).to(device).eval()
    if group=='local_cross_attention':model.local=LocalCrossCorrelation(cfg['model']['local_radius_fine_cells']).to(device)
    if group not in ['original_refiner','local_cross_attention']:raise ValueError(group)
    missing=model.load_state_dict(checkpoint,strict=False)
    assert not missing.unexpected_keys
    assert all(k.startswith('local.cross_') for k in missing.missing_keys)
    if group=='original_refiner':assert not missing.missing_keys
    return model.eval()


def fixed_supervision(source,coarse,A,sar_mask,optical_mask):
    truth=source@A[:,:2].float().T+A[:,2].float()
    inside=((truth-coarse).abs()<=6+1e-6).all(-1)
    bounds=((truth>=0)&(truth<=255)).all(-1)
    def sample_mask(mask,xy):
        if not len(xy):return torch.empty(0,dtype=torch.bool,device=xy.device)
        grid=(2*(xy+.5)/256-1)[None,None]
        return F.grid_sample(mask[None,None].float(),grid,mode='bilinear',align_corners=False)[0,0,0]>.999
    visible=bounds&sample_mask(sar_mask,truth)&sample_mask(optical_mask,source)
    return truth,visible,visible&inside
