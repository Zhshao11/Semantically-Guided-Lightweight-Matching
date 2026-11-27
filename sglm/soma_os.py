"""SOMA OS-training adaptation. Geometry corrections are explicit, upstream untouched."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def resize_flow(flow,size):
    h,w=flow.shape[-2:];rh,rw=size
    return F.interpolate(flow,size=size,mode='bilinear',align_corners=False)*flow.new_tensor([rw/w,rh/h])[None,:,None,None]


def theta_to_flow(theta,H,W):
    grid=F.affine_grid(theta,(len(theta),1,H,W),align_corners=False)
    identity=theta.new_tensor([[1.,0,0],[0,1,0]])[None].expand_as(theta)
    base=F.affine_grid(identity,(len(theta),1,H,W),align_corners=False)
    return ((grid-base)*theta.new_tensor([W/2,H/2])).permute(0,3,1,2)


def refinement_forward(self,x_feat,y_feat,flow,logits=None):
    B,C,H,W=x_feat.shape
    if flow.shape[-2:]!=(H,W):flow=resize_flow(flow,(H,W))
    identity=flow.new_tensor([[1.,0,0],[0,1,0]])[None].expand(B,-1,-1)
    grid=F.affine_grid(identity,(B,C,H,W),align_corners=False)
    norm=flow.permute(0,2,3,1)/flow.new_tensor([W/2,H/2])
    warped=F.grid_sample(y_feat,grid+norm,align_corners=False)
    features=[x_feat,warped,self.disp_emb(flow)]
    if self.concat_logits and logits is not None:
        if logits.shape[-2:]!=(H,W):logits=F.interpolate(logits,(H,W),mode='bilinear',align_corners=False)
        features.append(logits)
    out=self.out_conv(self.encoder(torch.cat(features,1)))
    return out[:,:2],out[:,2:3]


def glam_forward(self,feats1,feats2):
    f16a=self.reducer_16_1(feats1['16']);f16b=self.reducer_16_2(feats2['16'])
    f32a=F.interpolate(self.reducer_32_1(feats1['32']),f16a.shape[-2:],mode='bilinear',align_corners=False)
    f32b=F.interpolate(self.reducer_32_2(feats2['32']),f16b.shape[-2:],mode='bilinear',align_corners=False)
    theta=self.coarse_matcher(torch.cat([f16a,f32a],1),torch.cat([f16b,f32b],1))
    full=feats1['1'].shape[-2:];h8,w8=feats1['8'].shape[-2:];h4,w4=feats1['4'].shape[-2:];h2,w2=feats1['2'].shape[-2:]
    flow8=theta_to_flow(theta,*full);dt8,p8=self.fine_matcher_8(feats1['8'],feats2['8'],theta)
    delta8=theta_to_flow(dt8,h8,w8);t8=self.compose(theta,dt8)
    flow4=theta_to_flow(t8,*full);dt4,p4=self.fine_matcher_4(feats1['4'],feats2['4'],t8)
    delta4=theta_to_flow(dt4,h4,w4);t4=self.compose(t8,dt4)
    base2=theta_to_flow(t4,h2,w2);d2,c2=self.fine_matcher_2(feats1['2'],feats2['2'],base2)
    combined=resize_flow(base2+d2,full);base2_full=resize_flow(base2,full);d2_full=resize_flow(d2,full)
    c2=F.interpolate(c2,full,mode='bilinear',align_corners=False)
    d1,c1=self.fine_matcher_1(feats1['1'],feats2['1'],combined,logits=c2)
    return dict(coarse_theta=theta,flow_8_pred=p8,flow_8_delta=delta8,flow_4_pred=p4,flow_4_delta=delta4,
        flow_2_delta=d2_full,flow_1_delta=d1,flow_final=combined+d1,certainty=c2+c1,
        flow_8=flow8,flow_4=flow4,flow_2=base2_full,flow_1=combined)


def build_soma(config,device='cuda'):
    root=Path(config['upstream_source']).resolve();sys.path.insert(0,str(root))
    import models.encoder as encoder_mod
    import models.matcher as matcher_mod
    if not Path(encoder_mod.__file__).resolve().is_relative_to(root):raise RuntimeError('Wrong SOMA import namespace')
    dino=torch.hub.load(config['dinov2_source'],'dinov2_vitl14_reg',source='local',pretrained=False)
    dino.load_state_dict(torch.load(config['dinov2_weights'],map_location='cpu',weights_only=False),strict=True)
    dino.eval().requires_grad_(False)
    def finite_or_raise(tensor,name):
        if not bool(torch.isfinite(tensor).all()):raise FloatingPointError('Nonfinite SOMA '+name)
    encoder_mod.check_na=finite_or_raise
    # Construct the exact upstream branches using the explicitly supplied proper DINO.
    class OSEncoder(nn.Module):
        def __init__(self):
            super().__init__();self.dino_model=dino
            self.opt_encoder=encoder_mod.OPTEncoder(dino);self.sar_encoder=encoder_mod.SAREncoder(dino)
        def train(self,mode=True):
            super().train(mode);self.dino_model.eval();return self
        def forward(self,image,label):
            return {label:(self.opt_encoder if label=='opt' else self.sar_encoder)(image)}
    class OSRefine(matcher_mod.RefineMatcher):
        forward=refinement_forward
    class OSGLAM(matcher_mod.GLAM):
        compose=staticmethod(matcher_mod.compose_theta)
        forward=glam_forward
        def __init__(self):
            super().__init__()
            # Same learned parameter schema as upstream; only geometric execution changes.
            self.fine_matcher_2.__class__=OSRefine;self.fine_matcher_1.__class__=OSRefine
    encoder=OSEncoder().to(device);matcher=OSGLAM().to(device)
    return encoder,matcher


def prepare_images(batch,size=512,device='cuda'):
    optical=batch['optical'].to(device,non_blocking=True);sar=batch['sar_aug'].to(device,non_blocking=True).repeat(1,3,1,1)
    return (F.interpolate(optical,(size,size),mode='bilinear',align_corners=False),
            F.interpolate(sar,(size,size),mode='bilinear',align_corners=False))


def inverse_targets(batch,size=512,device='cuda'):
    A=batch['A_forward'].to(device,dtype=torch.float32);B=len(A)
    homogeneous=torch.eye(3,device=device)[None].repeat(B,1,1);homogeneous[:,:2]=A
    inv=torch.linalg.inv(homogeneous)[:,:2]
    axis=(torch.arange(size,device=device).float()+.5)*256/size-.5
    yy,xx=torch.meshgrid(axis,axis,indexing='ij');grid=torch.stack([xx,yy],-1)[None].expand(B,-1,-1,-1)
    xy=torch.einsum('bij,bhwj->bhwi',inv[:,:,:2],grid)+inv[:,:,2][:,None,None,:]
    flow=(xy-grid).permute(0,3,1,2)*(size/256)
    valid=((xy>=0)&(xy<=255)).all(-1)
    sar_mask=F.interpolate(batch['sar_aug_valid_mask'].to(device)[:,None].float(),(size,size),mode='nearest')[:,0]>.999
    valid&=sar_mask
    return flow,valid


def soma_loss(output,truth,valid):
    """Original compound loss structure, explicit valid mask and corrected units."""
    if not bool(valid.any()):raise ValueError('No valid training correspondences')
    pred=output['flow_final'];mask=valid[:,None].float()
    robust=lambda x:((x.float().square()+1e-12).sqrt()*mask).sum()/(2*mask.sum())
    warp=robust(pred-truth)
    consistency=[]
    for level in ['8','4']:
        p=output['flow_'+level+'_pred'];delta=output['flow_'+level+'_delta']
        m=F.interpolate(mask,p.shape[-2:],mode='nearest')
        consistency.append((((p.float()-delta.float()).square().sum(1,keepdim=True)*m).sum()/m.sum().clamp_min(1)).clamp_min(1e-12).sqrt())
    delta=0.
    for level,weight in [('8',.125),('4',.25),('2',.5),('1',1.)]:
        d=output['flow_'+level+'_delta']
        if d.shape[-2:]!=truth.shape[-2:]:d=resize_flow(d,truth.shape[-2:])
        base=output['flow_'+level]
        delta=delta+weight*(((d-(truth-base)).abs()*mask).sum()/(2*mask.sum()))
    error=(pred-truth).float().norm(dim=1,keepdim=True)
    cert_target=torch.exp(-error).detach()
    cert=((output['certainty'].float().sigmoid()-cert_target).square()*mask).sum()/mask.sum()
    # Original4x4 patch-uniformity diagnostic with valid-weighted patch means.
    patch=[];h,w=truth.shape[-2:]
    for y in range(4):
        for x in range(4):
            m=mask[:,:,y*h//4:(y+1)*h//4,x*w//4:(x+1)*w//4]
            e=error[:,:,y*h//4:(y+1)*h//4,x*w//4:(x+1)*w//4]
            patch.append((e*m).sum((1,2,3))/m.sum((1,2,3)).clamp_min(1))
    uniform=torch.stack(patch,1).std(1).mean()
    consistency=sum(consistency);total=warp+.5*consistency+.1*uniform+.1*delta+.1*cert
    return total,dict(total=float(total.detach()),warp=float(warp.detach()),consistency=float(consistency.detach()),
        uniform=float(uniform.detach()),delta=float(delta.detach()),certainty=float(cert.detach()),valid_fraction=float(valid.float().mean()))


@torch.no_grad()
def dense_to_matches(flow,certainty,grid=32):
    """d(ySAR)=xOPT-ySAR in512 units -> predicted OPT->SAR pairs in256 units."""
    n=flow.shape[-1];axis=(torch.arange(grid,device=flow.device).float()+.5)*256/grid-.5
    yy,xx=torch.meshgrid(axis,axis,indexing='ij');sar=torch.stack([xx,yy],-1).reshape(-1,2)
    sample_grid=(2*(sar+.5)/256-1)[None,None]
    delta=F.grid_sample(flow.float(),sample_grid,align_corners=False)[0,:,0].T*(256/n)
    score=F.grid_sample(certainty.float(),sample_grid,align_corners=False)[0,0,0].sigmoid()
    return sar+delta,sar,score
