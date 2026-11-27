"""Thin official-inference adapters; no model modifications, training or GT access."""
import hashlib,json,sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image


def stable_pair_seed(pair_id,seed):
    return int.from_bytes(hashlib.sha256(f'{seed}/{pair_id}/external-reference'.encode()).digest()[:4],'little')%(2**31-1)


def restore_official_coordinates(points,original_hw,resized_hw):
    """Preserve official wrapper's multiplicative xy restore, not a new halfpixel rule."""
    h,w=original_hw;rh,rw=resized_hw
    return np.asarray(points,dtype=np.float32)*np.array([w/rw,h/rh],np.float32)


def valid_correspondences(source,target,score,optical_mask,sar_mask):
    source=np.asarray(source,np.float32).reshape(-1,2);target=np.asarray(target,np.float32).reshape(-1,2)
    score=np.asarray(score,np.float32).reshape(-1)
    if source.shape!=target.shape or len(source)!=len(score):raise ValueError('Misaligned predicted correspondences')
    valid=np.isfinite(source).all(1)&np.isfinite(target).all(1)&np.isfinite(score)
    for xy,mask in [(source,optical_mask),(target,sar_mask)]:
        h,w=mask.shape;valid&=(xy[:,0]>=0)&(xy[:,0]<=w-1)&(xy[:,1]>=0)&(xy[:,1]<=h-1)
        if len(xy):
            safe=np.where(np.isfinite(xy),xy,-1000).astype(np.float32)
            sampled=cv2.remap(mask.astype(np.float32),safe[:,0,None],safe[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)[:,0]
            valid&=sampled>.999
    return source[valid],target[valid],score[valid],valid


def cpu_input(item):
    """Exact inverse of existing uint8 /255 loader; no SAR log or semantic preprocessing."""
    optical=np.rint(item['optical'].numpy().transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
    sar=np.rint(item['sar_aug'][0].numpy()*255).clip(0,255).astype(np.uint8)
    return dict(pair_id=item['pair_id'],optical=optical,sar=sar,
        optical_mask=item['optical_valid_mask'].numpy(),sar_mask=item['sar_aug_valid_mask'].numpy())


class OfficialReference:
    def __init__(self,name,protocol,config):
        self.name=name;self.protocol=protocol;self.config=config
        root=Path(protocol['source_root']);spec=protocol['method_protocols'][name]
        if name=='T05':
            from .t02.model import B0Full
            saved=torch.load('runs/t05_frozen_v1/checkpoint.pt',map_location='cpu',weights_only=False)
            self.model=B0Full(config).cuda().eval();self.model.load_state_dict(saved['model'],strict=True)
        elif name=='XoFTR':
            sys.path.insert(0,str((root/'XoFTR').resolve()))
            from src.xoftr import XoFTR
            from src.config.default import get_cfg_defaults
            from src.utils.data_io import lower_config
            cfg=lower_config(get_cfg_defaults(inference=True))
            assert cfg['xoftr']['match_coarse']['thr']==spec['coarse_threshold'] and cfg['xoftr']['fine']['thr']==spec['fine_threshold']
            assert cfg['xoftr']['fine']['denser']==spec['denser']
            self.model=XoFTR(cfg['xoftr']);saved=torch.load(root/spec['checkpoint'],map_location='cpu',weights_only=False)
            self.model.load_state_dict(saved['state_dict'],strict=True);self.model.cuda().eval()
        elif name=='MINIMA_RoMa':
            sys.path.insert(0,str((root/'MINIMA/third_party/RoMa_minima').resolve()))
            from romatch import roma_outdoor
            weights=torch.load(root/spec['checkpoint'],map_location='cpu',weights_only=False)
            dino=torch.load(root/spec['dinov2'],map_location='cpu',weights_only=False)
            self.model=roma_outdoor(device='cuda',weights=weights,dinov2_weights=dino,
                coarse_res=spec['coarse_resolution'],upsample_res=spec['upsample_resolution'])
            self.model.eval();self.model.encoder.dinov2_vitl14[0].eval()
            # Upstream CNNandDinov2.train delegates only to cnn.train and leaves
            # its own metadata flag True. Forward never reads that flag; all BN
            # and DINO already eval. Make the audit flag explicit without editing upstream.
            self.model.encoder.training=False
        else:raise ValueError(name)
        self.model.requires_grad_(False)
        if name=='MINIMA_RoMa':self.model.encoder.dinov2_vitl14[0].requires_grad_(False)
        self.model_parameters=sum(p.numel() for p in self.model.parameters())
        self.hidden_dino_parameters=sum(p.numel() for p in self.model.encoder.dinov2_vitl14[0].parameters()) if name=='MINIMA_RoMa' else 0
        assert not any(m.training for m in self.model.modules())

    @torch.inference_mode()
    def predict(self,inputs):
        optical,sar=inputs['optical'],inputs['sar'];spec=self.protocol['method_protocols'][self.name]
        if self.name=='T05':
            args=[torch.from_numpy(optical.copy()).permute(2,0,1)[None].float().cuda()/255,
                torch.from_numpy(sar.copy())[None,None].float().cuda()/255,
                torch.from_numpy(inputs['optical_mask'])[None].cuda(),torch.from_numpy(inputs['sar_mask'])[None].cuda()]
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=self.model.predict(*args)[0]
            source,target,score=[pred[k].float().cpu().numpy() for k in ('source','target','score')]
        elif self.name=='XoFTR':
            size=spec['input_resize'];gray=optical[:,:,0] # audited OS optical channels identical
            images=[torch.from_numpy(cv2.resize(im,(size,size),interpolation=cv2.INTER_LINEAR))[None,None].float().cuda()/255 for im in (gray,sar)]
            batch=dict(image0=images[0],image1=images[1]);self.model(batch)
            source,target=[restore_official_coordinates(batch[k].float().cpu().numpy(),(256,256),(size,size)) for k in ('mkpts0_f','mkpts1_f')]
            score=batch['mconf_f'].float().cpu().numpy()
        else:
            # Match official outer640 OpenCV resize -> RGB PIL -> RoMa560/864 transforms.
            size=spec['outer_resize'];a=Image.fromarray(cv2.resize(optical,(size,size)))
            b=Image.fromarray(cv2.resize(np.repeat(sar[:,:,None],3,2),(size,size)))
            warp,certainty=self.model.match(a,b,batched=False)
            seed=stable_pair_seed(inputs['pair_id'],self.config['seed'])
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            matches,score=self.model.sample(warp,certainty,num=spec['sample_count'])
            aa,bb=self.model.to_pixel_coordinates(matches,size,size,size,size)
            source,target=[restore_official_coordinates(x.float().cpu().numpy(),(256,256),(size,size)) for x in (aa,bb)]
            score=score.float().cpu().numpy()
        raw_count=len(score)
        source,target,score,valid=valid_correspondences(source,target,score,inputs['optical_mask'],inputs['sar_mask'])
        return dict(source=source,target=target,score=score,raw_matches=raw_count,valid_matches=int(valid.sum()),removed_invalid=int((~valid).sum()))
