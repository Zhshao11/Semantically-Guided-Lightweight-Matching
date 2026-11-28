"""Thin native-weight adapters. No model architecture or forward computation patches."""
import copy,hashlib,importlib,sys
from pathlib import Path
import cv2
import numpy as np
import torch
from .sar_reference_geometry import dense_correspondences,filter_correspondences,restore_resized_coordinates


def install_profiler_namespace_alias():
    # Upstream imports PL1's profiler namespace. PL2 re-exports identical-purpose
    # profiler classes from profilers; inference uses only PassThroughProfiler.
    try:importlib.import_module('pytorch_lightning.profiler');return False
    except ModuleNotFoundError:
        sys.modules['pytorch_lightning.profiler']=importlib.import_module('pytorch_lightning.profilers');return True


class SARReference:
    def __init__(self,name,protocol,config,device='cuda'):
        self.name=name;self.protocol=protocol;self.config=config;self.device=device;self.compatibility={}
        root=Path(protocol['source_root']).resolve()
        if name.startswith('T05'):
            from .t02.model import B0Full
            from .t05 import UniformB0
            saved=torch.load('runs/t05_frozen_v1/checkpoint.pt',map_location='cpu',weights_only=False)
            self.model=(B0Full(config) if name=='T05_Full' else UniformB0(config,k=256)).to(device).eval()
            self.model.load_state_dict(saved['model'],strict=True)
        elif name.startswith('FHReg_GUSO'):
            spec=protocol['FHReg_GUSO'];path=root/spec['weight'];assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
            sys.path.insert(0,str(root/'GUSO'));self.compatibility['lightning_profiler_namespace_alias']=install_profiler_namespace_alias()
            from fhreg.fhreg import FHReg
            from utils.cvpr_ds_config import default_cfg
            assert Path(sys.modules['fhreg.fhreg'].__file__).resolve().is_relative_to(root/'GUSO')
            native=copy.deepcopy(default_cfg);assert native['match_coarse']['thr']==spec['coarse_threshold'] and native['fine']['thr']==spec['fine_threshold']
            self.model=FHReg(native,model_training=False)
            saved=torch.load(path,map_location='cpu',weights_only=False)
            self.model.load_state_dict(saved['state_dict'] if 'state_dict' in saved else saved,strict=True)
            self.model.to(device).eval()
        elif name=='GDROS_WHU':
            spec=protocol[name];path=root/spec['weight'];assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
            sys.path.insert(0,str(root/'GDROS'))
            from configs.default import get_cfg
            from core.GDROSnet.FeatureFormer.gdros import GDROS
            from core.LSRnet.LSmodel import LSR
            assert Path(sys.modules['core.GDROSnet.FeatureFormer.gdros'].__file__).resolve().is_relative_to(root/'GDROS')
            cfg=get_cfg();self.model=GDROS(cfg[cfg.transformer]);saved=torch.load(path,map_location='cpu',weights_only=False)
            # Native test.py loads DataParallel state; removing its namespace is exact.
            self.model.load_state_dict({k.removeprefix('module.'):v for k,v in saved.items()},strict=True)
            self.model.to(device).eval();self.native_lsr=LSR
        else:raise ValueError(name)
        self.model.requires_grad_(False);assert not any(m.training for m in self.model.modules())
        self.parameters=sum(p.numel() for p in self.model.parameters())

    def prepare(self,data):
        optical,sar=data['optical'],data['sar']
        if self.name=='GDROS_WHU':
            size=self.protocol[self.name]['input_resize']
            images=[cv2.resize(im,(size,size),interpolation=cv2.INTER_LINEAR) for im in [optical,np.repeat(sar[:,:,None],3,2)]]
            return [torch.from_numpy(im.copy()).permute(2,0,1)[None].float().to(self.device) for im in images]
        if self.name.startswith('FHReg_GUSO'):
            size=self.protocol[self.name]['input_resize']
            optical=cv2.resize(optical,(size,size),interpolation=cv2.INTER_LINEAR)
            sar=cv2.resize(sar,(size,size),interpolation=cv2.INTER_LINEAR)
            return [(torch.from_numpy(im.transpose(2,0,1).copy())[None].float()/255).to(self.device) for im in [optical,np.repeat(sar[:,:,None],3,2)]]
        return [torch.from_numpy(optical.copy()).permute(2,0,1)[None].float().to(self.device)/255,
            torch.from_numpy(sar.copy())[None,None].float().to(self.device)/255,
            torch.from_numpy(data['optical_mask'])[None].to(self.device),torch.from_numpy(data['sar_mask'])[None].to(self.device)]

    @torch.inference_mode()
    def predict(self,data,include_native_dense=False):
        inputs=self.prepare(data);supplementary={}
        if self.name.startswith('T05'):
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=self.model.predict(*inputs)[0]
            source,target,score=[pred[k].float().cpu().numpy() for k in ['source','target','score']]
        elif self.name.startswith('FHReg_GUSO'):
            batch=dict(image_opt=inputs[0],image_sar=inputs[1]);result=self.model(batch)
            source,target,score=[result[k].float().cpu().numpy() for k in ['mkpts0_f','mkpts1_f','mconf_f']]
            size=self.protocol[self.name]['input_resize']
            source=restore_resized_coordinates(source,data['sar'].shape,(size,size));target=restore_resized_coordinates(target,data['sar'].shape,(size,size))
        else:
            _,flow=self.model(*inputs,iters=self.protocol[self.name]['iterations'],test_mode=True)
            raw=flow[0].float().cpu().numpy();source,target,score=dense_correspondences(raw)
            if include_native_dense:
                spec=self.protocol[self.name];h,w=flow.shape[-2:]
                native=self.native_lsr(1,h,w,flow,spec['native_LSR_points'])[0].float().numpy()
                supplementary=dict(raw_flow=raw,native_lsr_flow=native)
        raw_count=len(source);source,target,score,valid=filter_correspondences(source,target,score,data['optical_mask'],data['sar_mask'])
        return dict(source=source,target=target,score=score,raw_matches=raw_count,valid_matches=int(valid.sum()),removed_invalid=int((~valid).sum()),**supplementary)

    @torch.inference_mode()
    def predict_native_lsr(self,data):
        if self.name!='GDROS_WHU':raise ValueError(self.name)
        inputs=self.prepare(data);_,flow=self.model(*inputs,iters=self.protocol[self.name]['iterations'],test_mode=True)
        h,w=flow.shape[-2:];return self.native_lsr(1,h,w,flow,self.protocol[self.name]['native_LSR_points'])[0].float().numpy()
