"""Online per-presentation augmentation with reproducible explicit draw tokens."""
import hashlib
import itertools
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset,Sampler
from ..config import resolve_data_root
from ..data import _read_patch
from ..geometry import warp_image
from ..t02.data import sample_transform


def seed_for_presentation(seed,pair_id,token):
    return int.from_bytes(hashlib.sha256(f'{seed}/{pair_id}/{token}/T03-online-v1'.encode()).digest()[:8],'little')


def permutation_for_epoch(length,epoch,seed):
    return torch.randperm(length,generator=torch.Generator().manual_seed(seed+epoch)).tolist()


class PresentationBatchSampler(Sampler):
    def __init__(self,length,batch_size,seed):
        self.length,self.batch_size,self.seed=length,batch_size,seed
        self.epoch=0
    def __len__(self): return (self.length+self.batch_size-1)//self.batch_size
    def __iter__(self):
        order=permutation_for_epoch(self.length,self.epoch,self.seed)
        for start in range(0,self.length,self.batch_size):
            yield [(index,self.epoch*self.length+position) for position,index in enumerate(order[start:start+self.batch_size],start)]


class FullAffinePairs256(Dataset):
    def __init__(self,root,plan,split,config,preload=False):
        self.root=resolve_data_root(root)
        self.plan=json.loads(Path(plan).read_text()) if isinstance(plan,(str,Path)) else plan
        self.samples=self.plan[split]
        self.split,self.config=split,config
        self._counter=itertools.count();self._verified=set()
        self.images=[self.read_images(i) for i in range(len(self.samples))] if preload else None
    def __len__(self): return len(self.samples)
    def read_images(self,index):
        sample=self.samples[index]
        paths=[self.root/sample[k] for k in ('optical_path','sar_path')]
        if index not in self._verified:
            for path,key in zip(paths,('optical_sha256','sar_sha256')):
                if not path.resolve().is_relative_to(self.root) or hashlib.sha256(path.read_bytes()).hexdigest()!=sample[key]:
                    raise ValueError('Full-data input hash changed')
            self._verified.add(index)
        optical,sar=_read_patch(paths[0],True),_read_patch(paths[1],False)
        if optical.shape!=(256,256,3) or sar.shape!=(256,256): raise ValueError('Invalid T03 image shape')
        return optical,sar
    def __getitem__(self,key):
        if isinstance(key,tuple): index,token=key
        else: index,token=key,next(self._counter)
        sample=self.samples[index]
        optical,sar=self.images[index] if self.images is not None else self.read_images(index)
        if self.split=='train':
            seed=seed_for_presentation(self.config['seed'],sample['pair_id'],token)
            matrix,_=sample_transform(np.random.default_rng(seed),self.config)
        else:
            seed=None;token=-1
            matrix=np.asarray(sample['A_forward'],float)
        augmented,valid=warp_image(sar,matrix,mask_min_weight=self.config['augmentation']['valid_mask_min_weight'])
        return dict(pair_id=sample['pair_id'],optical=torch.from_numpy(optical.copy()).permute(2,0,1).float()/255,
            sar_aug=torch.from_numpy(augmented.copy())[None].float()/255,A_forward=torch.from_numpy(matrix.astype(np.float32)),
            optical_valid_mask=torch.ones(256,256,dtype=torch.bool),sar_aug_valid_mask=torch.from_numpy(valid),
            draw_token=token,augmentation_seed='' if seed is None else str(seed))


def worker_init(worker_id):
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
