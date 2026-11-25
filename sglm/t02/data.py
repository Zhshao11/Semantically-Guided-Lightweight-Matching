"""Fixed plans for small-sample debugging, or explicit epoch-varying online augmentation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from ..config import resolve_data_root
from ..data import _read_patch
from ..geometry import affine_from_parameters, warp_image


def load_t02_config(path='configs/t02_b0_full_v1.json'):
    config=json.loads(Path(path).read_text())
    if config['input_size']!=256 or config['model']['name']!='B0-Full':
        raise ValueError('T02 requires B0-Full with 256 inputs')
    if config['augmentation']['matrix_order_rightmost_first']!='T_translation @ T_center @ R_rotation @ H_shear_xy @ S_uniform @ T_minus_center':
        raise ValueError('Unsupported matrix order')
    return config


def sample_transform(rng,config):
    aug=config['augmentation']
    angle=float(rng.uniform(*aug['rotation_deg']))
    scale=float(rng.uniform(*aug['uniform_scale']))
    translation=rng.uniform(*aug['translation_px_each_axis'],size=2)
    shear_deg=rng.uniform(*aug['shear_deg_each_axis'],size=2)
    matrix=affine_from_parameters(256,256,angle,(scale,scale),np.tan(np.deg2rad(shear_deg)),translation)
    return matrix,dict(rotation_deg=angle,uniform_scale=scale,translation_px=translation.tolist(),shear_deg=shear_deg.tolist())


def create_plan(config,train_manifest,reference_manifest):
    selected=[]
    for split,path,count in [('train',train_manifest,config['sampling']['train_pairs']),('val',reference_manifest,config['sampling']['val_pairs'])]:
        rows=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
        rows=sorted([p for p in rows if p['dataset']=='OSdataset' and p['resolution']==256 and p['split']==split and p['usable']],key=lambda p:p['pair_id'])
        if len(rows)<count or len({p['pair_id'] for p in rows})!=len(rows):
            raise ValueError('Insufficient/duplicate plan candidates')
        rng=np.random.default_rng(config['seed']+(0 if split=='train' else 1))
        indices=sorted(rng.choice(len(rows),count,replace=False).tolist())
        for index in indices:
            pair=rows[index]
            seed=int.from_bytes(hashlib.sha256(f"{config['seed']}/{pair['pair_id']}/T02".encode()).digest()[:8],'little')
            matrix,params=sample_transform(np.random.default_rng(seed),config)
            selected.append(dict(pair_id=pair['pair_id'],split=split,
                optical_path=pair['optical']['path'],sar_path=pair['sar']['path'],
                optical_sha256=pair['optical']['sha256'],sar_sha256=pair['sar']['sha256'],
                A_forward=matrix.tolist(),parameters=params))
    return dict(protocol=config['protocol'],config=config,
        source_train_manifest_sha256=hashlib.sha256(Path(train_manifest).read_bytes()).hexdigest(),
        source_reference_manifest_sha256=hashlib.sha256(Path(reference_manifest).read_bytes()).hexdigest(),samples=selected)


class AffinePairs256(Dataset):
    def __init__(self,root,plan,split='train',mode='fixed',preload=True):
        self.root=resolve_data_root(root)
        self.plan=json.loads(Path(plan).read_text()) if isinstance(plan,(str,Path)) else plan
        self.config=self.plan['config']
        self.samples=[p for p in self.plan['samples'] if p['split']==split]
        if not self.samples or mode not in ('fixed','online'):
            raise ValueError('Empty split or unsupported perturbation mode')
        self.mode,self.epoch=mode,0
        self.images=[self.read_images(p) for p in self.samples] if preload else None

    def read_images(self,pair):
        paths=[self.root/pair[k] for k in ('optical_path','sar_path')]
        for path,key in zip(paths,('optical_sha256','sar_sha256')):
            if not path.resolve().is_relative_to(self.root) or hashlib.sha256(path.read_bytes()).hexdigest()!=pair[key]:
                raise ValueError(f'Input image path/hash changed: {path}')
        optical,sar=_read_patch(paths[0],True),_read_patch(paths[1],False)
        if optical.shape!=(256,256,3) or sar.shape!=(256,256):
            raise ValueError('Expected 256 optical RGB-repeated and grayscale SAR')
        return optical,sar

    def set_epoch(self,epoch):
        self.epoch=int(epoch)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        pair=self.samples[index]
        optical,sar=self.images[index] if self.images is not None else self.read_images(pair)
        matrix=np.asarray(pair['A_forward'],dtype=np.float64)
        if self.mode=='online':
            seed=int.from_bytes(hashlib.sha256(f"{self.config['seed']}/{pair['pair_id']}/{self.epoch}/online".encode()).digest()[:8],'little')
            matrix,_=sample_transform(np.random.default_rng(seed),self.config)
        augmented,valid=warp_image(sar,matrix,mask_min_weight=self.config['augmentation']['valid_mask_min_weight'])
        return dict(pair_id=pair['pair_id'],optical=torch.from_numpy(optical.copy()).permute(2,0,1).float()/255,
            sar_aug=torch.from_numpy(augmented.copy())[None].float()/255,A_forward=torch.from_numpy(matrix.astype(np.float32)),
            optical_valid_mask=torch.ones(256,256,dtype=torch.bool),sar_aug_valid_mask=torch.from_numpy(valid))
