"""Shared SOMA training/evaluation setup and independent256-pixel affine metrics."""
import contextlib,hashlib,json,os,time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from .soma_os import build_soma,prepare_images,inverse_targets,dense_to_matches
from .external_reference import valid_correspondences
from .evaluation import estimate_from_correspondences,fixed_synthetic_probes,evaluate_transform,summarize_evaluations
from .t03.data import FullAffinePairs256


def atomic_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)


def setup(config_path='configs/soma_os_training_v1.json'):
    config=json.loads(Path(config_path).read_text());dataset_config=json.loads((Path(config['reports'])/'dataset_config.json').read_text())
    plan=json.loads(Path(config['data_plan']).read_text());torch.set_num_threads(4)
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    return config,dataset_config,plan


def amp_context(config):
    return torch.autocast('cuda',dtype=torch.bfloat16 if config['training']['precision']=='bf16' else torch.float16)


class SOMAModel(nn.Module):
    def __init__(self,config):
        super().__init__();self.encoder,self.matcher=build_soma(config)
    def forward(self,optical,sar):
        return self.matcher(self.encoder(sar,'sar')['sar'],self.encoder(optical,'opt')['opt'])


def checkpoint(path,model,optimizer,scaler,config,epoch,updates,best,next_update=0,scheduler=None):
    # Frozen pretrainedDINO is identified bySHA and loaded separately; don't duplicate
    # its three state_dict aliases in every checkpoint/optimizer save.
    state={k:v.detach().cpu() for k,v in model.state_dict().items() if 'dino_model' not in k}
    obj=dict(model=state,config=config,epoch=epoch,updates=updates,next_update=next_update,best=best,
        plan_sha256=hashlib.sha256(Path(config['data_plan']).read_bytes()).hexdigest(),
        dinov2_sha256=hashlib.sha256(Path(config['dinov2_weights']).read_bytes()).hexdigest())
    if optimizer is not None:obj['optimizer']=optimizer.state_dict();obj['scaler']=scaler.state_dict()
    if scheduler is not None:obj['scheduler']=scheduler.state_dict()
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)


def load_model_state(model,path,config):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['plan_sha256']==hashlib.sha256(Path(config['data_plan']).read_bytes()).hexdigest()
    incompatible=model.load_state_dict(saved['model'],strict=False)
    assert not incompatible.unexpected_keys and all('dino_model' in k for k in incompatible.missing_keys)
    return saved


@torch.no_grad()
def predict(model,item,config):
    batch={k:item[k][None] for k in ['optical','sar_aug','optical_valid_mask','sar_aug_valid_mask']}
    optical,sar=prepare_images(batch,config['native_size'])
    with amp_context(config):out=model(optical,sar)
    src,dst,score=dense_to_matches(out['flow_final'],out['certainty'],config['evaluation']['correspondence_grid'])
    aa=[x.float().cpu().numpy() for x in [src,dst,score]]
    return valid_correspondences(*aa,item['optical_valid_mask'].numpy(),item['sar_aug_valid_mask'].numpy())[:3]


@torch.no_grad()
def validate(model,dataset,config,evaluator,limit=None,progress=None):
    model.eval();records=[]
    n=len(dataset) if limit is None else min(limit,len(dataset))
    for i in range(n):
        item=dataset[i];aa=predict(model,item,config)
        fit=estimate_from_correspondences(*aa,evaluator,seed=config['evaluation']['solver_seed'])
        probes,truth=fixed_synthetic_probes(item['A_forward'].numpy(),grid=evaluator['probe_grid'],
            optical_mask=item['optical_valid_mask'],sar_aug_mask=item['sar_aug_valid_mask'])
        ev=evaluate_transform(fit['matrix'],probes,truth,evaluator['success_thresholds_px'],status=fit['status'])
        records.append(dict(pair_id=item['pair_id'],matches=len(aa[0]),estimation=fit,evaluation=ev))
        if progress is not None and (i+1)%100==0:progress(i+1,n)
    summary=summarize_evaluations([r['evaluation'] for r in records]);errors=[r['evaluation']['rmse_px'] for r in records if r['evaluation']['solved']]
    summary['solved_p90_rmse_px']=float(np.percentile(errors,90)) if errors else None
    return dict(summary=summary,pairs=records)


class EpochMicroSampler(torch.utils.data.Sampler):
    def __init__(self,n,seed,rank,world,micro,effective,epoch,start_update=0):
        self.n,self.seed,self.rank,self.world,self.micro,self.effective,self.epoch,self.start=n,seed,rank,world,micro,effective,epoch,start_update
        assert n%effective==0 and effective%(world*micro)==0
        self.accum=effective//(world*micro)
    def __len__(self):return (self.n//self.effective-self.start)*self.accum
    def __iter__(self):
        order=torch.randperm(self.n,generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
        for u in range(self.start,self.n//self.effective):
            for a in range(self.accum):
                start=u*self.effective+(a*self.world+self.rank)*self.micro
                yield [(order[pos],self.epoch*self.n+pos) for pos in range(start,start+self.micro)]
