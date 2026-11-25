"""Epoch-based full-data training and fixed independent validation."""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..t02.model import B0Full
from ..t02.loss import b0_loss
from ..diagnostics import bn_snapshot,bn_unchanged
from ..evaluation import estimate_from_correspondences,fixed_synthetic_probes,evaluate_transform,summarize_evaluations
from .data import FullAffinePairs256,PresentationBatchSampler,worker_init


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def state_digest(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def checkpoint(path,model,optimizer,epoch,step,best,config,plan_hash,pending_validation=False):
    value=dict(model_name='B0-Full',model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,step=step,
        best=best,config=config,plan_sha256=plan_hash,pending_validation=pending_validation,
        rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate()))
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)


@torch.no_grad()
def validate(model,dataset,config,out,epoch,status_callback):
    model.eval()
    if any(m.training for m in model.modules()): raise AssertionError('Not all eval')
    before=bn_snapshot(model)
    records=[]
    for index,item in enumerate(dataset):
        args=[item[k][None].cuda() for k in ('optical','sar_aug','optical_valid_mask','sar_aug_valid_mask')]
        with torch.autocast('cuda',dtype=torch.bfloat16): prediction=model.predict(*args)[0]
        source,target,scores=[prediction[k].float().cpu().numpy() for k in ('source','target','score')]
        estimate=estimate_from_correspondences(source,target,scores,config['evaluation'],seed=config['seed'])
        probes,truth=fixed_synthetic_probes(item['A_forward'].numpy(),grid=config['evaluation']['probe_grid'],
            optical_mask=item['optical_valid_mask'],sar_aug_mask=item['sar_aug_valid_mask'])
        evaluated=evaluate_transform(estimate['matrix'],probes,truth,config['evaluation']['success_thresholds_px'],status=estimate['status'])
        records.append(dict(pair_id=item['pair_id'],estimation=estimate,evaluation=evaluated))
        if (index+1)%100==0: status_callback(validation_pairs_done=index+1)
    if len(records)!=952 or not bn_unchanged(before,model): raise AssertionError('Validation count or BN state changed')
    result=dict(epoch=epoch,summary=summarize_evaluations([r['evaluation'] for r in records]),
        model_eval=True,BN_buffers_unchanged=True,reference='fixed independent probes; failures included',pairs=records)
    atomic_json(out/f'validation_epoch_{epoch:03d}.json',result)
    return result
