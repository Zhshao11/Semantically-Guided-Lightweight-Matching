"""T02 fixed-sample BF16 training with step-exact data schedule and resumable state."""
import argparse
import hashlib
import json
import math
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from sglm.t02.data import load_t02_config,AffinePairs256
from sglm.t02.model import B0Full
from sglm.t02.loss import b0_loss


def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def batch_indices(length,batch_size,step,seed):
    batches=math.ceil(length/batch_size)
    epoch,offset=divmod(step,batches)
    generator=torch.Generator().manual_seed(seed+epoch)
    permutation=torch.randperm(length,generator=generator).tolist()
    return epoch,permutation[offset*batch_size:(offset+1)*batch_size]


def save_checkpoint(path,model,optimizer,step,config,plan_hash,precision_evidence):
    checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,config=config,
        plan_sha256=plan_hash,rng=capture_rng(),precision_evidence=precision_evidence,
        model_name='B0-Full',schedule='epoch=step//ceil(N/B); permutation=randperm(seed+epoch)',
        inference_inputs=['optical','sar_aug','optical_valid_mask','sar_aug_valid_mask'])
    temporary=path.with_suffix('.tmp')
    torch.save(checkpoint,temporary)
    os.replace(temporary,path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--plan',required=True)
    ap.add_argument('--config',default='configs/t02_b0_full_v1.json')
    ap.add_argument('--output',required=True)
    ap.add_argument('--resume')
    ap.add_argument('--stop-step',type=int)
    ap.add_argument('--device',default='cuda')
    a=ap.parse_args()
    config=load_t02_config(a.config)
    if not a.device.startswith('cuda') or not torch.cuda.is_bf16_supported():
        raise RuntimeError('T02 acceptance requires a CUDA device supporting BF16')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    plan_hash=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest()
    dataset=AffinePairs256(a.root,a.plan,'train',mode='fixed')
    model=B0Full(config).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    step=0
    precision_evidence={}
    if a.resume:
        checkpoint=torch.load(a.resume,map_location=a.device,weights_only=False)
        if checkpoint['config']!=config or checkpoint['plan_sha256']!=plan_hash or checkpoint['model_name']!='B0-Full':
            raise ValueError('Resume config/model/fixed-plan mismatch')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        step=checkpoint['step'];precision_evidence=checkpoint['precision_evidence']
        restore_rng(checkpoint['rng'])
        with (out/'resume_history.jsonl').open('a') as f:
            f.write(json.dumps(dict(resumed_from_step=step,checkpoint=str(a.resume),optimizer_state_entries=len(optimizer.state),rng_restored=True))+'\n')
    elif (out/'train.jsonl').exists():
        raise FileExistsError('Use a new output directory or explicit --resume')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'environment.json').write_text(json.dumps(dict(torch=torch.__version__,device=torch.cuda.get_device_name(),
        bf16_supported=torch.cuda.is_bf16_supported(),parameters=sum(p.numel() for p in model.parameters()),plan_sha256=plan_hash),indent=2))
    if not a.resume:
        save_checkpoint(out/'initial.pt',model,optimizer,0,config,plan_hash,precision_evidence)
    stop=config['training']['total_steps'] if a.stop_step is None else a.stop_step
    if not step<stop<=config['training']['total_steps']:
        raise ValueError('stop-step must advance checkpoint and not exceed configured total steps')
    model.train()
    begun=time.monotonic()
    while step<stop:
        epoch,indices=batch_indices(len(dataset),config['training']['batch_size'],step,config['seed'])
        dataset.set_epoch(epoch)
        batch=default_collate([dataset[i] for i in indices])
        batch={k:v.to(a.device) if torch.is_tensor(v) else v for k,v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        # Coarse network uses BF16. Geometry/labels/loss and local sampling retain FP32.
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(batch['optical'],batch['sar_aug'],batch['optical_valid_mask'],batch['sar_aug_valid_mask'])
        total,losses=b0_loss(model,output,batch,config,step)
        if not torch.isfinite(total):
            raise FloatingPointError(f'Nonfinite T02 loss at {step}')
        total.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['training']['gradient_clip_norm'],error_if_nonfinite=True)
        optimizer.step()
        step+=1
        if not precision_evidence:
            precision_evidence=dict(coarse_feature_dtype=str(output['optical_coarse'].dtype),
                descriptor_dtype=str(output['optical_descriptors'].dtype),correlation_dtype=str(output['coarse_correlation'].dtype),
                parameters_dtype=str(next(model.parameters()).dtype),loss_dtype=str(total.dtype),
                backward_finite=True,coarse_feature_shape=list(output['optical_coarse'].shape),
                fine_feature_shape=list(output['optical_fine'].shape),descriptor_shape=list(output['optical_descriptors'].shape),
                correlation_shape=list(output['coarse_correlation'].shape))
            assert output['coarse_correlation'].dtype==torch.bfloat16
        record=dict(step=step,epoch=epoch,pair_ids=batch['pair_id'],total=float(total.detach()),grad_norm=float(norm),
                    seconds_this_session=time.monotonic()-begun,**losses)
        with (out/'train.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        if step==1 or step%50==0:
            print(json.dumps(record),flush=True)
        if step%config['training']['checkpoint_interval']==0 or step==stop:
            save_checkpoint(out/'last.pt',model,optimizer,step,config,plan_hash,precision_evidence)
    save_checkpoint(out/f'step_{step:06d}.pt',model,optimizer,step,config,plan_hash,precision_evidence)
    print(json.dumps(dict(completed_step=step,checkpoint=str(out/'last.pt'),precision_evidence=precision_evidence)),flush=True)


if __name__=='__main__':
    main()
