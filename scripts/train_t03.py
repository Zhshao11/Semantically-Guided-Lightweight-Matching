"""T03 full-data B0-Full, same optimizer/loss/batch; epoch-boundary restart checkpoints."""
import argparse
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sglm.t02.model import B0Full
from sglm.t02.loss import b0_loss
from sglm.t03.data import FullAffinePairs256,PresentationBatchSampler,worker_init
from sglm.t03.runner import atomic_json,state_digest,checkpoint,validate


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True);ap.add_argument('--plan',required=True)
    ap.add_argument('--config',required=True);ap.add_argument('--diagnostic-gate',required=True)
    ap.add_argument('--preflight',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--resume')
    a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    status=dict(pid=os.getpid(),phase='initializing',epoch=0,step=0)
    def update(**kwargs):
        status.update(kwargs);status['updated_unix']=time.time();atomic_json(out/'status.json',status)
    try:
        gate=json.loads(Path(a.diagnostic_gate).read_text());preflight=json.loads(Path(a.preflight).read_text())
        if not gate['T03_gate_passed'] or not gate['reproduction_A_exact'] or not preflight['passed']:
            raise RuntimeError('T02 reproduction/state and full-data preflight must pass')
        config=json.loads(Path(a.config).read_text());plan=json.loads(Path(a.plan).read_text())
        plan_hash=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest()
        if plan_hash!=preflight['full_plan_sha256']: raise ValueError('Preflight plan mismatch')
        if config['model']!=preflight['T02_model'] or config['evaluation']!=preflight['T02_evaluation']:
            raise ValueError('Model or RANSAC/evaluation changed')
        if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 CUDA required')
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
        model=B0Full(config).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
        initial_hash=state_digest(model)
        if not a.resume and initial_hash!=preflight['from_scratch_initial_state_sha256']:
            raise ValueError('From-scratch initialization differs from preflight')
        start_epoch,step,best=0,0,dict(epoch=None,score=None)
        pending_validation=False
        if a.resume:
            saved=torch.load(a.resume,map_location='cuda',weights_only=False)
            if saved['config']!=config or saved['plan_sha256']!=plan_hash: raise ValueError('Resume mismatch')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            start_epoch,step,best=saved['epoch'],saved['step'],saved['best']
            pending_validation=saved.get('pending_validation',False)
            torch.set_rng_state(saved['rng']['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in saved['rng']['cuda']])
            random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy'])
            for filename,key,limit in [('epoch_summaries.jsonl','epoch',start_epoch),('steps.jsonl','step',step),('validation_summaries.jsonl','epoch',start_epoch)]:
                log=out/filename
                if log.exists():
                    rows=[json.loads(x) for x in log.read_text().splitlines() if x.strip()]
                    discarded=[r for r in rows if r[key]>limit]
                    if discarded:
                        (out/(filename+'.orphaned')).write_text(''.join(json.dumps(r)+'\n' for r in discarded))
                        log.write_text(''.join(json.dumps(r)+'\n' for r in rows if r[key]<=limit))
        elif (out/'epoch_summaries.jsonl').exists() or (out/'initial.pt').exists():
            raise FileExistsError('New run directory or --resume required')
        atomic_json(out/'config.json',config)
        atomic_json(out/'environment.json',dict(torch=torch.__version__,device=torch.cuda.get_device_name(),
            physical_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),plan_sha256=plan_hash,
            initialization='from scratch',initial_state_sha256=initial_hash,parameters=sum(p.numel() for p in model.parameters()),
            learning_rate=config['training']['learning_rate'],code_revision=os.popen('git rev-parse HEAD').read().strip()))
        update(phase='loading_validation')
        validation=FullAffinePairs256(a.root,plan,'val',config,preload=True)
        training=FullAffinePairs256(a.root,plan,'train',config,preload=False)
        if len(training)!=7912 or len(validation)!=952: raise ValueError('Incorrect split counts')
        sampler=PresentationBatchSampler(len(training),config['training']['batch_size'],config['seed'])
        loader=DataLoader(training,batch_sampler=sampler,num_workers=config['training']['num_workers'],
            worker_init_fn=worker_init,pin_memory=True,persistent_workers=True,prefetch_factor=2,
            generator=torch.Generator().manual_seed(config['seed']))
        if not a.resume: checkpoint(out/'initial.pt',model,optimizer,0,0,best,config,plan_hash)
        def run_validation(epoch):
            nonlocal best
            update(phase='validating',epoch=epoch,step=step,validation_pairs_done=0)
            result=validate(model,validation,config,out,epoch,update)
            score=result['summary'][config['validation']['best_metric']]
            if best['score'] is None or score>best['score']:
                best=dict(epoch=epoch,score=score)
                checkpoint(out/'best.pt',model,optimizer,epoch,step,best,config,plan_hash)
            checkpoint(out/'last.pt',model,optimizer,epoch,step,best,config,plan_hash)
            with (out/'validation_summaries.jsonl').open('a') as f: f.write(json.dumps(dict(epoch=epoch,**result['summary'],best=best))+'\n')
            print(json.dumps(dict(phase='validated',epoch=epoch,summary=result['summary'],best=best)),flush=True)
        if pending_validation: run_validation(start_epoch)
        for epoch in range(start_epoch,config['training']['epochs']):
            model.train();sampler.epoch=epoch
            sums={k:0. for k in ('total','coarse','fine','positive_coarse_accuracy','fine_training_pairs')}
            trace=hashlib.sha256();count=0;begun=time.monotonic()
            update(phase='training',epoch=epoch+1,step=step,batches_done=0,batches_total=len(sampler))
            for batch in loader:
                for uid,seed,matrix in zip(batch['pair_id'],batch['augmentation_seed'],batch['A_forward']):
                    trace.update(uid.encode());trace.update(seed.encode());trace.update(matrix.numpy().tobytes())
                batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(batch['optical'],batch['sar_aug'],batch['optical_valid_mask'],batch['sar_aug_valid_mask'])
                loss,metrics=b0_loss(model,output,batch,config,step)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite T03 loss')
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config['training']['gradient_clip_norm'],error_if_nonfinite=True)
                optimizer.step();step+=1;count+=1
                sums['total']+=float(loss.detach())
                for key in sums:
                    if key!='total': sums[key]+=metrics[key]
                with (out/'steps.jsonl').open('a') as f:
                    f.write(json.dumps(dict(epoch=epoch+1,step=step,total=float(loss.detach()),grad_norm=float(grad),**metrics))+'\n')
                if count%50==0:
                    update(batches_done=count,step=step,last_loss=float(loss.detach()))
            summary=dict(epoch=epoch+1,step=step,batches=count,train_pairs=7912,seconds=time.monotonic()-begun,
                learning_rate=config['training']['learning_rate'],sample_and_augmentation_stream_sha256=trace.hexdigest(),
                **{key:value/count for key,value in sums.items()})
            with (out/'epoch_summaries.jsonl').open('a') as f: f.write(json.dumps(summary)+'\n')
            print(json.dumps(dict(phase='trained_epoch',**summary)),flush=True)
            due=epoch+1 in config['validation']['epochs']
            checkpoint(out/'last.pt',model,optimizer,epoch+1,step,best,config,plan_hash,pending_validation=due)
            if due: run_validation(epoch+1)
        update(phase='completed',epoch=config['training']['epochs'],step=step,best=best)
    except Exception as exc:
        update(phase='failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
