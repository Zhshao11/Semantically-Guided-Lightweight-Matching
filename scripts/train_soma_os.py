"""OS-only SOMA adaptation: DDP, deterministic online stream, automatic val checkpointing."""
import argparse,contextlib,datetime,hashlib,json,os,time,traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from sglm.soma_training import setup,SOMAModel,amp_context,atomic_json,validate,EpochMicroSampler,checkpoint,load_model_state
from sglm.soma_os import prepare_images,inverse_targets,soma_loss
from sglm.t03.data import FullAffinePairs256,worker_init


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--resume',default=None);a=ap.parse_args()
    local=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    torch.cuda.set_device(local)
    if world>1:dist.init_process_group('nccl',timeout=datetime.timedelta(hours=2))
    config,dsconfig,plan=setup();spec=config['training'];out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    status=dict(pid=os.getpid(),world_size=world,physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),phase='initializing')
    def update(**kw):
        status.update(kw);status['updated_unix']=time.time()
        if rank==0:atomic_json(out/'status.json',status)
    def barrier():
        if world>1:dist.barrier(device_ids=[local])
    try:
        preflight=json.loads((Path(config['reports'])/'preflight_micro1.json').read_text());assert preflight['passed']
        frozen=json.loads((Path(config['reports'])/'launch_freeze.json').read_text());assert frozen['protocol']==config
        for path,digest in frozen['input_source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        model=SOMAModel(config);params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW([{'params':[p for p in model.matcher.parameters() if p.requires_grad],'lr':spec['matcher_lr']},
            {'params':[p for p in model.encoder.parameters() if p.requires_grad],'lr':spec['encoder_lr']}],weight_decay=spec['weight_decay'])
        scaler=torch.amp.GradScaler('cuda',enabled=spec['precision']!='bf16',init_scale=128.)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=spec['plateau_factor'],patience=spec['plateau_patience_epochs'])
        training=FullAffinePairs256(a.root,plan,'train',dsconfig,preload=False)
        validation=FullAffinePairs256(a.root,plan,'val',dsconfig,preload=True) if rank==0 else None
        updates=0;epoch_start=0;start_update=0;best=dict(epoch=None,SR5=None,bad_epochs=0)
        if a.resume:
            saved=load_model_state(model,a.resume,config);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
            if 'scheduler' in saved:scheduler.load_state_dict(saved['scheduler'])
            epoch_start=saved['epoch'];updates=saved['updates'];start_update=saved.get('next_update',0);best=saved['best']
        elif (out/'last.pt').exists() or (out/'steps.jsonl').exists():raise FileExistsError('Newrun directory or resume required')
        wrapper=DDP(model,device_ids=[local],find_unused_parameters=True,broadcast_buffers=False) if world>1 else model
        if rank==0:
            atomic_json(out/'config.json',config)
            atomic_json(out/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
                world_size=world,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in params),
                proper_DINO_registers=model.encoder.dino_model.num_register_tokens,
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('sglm/soma_os.py'),Path('sglm/soma_training.py'),Path('scripts/train_soma_os.py')]},
                test_not_loaded_in_training=True))
        barrier();updates_per_epoch=len(training)//spec['effective_batch_size'];accum=spec['effective_batch_size']//(world*spec['micro_batch_size'])
        base_lrs=[spec['matcher_lr'],spec['encoder_lr']]
        for epoch in range(epoch_start,spec['epochs']):
            model.train();wrapper.train()
            if epoch<spec['warmup_epochs']:
                for g,lr in zip(optimizer.param_groups,base_lrs):g['lr']=lr*(epoch+1)/spec['warmup_epochs']
            sampler=EpochMicroSampler(len(training),config['seed'],rank,world,spec['micro_batch_size'],spec['effective_batch_size'],epoch,start_update)
            loader=DataLoader(training,batch_sampler=sampler,num_workers=spec['num_workers'],worker_init_fn=worker_init,pin_memory=True,
                persistent_workers=False,generator=torch.Generator().manual_seed(config['seed']+epoch+rank))
            optimizer.zero_grad(set_to_none=True);epoch_sum=0.;epoch_count=0;begin=time.perf_counter();pending_metrics=0.
            update(phase='training',epoch=epoch+1,updates=updates,updates_in_epoch=start_update,updates_per_epoch=updates_per_epoch,best=best)
            for i,batch in enumerate(loader):
                sync=(i+1)%accum==0
                scope=wrapper.no_sync() if world>1 and not sync else contextlib.nullcontext()
                with scope:
                    optical,sar=prepare_images(batch,config['native_size']);truth,valid=inverse_targets(batch,config['native_size'])
                    with amp_context(config):output=wrapper(optical,sar);loss,metrics=soma_loss(output,truth,valid)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite SOMA loss')
                    scaler.scale(loss/accum).backward();pending_metrics+=metrics['total']/accum
                if not sync:continue
                scaler.unscale_(optimizer);grad=torch.nn.utils.clip_grad_norm_(params,spec['gradient_clip_norm'])
                finite=torch.tensor([int(torch.isfinite(grad))],device='cuda')
                if world>1:dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not bool(finite):raise FloatingPointError('Nonfinite SOMA gradients; inspect rather than silentfailed updates')
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);updates+=1;epoch_count+=1
                values=torch.tensor([pending_metrics],device='cuda')
                if world>1:dist.all_reduce(values);values/=world
                epoch_sum+=float(values);pending_metrics=0.;completed=start_update+epoch_count
                if rank==0:
                    row=dict(epoch=epoch+1,update=updates,update_in_epoch=completed,total=float(values),grad_norm=float(grad),
                        matcher_lr=optimizer.param_groups[0]['lr'],encoder_lr=optimizer.param_groups[1]['lr'])
                    with (out/'steps.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                    if updates%25==0:update(updates=updates,updates_in_epoch=completed,last_loss=float(values),seconds_per_update=(time.perf_counter()-begin)/epoch_count)
                    if updates%spec['checkpoint_interval_updates']==0:
                        checkpoint(out/'last.pt',model,optimizer,scaler,config,epoch,updates,best,completed,scheduler)
            barrier()
            if rank==0:
                summary=dict(epoch=epoch+1,updates=updates,mean_loss=epoch_sum/max(epoch_count,1),seconds=time.perf_counter()-begin)
                with (out/'epoch_summaries.jsonl').open('a') as stream:stream.write(json.dumps(summary)+'\n')
                update(phase='validating',epoch=epoch+1,updates=updates,validation_done=0)
                result=validate(model,validation,config,dsconfig['evaluation'],progress=lambda done,total:update(validation_done=done,validation_total=total))
                score=result['summary']['success_rate@5px']
                if best['SR5'] is None or score>best['SR5']:
                    best=dict(epoch=epoch+1,SR5=score,bad_epochs=0)
                    checkpoint(out/'best.pt',model,None,None,config,epoch+1,updates,best)
                else:best['bad_epochs']+=1
                scheduler.step(score)
                checkpoint(out/'last.pt',model,optimizer,scaler,config,epoch+1,updates,best,0,scheduler)
                atomic_json(out/f'validation_epoch_{epoch+1:03d}.json',result)
                with (out/'validation_summaries.jsonl').open('a') as stream:stream.write(json.dumps(dict(epoch=epoch+1,**result['summary'],best=best))+'\n')
                print(json.dumps(dict(phase='validated',epoch=epoch+1,summary=result['summary'],best=best)),flush=True)
                stop=(epoch+1>=spec['minimum_epochs'] and best['bad_epochs']>=spec['early_stop_patience_epochs'])
            else:stop=False
            flag=torch.tensor([int(stop)],device='cuda')
            if world>1:dist.broadcast(flag,src=0)
            start_update=0;barrier()
            if bool(flag):break
        if rank==0:
            update(phase='training_completed',epoch=epoch+1,updates=updates,best=best)
            atomic_json(out/'training_finished.json',dict(completed=True,epoch=epoch+1,updates=updates,best=best,final_test_not_yet_used=True))
        barrier()
    except Exception as error:
        update(phase='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
