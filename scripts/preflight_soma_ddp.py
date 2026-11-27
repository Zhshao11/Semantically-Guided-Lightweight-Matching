"""Bounded real4GPU throughput check; discarded weights, no test access."""
import argparse,datetime,json,os,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from sglm.soma_training import setup,SOMAModel,amp_context,atomic_json,EpochMicroSampler
from sglm.soma_os import prepare_images,inverse_targets,soma_loss
from sglm.t03.data import FullAffinePairs256,worker_init


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    local=int(os.environ['LOCAL_RANK']);rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(local)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=20))
    config,dsconfig,plan=setup();model=SOMAModel(config).train();spec=config['training']
    params=[p for p in model.parameters() if p.requires_grad];wrapper=DDP(model,device_ids=[local],find_unused_parameters=True,broadcast_buffers=False)
    opt=torch.optim.AdamW(params,lr=spec['matcher_lr'],weight_decay=spec['weight_decay']);scaler=torch.amp.GradScaler('cuda',enabled=spec['precision']!='bf16',init_scale=128.)
    ds=FullAffinePairs256(a.root,plan,'train',dsconfig,preload=False);sampler=EpochMicroSampler(len(ds),config['seed'],rank,world,1,world,0)
    loader=DataLoader(ds,batch_sampler=sampler,num_workers=2,worker_init_fn=worker_init,pin_memory=True)
    times=[];losses=[];torch.cuda.reset_peak_memory_stats()
    for i,batch in enumerate(loader):
        if i>=15:break
        dist.barrier(device_ids=[local]);torch.cuda.synchronize();begin=time.perf_counter();opt.zero_grad(set_to_none=True)
        optimg,sar=prepare_images(batch,config['native_size']);gt,valid=inverse_targets(batch,config['native_size'])
        with amp_context(config):output=wrapper(optimg,sar);loss,metrics=soma_loss(output,gt,valid)
        assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
        scaler.step(opt);scaler.update();torch.cuda.synchronize();elapsed=torch.tensor([time.perf_counter()-begin],device='cuda')
        dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
        if i>=3:times.append(float(elapsed));losses.append(float(loss.detach()))
        if rank==0:print('DDPpreflight',i+1,float(elapsed),float(loss.detach()),flush=True)
    if rank==0:
        single=json.loads((Path(config['reports'])/'preflight_micro1.json').read_text());seconds=float(np.mean(times));val=single['full_validation_estimate_seconds']
        epoch_train=seconds*(7912//world);epoch_total=epoch_train+val
        result=dict(passed=True,world_size=world,effective_batch=world,mean_update_seconds=seconds,median_update_seconds=float(np.median(times)),
            images_per_second=world/seconds,train_epoch_estimate_seconds=epoch_train,validation_epoch_estimate_seconds=val,
            base_epoch_estimate_seconds=epoch_total,estimated_hours_20epochs=20*epoch_total/3600,
            estimated_hours_100epochs=100*epoch_total/3600,
            estimated_range_hours_100epochs=[100*epoch_total/3600*.85,100*epoch_total/3600*1.35],
            estimated_range_hours_20epochs=[20*epoch_total/3600*.85,20*epoch_total/3600*1.35],
            peak_perGPU_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
            weights_discarded=True,no_test_used=True,losses=losses)
        atomic_json(Path(config['reports'])/'throughput_estimate.json',result);print(json.dumps(result,indent=2))
    dist.barrier(device_ids=[local]);dist.destroy_process_group()


if __name__=='__main__':main()
