"""Real data optimizer updates and throughput before launching the long OS run."""
import argparse,json,os,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sglm.soma_training import setup,SOMAModel,amp_context,atomic_json,validate,EpochMicroSampler
from sglm.soma_os import prepare_images,inverse_targets,soma_loss
from sglm.t03.data import FullAffinePairs256,worker_init


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--micro',type=int,default=1);a=ap.parse_args()
    config,dsconfig,plan=setup();spec=config['training'];out=Path(config['reports']);out.mkdir(exist_ok=True)
    model=SOMAModel(config).train();params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW([{'params':[p for p in model.matcher.parameters() if p.requires_grad],'lr':spec['matcher_lr']},
        {'params':[p for p in model.encoder.parameters() if p.requires_grad],'lr':spec['encoder_lr']}],weight_decay=spec['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=spec['precision']!='bf16',init_scale=128.)
    ds=FullAffinePairs256(a.root,plan,'train',dsconfig,preload=False)
    sampler=EpochMicroSampler(len(ds),config['seed'],0,1,a.micro,a.micro,0)
    loader=DataLoader(ds,batch_sampler=sampler,num_workers=4,worker_init_fn=worker_init,pin_memory=True)
    times=[];rows=[];torch.cuda.reset_peak_memory_stats()
    total=config['preflight']['warmup_micro_steps']+config['preflight']['timed_micro_steps']
    for i,batch in enumerate(loader):
        if i>=total:break
        torch.cuda.synchronize();begin=time.perf_counter();opt.zero_grad(set_to_none=True)
        optical,sar=prepare_images(batch,config['native_size']);gt,valid=inverse_targets(batch,config['native_size'])
        with amp_context(config):output=model(optical,sar);loss,metrics=soma_loss(output,gt,valid)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_(params,spec['gradient_clip_norm']);assert torch.isfinite(norm),'Nonfinite training gradients'
        scaler.step(opt);scaler.update();torch.cuda.synchronize();elapsed=time.perf_counter()-begin
        rows.append(dict(step=i+1,seconds=elapsed,grad_norm=float(norm),**metrics));print(rows[-1],flush=True)
        if i>=config['preflight']['warmup_micro_steps']:times.append(elapsed)
    validation=FullAffinePairs256(a.root,plan,'val',dsconfig,preload=False)
    begin=time.perf_counter();result=validate(model,validation,config,dsconfig['evaluation'],limit=config['preflight']['measure_validation_pairs'])
    val_sec=(time.perf_counter()-begin)/result['summary']['image_pairs']
    mean=float(np.mean(times));record=dict(passed=True,micro_batch=a.micro,timed_steps=len(times),mean_step_seconds=mean,
        median_step_seconds=float(np.median(times)),single_gpu_images_per_second=a.micro/mean,
        validation_seconds_per_pair=val_sec,full_validation_estimate_seconds=val_sec*952,
        peak_training_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in params),
        proper_dino_registers_frozen=not any(p.requires_grad for p in model.encoder.dino_model.parameters()),
        finite_updates=True,no_test_used=True,rows=rows,initial_diagnostic_validation=result['summary'])
    atomic_json(out/f'preflight_micro{a.micro}.json',record);print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
