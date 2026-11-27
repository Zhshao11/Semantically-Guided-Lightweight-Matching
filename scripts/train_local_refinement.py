"""Paired local-only fine tuning: identical frozen features, candidates and labels."""
import argparse,hashlib,json,os,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sglm.local_interaction import build_local_group,fixed_supervision
from sglm.t03.data import FullAffinePairs256,PresentationBatchSampler,worker_init
from sglm.experiment_io import write_csv


def digest(model,local=False):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith('local.')!=local:continue
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    p=json.loads(Path('configs/local_interaction_v1.json').read_text());out=Path('reports/local_interaction_v1');out.mkdir(parents=True,exist_ok=True)
    run=Path('runs/local_interaction_v1');run.mkdir(parents=True,exist_ok=True)
    if (run/'training_completed.json').exists() or (run/'original_refiner_last.pt').exists():raise FileExistsError('Run exists')
    assert hashlib.sha256(Path(p['checkpoint']).read_bytes()).hexdigest()==p['checkpoint_sha256']
    saved=torch.load(p['checkpoint'],map_location='cpu',weights_only=False);cfg=saved['config']
    dsconfig=json.loads(Path('reports/t03_setup_v1/main_config.json').read_text());dsconfig['seed']=p['seed']
    torch.set_num_threads(4);torch.manual_seed(p['seed']);np.random.seed(p['seed']);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    models={g:build_local_group(cfg,saved['model'],g) for g in p['groups']}
    initial={g:digest(m) for g,m in models.items()};assert len(set(initial.values()))==1
    for m in models.values():
        m.requires_grad_(False);m.local.requires_grad_(True);m.eval();m.local.train()
    optimizers={g:torch.optim.AdamW(m.local.parameters(),lr=p['training']['lr'],weight_decay=p['training']['weight_decay']) for g,m in models.items()}
    ds=FullAffinePairs256(a.root,'reports/t03_setup_v1/full_data_plan.json','train',dsconfig,preload=False)
    assert len(ds)==7912
    sampler=PresentationBatchSampler(len(ds),4,p['seed'])
    loader=DataLoader(ds,batch_sampler=sampler,num_workers=4,worker_init_fn=worker_init,pin_memory=True,
        persistent_workers=True,generator=torch.Generator().manual_seed(p['seed']))
    steps=0;rows=[];epochs=[];begin=time.perf_counter();initial_regression=[];zero_batches=0
    for epoch in range(p['training']['epochs']):
        sampler.epoch=epoch;stream=hashlib.sha256();epoch_rows=[]
        for batch in loader:
            for uid,seed,A in zip(batch['pair_id'],batch['augmentation_seed'],batch['A_forward']):
                stream.update(uid.encode());stream.update(seed.encode());stream.update(A.numpy().tobytes())
            args=[batch[k].cuda(non_blocking=True) for k in ['optical','sar_aug','optical_valid_mask','sar_aug_valid_mask']]
            matrices=batch['A_forward'].cuda()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):output=models['original_refiner'](*args)
            packets=[]
            for i,A in enumerate(matrices):
                s,c,conf=models['original_refiner'].coarse_matches(output,i)
                truth,visible,inside=fixed_supervision(s,c,A,args[3][i],args[2][i])
                packets.append((s,c,truth,inside))
            reference_valid=None;selected_count=sum(len(x[0]) for x in packets);supervised_count=None
            metrics={}
            for group,m in models.items():
                optimizers[group].zero_grad(set_to_none=True)
                loss=sum((parameter.sum()*0 for parameter in m.local.parameters()));count=0;valid_sets=[];initial_outputs=[]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    for i,(source,centers,truth,inside) in enumerate(packets):
                        local=m.local(output['optical_fine'][i:i+1],output['sar_fine'][i:i+1],source,centers,args[2][i:i+1],args[3][i:i+1])
                        valid=local.get('valid',torch.ones(len(source),device='cuda',dtype=torch.bool));valid_sets.append(valid)
                        supervised=inside&valid;count+=int(supervised.sum())
                        if supervised.any():loss=loss+F.smooth_l1_loss(local['target'][supervised].float(),truth[supervised].float())/len(args[0])
                        if steps==0:initial_outputs.append(local['target'].detach().clone())
                if reference_valid is None:reference_valid=valid_sets;supervised_count=count;initial_regression=initial_outputs
                else:
                    assert all(torch.equal(a,b) for a,b in zip(reference_valid,valid_sets)) and count==supervised_count
                    if steps==0:assert all(torch.equal(a,b) for a,b in zip(initial_regression,initial_outputs))
                weighted=p['training']['loss_weight']*loss;assert torch.isfinite(weighted)
                weighted.backward();grad=torch.nn.utils.clip_grad_norm_(m.local.parameters(),p['training']['gradient_clip'],error_if_nonfinite=True)
                optimizers[group].step();metrics[group]=dict(loss=float(loss.detach()),grad=float(grad),supervised=count)
            steps+=1;zero_batches+=int(supervised_count==0)
            record=dict(epoch=epoch+1,step=steps,selected_refinements=selected_count,supervised_correspondences=supervised_count,
                **{g+'_loss':v['loss'] for g,v in metrics.items()},**{g+'_grad':v['grad'] for g,v in metrics.items()})
            rows.append(record);epoch_rows.append(record)
            if steps%100==0:
                (run/'status.json').write_text(json.dumps(dict(phase='paired_training',epoch=epoch+1,step=steps,total_updates=p['training']['updates'],
                    mean_seconds_per_update=(time.perf_counter()-begin)/steps,**metrics),indent=2));print('paired',steps,metrics,flush=True)
        epochs.append(dict(epoch=epoch+1,updates=steps,stream_sha256=stream.hexdigest(),
            **{g+'_loss':float(np.mean([r[g+'_loss'] for r in epoch_rows])) for g in p['groups']}))
        for group,m in models.items():
            assert digest(m)==initial[group]
            torch.save(dict(model=m.state_dict(),group=group,epoch=epoch+1,updates=steps,protocol=p,config=cfg,
                frozen_state_sha256=initial[group],optimizer=optimizers[group].state_dict()),run/f'{group}_last.pt')
        write_csv(out/'training_steps.csv',rows);write_csv(out/'training_epochs.csv',epochs)
    assert steps==p['training']['updates']
    result=dict(passed=True,updates_per_group=steps,epochs=p['training']['epochs'],train_pairs=7912,
        candidates_and_supervision_identical=True,frozen_encoder_coarse_and_BN_unchanged=True,
        initial_prediction_bitwise_identical_on_first_real_batch=True,zero_supervision_batches=zero_batches,
        frozen_sha256=initial,actual_parameters={g:sum(x.numel() for x in m.parameters()) for g,m in models.items()},
        trainable_parameters={g:sum(x.numel() for x in m.local.parameters()) for g,m in models.items()},
        wall_seconds=time.perf_counter()-begin,test_used=False,no_semantics=True,no_dynamic_budget=True)
    (run/'training_completed.json').write_text(json.dumps(result,indent=2));(out/'training_acceptance.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
