"""T09 paired training-order seeds; sameT08 losses/model, corrected fixedshuffle."""
import argparse,hashlib,json,os,time,random
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from sglm.t02.data import AffinePairs256
from sglm.t08_auxiliary import AuxiliaryB0,sparse_matching_loss
from sglm.semantic_targets import weighted_semantic_kl
from sglm.t03.runner import state_digest,atomic_json,validate
from sglm.diagnostics import bn_snapshot,bn_unchanged


class TrainingPairs(Dataset):
    def __init__(self,root,plan,group):
        self.base=AffinePairs256(root,plan,split='train',preload=True)
        cache=Path('runs/t09_shuffle_parent_v1') if group=='MS-shuffle' else Path('runs/t07_semantic_cache_v1')
        self.prob=np.load(cache/'conditional7.npy',mmap_mode='r');self.weight=np.load(cache/'weight.npy',mmap_mode='r')
        ids=json.loads(Path('runs/t07_semantic_cache_v1/pair_ids.json').read_text());assert ids==[r['pair_id'] for r in self.base.samples]
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        sample=self.base[i];p=np.array(self.prob[i],dtype=np.float32,copy=True);p/=np.maximum(p.sum(0,keepdims=True),1e-12)
        sample['pseudo7']=torch.from_numpy(p);sample['pseudo_weight']=torch.from_numpy(np.array(self.weight[i],copy=True));return sample


def matching_frozen_digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        if name.startswith('semantic_head.'):continue
        # All matching buffers must stay frozen; only matching parameters in declared prefixes can change.
        if name in dict(model.named_parameters()) and name.startswith(model.prefixes):continue
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--group',choices=['M','MS','MS-shuffle'],required=True);ap.add_argument('--seed',type=int,choices=[20261126,20261127,20261128],required=True);a=ap.parse_args()
    p=json.loads(Path('configs/t08_protocol_v1.json').read_text());t09=json.loads(Path('configs/t09_protocol_v1.json').read_text());p['seed']=a.seed;p['protocol']=t09['protocol'];p['T09']=t09;setup=json.loads(Path('reports/t08_setup_v1/setup.json').read_text());pre=json.loads(Path('reports/t08_preflight_v1/diagnosis.json').read_text());assert pre['training_source_pairs']==128
    test=Path('reports/t08_preflight_v1/tests.txt').read_text();assert 'passed' in test and 'failed' not in test
    out=Path('runs')/(f't09_seed{a.seed}_'+a.group+'_v1');out.mkdir(exist_ok=True)
    if (out/'steps.jsonl').exists() or (out/'last.pt').exists():raise FileExistsError('Do not overwrite T08 run')
    label_root=Path('runs/t09_shuffle_parent_v1') if a.group=='MS-shuffle' else Path('runs/t07_semantic_cache_v1')
    label_paths=[label_root/'conditional7.npy',label_root/'weight.npy']+([label_root/'donor_flat_index.npy'] if a.group=='MS-shuffle' else [])
    label_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in label_paths}
    label_epoch_checks=[];gradient_rows=[]
    start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(p['seed']);np.random.seed(p['seed']);random.seed(p['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    cp=Path(p['matcher_checkpoint']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==p['matcher_sha256'];saved=torch.load(cp,map_location='cpu',weights_only=False);cfg=saved['config'];cfg['training']['fine_gt_bootstrap_steps']=0
    model=AuxiliaryB0(cfg,256,a.group!='M',p['trainable_prefixes']);missing=model.load_state_dict(saved['model'],strict=False)
    assert not missing.unexpected_keys and all(key.startswith('semantic_head.') for key in missing.missing_keys)
    if a.group!='M':
        h=torch.load('runs/t07_semantic_head_v1/last.pt',map_location='cpu',weights_only=False)['head'];semantic={key[len('semantic.'):]:v for key,v in h.items() if key.startswith('semantic.')};model.semantic_head.load_state_dict(semantic,strict=True)
    model=model.cuda();model.eval();initial_matching_hash=state_digest(model.export_matching_model());frozen_before=matching_frozen_digest(model);initial_semantic_hash=None
    if hasattr(model,'semantic_head'):initial_semantic_hash=state_digest(model.semantic_head)
    trainable=[(n,v) for n,v in model.named_parameters() if v.requires_grad];opt=torch.optim.AdamW([v for n,v in trainable],lr=p['train']['lr'],weight_decay=p['train']['weight_decay'])
    atomic_json(out/'config.json',p);atomic_json(out/'architecture.json',dict(trainable={n:v.numel() for n,v in trainable},all_matching_parameters=sum(v.numel() for v in model.export_matching_model().parameters()),auxiliary_parameters=sum(v.numel() for v in model.semantic_head.parameters()) if hasattr(model,'semantic_head') else 0,initial_matching_state_sha256=initial_matching_hash,initial_semantic_state_sha256=initial_semantic_hash,frozen_state_sha256=frozen_before,all_backbone_BN_running_stats_frozen=True))
    training=TrainingPairs(a.root,p['training_plan'],a.group);validation=AffinePairs256(a.root,p['validation_plan'],split='val',preload=True);assert len(training)==7912 and len(validation)==952
    status={};best=dict(epoch=None,score=None);step=0;history=[];validations=[];streams=[]
    def update(**kwargs):status.update(kwargs);atomic_json(out/'status.json',dict(group=a.group,**status))
    def save(name,epoch):
        temporary=out/(name+'.tmp');torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,step=step,config=cfg,protocol=p,group=a.group,best=best,matcher_initial_sha256=p['matcher_sha256'],training_plan_sha256=setup['plan_sha256']['training_plan.json']),temporary);temporary.replace(out/name)
    def check_labels(epoch,phase):
        current={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in label_hashes}
        assert current==label_hashes
        label_epoch_checks.append(dict(epoch=epoch,phase=phase,sha256=current));atomic_json(out/'label_epoch_checks.json',label_epoch_checks)
    def run_validation(epoch,k=256):
        current=model.export_matching_model();current.k=k;folder=out/f'K{k}';folder.mkdir(exist_ok=True);update(phase='validating',epoch=epoch,step=step,K=k)
        result=validate(current,validation,cfg,folder,epoch,update);validations.append(dict(epoch=epoch,K=k,**result['summary']));del current
        return result
    initial=run_validation(0);old=json.loads(Path('reports/t05_results_v1/K256_validation.json').read_text())['pairs'];assert [r['evaluation'] for r in initial['pairs']]==[r['evaluation'] for r in old]
    best=dict(epoch=0,score=initial['summary']['success_rate@5px']);save('best.pt',0)
    for epoch in range(p['train']['epochs']):
        check_labels(epoch+1,'before');model.train();before_bn=bn_snapshot(model.backbone);order=np.random.default_rng(p['seed']+epoch).permutation(len(training)).tolist();trace=hashlib.sha256();n=0;sums=dict(total=0.,matching=0.,KL=0.,coarse=0.,fine=0.)
        loader=DataLoader(training,batch_size=4,sampler=order,num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(p['seed']+epoch),persistent_workers=False)
        update(phase='training',epoch=epoch+1,step=step)
        for batch in loader:
            for uid,matrix in zip(batch['pair_id'],batch['A_forward']):trace.update(uid.encode());trace.update(matrix.numpy().tobytes())
            batch={key:value.cuda(non_blocking=True) if torch.is_tensor(value) else value for key,value in batch.items()};opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):output=model(batch['optical'],batch['sar_aug'],batch['optical_valid_mask'],batch['sar_aug_valid_mask'])
            matching,metrics=sparse_matching_loss(model,output,batch,cfg)
            kl=weighted_semantic_kl(output['semantic_logits'],batch['pseudo7'],batch['pseudo_weight']) if a.group!='M' else matching.new_zeros(())
            total=matching+p['train']['KL_weight']*kl
            if not torch.isfinite(total):raise FloatingPointError('Nonfinite T08 loss')
            if step+1 in t09['gradient_audit']['trainer_steps']:
                encoder=[v for name,v in trainable if name.startswith(model.prefixes) and not name.startswith('backbone.descriptor.')]
                mg=torch.autograd.grad(matching,encoder,retain_graph=True,allow_unused=True)
                sg=torch.autograd.grad(p['train']['KL_weight']*kl,encoder,retain_graph=True,allow_unused=True) if a.group!='M' else [None]*len(encoder)
                mn=torch.stack([g.float().square().sum() for g in mg if g is not None]).sum().sqrt()
                sn=torch.stack([g.float().square().sum() for g in sg if g is not None]).sum().sqrt() if any(g is not None for g in sg) else mn.new_zeros(())
                dot=sum((x.float()*y.float()).sum() for x,y in zip(mg,sg) if x is not None and y is not None)
                cosine=float(dot/(mn*sn)) if sn>0 and mn>0 else None
                gradient_rows.append(dict(seed=a.seed,group=a.group,step=step+1,epoch=epoch+1,KL=float(kl.detach()),matching_loss=float(matching.detach()),semantic_encoder_L2=float(sn),matching_encoder_L2=float(mn),semantic_over_matching=float(sn/mn) if mn>0 else None,cosine=cosine,pre_globalclip=True,pair_ids=batch['pair_id']))
                atomic_json(out/'semantic_gradient_probes.json',gradient_rows)
            total.backward();grad=torch.nn.utils.clip_grad_norm_([v for n,v in trainable],p['train']['gradient_clip'],error_if_nonfinite=True);opt.step();step+=1;n+=1
            row=dict(epoch=epoch+1,step=step,total=float(total.detach()),matching=float(matching.detach()),KL=float(kl.detach()),grad_norm=float(grad),**metrics)
            with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            for name in sums:sums[name]+=row[name]
            if n%100==0:update(step=step,last_loss=row['total'],batches=n);print(a.group,epoch+1,step,row['total'],row['KL'],flush=True)
        check_labels(epoch+1,'after');assert bn_unchanged(before_bn,model.backbone) and matching_frozen_digest(model)==frozen_before
        streams.append(trace.hexdigest());history.append(dict(epoch=epoch+1,step=step,**{name:value/n for name,value in sums.items()},sample_affine_stream_sha256=streams[-1]));save('last.pt',epoch+1)
        atomic_json(out/'history.json',history)
        if epoch+1 in p['train']['validation_epochs']:
            result=run_validation(epoch+1);score=result['summary']['success_rate@5px']
            if score>best['score']:best=dict(epoch=epoch+1,score=score);save('best.pt',epoch+1)
            print('validated',a.group,epoch+1,score,flush=True)
        atomic_json(out/'validation_history.json',validations)
    assert step==p['train']['total_updates'];save('last.pt',p['train']['epochs'])
    # The fixed-last checkpoint is primary; export contains no auxiliary branch or teacher.
    infer=model.export_matching_model();torch.save(dict(model=infer.state_dict(),config=cfg,protocol=p,group=a.group,epoch=p['train']['epochs'],step=step),out/'inference_last.pt')
    atomic_json(out/'validation_history.json',validations)
    summary=dict(seed=a.seed,label_files_sha256=label_hashes,label_epoch_checks=label_epoch_checks,semantic_gradient_probes=gradient_rows,group=a.group,steps=step,epochs=p['train']['epochs'],train_pairs=7912,training_plan_sha256=setup['plan_sha256']['training_plan.json'],initial_matching_state_sha256=initial_matching_hash,initial_semantic_state_sha256=initial_semantic_hash,frozen_matching_state_unchanged=matching_frozen_digest(model)==frozen_before,sample_affine_stream_sha256=streams,history=history,validation_history=validations,best=best,primary='last epoch4 not validationbest',inference_has_semantic_head=any('semantic' in k for k in infer.state_dict()),inference_parameters=sum(v.numel() for v in infer.parameters()),inference_checkpoint_sha256=hashlib.sha256((out/'inference_last.pt').read_bytes()).hexdigest(),last_checkpoint_sha256=hashlib.sha256((out/'last.pt').read_bytes()).hexdigest(),seconds=time.monotonic()-start,OS_semantic_GT=False,SAR_semantic_loss=False,pseudoKL_updates_shared_coarse_encoder=a.group!='M',protocol=p)
    atomic_json(out/'summary.json',summary);update(phase='completed',step=step,epoch=p['train']['epochs']);print(a.group,'completed',step,flush=True)


if __name__=='__main__':main()
