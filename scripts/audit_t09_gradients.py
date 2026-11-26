"""Matched training-source replay: KL and coarseencoder gradients, no weight retuning."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from sglm.t02.data import AffinePairs256
from sglm.t08_auxiliary import AuxiliaryB0,sparse_matching_loss
from sglm.semantic_targets import weighted_semantic_kl
from sglm.experiment_io import write_csv


def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    p=json.loads(Path('configs/t08_protocol_v1.json').read_text());base=torch.load(p['matcher_checkpoint'],map_location='cpu',weights_only=False);cfg=base['config'];cfg['training']['fine_gt_bootstrap_steps']=0
    ds=AffinePairs256('/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0','reports/t08_setup_v1/diagnostic_plan.json',split='train',preload=True);ids=json.loads(Path('runs/t07_semantic_cache_v1/pair_ids.json').read_text());index={uid:i for i,uid in enumerate(ids)}
    caches={name:dict(p=np.load(Path(root)/'conditional7.npy',mmap_mode='r'),w=np.load(Path(root)/'weight.npy',mmap_mode='r')) for name,root in [('MS','runs/t07_semantic_cache_v1'),('T08-shuffle','runs/t08_shuffle_v1'),('T09-shuffle','runs/t09_shuffle_parent_v1')]};rows=[]
    for checkpoint in ['common_initial','T08_last']:
        for label,cache in caches.items():
            model=AuxiliaryB0(cfg,256,True)
            if checkpoint=='common_initial':
                model.load_state_dict(base['model'],strict=False);head=torch.load('runs/t07_semantic_head_v1/last.pt',map_location='cpu',weights_only=False)['head'];model.semantic_head.load_state_dict({key[len('semantic.'):]:v for key,v in head.items() if key.startswith('semantic.')})
            else:
                # Correctedlabels can also be probed at the ORIGINALshuffle endpoint, without retraining it.
                name='MS' if label=='MS' else 'MS-shuffle';state=torch.load(f'runs/t08_{name}_v1/last.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True)
            model=model.cuda().train();encoder=[v for n,v in model.named_parameters() if v.requires_grad and n.startswith(model.prefixes) and not n.startswith('backbone.descriptor.')]
            # Reset semanticBN each replay batch to avoid replay-order effects; checkpoint files stay untouched.
            initial_buffers={n:v.detach().clone() for n,v in model.semantic_head.named_buffers()}
            for start in range(0,128,4):
                for n,v in model.semantic_head.named_buffers():v.copy_(initial_buffers[n])
                items=[ds[i] for i in range(start,start+4)];batch={key:torch.stack([item[key] for item in items]).cuda() for key in ['optical','sar_aug','optical_valid_mask','sar_aug_valid_mask','A_forward']};batch_ids=[item['pair_id'] for item in items];selected=[index[uid] for uid in batch_ids]
                target=torch.from_numpy(np.array(cache['p'][selected],dtype=np.float32)).cuda();target/=target.sum(1,keepdim=True);weight=torch.from_numpy(np.array(cache['w'][selected],copy=True)).cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):output=model(batch['optical'],batch['sar_aug'],batch['optical_valid_mask'],batch['sar_aug_valid_mask'])
                match,_=sparse_matching_loss(model,output,batch,cfg);kl=weighted_semantic_kl(output['semantic_logits'],target,weight)
                sg=torch.autograd.grad(kl,encoder,retain_graph=True,allow_unused=True);mg=torch.autograd.grad(match,encoder,allow_unused=True)
                sn=sum(float(g.float().square().sum()) for g in sg if g is not None)**.5;mn=sum(float(g.float().square().sum()) for g in mg if g is not None)**.5;dot=sum(float((s.float()*m.float()).sum()) for s,m in zip(sg,mg) if s is not None and m is not None)
                rows.append(dict(checkpoint=checkpoint,label_version=label,batch=start//4,KL=float(kl.detach()),matching_loss=float(match.detach()),semantic_encoder_L2=sn,matching_encoder_L2=mn,semantic_over_matching=sn/mn if mn else None,cosine=dot/(sn*mn) if sn*mn else None,accepted_fraction=float((weight>0).float().mean()),mean_weight=float(weight.mean()),pair_ids=';'.join(batch_ids)))
            print('gradient replay',checkpoint,label,flush=True);del model
    summary=[]
    for checkpoint in ['common_initial','T08_last']:
        for label in caches:
            group=[r for r in rows if r['checkpoint']==checkpoint and r['label_version']==label];summary.append(dict(checkpoint=checkpoint,label_version=label,matched_batches=len(group),**{name:float(np.mean([r[name] for r in group])) for name in ['KL','matching_loss','semantic_encoder_L2','matching_encoder_L2','semantic_over_matching','accepted_fraction','mean_weight']}))
    out=Path('reports/t09_audit_v1');write_csv(out/'gradient_replay_by_batch.csv',rows);write_csv(out/'gradient_replay_summary.csv',summary)
    (out/'gradient_replay.json').write_text(json.dumps(dict(rows=summary,matched_train_pairs=128,lambda_sem=1.,semantic_BN='train mode onthrowawaycopy, resetbuffers foreachbatch',matching_BN='frozen',norm='preclip L2 coarseencoder; excludes descriptor projection/auxhead',historical_logged_gradients=False,scope='matched diagnostic replay at commoninitial and saved T08final snapshots; not trajectory-equivalent historical measurements',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
