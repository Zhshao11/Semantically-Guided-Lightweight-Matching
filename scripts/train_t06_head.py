"""Only appearance-head optimization on cached frozen train features/labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sglm.t06_appearance import AppearanceHead


def ranking_metrics(scores,labels):
    y=labels[labels>=0].astype(int);s=scores[labels>=0]
    order=np.argsort(-s,kind='stable');y=y[order];s=s[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1];tp=np.cumsum(y)[ends];n=ends+1
    positives=int(y.sum());negatives=len(y)-positives
    recall=tp/max(positives,1);precision=tp/n
    ap=float((np.diff(np.r_[0,recall])*precision).sum())
    fpr=(n-tp)/max(negatives,1);auc=float(np.trapezoid(np.r_[0,recall],np.r_[0,fpr]))
    return dict(average_precision=ap,ROC_AUC=auc,positive_fraction=positives/len(y),supervised_queries=len(y))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',default='runs/t06_reliability_v1')
    ap.add_argument('--output',required=True);ap.add_argument('--width',type=int,default=16);ap.add_argument('--small-fit',action='store_true')
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists():raise FileExistsError('Do not overwrite trained head')
    setup=json.loads(Path('reports/t06_setup_v1/setup.json').read_text());p=setup['protocol'];cache=Path(a.cache)
    summary=json.loads((cache/'summary.json').read_text());assert summary['labels_from_train_only'] and summary['matcher_sha256']==p['matcher_sha256']
    pair_ids=json.loads((cache/'pair_ids.json').read_text());assert len(pair_ids)==7912 and all('/train/' in uid for uid in pair_ids)
    features=np.load(cache/'optical_features.npy',mmap_mode='r');sar=np.load(cache/'sar_mean.npy',mmap_mode='r');labels=np.load(cache/'geometric_correctness.npy',mmap_mode='r')
    probe=setup['probe_indices'];indices=probe if a.small_fit else list(range(7912))
    torch.set_num_threads(4);torch.manual_seed(p['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    head=AppearanceHead(a.width).cuda();optimizer=torch.optim.AdamW(head.parameters(),lr=p['head']['learning_rate'],weight_decay=p['head']['weight_decay'])
    pos_weight=torch.tensor(summary['label_counts']['negative']/summary['label_counts']['positive'],device='cuda')
    def batch(ids):
        return [torch.from_numpy(np.array(array[ids],copy=True)).cuda() for array in [features,sar,labels]]
    def loss_fn(logits,y):
        valid=y>=0;target=y.clamp_min(0).float()
        losses=F.binary_cross_entropy_with_logits(logits.flatten(1),target,pos_weight=pos_weight,reduction='none')
        return losses[valid].mean()
    @torch.no_grad()
    def evaluate():
        head.eval();scores=[];ys=[];losses=[]
        for start in range(0,len(probe),16):
            fo,fs,y=batch(probe[start:start+16]);logits=head(fo,fs);scores.append(logits.cpu().numpy().reshape(-1));ys.append(y.cpu().numpy().reshape(-1));losses.append(float(loss_fn(logits,y)))
        return dict(loss=float(np.mean(losses)),**ranking_metrics(np.concatenate(scores),np.concatenate(ys)))
    initial=evaluate();history=[];step=0
    epochs=1 if a.small_fit else p['head']['epochs'];batch_size=p['head']['batch_size']
    for epoch in range(epochs):
        head.train();rng=np.random.default_rng(p['seed']+epoch);order=np.array(indices)[rng.permutation(len(indices))]
        if a.small_fit:
            batches=[rng.choice(indices,batch_size,replace=False).tolist() for _ in range(p['head']['small_fit_steps'])]
        else:batches=[order[i:i+batch_size].tolist() for i in range(0,len(order),batch_size)]
        losses=[]
        for ids in batches:
            fo,fs,y=batch(ids);optimizer.zero_grad(set_to_none=True);logits=head(fo,fs);loss=loss_fn(logits,y)
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),p['head']['gradient_clip'],error_if_nonfinite=True);optimizer.step();step+=1;losses.append(float(loss.detach()))
        probe_result=evaluate();row=dict(epoch=epoch+1,step=step,train_loss=float(np.mean(losses)),probe=probe_result)
        history.append(row);print(json.dumps(row),flush=True)
        torch.save(dict(head=head.state_dict(),width=a.width,epoch=epoch+1,step=step,optimizer=optimizer.state_dict(),matcher_sha256=p['matcher_sha256'],
            train_plan_sha256=setup['train_plan_sha256'],protocol=p,selection='last fixed budget; no validation selection'),out/'last.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))
    result=dict(initial_train_probe=initial,final_train_probe=history[-1]['probe'],history=history,
        head_parameters=sum(t.numel() for t in head.parameters()),width=a.width,small_fit=a.small_fit,
        train_pairs=len(indices),steps=step,backbone_instantiated_in_training=False,backbone_gradients=False,descriptor_gradients=False,
        validation_labels_used=False,test_used=False,ambiguity_margin_used=False,
        cache_summary_sha256=hashlib.sha256((cache/'summary.json').read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256((out/'last.pt').read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['final_train_probe']),flush=True)


if __name__=='__main__':main()
