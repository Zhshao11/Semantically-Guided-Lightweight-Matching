"""Semantic head/scorer only; identical T06 reliability targets, flow and update budget."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sglm.t07_semantic import SemanticScore
from sglm.semantic_targets import weighted_semantic_kl
from train_t06_head import ranking_metrics


def main():
    out=Path('runs/t07_semantic_head_v1');out.mkdir(exist_ok=True)
    if (out/'last.pt').exists():raise FileExistsError('No overwrite')
    p=json.loads(Path('configs/t07_protocol_v1.json').read_text());setup=json.loads(Path('reports/t06_setup_v1/setup.json').read_text());cache=Path('runs/t06_reliability_v1');semantic=Path('runs/t07_semantic_cache_v1')
    rel=json.loads((cache/'summary.json').read_text());sem=json.loads((semantic/'summary.json').read_text())
    assert sem['train_only'] and sem['teacher_license_research_confirmed'] and sem['pairs']==7912
    ids=json.loads((cache/'pair_ids.json').read_text());assert ids==json.loads((semantic/'pair_ids.json').read_text()) and all('/train/' in uid for uid in ids)
    assert rel['train_plan_sha256']==sem['train_plan_sha256']==setup['train_plan_sha256']
    arrays=[np.load(cache/name,mmap_mode='r') for name in ['optical_features.npy','sar_mean.npy','geometric_correctness.npy']]+[np.load(semantic/name,mmap_mode='r') for name in ['conditional7.npy','weight.npy']]
    torch.set_num_threads(4);torch.manual_seed(p['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    head=SemanticScore().cuda();opt=torch.optim.AdamW(head.parameters(),lr=p['head']['learning_rate'],weight_decay=p['head']['weight_decay']);pos_weight=torch.tensor(rel['label_counts']['negative']/rel['label_counts']['positive'],device='cuda');probe=setup['probe_indices']
    def batch(ids):
        values=[torch.from_numpy(np.array(array[ids],copy=True)).cuda() for array in arrays]
        # Float16 disk compression renormalized to a valid detached conditional distribution.
        values[3]=values[3].float();values[3]=values[3]/values[3].sum(1,keepdim=True).clamp_min(1e-12);return values
    def losses(fields,y,target,w):
        valid=y>=0;bce=F.binary_cross_entropy_with_logits(fields['logits'].flatten(1),y.clamp_min(0).float(),pos_weight=pos_weight,reduction='none')[valid].mean()
        kl=weighted_semantic_kl(fields['semantic_logits'],target,w)
        return bce,kl,bce+p['semantic_student']['KL_weight']*kl
    @torch.no_grad()
    def evaluate():
        head.eval();scores=[];ys=[];b=[];ks=[];class_mass=np.zeros(7)
        for start in range(0,len(probe),16):
            fo,fs,y,t,w=batch(probe[start:start+16]);fields=head.fields(fo,fs);bc,kl,_=losses(fields,y,t,w);scores.append(fields['logits'].cpu().numpy().reshape(-1));ys.append(y.cpu().numpy().reshape(-1));b.append(float(bc));ks.append(float(kl));class_mass+=fields['probabilities'].sum((0,2,3)).cpu().numpy()
        return dict(BCE=float(np.mean(b)),KL=float(np.mean(ks)),student_class_fraction=(class_mass/class_mass.sum()).tolist(),**ranking_metrics(np.concatenate(scores),np.concatenate(ys)))
    initial=evaluate();history=[];step=0;stream=hashlib.sha256()
    for epoch in range(p['head']['epochs']):
        head.train();order=np.random.default_rng(p['seed']+epoch).permutation(7912);b=[];ks=[]
        for start in range(0,7912,32):
            ids=order[start:start+32].tolist();stream.update(np.asarray(ids,dtype=np.int64).tobytes());fo,fs,y,t,w=batch(ids);opt.zero_grad(set_to_none=True)
            fields=head.fields(fo,fs);bc,kl,total=losses(fields,y,t,w)
            if not torch.isfinite(total):raise FloatingPointError('Nonfinite head loss')
            total.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),p['head']['gradient_clip'],error_if_nonfinite=True);opt.step();step+=1;b.append(float(bc.detach()));ks.append(float(kl.detach()))
        row=dict(epoch=epoch+1,step=step,BCE=float(np.mean(b)),KL=float(np.mean(ks)),probe=evaluate());history.append(row);print(json.dumps(row),flush=True)
        torch.save(dict(head=head.state_dict(),step=step,epoch=epoch+1,matcher_sha256=p['matcher_sha256'],teacher_sha256=sem['teacher_checkpoint_sha256'],train_plan_sha256=setup['train_plan_sha256'],protocol=p,selection='last fixed12epochs; no validation selection'),out/'last.pt');(out/'history.json').write_text(json.dumps(history,indent=2))
    # Reconstruct A_cap's exact T06 sample stream for explicit equality.
    reference=hashlib.sha256()
    for epoch in range(12):
        order=np.random.default_rng(p['seed']+epoch).permutation(7912)
        for start in range(0,7912,32):reference.update(np.asarray(order[start:start+32],dtype=np.int64).tobytes())
    assert stream.hexdigest()==reference.hexdigest()
    summary=dict(protocol=p,steps=step,train_pairs=7912,head_parameters=sum(v.numel() for v in head.parameters()),backbone_instantiated=False,backbone_gradients=False,teacher_at_inference=False,appearance_control_sample_stream_equal=True,sample_stream_sha256=stream.hexdigest(),train_plan_sha256=setup['train_plan_sha256'],initial_train_probe=initial,final_train_probe=history[-1]['probe'],history=history,semantic_cache_summary_sha256=hashlib.sha256((semantic/'summary.json').read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256((out/'last.pt').read_bytes()).hexdigest(),selection='last',validation_training_labels=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('head complete',step,flush=True)


if __name__=='__main__':main()
