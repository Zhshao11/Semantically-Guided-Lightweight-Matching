"""License-explicit B2 replacement; paired RGB/gray streams, no community tensors."""
import argparse,hashlib,json,os,random
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForImageClassification,SegformerForSemanticSegmentation,SegformerConfig,SegformerImageProcessor
from train_t06_teacher import Crops,evaluate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['TR','TG'],required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists() or (out/'loss.jsonl').exists():raise FileExistsError('Do not overwrite teacher')
    p=json.loads(Path('configs/t07_protocol_v1.json').read_text());cfg=p['replacement_teacher'];root=Path('runs/t06_teacher_data')
    initial=Path('runs/t07_official_init');audit=json.loads(Path('reports/t07_teacher_v1/initialization_license.json').read_text())
    for row in audit['files']:assert hashlib.sha256((initial/row['file']).read_bytes()).hexdigest()==row['sha256']
    torch.set_num_threads(4);torch.manual_seed(p['seed']);np.random.seed(p['seed']);random.seed(p['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    configuration=SegformerConfig.from_json_file('reports/t06_training_v1/initial_teacher_config.json')
    model=SegformerForSemanticSegmentation(configuration)
    source=SegformerForImageClassification.from_pretrained(initial,local_files_only=True)
    model.segformer.load_state_dict(source.segformer.state_dict(),strict=True);del source
    init_sha=hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    model=model.cuda();processor=SegformerImageProcessor.from_pretrained('runs/teacher_preflight_v1/teacher_candidate',local_files_only=True)
    model.config.save_pretrained(out/'model_config');processor.save_pretrained(out/'model_config')
    manifest=json.loads((root/'manifest.json').read_text());old=json.loads(Path('runs/loveda_val_t05_v1/manifest.json').read_text())
    rng=np.random.default_rng(p['seed']);rows=sorted(manifest['train'],key=lambda r:r['pair_id']);plan=[]
    while len(plan)<cfg['steps']*4:
        for i in rng.permutation(len(rows)):
            plan.append(dict(pair_id=rows[int(i)]['pair_id'],crop_xy=rng.integers(0,513,size=2).tolist()))
            if len(plan)==cfg['steps']*4:break
    (out/'presentations.json').write_text(json.dumps(plan,separators=(',',':')))
    ds=Crops(root,manifest,plan,processor,a.mode=='TG');loader=DataLoader(ds,batch_size=2,num_workers=2,pin_memory=True,shuffle=False,
        generator=torch.Generator().manual_seed(p['seed']),persistent_workers=True)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.01)
    stream=hashlib.sha256();counts=np.zeros(8,np.int64);step=0;total=0.;best=dict(step=0,score=-1.);history=[]
    model.train();opt.zero_grad(set_to_none=True)
    for micro,(pixels,labels,fingerprints,n) in enumerate(loader):
        for v in fingerprints:stream.update(v.encode())
        counts+=n.numpy().sum(0);pixels=pixels.cuda(non_blocking=True);labels=labels.cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(pixels).logits
        logits=F.interpolate(logits.float(),size=(512,512),mode='bilinear',align_corners=False)
        loss=F.cross_entropy(logits,labels,ignore_index=255,reduction='sum')/(labels!=255).sum().clamp_min(1)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite')
        (loss/2).backward();total+=float(loss.detach())/2
        if (micro+1)%2:continue
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);step+=1
        with (out/'loss.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=total))+'\n')
        if step%50==0:print(a.mode,step,total,flush=True)
        total=0.
        if step%cfg['validation_every']==0:
            val=evaluate(model,processor,root,manifest,old,step,out);score=val['groups']['fixed64/gray_RGB']['mean_IoU']
            if score>best['score']:
                best=dict(step=step,score=score);torch.save(dict(model=model.state_dict(),step=step,mode=a.mode,initial_state_sha256=init_sha),out/'best.pt')
            torch.save(dict(model=model.state_dict(),step=step,mode=a.mode,initial_state_sha256=init_sha),out/'last.pt')
            history.append(dict(step=step,RGB_mIoU=val['groups']['fixed64/RGB']['mean_IoU'],gray_mIoU=score,best=best.copy()));(out/'history.json').write_text(json.dumps(history,indent=2));model.train()
            print('validated',a.mode,step,score,flush=True)
        (out/'status.json').write_text(json.dumps(dict(step=step,total=cfg['steps'],mode=a.mode)))
    summary=dict(mode=a.mode,steps=step,best=best,history=history,initial_state_sha256=init_sha,initialization=audit,
        presentations_sha256=hashlib.sha256((out/'presentations.json').read_bytes()).hexdigest(),cropped_GT_stream_sha256=stream.hexdigest(),GT_counts=counts.tolist(),
        protocol=p,community_weights_used=False,teacher_weight_release=False,license='non-commercial research/evaluation only')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(dict(mode=a.mode,best=best,steps=step)),flush=True)


if __name__=='__main__':main()
