"""Controlled T0/TR/TG; full LoveDA training split, shared saved crops and budget."""
import argparse
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import random
import numpy as np
from PIL import Image,ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from transformers import SegformerForSemanticSegmentation,SegformerImageProcessor
from sglm.semantic_evaluation import loveda_confusion,loveda_metrics


class Crops(Dataset):
    def __init__(self,root,manifest,plan,processor,gray):
        self.root=Path(root);self.rows={r['pair_id']:r for r in manifest['train']};self.plan=plan;self.processor=processor;self.gray=gray
    def __len__(self):return len(self.plan)
    def __getitem__(self,index):
        p=self.plan[index];r=self.rows[p['pair_id']];x,y=p['crop_xy'];box=(x,y,x+512,y+512)
        with Image.open(self.root/r['files']['image']['path']) as im:image=im.convert('RGB').crop(box)
        with Image.open(self.root/r['files']['label']['path']) as im:label=np.array(im.crop(box),dtype=np.int64)
        if not np.isin(label,np.arange(8)).all():raise ValueError('Unexpected LoveDA IDs')
        counts=np.bincount(label.ravel(),minlength=8)
        if self.gray:image=ImageOps.grayscale(image).convert('RGB')
        pixel=self.processor(images=image,return_tensors='pt')['pixel_values'][0]
        label[label==0]=255
        fingerprint=hashlib.sha256((p['pair_id']+str(p['crop_xy'])).encode()+label.tobytes()).hexdigest()
        return pixel,torch.from_numpy(label),fingerprint,torch.from_numpy(counts)


@torch.inference_mode()
def evaluate(model,processor,root,manifest,old8,step,out):
    model.eval();groups={};records=[]
    for group,rows,source_root in [('fixed64',manifest['val64'],Path(root)),('old8',old8['samples'],Path('runs/loveda_val_t05_v1'))]:
        for mode in ['RGB','gray_RGB']:
            confusion=np.zeros((8,8),np.int64);per_domain={}
            for r in rows:
                with Image.open(source_root/r['files']['image']['path']) as im:image=im.convert('RGB')
                with Image.open(source_root/r['files']['label']['path']) as im:gt=np.array(im)
                if mode=='gray_RGB':image=ImageOps.grayscale(image).convert('RGB')
                tensor=processor(images=image,return_tensors='pt')['pixel_values'].cuda()
                logits=model(tensor).logits.float();native=F.interpolate(logits,size=gt.shape,mode='bilinear',align_corners=False)
                pred=native.softmax(1).argmax(1)[0].byte().cpu().numpy()
                c=loveda_confusion(gt,pred);confusion+=c;per_domain.setdefault(r['domain'],np.zeros((8,8),np.int64));per_domain[r['domain']]+=c
                records.append(dict(group=group,mode=mode,pair_id=r['pair_id'],metrics=loveda_metrics(c)))
            groups[group+'/'+mode]=dict(**loveda_metrics(confusion),domain_metrics={name:loveda_metrics(c) for name,c in per_domain.items()})
    result=dict(step=step,groups=groups,pairs=records,model_eval=True,GT_zero_ignored=True,predicted_zero_penalized=True)
    (out/f'validation_{step:04d}.json').write_text(json.dumps(result,indent=2));return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['T0','TR','TG'],required=True)
    ap.add_argument('--data',default='runs/t06_teacher_data');ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Completed teacher run must not be overwritten')
    p=json.loads(Path('configs/t06_protocol_v1.json').read_text());cfg=p['teacher'];root=Path(a.data)
    manifest=json.loads((root/'manifest.json').read_text());assert len(manifest['train'])==2522 and len(manifest['val64'])==64
    assert not ({r['pair_id'] for r in manifest['val64']}&{r['pair_id'] for r in json.loads(Path('runs/loveda_val_t05_v1/manifest.json').read_text())['samples']})
    weights=Path('runs/teacher_preflight_v1/teacher_candidate');teacher_cfg=json.loads(Path('configs/semantic_teacher_preflight_v1.json').read_text())
    assert hashlib.sha256((weights/'pytorch_model.bin').read_bytes()).hexdigest()==teacher_cfg['weights_sha256']
    torch.set_num_threads(4);torch.manual_seed(p['seed']);np.random.seed(p['seed']);random.seed(p['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    model=SegformerForSemanticSegmentation.from_pretrained(weights,local_files_only=True).cuda()
    processor=SegformerImageProcessor.from_pretrained(weights,local_files_only=True)
    assert model.config.num_labels==8 and model.config.id2label[0]=='Ignore' and not processor.do_reduce_labels
    assert processor.size==dict(height=512,width=512) and int(processor.resample)==2 and processor.rescale_factor==1/255
    old8=json.loads(Path('runs/loveda_val_t05_v1/manifest.json').read_text())
    initial=evaluate(model,processor,root,manifest,old8,0,out)
    (out/'config.json').write_text(json.dumps(p,indent=2));history=[]
    initial_score=initial['groups']['fixed64/gray_RGB']['mean_IoU'];best=dict(step=0,score=initial_score)
    torch.save(dict(model=model.state_dict(),step=0,mode=a.mode,initial_weight_sha256=teacher_cfg['weights_sha256']),out/'best.pt')
    if a.mode=='T0':
        (out/'summary.json').write_text(json.dumps(dict(mode='T0',steps=0,best=best,last_validation=initial,initial_weight_sha256=teacher_cfg['weights_sha256'],license='unresolved; no weight release'),indent=2));return
    # Identical deterministic stream in both independent processes, pre-saved before updates.
    rng=np.random.default_rng(p['seed']);train=sorted(manifest['train'],key=lambda r:r['pair_id']);presentations=[]
    while len(presentations)<cfg['steps']*cfg['effective_batch']:
        for idx in rng.permutation(len(train)):
            r=train[int(idx)];xy=rng.integers(0,513,size=2).tolist()
            presentations.append(dict(pair_id=r['pair_id'],crop_xy=xy))
            if len(presentations)==cfg['steps']*cfg['effective_batch']:break
    (out/'presentations.json').write_text(json.dumps(presentations,separators=(',',':')))
    plan_sha=hashlib.sha256((out/'presentations.json').read_bytes()).hexdigest()
    ds=Crops(root,manifest,presentations,processor,a.mode=='TG')
    loader=DataLoader(ds,batch_size=cfg['micro_batch'],shuffle=False,num_workers=2,pin_memory=True,
        generator=torch.Generator().manual_seed(p['seed']),persistent_workers=True)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    stream=hashlib.sha256();gt_counts=np.zeros(8,np.int64);step=0;loss_sum=0.;opt.zero_grad(set_to_none=True);model.train()
    for micro,(pixels,labels,fingerprints,counts) in enumerate(loader):
        for fingerprint in fingerprints:stream.update(fingerprint.encode())
        gt_counts+=counts.numpy().sum(0);pixels=pixels.cuda(non_blocking=True);labels=labels.cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(pixels).logits
        logits=F.interpolate(logits.float(),size=(512,512),mode='bilinear',align_corners=False)
        valid=labels!=255
        loss=F.cross_entropy(logits,labels,ignore_index=255,reduction='sum')/valid.sum().clamp_min(1)
        if not torch.isfinite(loss):raise FloatingPointError('Teacher nonfinite loss')
        (loss/cfg['accumulation']).backward();loss_sum+=float(loss.detach())/cfg['accumulation']
        if (micro+1)%cfg['accumulation']:continue
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_norm'],error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);step+=1
        if step%20==0:
            (out/'status.json').write_text(json.dumps(dict(mode=a.mode,step=step,total=cfg['steps'],loss=loss_sum)))
            print('training',a.mode,step,loss_sum,flush=True)
        with (out/'loss.jsonl').open('a') as stream_file:stream_file.write(json.dumps(dict(step=step,loss=loss_sum))+'\n')
        loss_sum=0.
        if step in cfg['validation_steps']:
            result=evaluate(model,processor,root,manifest,old8,step,out);score=result['groups']['fixed64/gray_RGB']['mean_IoU']
            if score>best['score']:
                best=dict(step=step,score=score);torch.save(dict(model=model.state_dict(),step=step,mode=a.mode,initial_weight_sha256=teacher_cfg['weights_sha256']),out/'best.pt')
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,mode=a.mode,initial_weight_sha256=teacher_cfg['weights_sha256']),out/'last.pt')
            history.append(dict(step=step,RGB_mIoU=result['groups']['fixed64/RGB']['mean_IoU'],gray_mIoU=score,best=best.copy()))
            (out/'history.json').write_text(json.dumps(history,indent=2));model.train();print('validated',a.mode,step,score,flush=True)
    assert step==cfg['steps']
    result=dict(mode=a.mode,steps=step,best=best,history=history,last_validation_step=step,
        initial_weight_sha256=teacher_cfg['weights_sha256'],presentations_sha256=plan_sha,cropped_GT_stream_sha256=stream.hexdigest(),
        GT_counts_in_training=gt_counts.tolist(),effective_batch=cfg['effective_batch'],training_images=2522,validation_images=64,longitudinal_images=8,
        only_between_run_change='TR RGB vsTG PIL L repeatedRGB; same sample/crop/optimizer schedule',
        class_zero_used_as_land_cover=False,GT0_loss_masked=True,license='unresolved; original and derived weights remain private',mixed_RGB_gray_branch=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'status.json').write_text(json.dumps(dict(mode=a.mode,phase='completed',steps=step)))


if __name__=='__main__':main()
