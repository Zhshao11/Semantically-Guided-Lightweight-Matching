"""Exploratory train-only teacher preview; never generates matchability labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from sglm.semantic_targets import aggregate_semantic_targets, teacher_logits_to_native_probabilities


PALETTE=np.array([[0,0,0],[200,200,200],[220,60,60],[240,220,50],
                  [40,100,230],[170,120,70],[30,150,60],[150,210,90]],np.uint8)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True);ap.add_argument('--weights',required=True)
    ap.add_argument('--plan',default='reports/t02_b0_full_v1/fixed_plan.json')
    ap.add_argument('--config',default='configs/semantic_teacher_preflight_v1.json')
    ap.add_argument('--output',required=True);ap.add_argument('--device',default='cpu')
    a=ap.parse_args();torch.set_num_threads(4)
    cfg=json.loads(Path(a.config).read_text());weights=Path(a.weights)
    digest=hashlib.sha256((weights/'pytorch_model.bin').read_bytes()).hexdigest()
    if digest!=cfg['weights_sha256']:raise ValueError('Unverified teacher weights')
    model=SegformerForSemanticSegmentation.from_pretrained(weights,local_files_only=True).to(a.device).eval()
    if model.config.hidden_sizes!=cfg['expected_hidden_sizes'] or model.config.depths!=cfg['expected_depths']:
        raise ValueError('Not expected B2 architecture')
    names=[model.config.id2label[i] for i in range(model.config.num_labels)]
    if [s.lower() for s in names]!=[s.lower() for s in cfg['source_classes']]:raise ValueError('Class order mismatch')
    processor=SegformerImageProcessor.from_pretrained(weights,local_files_only=True)
    pp=cfg['preprocessing']
    assert processor.do_resize and processor.size==dict(height=512,width=512) and int(processor.resample)==2
    assert processor.do_rescale and abs(processor.rescale_factor-pp['rescale'])<1e-12
    assert processor.do_normalize and processor.image_mean==pp['mean'] and processor.image_std==pp['std']
    assert processor.do_reduce_labels==pp['do_reduce_labels']
    plan=json.loads(Path(a.plan).read_text());samples=[s for s in plan['samples'] if s['split']=='train'][:cfg['sample_count']]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);records=[];panels=[]
    before={k:v.clone() for k,v in model.named_buffers()}
    with torch.inference_mode():
        for sample in samples:
            path=Path(a.root)/sample['optical_path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=sample['optical_sha256']:raise ValueError('Image hash mismatch')
            with Image.open(path) as original:
                if original.mode!='L' or original.size!=(256,256):raise ValueError('Expected audited grayscale uint8 train optical')
                rgb=original.convert('RGB')
            pixel=processor(images=rgb,return_tensors='pt')['pixel_values']
            # Independently verify one rescale, RGB order, resize and normalization.
            manual=np.asarray(rgb.resize((512,512),Image.Resampling.BILINEAR)).astype(np.float32)/255
            manual=(manual-np.array(pp['mean'],np.float32))/np.array(pp['std'],np.float32)
            assert np.allclose(pixel.numpy()[0],manual.transpose(2,0,1),atol=2e-6)
            p=teacher_logits_to_native_probabilities(model(pixel.to(a.device)).logits,(256,256))
            aggregate=aggregate_semantic_targets(p,torch.ones((1,256,256),device=a.device,dtype=torch.bool),
                cfg['class_mapping'],**{k:v for k,v in cfg['aggregation'].items() if k!='policy'})
            raw=p[0].cpu().numpy();grouped=aggregate['probabilities'][0].cpu().numpy()
            mask=aggregate['mask'][0].cpu().numpy();hard=raw.argmax(0)
            basename=sample['pair_id'].replace('/','_')
            np.savez_compressed(out/(basename+'.npz'),raw8_native_probabilities=raw,
                **{k:v[0].cpu().numpy() for k,v in aggregate.items()})
            record=dict(pair_id=sample['pair_id'],split='train',optical_path=sample['optical_path'],
                optical_sha256=sample['optical_sha256'],raw_argmax_fraction={names[i]:float((hard==i).mean()) for i in range(8)},
                raw_mean_probability={names[i]:float(raw[i].mean()) for i in range(8)},
                native_max_probability_mean=float(raw.max(0).mean()),
                coarse_retained_mass_mean=float(aggregate['retained_mass'].mean()),
                coarse_conditional_confidence_mean=float(aggregate['confidence'].mean()),mask_fraction=float(mask.mean()))
            records.append(record)
            panel=Image.new('RGB',(256*4,282),'white');draw=ImageDraw.Draw(panel)
            images=[rgb,Image.fromarray(PALETTE[hard]),
                Image.fromarray(PALETTE[1:][grouped.argmax(0)]).resize((256,256),Image.Resampling.NEAREST),
                Image.fromarray(np.repeat((mask*255).astype(np.uint8)[...,None],3,2)).resize((256,256),Image.Resampling.NEAREST)]
            for j,(im,title) in enumerate(zip(images,['train optical','raw8 (black=Ignore)','conditional7 (inspect mask)','mask (white=valid)'])):
                panel.paste(im,(j*256,26));draw.text((j*256+3,5),title,fill='black')
            panel.save(out/(basename+'.png'));panels.append(panel)
    assert all(not module.training for module in model.modules())
    assert all(torch.equal(v,dict(model.named_buffers())[k]) for k,v in before.items())
    sheet=Image.new('RGB',(1024,282*len(panels)),'white')
    for i,panel in enumerate(panels):sheet.paste(panel,(0,i*282))
    sheet.save(out/'teacher_preview_sheet.png')
    report=dict(protocol=cfg['protocol'],model_id=cfg['model_id'],revision=cfg['revision'],weights_sha256=digest,
        torch=torch.__version__,transformers=__import__('transformers').__version__,device=a.device,
        source_classes=names,config=cfg,eval=True,buffers_unchanged=True,preprocessing_independently_verified=True,
        semantic_GT_available=False,accuracy_claim=False,production_teacher_ready=False,
        preview_only=True,matchability_labels_generated=False,samples=records)
    (out/'teacher_preview.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('config','samples')},indent=2))


if __name__=='__main__':main()
