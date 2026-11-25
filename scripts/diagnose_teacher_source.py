"""Same fixed B2 weights/processor: paired RGB vs gray LoveDA Val, OS train only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import torch
from transformers import SegformerForSemanticSegmentation,SegformerImageProcessor
from sglm.semantic_targets import teacher_logits_to_native_probabilities
from sglm.semantic_evaluation import loveda_confusion,loveda_metrics


COLORS=np.array([[0,0,0],[200,200,200],[220,60,60],[240,220,50],[40,100,230],[170,120,70],[30,150,60],[150,210,90]],np.uint8)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',required=True);ap.add_argument('--os-root',required=True)
    ap.add_argument('--weights',default='runs/teacher_preflight_v1/teacher_candidate');ap.add_argument('--output',required=True)
    a=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(20261126)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path('configs/semantic_teacher_preflight_v1.json').read_text());weight=Path(a.weights)
    sha=hashlib.sha256((weight/'pytorch_model.bin').read_bytes()).hexdigest();assert sha==cfg['weights_sha256']
    model=SegformerForSemanticSegmentation.from_pretrained(weight,local_files_only=True).cuda().eval()
    processor=SegformerImageProcessor.from_pretrained(weight,local_files_only=True)
    assert [model.config.id2label[i] for i in range(8)]==cfg['source_classes']
    assert processor.size==dict(height=512,width=512) and not processor.do_reduce_labels
    assert processor.rescale_factor==1/255 and processor.image_mean==cfg['preprocessing']['mean'] and processor.image_std==cfg['preprocessing']['std']
    assert model.config.hidden_sizes==cfg['expected_hidden_sizes'] and model.config.depths==cfg['expected_depths']
    buffers={k:v.clone() for k,v in model.named_buffers()}
    source_root=Path(a.source_root);manifest=json.loads((source_root/'manifest.json').read_text())
    totals={mode:np.zeros((8,8),np.int64) for mode in ('RGB','gray_RGB')};domain_totals={}
    rows=[];panels=[];label_values=set()
    def infer(image):
        tensor=processor(images=image,return_tensors='pt')['pixel_values'].cuda()
        logits=model(tensor).logits
        p=teacher_logits_to_native_probabilities(logits,(image.height,image.width))[0].cpu().numpy()
        return p.argmax(0).astype(np.uint8),dict(raw_prediction_fraction={cfg['source_classes'][i]:float((p.argmax(0)==i).mean()) for i in range(8)},
            raw_mean_probability={cfg['source_classes'][i]:float(p[i].mean()) for i in range(8)},max_probability_mean=float(p.max(0).mean()))
    with torch.inference_mode():
        for sample in manifest['samples']:
            files={}
            for key in ('image','label'):
                info=sample['files'][key];path=source_root/info['path']
                assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256'];files[key]=path
            with Image.open(files['image']) as image:rgb=image.convert('RGB')
            gray=ImageOps.grayscale(rgb).convert('RGB')
            with Image.open(files['label']) as label:gt=np.asarray(label)
            if gt.ndim!=2 or gt.shape!=(rgb.height,rgb.width):raise ValueError('Label dimensions/channel mismatch')
            label_values.update(np.unique(gt).tolist());preds={}
            for mode,image in [('RGB',rgb),('gray_RGB',gray)]:
                pred,distribution=infer(image);confusion=loveda_confusion(gt,pred);totals[mode]+=confusion
                key=sample['domain']+'/'+mode;domain_totals.setdefault(key,np.zeros((8,8),np.int64));domain_totals[key]+=confusion
                record=dict(pair_id=sample['pair_id'],domain=sample['domain'],mode=mode,
                    native_hw=list(gt.shape),GT_ignored_fraction=float((gt==0).mean()),GT_value_histogram={str(i):int((gt==i).sum()) for i in range(8)},
                    metrics=loveda_metrics(confusion,cfg['source_classes']),**distribution)
                rows.append(record);preds[mode]=pred
                Image.fromarray(pred).save(out/(sample['pair_id'].replace('/','_')+'_'+mode+'_pred.png'))
            panel=Image.new('RGB',(1280,282),'white');draw=ImageDraw.Draw(panel)
            for j,(image,title) in enumerate([(rgb,'LoveDA RGB'),(gray,'same image gray RGB'),(Image.fromarray(COLORS[gt]),'GT0 black: ignored'),(Image.fromarray(COLORS[preds['RGB']]),'RGB prediction'),(Image.fromarray(COLORS[preds['gray_RGB']]),'gray prediction')]):
                panel.paste(image.resize((256,256),Image.Resampling.NEAREST if j>=2 else Image.Resampling.BILINEAR),(j*256,26));draw.text((j*256+3,5),sample['pair_id']+' '+title,fill='black')
            panels.append(panel);print('source evaluated',sample['pair_id'],flush=True)
        os_rows=[]
        for sample in [r for r in json.loads(Path('reports/t02_b0_full_v1/fixed_plan.json').read_text())['samples'] if r['split']=='train'][:8]:
            path=Path(a.os_root)/sample['optical_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==sample['optical_sha256']
            with Image.open(path) as image:rgb=image.convert('RGB')
            _,distribution=infer(rgb);os_rows.append(dict(pair_id=sample['pair_id'],split='train',semantic_GT=False,**distribution))
    assert all(not m.training for m in model.modules())
    assert all(torch.equal(v,dict(model.named_buffers())[k]) for k,v in buffers.items())
    sheet=Image.new('RGB',(1280,282*len(panels)),'white')
    for i,panel in enumerate(panels):sheet.paste(panel,(0,i*282))
    sheet.save(out/'source_RGB_gray_review.png')
    result=dict(protocol='T05-C-fixed-source-domain-control-v1',weights_sha256=sha,teacher_revision=cfg['revision'],
        source_manifest_sha256=hashlib.sha256((source_root/'manifest.json').read_bytes()).hexdigest(),source_manifest=manifest,
        class_mapping_verified=dict(native_GT_values=sorted(label_values),GT_zero_ignored=True,GT1_to_7_match_teacher1_to_7=True,
            predicted_zero_penalized=True,no_reduce_labels=True,teacher_id2label=cfg['source_classes']),
        input_policy='identical saved512 bilinear processor for RGB and PIL L->RGB; logits restored to native size, GT never resized',
        native_source='LoveDA1024; OS256. Same512 network input does not imply equal physical ground resolution',
        precision='FP32, eval, inference_mode',torch=torch.__version__,transformers=__import__('transformers').__version__,
        summary={mode:loveda_metrics(c,cfg['source_classes']) for mode,c in totals.items()},
        domain_summary={key:loveda_metrics(c,cfg['source_classes']) for key,c in domain_totals.items()},
        pairs=rows,OS_train_diagnostic=os_rows,OS_semantic_accuracy_claim=False,BN_or_model_buffers_changed=False,
        teacher_training_performed=False,community_checkpoint_license='unresolved; not free redistribution/deployment permission')
    (out/'teacher_source_control.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
