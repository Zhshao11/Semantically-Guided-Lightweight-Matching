"""Review replacement teacher on the SAME32 OS training sources; no val/test cache."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from transformers import SegformerForSemanticSegmentation,SegformerImageProcessor,SegformerConfig
from sglm.semantic_targets import teacher_logits_to_native_probabilities,aggregate_semantic_targets,semantic_supervision_fields
from review_t06_teachers import PALETTE


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    out=Path('runs/t07_teacher_review_v1');out.mkdir(exist_ok=True);torch.set_num_threads(4)
    cfg=json.loads(Path('configs/semantic_teacher_preflight_v1.json').read_text());rows=json.loads(Path('reports/t06_setup_v1/OS_train_review.json').read_text())['samples']
    r=Path('runs/t07_teacher_TG_v1');state=torch.load(r/'best.pt',map_location='cpu',weights_only=False)
    model=SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(r/'model_config',local_files_only=True));model.load_state_dict(state['model'],strict=True);model=model.cuda().eval()
    processor=SegformerImageProcessor.from_pretrained(r/'model_config',local_files_only=True);records=[]
    with torch.inference_mode():
        for start in range(0,32,8):
            sheet=Image.new('RGB',(1024,8*282),'white')
            for j,row in enumerate(rows[start:start+8]):
                uid=row['pair_id'];assert '/train/' in uid
                path=Path(a.root)/row['optical_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['optical_sha256']
                with Image.open(path) as im:image=im.convert('L').convert('RGB')
                pixel=processor(images=image,return_tensors='pt')['pixel_values'].cuda()
                raw=teacher_logits_to_native_probabilities(model(pixel).logits,(256,256));agg=aggregate_semantic_targets(raw,torch.ones(1,256,256,device='cuda',dtype=torch.bool),cfg['class_mapping'],**{k:v for k,v in cfg['aggregation'].items() if k!='policy'});f=semantic_supervision_fields(agg)
                pred=raw.argmax(1)[0].cpu().numpy();conf=raw.max(1).values[0].cpu().numpy();mask=agg['mask'][0].cpu().numpy();weight=f['supervision_weight'][0].cpu().numpy();conditional=f['conditional_probabilities'][0].cpu().numpy()
                rec=dict(pair_id=uid,split='train',predicted_fractions={name:float((pred==i).mean()) for i,name in enumerate(cfg['source_classes'])},mean_max_probability=float(conf.mean()),mask_fraction=float(mask.mean()),weight_mean=float(weight.mean()),class_mask_mass={name:float((weight*conditional[i]).mean()) for i,name in enumerate(cfg['target_classes'])})
                rec['collapse_flag']=rec['predicted_fractions']['Background']>=.95 and rec['mean_max_probability']>=.90;records.append(rec)
                np.savez_compressed(out/(uid.replace('/','_')+'.npz'),raw8=raw[0].cpu().numpy().astype(np.float16),conditional7=conditional,mask=mask,weight=weight)
                mask256=np.repeat(np.repeat(mask,8,0),8,1);weighted=np.repeat(np.repeat(weight,8,0),8,1)
                masked=PALETTE[pred].copy();masked[~mask256]=[240,0,240]
                heat=np.stack([weighted*255,np.zeros_like(weighted),(1-weighted)*255],2).astype(np.uint8)
                for col,(label,pic) in enumerate([('optical',image),('TG official init',Image.fromarray(PALETTE[pred])),('accepted mask / magenta ignore',Image.fromarray(masked)),('weight red high',Image.fromarray(heat))]):
                    sheet.paste(pic,(col*256,j*282+26));ImageDraw.Draw(sheet).text((col*256+3,j*282+5),uid.rsplit('/',1)[-1]+' '+label,fill='black')
            sheet.save(out/f'review_page_{start//8+1}.png')
    result=dict(checkpoint_sha256=hashlib.sha256((r/'best.pt').read_bytes()).hexdigest(),step=state['step'],records=records,OS_semantic_GT=False,OS_accuracy=None,all_background_flags=sum(x['collapse_flag'] for x in records),visual_review='pending; inspect mask overlays before freezing',test_used=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(step=state['step'],all_background_flags=result['all_background_flags'],mean_mask=float(np.mean([r['mask_fraction'] for r in records])))))


if __name__=='__main__':main()
