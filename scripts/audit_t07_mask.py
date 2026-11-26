"""Independent source-GT mask audit; OS has no GT and remains visual only."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import SegformerConfig,SegformerForSemanticSegmentation,SegformerImageProcessor
from sglm.semantic_targets import teacher_logits_to_native_probabilities,aggregate_semantic_targets,semantic_supervision_fields
from sglm.experiment_io import write_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='runs/t07_teacher_TG_v1');a=ap.parse_args();r=Path(a.run);torch.set_num_threads(4)
    state=torch.load(r/'best.pt',map_location='cpu',weights_only=False);model=SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(r/'model_config',local_files_only=True));model.load_state_dict(state['model']);model=model.cuda().eval()
    processor=SegformerImageProcessor.from_pretrained(r/'model_config',local_files_only=True);cfg=json.loads(Path('configs/semantic_teacher_preflight_v1.json').read_text());rows=json.loads(Path('runs/t06_teacher_data/manifest.json').read_text())['val64'];records=[];class_rows=[]
    with torch.inference_mode():
        for row in rows:
            with Image.open(Path('runs/t06_teacher_data')/row['files']['image']['path']) as im:image=im.convert('L').convert('RGB')
            with Image.open(Path('runs/t06_teacher_data')/row['files']['label']['path']) as im:gt=np.array(im)
            raw=teacher_logits_to_native_probabilities(model(processor(images=image,return_tensors='pt')['pixel_values'].cuda()).logits,gt.shape)
            # Fixed32 coarse cells on native1024: area factor32, no change to OS factor8.
            aggregation=aggregate_semantic_targets(raw,torch.ones(1,*gt.shape,device='cuda',dtype=torch.bool),cfg['class_mapping'],factor=32,min_valid_fraction=1.,min_retained_mass=.8,min_confidence=.6)
            fields=semantic_supervision_fields(aggregation);pred=(fields['conditional_probabilities'].argmax(1)[0]+1).cpu().numpy();mask=aggregation['mask'][0].cpu().numpy();weight=fields['supervision_weight'][0].cpu().numpy()
            pred=np.repeat(np.repeat(pred,32,0),32,1);mask=np.repeat(np.repeat(mask,32,0),32,1);weight=np.repeat(np.repeat(weight,32,0),32,1)
            valid=gt>0;accepted=valid&mask;wrong=accepted&(pred!=gt)
            records.append(dict(pair_id=row['pair_id'],valid_pixels=int(valid.sum()),accepted_pixels=int(accepted.sum()),wrong_accepted_pixels=int(wrong.sum()),accepted_wrong_fraction=float(wrong.sum()/max(accepted.sum(),1)),accepted_fraction=float(accepted.sum()/valid.sum()),weighted_wrong_mass=float((weight*(pred!=gt)*valid).sum()/valid.sum())))
            for c in range(1,8):
                selected=accepted&(pred==c);class_rows.append(dict(pair_id=row['pair_id'],class_name=cfg['target_classes'][c-1],accepted_pixels=int(selected.sum()),wrong_pixels=int((selected&(gt!=c)).sum())))
    out=Path('reports/t07_teacher_v1');write_csv(out/'source_mask_pairs.csv',records)
    total=[]
    for name in cfg['target_classes']:
        group=[x for x in class_rows if x['class_name']==name];n=sum(x['accepted_pixels'] for x in group);w=sum(x['wrong_pixels'] for x in group);total.append(dict(class_name=name,accepted_pixels=n,wrong_pixels=w,accepted_coarse_semantic_precision=1-w/max(n,1)))
    write_csv(out/'source_mask_classes.csv',total)
    result=dict(step=state['step'],source_val64=True,OS_semantic_accuracy=False,definition='coarse32 majority conditional class expanded to native1024; compare all valid GT pixels in accepted cells; mixed boundaries count as errors too',accepted_fraction=sum(r['accepted_pixels'] for r in records)/sum(r['valid_pixels'] for r in records),wrong_fraction_among_accepted=sum(r['wrong_accepted_pixels'] for r in records)/sum(r['accepted_pixels'] for r in records),classes=total)
    (out/'source_mask_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
