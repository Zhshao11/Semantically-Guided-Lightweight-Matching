"""Train-only OS previews with per-image blinded teacher order and retained mass."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image,ImageDraw
import torch
from transformers import SegformerForSemanticSegmentation,SegformerImageProcessor
from sglm.semantic_targets import teacher_logits_to_native_probabilities,aggregate_semantic_targets,semantic_supervision_fields


PALETTE=np.array([[0,0,0],[200,200,200],[220,60,60],[240,220,50],[40,100,230],[170,120,70],[30,150,60],[150,210,90]],np.uint8)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',default='runs/t06_OS_review_v1');a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    cfg=json.loads(Path('configs/semantic_teacher_preflight_v1.json').read_text())
    review=json.loads(Path('reports/t06_setup_v1/OS_train_review.json').read_text())['samples'];assert len(review)==32 and all(r['split']=='train' for r in review)
    weights=Path('runs/teacher_preflight_v1/teacher_candidate');model=SegformerForSemanticSegmentation.from_pretrained(weights,local_files_only=True).cuda().eval()
    processor=SegformerImageProcessor.from_pretrained(weights,local_files_only=True)
    images={};records=[];predictions={};checkpoint_ids={}
    for row in review:
        path=Path(a.root)/row['optical_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['optical_sha256']
        with Image.open(path) as im:images[row['pair_id']]=im.convert('RGB')
    with torch.inference_mode():
        for version in ['T0','TR','TG']:
            path=Path('runs')/('t06_teacher_'+version)/'best.pt';state=torch.load(path,map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'],strict=True);checkpoint_ids[version]=dict(step=state['step'],sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for row in review:
                uid=row['pair_id'];image=images[uid];pixel=processor(images=image,return_tensors='pt')['pixel_values'].cuda()
                raw=teacher_logits_to_native_probabilities(model(pixel).logits,(256,256))
                aggregate=aggregate_semantic_targets(raw,torch.ones(1,256,256,device='cuda',dtype=torch.bool),cfg['class_mapping'],
                    **{k:v for k,v in cfg['aggregation'].items() if k!='policy'})
                fields=semantic_supervision_fields(aggregate);pred=raw.argmax(1)[0].byte().cpu().numpy();confidence=raw.max(1).values[0].cpu().numpy()
                uid_safe=uid.replace('/','_');np.savez_compressed(out/(version+'_'+uid_safe+'.npz'),raw8_probabilities=raw[0].cpu().numpy().astype(np.float16),
                    **{key:value[0].cpu().numpy() for key,value in fields.items()},mask=aggregate['mask'][0].cpu().numpy())
                record=dict(version=version,pair_id=uid,split='train',semantic_GT=False,
                    predicted_fractions={cfg['source_classes'][i]:float((pred==i).mean()) for i in range(8)},
                    mean_max_probability=float(confidence.mean()),valid_class_mass_mean=float(fields['valid_class_mass'].mean()),
                    conditional_reliability_mean=float(fields['distribution_reliability'].mean()),supervision_weight_mean=float(fields['supervision_weight'].mean()),
                    supervision_mask_fraction=float(aggregate['mask'].float().mean()),
                    effective_class_mass={cfg['target_classes'][i]:float(fields['effective_class_mass'][0,i].mean()) for i in range(7)})
                record['high_confidence_all_background_flag']=record['predicted_fractions']['Background']>=.95 and record['mean_max_probability']>=.90
                records.append(record);predictions[(version,uid)]=(pred,confidence)
            print('review inferred',version,32,flush=True)
    rng=random.Random(20261126);keys={};worksheets=[]
    for start in range(0,32,8):
        sheet=Image.new('RGB',(1024,8*282),'white');heat=sheet.copy()
        for j,row in enumerate(review[start:start+8]):
            uid=row['pair_id'];order=['T0','TR','TG'];rng.shuffle(order);keys[uid]=dict(zip(['P','Q','R'],order))
            for canvas in [sheet,heat]:
                canvas.paste(images[uid],(0,j*282+26));ImageDraw.Draw(canvas).text((3,j*282+5),uid+' optical',fill='black')
            for column,(letter,version) in enumerate(zip(['P','Q','R'],order),1):
                pred,conf=predictions[(version,uid)];sheet.paste(Image.fromarray(PALETTE[pred]),(column*256,j*282+26))
                rgb=np.stack([conf*255,np.zeros_like(conf),(1-conf)*255],2).astype(np.uint8);heat.paste(Image.fromarray(rgb),(column*256,j*282+26))
                ImageDraw.Draw(sheet).text((column*256+3,j*282+5),letter+' land-cover prediction',fill='black')
                ImageDraw.Draw(heat).text((column*256+3,j*282+5),letter+' confidence: red high',fill='black')
            worksheets.append(dict(pair_id=uid,P_observation='',Q_observation='',R_observation='',note='version order differs per image; no GT accuracy scoring'))
        sheet.save(out/f'blind_predictions_page_{start//8+1}.png');heat.save(out/f'blind_confidence_page_{start//8+1}.png')
    (out/'keys_private.json').write_text(json.dumps(keys,indent=2));(out/'blind_review_worksheet.json').write_text(json.dumps(worksheets,indent=2))
    summary={}
    for version in ['T0','TR','TG']:
        rows=[r for r in records if r['version']==version]
        summary[version]=dict(high_confidence_all_background_flags=sum(r['high_confidence_all_background_flag'] for r in rows),
            count=32,semantic_accuracy=None,supervision_mask_fraction=float(np.mean([r['supervision_mask_fraction'] for r in rows])),
            supervision_weight_mean=float(np.mean([r['supervision_weight_mean'] for r in rows])),
            valid_class_mass_mean=float(np.mean([r['valid_class_mass_mean'] for r in rows])),
            effective_class_mass={name:float(np.mean([r['effective_class_mass'][name] for r in rows])) for name in cfg['target_classes']})
    (out/'summary.json').write_text(json.dumps(dict(checkpoints=checkpoint_ids,summary=summary,records=records,
        supervised_labels_generated_for_full_OS_dataset=False,SC_training_started=False,blind_version_order=True,human_review_completed=False,
        flag_definition='predicted background>=95% and mean max raw probability>=90%; output-pattern flag, not a semantic error rate',
        weighting='a_i times [1-H(p_conditional)/log7] times fixed valid/mass/confidence mask; retain Ignore mass',
        semantic_GT=False,OS_accuracy_claim=False,license='community/derived weights unresolved, kept private'),indent=2))


if __name__=='__main__':main()
