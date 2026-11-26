"""Offline teacher on TRAIN optical only; stopped conditional targets and absolute mass."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import SegformerConfig,SegformerForSemanticSegmentation,SegformerImageProcessor
from sglm.t07_semantic import abstain_semantic_weight
from sglm.semantic_targets import teacher_logits_to_native_probabilities,aggregate_semantic_targets,semantic_supervision_fields


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    frozen=json.loads(Path('reports/t07_teacher_v1/teacher_freeze.json').read_text());assert frozen['technical_decision']=='freeze_for_research_with_explicit_abstention' and frozen['license_research_use_confirmed']
    out=Path('runs/t07_semantic_cache_v1');out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Completed cache cannot be overwritten')
    planpath=Path('reports/t06_setup_v1/training_plan.json');plan=json.loads(planpath.read_text())['samples'];assert len(plan)==7912 and all('/train/' in r['pair_id'] and r['split']=='train' for r in plan)
    assert hashlib.sha256(planpath.read_bytes()).hexdigest()==json.loads(Path('reports/t06_setup_v1/setup.json').read_text())['train_plan_sha256']
    cp=Path(frozen['checkpoint_path']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==frozen['checkpoint_sha256'];run=cp.parent
    model=SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(run/'model_config',local_files_only=True));model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'],strict=True);model=model.cuda().eval().requires_grad_(False)
    processor=SegformerImageProcessor.from_pretrained(run/'model_config',local_files_only=True);cfg=json.loads(Path('configs/semantic_teacher_preflight_v1.json').read_text());torch.set_num_threads(4)
    targets=np.lib.format.open_memmap(out/'conditional7.npy',mode='w+',dtype=np.float16,shape=(7912,7,32,32));weights=np.lib.format.open_memmap(out/'weight.npy',mode='w+',dtype=np.float32,shape=(7912,32,32));retained=np.lib.format.open_memmap(out/'retained_mass.npy',mode='w+',dtype=np.float32,shape=(7912,32,32))
    records=[];mass=np.zeros(7);quarantine=set(frozen['abstention']['image_ids']);total_masks=0;total_before_masks=0;raw_counts=np.zeros(8,dtype=np.int64)
    with torch.inference_mode():
        for start in range(0,len(plan),8):
            batch=plan[start:start+8];images=[]
            for row in batch:
                path=Path(a.root)/row['optical_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['optical_sha256']
                with Image.open(path) as im:images.append(im.convert('L').convert('RGB'))
            pixel=processor(images=images,return_tensors='pt')['pixel_values'].cuda();raw=teacher_logits_to_native_probabilities(model(pixel).logits,(256,256));agg=aggregate_semantic_targets(raw,torch.ones(len(batch),256,256,device='cuda',dtype=torch.bool),cfg['class_mapping'],**{k:v for k,v in cfg['aggregation'].items() if k!='policy'});fields=semantic_supervision_fields(agg)
            pred=raw.argmax(1);raw_counts+=torch.bincount(pred.flatten(),minlength=8).cpu().numpy();confidence=raw.max(1).values.mean((1,2));background=(pred==1).float().mean((1,2));flag=(background>=.95)&(confidence>=.9)
            disabled=[cfg['target_classes'].index(name) for name in frozen['abstention']['disabled_dominant_classes']]
            class_abstained=abstain_semantic_weight(fields['conditional_probabilities'],fields['supervision_weight'],disabled,torch.zeros(len(batch),dtype=torch.bool,device='cuda'))
            w=class_abstained.cpu().numpy();p=fields['conditional_probabilities'].cpu().numpy();total_before_masks+=int(agg['mask'].sum())
            for j,row in enumerate(batch):
                reason='fixed32 visual audit ambiguity' if row['pair_id'] in quarantine else 'same T06 high-confidence all-background flag' if flag[j] else None
                if reason:w[j]=0
                records.append(dict(pair_id=row['pair_id'],quarantined=reason is not None,reason=reason,raw_background_fraction=float(background[j]),raw_confidence_mean=float(confidence[j]),weight_mean=float(w[j].mean())))
            stop=start+len(batch);targets[start:stop]=p;weights[start:stop]=w;retained[start:stop]=fields['valid_class_mass'].cpu().numpy();mass+=(p*w[:,None]).sum((0,2,3));total_masks+=int((w>0).sum())
            if stop%200==0:print('cached',stop,flush=True)
    for array in [targets,weights,retained]:array.flush()
    result=dict(pairs=7912,train_only=True,teacher_checkpoint_sha256=frozen['checkpoint_sha256'],teacher_license_research_confirmed=True,train_plan_sha256=hashlib.sha256(planpath.read_bytes()).hexdigest(),class_effective_mass=(mass/(7912*1024)).tolist(),raw_class_fractions=(raw_counts/raw_counts.sum()).tolist(),positive_weight_fraction=total_masks/(7912*1024),pre_abstention_mask_fraction=total_before_masks/(7912*1024),quarantined_images=sum(r['quarantined'] for r in records),records=records,teacher_used_at_inference=False,validation_or_test_teacher_cache=False)
    (out/'pair_ids.json').write_text(json.dumps([r['pair_id'] for r in plan]));(out/'summary.json').write_text(json.dumps(result,indent=2));(out/'status.json').write_text(json.dumps(dict(phase='completed',pairs=7912)));print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
