"""Same-cohort initialized/trained coarse matches and predicted affine registration.

All 48 frozen cases are included. GT is used only after prediction for diagnostic
line colors and external probe errors; it never provides a registration matrix.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from sglm.t02.data import AffinePairs256
from sglm.t02.model import B0Full
from sglm.geometry import invert_affine,transform_points,warp_image
from sglm.evaluation import estimate_from_correspondences,fixed_synthetic_probes,evaluate_transform


def font(size=13):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',):
        if Path(path).exists():
            return ImageFont.truetype(path,size)
    return ImageFont.load_default()


FONT=font()
SMALL=font(11)


def coarse_panel(optical,sar,source,target,A,stage):
    image=Image.new('RGB',(512,302),'#ffffff')
    image.paste(Image.fromarray(optical),(0,46))
    image.paste(Image.fromarray(sar).convert('RGB'),(256,46))
    draw=ImageDraw.Draw(image)
    draw.text((7,3),f'{stage}: predicted coarse matches = {len(source)}',fill='#111111',font=FONT)
    draw.text((7,22),'Shown <=64, uniform source order; green <5 / amber <10 / red >=10 px',fill='#333333',font=SMALL)
    if len(source):
        indices=np.linspace(0,len(source)-1,min(64,len(source)),dtype=int)
        # Diagnostic only. No selection, model decision or geometric fit uses these GT errors.
        errors=np.linalg.norm(target-transform_points(source,A),axis=1)
        for i in indices:
            color='#24e67f' if errors[i]<5 else ('#ffc94d' if errors[i]<10 else '#ff5b65')
            sx,sy=source[i]+[0,46]
            tx,ty=target[i]+[256,46]
            draw.line((sx,sy,tx,ty),fill=color,width=1)
            draw.ellipse((sx-1.5,sy-1.5,sx+1.5,sy+1.5),fill=color)
            draw.ellipse((tx-1.5,ty-1.5,tx+1.5,ty+1.5),fill=color)
    else:
        draw.rectangle((70,153,443,196),fill='#111111')
        draw.text((81,166),'No accepted matches at the fixed threshold',fill='white',font=FONT)
    return image


def registration_panel(optical,sar,valid_mask,estimate,evaluation,stage):
    image=Image.new('RGB',(512,302),'white')
    draw=ImageDraw.Draw(image)
    err=evaluation['rmse_px']
    err_text='undefined' if err is None else f'{err:.2f} px'
    good=evaluation['success']['5']
    draw.text((7,3),f'{stage}: probe RMSE {err_text} | '+('PASS <5px' if good else 'FAIL <5px'),
              fill='#16713f' if good else '#b32633',font=FONT)
    draw.text((7,22),'Left: optical | Right: optical / predicted-registered SAR checkerboard',fill='#333333',font=SMALL)
    image.paste(Image.fromarray(optical),(0,46))
    if estimate['matrix'] is None:
        blank=Image.new('RGB',(256,256),'#282b30')
        bd=ImageDraw.Draw(blank)
        bd.line((18,18,238,238),fill='#a6434d',width=3)
        bd.line((238,18,18,238),fill='#a6434d',width=3)
        bd.text((20,100),'NO PREDICTED TRANSFORM',fill='white',font=SMALL)
        bd.text((20,124),estimate['status'],fill='#ffc0c6',font=SMALL)
        image.paste(blank,(256,46))
        return image,None
    # estimate maps optical -> augmented SAR. Its inverse alone registers SAR to optical.
    inverse=invert_affine(estimate['matrix'])
    registered,support=warp_image(sar,inverse)
    warped_valid,_=warp_image(valid_mask.astype(np.float32),inverse)
    support&=warped_valid>.999
    gray=np.repeat(registered[...,None],3,-1)
    y,x=np.mgrid[:256,:256]
    use_sar=(x//16+y//16)%2==0
    checker=optical.copy()
    checker[use_sar]=gray[use_sar]
    checker[~support]=[150,28,43]
    image.paste(Image.fromarray(checker),(256,46))
    registered_rgb=gray.copy();registered_rgb[~support]=[150,28,43]
    return image,registered_rgb


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--plan',required=True)
    ap.add_argument('--initial-checkpoint',required=True)
    ap.add_argument('--trained-checkpoint',required=True)
    ap.add_argument('--evaluation-directory',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--device',default='cuda')
    a=ap.parse_args()
    torch.set_num_threads(4)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    models={};checks={}
    plan_hash=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest()
    for stage,path in [('initial',a.initial_checkpoint),('trained',a.trained_checkpoint)]:
        ckpt=torch.load(path,map_location=a.device,weights_only=False)
        if ckpt['plan_sha256']!=plan_hash:
            raise ValueError('Checkpoint/plan mismatch')
        model=B0Full(ckpt['config']).to(a.device).eval();model.load_state_dict(ckpt['model'])
        models[stage]=model;checks[stage]=dict(step=ckpt['step'],sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
        config=ckpt['config']
    cases=[]
    for split in ('train','val'):
        dataset=AffinePairs256(a.root,a.plan,split)
        expected={}
        for stage,name in [('initial',f'initial_{split}.json'),('trained',f'final_{split}.json')]:
            expected[stage]={p['pair_id']:p for p in json.loads((Path(a.evaluation_directory)/name).read_text())['results']}
        for i,item in enumerate(dataset):
            predictions={}
            images=[item[k][None].to(a.device) for k in ('optical','sar_aug','optical_valid_mask','sar_aug_valid_mask')]
            optical=(item['optical'].permute(1,2,0).numpy()*255).round().astype(np.uint8)
            sar=(item['sar_aug'][0].numpy()*255).round().astype(np.uint8)
            mask=item['sar_aug_valid_mask'].numpy()
            A=item['A_forward'].numpy()
            panels={};registered_final=None
            for stage,model in models.items():
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    predicted=model.predict(*images)[0]
                source,target,coarse,scores=[predicted[k].float().cpu().numpy() for k in ('source','target','coarse_target','score')]
                estimate=estimate_from_correspondences(source,target,scores,config['evaluation'],seed=config['seed'])
                probes,reference=fixed_synthetic_probes(A,grid=config['evaluation']['probe_grid'],
                    optical_mask=item['optical_valid_mask'],sar_aug_mask=mask)
                evaluation=evaluate_transform(estimate['matrix'],probes,reference,status=estimate['status'])
                saved=expected[stage][item['pair_id']]
                if source.tolist()!=saved['predicted_source'] or target.tolist()!=saved['predicted_target'] or estimate['matrix']!=saved['estimation']['matrix']:
                    raise AssertionError('Visualization inference differs from saved evaluation')
                predictions[stage]=dict(coarse_source_xy=source.tolist(),coarse_target_xy=coarse.tolist(),
                    refined_target_xy=target.tolist(),scores=scores.tolist(),estimation=estimate,evaluation=evaluation,
                    local_window_source=predicted['local_window_source'])
                panels[stage+'_coarse']=coarse_panel(optical,sar,source,coarse,A,stage)
                panels[stage+'_registration'],registered=registration_panel(optical,sar,mask,estimate,evaluation,stage)
                if stage=='trained': registered_final=registered
            image=Image.new('RGB',(1024,630),'#f4f5f7')
            draw=ImageDraw.Draw(image)
            draw.text((9,5),item['pair_id']+' | SAME frozen images + A | GT used for diagnostics only',fill='#111111',font=FONT)
            image.paste(panels['initial_coarse'],(0,26));image.paste(panels['trained_coarse'],(512,26))
            image.paste(panels['initial_registration'],(0,328));image.paste(panels['trained_registration'],(512,328))
            name=split+'_'+item['pair_id'].rsplit('/',1)[1]
            image.save(out/(name+'.png'))
            if registered_final is not None:
                Image.fromarray(registered_final).save(out/(name+'_registered_sar.png'))
            record=dict(pair_id=item['pair_id'],split=split,case_png=name+'.png',
                registered_sar_png=name+'_registered_sar.png' if registered_final is not None else None,
                input_tensor_sha256=hashlib.sha256(item['optical'].numpy().tobytes()+item['sar_aug'].numpy().tobytes()).hexdigest(),
                A_forward=A.tolist(),predictions=predictions,
                registration_matrix_source='inverse of predicted optical_to_SAR_aug affine; no GT warp',
                coarse_display='up to 64 uniformly spaced in source order; full predictions retained in JSON',
                line_colors='diagnostic GT coarse endpoint error only, never used for filtering or fitting',
                prediction_and_matrix_equal_saved_evaluation=True)
            (out/(name+'.json')).write_text(json.dumps(record,indent=2))
            cases.append({k:record[k] for k in ('pair_id','split','case_png','registered_sar_png','prediction_and_matrix_equal_saved_evaluation')} |
                         dict(initial_rmse_px=predictions['initial']['evaluation']['rmse_px'],
                              trained_rmse_px=predictions['trained']['evaluation']['rmse_px']))
        # Deterministic contact preview: first four in the frozen plan, not best/worst cases.
        selected=[c for c in cases if c['split']==split][:4]
        preview=Image.new('RGB',(1024,630*2),'white')
        for j,c in enumerate(selected):
            tile=Image.open(out/c['case_png']).resize((512,315),Image.Resampling.LANCZOS)
            # 4 cases stacked full-width for readable labels at native case size via links.
            preview.paste(tile,((j%2)*512,(j//2)*315))
        preview=preview.crop((0,0,1024,630))
        preview.save(out/(split+'_first4_preview.png'))
    manifest=dict(plan_sha256=plan_hash,checkpoints=checks,cases=cases,all_fixed_samples_included=True,
        selection='all 32 train and all 16 val, frozen-plan order; contact previews first four',
        no_GT_windows_at_inference=True,GT_only_after_prediction_for_diagnostics=True)
    (out/'visualization_manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    for c in cases:
        error='undefined' if c['trained_rmse_px'] is None else f"{c['trained_rmse_px']:.3f}px"
        rows.append(f'<article data-split="{c["split"]}"><h2>{html.escape(c["pair_id"])} — trained probe RMSE {error}</h2>'
                    f'<a href="{c["case_png"]}"><img loading="lazy" src="{c["case_png"]}"></a><p><a href="{c["case_png"].replace(".png",".json")}">Full predicted correspondences and matrices</a></p></article>')
    document='''<!doctype html><meta charset="utf-8"><title>T02 same-sample prediction gallery</title>
<style>body{font:15px system-ui;margin:24px auto;max-width:1100px;background:#f4f5f7;color:#142334}img{max-width:100%;display:block}article{background:white;padding:16px;margin:20px 0}h2{font-size:18px}button{padding:8px 16px;margin:4px}</style>
<h1>T02: initialized / trained predictions on identical frozen inputs</h1>
<p>All 48 cases. Top: predicted coarse matches; bottom: optical and predicted-affine registration checkerboard. No transform is displayed as a failure, never as identity. Red borders are unsupported warp pixels. Match colors use GT only after prediction for diagnostics.</p>
<button onclick="filter('all')">All 48</button><button onclick="filter('train')">Train 32</button><button onclick="filter('val')">Val 16</button>
'''+''.join(rows)+'''<script>function filter(s){document.querySelectorAll('article').forEach(a=>a.hidden=s!=='all'&&a.dataset.split!==s)}</script>'''
    (out/'index.html').write_text(document)
    print(json.dumps(dict(cases=len(cases),all_equal_saved_evaluation=True,output=str(out))))


if __name__=='__main__':
    main()
