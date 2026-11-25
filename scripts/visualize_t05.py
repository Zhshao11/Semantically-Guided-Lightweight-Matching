"""Same fixed diagnostics: post-hoc success/failure examples, never used for tuning."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from sglm.t02.data import AffinePairs256
from sglm.geometry import transform_points,as_homogeneous,warp_image


def correspondence_panel(optical,sar,source,target,truth,scores):
    panel=Image.new('RGB',(512,256));panel.paste(Image.fromarray(optical),(0,0));panel.paste(Image.fromarray(sar).convert('RGB'),(256,0));draw=ImageDraw.Draw(panel)
    # Fixed score ordering, at most40 lines for legibility; metrics always use all matches.
    indices=np.argsort(-scores,kind='stable')[:40]
    for i in indices:
        a=tuple(source[i]);b=(target[i,0]+256,target[i,1]);color=(40,240,60) if np.linalg.norm(target[i]-truth[i])<5 else (255,70,60)
        draw.line([a,b],fill=color,width=1)
        for x,y in [a,b]:draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
    return panel


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',default='runs/t05_visuals')
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(Path('reports/t05_frozen_v1/validation_plan.json').read_text());ds=AffinePairs256(a.root,plan,split='val',preload=False)
    index={r['pair_id']:i for i,r in enumerate(ds.samples)}
    full=json.loads(Path('reports/t05_results_v1/K1024_diagnostics.json').read_text())['pairs']
    selected=[]
    for flag in [True,False]:
        group=sorted([r for r in full if r['evaluation']['success']['5']==flag],key=lambda r:r['evaluation']['rmse_px'])
        selected.extend(group[int(round(q*(len(group)-1)))]['pair_id'] for q in [.1,.4,.6,.9])
    records=[]
    for k in [1024,512,256,128]:
        rows={r['pair_id']:r for r in json.loads(Path(f'reports/t05_results_v1/K{k}_diagnostics.json').read_text())['pairs']}
        sheet=Image.new('RGB',(1280,288*len(selected)),'white')
        for ordinal,uid in enumerate(selected):
            r=rows[uid];item=ds[index[uid]]
            optical=(item['optical'].numpy().transpose(1,2,0)*255).astype(np.uint8);sar=(item['sar_aug'][0].numpy()*255).astype(np.uint8)
            source=np.array(r['predictions']['source'],float).reshape(-1,2);coarse=np.array(r['predictions']['coarse_target'],float).reshape(-1,2)
            target=np.array(r['predictions']['target'],float).reshape(-1,2);score=np.array(r['predictions']['scores'])
            truth=transform_points(source,item['A_forward'].numpy())
            left=correspondence_panel(optical,sar,source,coarse,truth,score)
            middle=correspondence_panel(optical,sar,source,target,truth,score)
            matrix=r['geometry']['estimation']['matrix']
            if matrix is not None:
                aligned,mask=warp_image(sar,np.linalg.inv(as_homogeneous(matrix))[:2])
                y,x=np.indices((256,256));check=(x//32+y//32)%2==0
                image=np.where(check[:,:,None],optical,np.repeat(aligned[:,:,None],3,2));image[~mask&~check]=[40,40,100]
            else:image=np.zeros_like(optical)
            panel=Image.new('RGB',(1280,288),'white');draw=ImageDraw.Draw(panel)
            error=r['evaluation']['rmse_px'];success=r['evaluation']['success']['5']
            title=f'K={k} {uid} final5px={success}, RMSE={error:.3f}px' if error is not None else f'K={k} {uid} solve failed'
            draw.text((3,3),title,fill='black');draw.text((3,17),'Coarse (top40; green error<5px)',fill='black');draw.text((515,17),'Refined (same selected correspondences)',fill='black');draw.text((1027,17),'Predicted registration checkerboard',fill='black')
            panel.paste(left,(0,32));panel.paste(middle,(512,32));panel.paste(Image.fromarray(image),(1024,32));sheet.paste(panel,(0,ordinal*288))
            records.append(dict(K=k,pair_id=uid,RMSE=error,success5px=success,all_predicted_matches=len(source),rendered_lines=min(40,len(source))))
        sheet.save(out/f'K{k}_same_success_failure_examples.png')
    (out/'visualization_manifest.json').write_text(json.dumps(dict(selection='K1024 success/failure groups, RMSE quantiles .1/.4/.6/.9; same8 pairs allK; posthoc illustrations only',
        correspondence_colors='GT-relative<5px green; otherwise red. GT is posthoc only',checkerboard='Optical / SAR_aug warped back by inverse predicted matrix; invalid estimates dark blue',records=records),indent=2))


if __name__=='__main__':main()
