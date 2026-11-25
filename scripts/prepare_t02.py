"""Freeze images, A_forward, masks and independent probes; render twenty GT review cards."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from sglm.t02.data import load_t02_config,create_plan,AffinePairs256
from sglm.geometry import transform_points
from sglm.evaluation import fixed_synthetic_probes


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--config',default='configs/t02_b0_full_v1.json')
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    config=load_t02_config(a.config)
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    plan=create_plan(config,config['training_manifest'],config['reference_manifest'])
    payload=json.dumps(plan,indent=2)+'\n'
    if (out/'fixed_plan.json').exists() and (out/'fixed_plan.json').read_text()!=payload:
        raise FileExistsError('Refusing to overwrite a different fixed T02 plan')
    (out/'fixed_plan.json').write_text(payload)
    dataset=AffinePairs256(a.root,plan)
    evidence=[]
    cards=[]
    for i in range(20):
        item=dataset[i]
        again=dataset[i]
        assert all(torch.equal(item[k],again[k]) for k in item if torch.is_tensor(item[k]))
        matrix=item['A_forward'].numpy()
        probes,targets=fixed_synthetic_probes(matrix,optical_mask=item['optical_valid_mask'],sar_aug_mask=item['sar_aug_valid_mask'])
        points=probes[np.linspace(0,len(probes)-1,12,dtype=int)]
        links=transform_points(points,matrix)
        optical=(item['optical'].permute(1,2,0).numpy()*255).round().astype(np.uint8)
        sar=np.repeat((item['sar_aug'][0].numpy()*255).round().astype(np.uint8)[...,None],3,-1)
        invalid=~item['sar_aug_valid_mask'].numpy()
        sar[invalid]=[150,20,20]
        card=Image.new('RGB',(512,282),'white')
        card.paste(Image.fromarray(optical),(0,26))
        card.paste(Image.fromarray(sar),(256,26))
        draw=ImageDraw.Draw(card)
        draw.text((4,6),item['pair_id']+' | known affine, not manual GT',fill='black')
        colors=['#00ff77','#ffdd00','#00cfff','#ff88ff']
        for j,(src,tgt) in enumerate(zip(points,links)):
            sx,sy=src+[0,26]
            tx,ty=tgt+[256,26]
            color=colors[j%len(colors)]
            draw.line((sx,sy,tx,ty),fill=color,width=1)
            for x,y in [(sx,sy),(tx,ty)]:
                draw.ellipse((x-2,y-2,x+2,y+2),outline=color,width=1)
        cards.append(card)
        evidence.append(dict(pair_id=item['pair_id'],repeat_read_bitwise_equal=True,
            A_forward=matrix.tolist(),optical_valid_fraction=float(item['optical_valid_mask'].float().mean()),
            sar_aug_valid_fraction=float(item['sar_aug_valid_mask'].float().mean()),
            valid_probe_count=len(probes),review_source_xy=points.tolist(),review_target_xy=links.tolist()))
    for sheet in range(2):
        image=Image.new('RGB',(1024,282*5),'white')
        for j,card in enumerate(cards[sheet*10:(sheet+1)*10]):
            image.paste(card,((j%2)*512,(j//2)*282))
        image.save(out/f'gt_review_sheet_{sheet+1:02d}.png')
    (out/'loader_review_20.json').write_text(json.dumps(dict(cases=evidence,passed=True,
        plan_sha256=hashlib.sha256((out/'fixed_plan.json').read_bytes()).hexdigest(),
        note='Synthetic GT assumes small residual original misregistration.'),indent=2))
    print(json.dumps(dict(train_pairs=len(dataset),val_pairs=len(AffinePairs256(a.root,plan,'val')),review_cases=20,passed=True)))


if __name__=='__main__':
    main()
