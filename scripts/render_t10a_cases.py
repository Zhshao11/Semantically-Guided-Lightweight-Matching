"""Fixedfirst16pair confusion cases, no effectsize-based image or query selection."""
import json,hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from sglm.geometry import warp_image


PALETTE=np.array([[200,200,200],[220,60,60],[240,220,50],[40,100,230],[170,120,70],[30,150,60],[150,210,90]],np.uint8)


def mark(draw,point,color,radius=5,kind='circle'):
    x,y=map(float,point)
    if kind=='cross':draw.line((x-radius,y,x+radius,y),fill=color,width=2);draw.line((x,y-radius,x,y+radius),fill=color,width=2)
    else:draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=2)


def main():
    source=Path('runs/t10a_candidates_v1');cases=json.loads((source/'cases.json').read_text());plan=json.loads(Path('reports/t10_setup_v1/plan.json').read_text())['samples'];root=Path('/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0');out=Path('runs/t10a_cases_v1');out.mkdir(exist_ok=True);names=['BG','Building','Road','Water','Barren','Forest','Agri']
    for start in range(0,16,4):
        sheet=Image.new('RGB',(1280,4*330),'white');draw=ImageDraw.Draw(sheet)
        for rowindex,case in enumerate(cases[start:start+4]):
            i=case['pair_index'];sample=plan[i];top=rowindex*330;array=np.load(source/case['file']);with_image=Image.open(root/sample['optical_path']);optical=with_image.convert('RGB');with_image.close()
            with Image.open(root/sample['sar_path']) as im:sar=np.array(im.convert('L'))
            aug,_=warp_image(sar,np.asarray(sample['A_forward']));aug=Image.fromarray(aug).convert('RGB');nominal=optical.copy()
            qprob=array['query_probability7'];label=qprob.argmax(1).reshape(32,32);semantic=Image.fromarray(PALETTE[label]).resize((256,256),resample=Image.Resampling.NEAREST);valid=array['query_semantic_valid'].reshape(32,32);shown=np.array(semantic);shown[~np.repeat(np.repeat(valid,8,0),8,1)]=[240,0,240];semantic=Image.fromarray(shown)
            if case['eligible']:
                q=case['query_index'];h=case['hard_id'];rids=np.asarray(case['matched_random_ids']);qx=(q%32*8+3.5,q//32*8+3.5);hy=(h%32*8+3.5,h//32*8+3.5);gt=case['GT_augmented_xy'];rx=np.stack([rids%32*8+3.5,rids//32*8+3.5],1)
                do=ImageDraw.Draw(optical);mark(do,qx,'lime',6);dn=ImageDraw.Draw(nominal);mark(dn,qx,'lime',6);nomxy=array['nominal_candidate_xy'];mark(dn,nomxy[h],'red',6);dn.line((*qx,*nomxy[h]),fill='red',width=1)
                da=ImageDraw.Draw(aug);mark(da,gt,'lime',6,'cross');mark(da,hy,'red',6)
                for rid,ry in zip(rids,rx):mark(da,ry,'cyan',4);mark(dn,nomxy[rid],'cyan',4)
                title=f"{i:03d} {sample['pair_id']} hardrank={case['hard_rank']+1} sim={case['semantic_similarity']:.3f} randmean={case['random_semantic_similarity']:.3f}"
            else:title=f"{i:03d} {sample['pair_id']}: no reliable hard+matchedrandom relation (shown, not replaced)"
            draw.text((4,top+3),title,fill='black')
            for col,(name,pic) in enumerate([('Optical: greenquery',optical),('SARaug: redwrong / cyanrandom',aug),('Nominalopt: greenquery-redcandidate',nominal),('Teacherpseudo: magentaunreliable',semantic)]):
                draw.text((col*256+3,top+22),name,fill='black');sheet.paste(pic,(col*256,top+43))
            if case['eligible']:
                for n,(labelname,values,color) in enumerate([('query',case['query_probability'],'green'),('hard',case['hard_probability'],'red'),('randommean',case['random_probability_mean'],'blue')]):
                    base=top+45+n*78;draw.text((1030,base),labelname,fill=color)
                    for c,v in enumerate(values):
                        yy=base+14+c*8;draw.text((1030,yy),names[c],fill='black',font=ImageFont.truetype('DejaVuSans.ttf',8));draw.rectangle((1095,yy+1,1095+float(v)*165,yy+6),fill=color)
            draw.text((3,top+306),'GreenSARcross = nominal A(query), not manualGT. All semantics externaloptical pseudo; original residual unknown.',fill='black')
        sheet.save(out/f'fixed_cases_page_{start//4+1}.png')
    (out/'case_order.json').write_text(json.dumps(dict(cases=cases,order='first16frozenpairs, includes ineligible, query deterministic UIDhash rotation then rank; no semanticdifference sorting',queryselection_uses_similarity=False,SARteacher=False,semanticGT=False),indent=2));print('Rendered4pages fixed16caseorder')


if __name__=='__main__':main()
