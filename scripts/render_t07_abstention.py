"""Before/after accepted-mask display using saved private probabilities, no teacher rerun."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
PALETTE=np.array([[0,0,0],[200,200,200],[220,60,60],[240,220,50],[40,100,230],[170,120,70],[30,150,60],[150,210,90]],np.uint8)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    frozen=json.loads(Path('reports/t07_teacher_v1/teacher_freeze.json').read_text());review=Path('runs/t07_teacher_review_v1');summ=json.loads((review/'summary.json').read_text());mapping={r['pair_id']:r for r in summ['records']};rows=json.loads(Path('reports/t06_setup_v1/OS_train_review.json').read_text())['samples']
    disabled=[frozen['target_classes'].index(c) for c in frozen['abstention']['disabled_dominant_classes']];excluded=set(frozen['abstention']['image_ids']);records=[]
    for start in range(0,32,8):
        sheet=Image.new('RGB',(1024,2256),'white')
        for j,row in enumerate(rows[start:start+8]):
            uid=row['pair_id'];data=np.load(review/(uid.replace('/','_')+'.npz'));pred=data['raw8'].argmax(0);p=data['conditional7'];before=data['weight'];after=before.copy();dominant=p.argmax(0)
            for c in disabled:after[dominant==c]=0
            if uid in excluded or mapping[uid]['collapse_flag']:after[:]=0
            mask_before=np.repeat(np.repeat(before>0,8,0),8,1);mask_after=np.repeat(np.repeat(after>0,8,0),8,1)
            with Image.open(Path(a.root)/row['optical_path']) as im:image=im.convert('RGB')
            first=PALETTE[pred].copy();second=first.copy();first[~mask_before]=[240,0,240];second[~mask_after]=[240,0,240]
            heat=np.repeat(np.repeat(after,8,0),8,1);heat=np.stack([heat*255,np.zeros_like(heat),(1-heat)*255],2).astype(np.uint8)
            for col,(label,pic) in enumerate([('optical',image),('before policy',Image.fromarray(first)),('after: magenta=abstain',Image.fromarray(second)),('after weight',Image.fromarray(heat))]):
                sheet.paste(pic,(col*256,j*282+26));ImageDraw.Draw(sheet).text((col*256+3,j*282+5),uid.rsplit('/',1)[-1]+' '+label,fill='black')
            records.append(dict(pair_id=uid,before_mask_fraction=float((before>0).mean()),after_mask_fraction=float((after>0).mean()),after_weight_mean=float(after.mean()),OS_semantic_accuracy=None))
        sheet.save(review/f'abstention_page_{start//8+1}.png')
    (review/'abstention_review.json').write_text(json.dumps(dict(records=records,source='saved32 batch1 preview arrays; production cache uses batch8 and freshFP32 logits; numerical quantization may slightly differ',checkpoint_sha256=frozen['checkpoint_sha256'],OS_semantic_GT=False),indent=2))


if __name__=='__main__':main()
