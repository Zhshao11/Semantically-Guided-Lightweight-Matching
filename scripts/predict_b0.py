"""GT-free B0 prediction entry: accepts optical/SAR images, optional masks, checkpoint."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from sglm.data import _read_patch
from sglm.t02.model import B0Full
from sglm.evaluation import estimate_from_correspondences


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--optical',required=True)
    ap.add_argument('--sar-aug',required=True)
    ap.add_argument('--optical-mask')
    ap.add_argument('--sar-mask')
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--device',default='cuda')
    a=ap.parse_args()
    torch.set_num_threads(4)
    checkpoint=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    model=B0Full(checkpoint['config']).to(a.device).eval()
    model.load_state_dict(checkpoint['model'])
    optical=_read_patch(Path(a.optical),True)
    sar=_read_patch(Path(a.sar_aug),False)
    def mask(path):
        if path is None:
            return torch.ones(1,256,256,dtype=torch.bool,device=a.device)
        array=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        if array is None or array.shape!=(256,256):
            raise ValueError('Invalid 256 mask')
        return torch.from_numpy(array>0)[None].to(a.device)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        prediction=model.predict(torch.from_numpy(optical).permute(2,0,1)[None].float().to(a.device)/255,
            torch.from_numpy(sar)[None,None].float().to(a.device)/255,mask(a.optical_mask),mask(a.sar_mask))[0]
    matches={k:prediction[k].float().cpu().numpy() for k in ('source','target','score')}
    geometry=estimate_from_correspondences(matches['source'],matches['target'],matches['score'],checkpoint['config']['evaluation'],seed=checkpoint['config']['seed'])
    result=dict(model='B0-Full',inference_reads_ground_truth=False,estimation=geometry,
        matches={k:v.tolist() for k,v in matches.items()},local_window_source=prediction['local_window_source'])
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(geometry))


if __name__=='__main__':
    main()
