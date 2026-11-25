"""Dense frozen train-only matcher labels and detached feature cache; no val/test reads."""
import argparse
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sglm.t02.data import AffinePairs256
from sglm.t05 import UniformB0
from sglm.t06_appearance import reliability_targets,sar_global
from sglm.diagnostics import bn_snapshot,bn_unchanged


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',default='runs/t06_reliability_v1');a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Completed labels must not be overwritten')
    setup=json.loads(Path('reports/t06_setup_v1/setup.json').read_text());p=setup['protocol']
    assert hashlib.sha256(Path(p['matcher_checkpoint']).read_bytes()).hexdigest()==p['matcher_sha256']
    planpath=Path('reports/t06_setup_v1/training_plan.json');assert hashlib.sha256(planpath.read_bytes()).hexdigest()==setup['train_plan_sha256']
    ds=AffinePairs256(a.root,planpath,split='train',preload=False);assert len(ds)==7912 and all(r['split']=='train' for r in ds.samples)
    saved=torch.load(p['matcher_checkpoint'],map_location='cpu',weights_only=False)
    model=UniformB0(saved['config'],1024).cuda().eval().requires_grad_(False);model.load_state_dict(saved['model'],strict=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    before=bn_snapshot(model);n=len(ds);maps={}
    maps['optical_features']=np.lib.format.open_memmap(out/'optical_features.npy',mode='w+',dtype=np.float32,shape=(n,64,32,32))
    maps['sar_mean']=np.lib.format.open_memmap(out/'sar_mean.npy',mode='w+',dtype=np.float32,shape=(n,64))
    offset=0
    with torch.inference_mode():
        for batch in DataLoader(ds,batch_size=4,num_workers=4,pin_memory=True,shuffle=False):
            args=[batch[k].cuda(non_blocking=True) for k in ['optical','sar_aug','optical_valid_mask','sar_aug_valid_mask']]
            with torch.autocast('cuda',dtype=torch.bfloat16):output=model(*args)
            targets=reliability_targets(output,batch['A_forward'].cuda(),args[3],p['labels'])
            count=len(args[0]);maps['optical_features'][offset:offset+count]=output['optical_coarse'].float().cpu().numpy()
            maps['sar_mean'][offset:offset+count]=sar_global(output).cpu().numpy()
            for key,tensor in targets.items():
                array=tensor.cpu().numpy()
                if key not in maps:maps[key]=np.lib.format.open_memmap(out/(key+'.npy'),mode='w+',dtype=array.dtype,shape=(n,1024))
                maps[key][offset:offset+count]=array
            offset+=count
            if offset%100==0:
                (out/'status.json').write_text(json.dumps(dict(pairs_done=offset,total=n)));print('labeled',offset,n,flush=True)
    assert bn_unchanged(before,model)
    for array in maps.values():array.flush()
    y=np.asarray(maps['geometric_correctness']);counts={name:int((y==value).sum()) for name,value in [('positive',1),('negative',0),('ignored',-1)]}
    finite=np.asarray(maps['ambiguity_margin']);finite=finite[np.isfinite(finite)]
    summary=dict(protocol=p,matcher_sha256=p['matcher_sha256'],train_plan_sha256=setup['train_plan_sha256'],pairs=n,
        coarse_queries=n*1024,label_counts=counts,label_fractions={k:v/y.size for k,v in counts.items()},
        optical_valid_fraction=float(maps['optical_valid'].mean()),GT_reliable_fraction=float(maps['GT_reliable'].mean()),
        positive_candidate_exists_fraction=float(maps['positive_candidate_exists'].mean()),ambiguity_margin_valid_fraction=float(maps['ambiguity_margin_valid'].mean()),
        margin_percentiles=np.percentile(finite,[0,10,50,90,100]).tolist(),
        features='FP32 promotion of frozen BF16 coarse outputs, exact cached input for FP32 head',
        BN_buffers_unchanged=True,backbone_requires_grad=False,labels_from_train_only=True,val_test_read=False,
        ambiguity_used_for_training=False,original_registration_residual='unknown;5..8px annulus is an explicit conservative ignore band, not measured residual')
    (out/'pair_ids.json').write_text(json.dumps([r['pair_id'] for r in ds.samples]));(out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='protocol'},indent=2))


if __name__=='__main__':main()
