"""Training-source equivalence to official data wrappers; no thresholds or metrics tuning."""
import argparse,gc,importlib.util,json,os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from sglm.external_reference import OfficialReference,cpu_input,valid_correspondences,stable_pair_seed
from sglm.t02.data import AffinePairs256


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    p=json.loads(Path('configs/external_references_v1.json').read_text());out=Path(p['output']);cfg=json.loads(Path(p['frozen']+'/candidate_config.json').read_text())
    torch.set_num_threads(p['timing']['threads']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    plan=json.loads(Path('reports/t10_setup_v1/plan.json').read_text());ds=AffinePairs256(a.root,dict(config=plan['config'],samples=plan['samples'][:2]),split='train',preload=False)
    records=[]
    for name in ['XoFTR','MINIMA_RoMa']:
        adapter=OfficialReference(name,p,cfg)
        if name=='XoFTR':
            from src.config.default import get_cfg_defaults
            from src.utils.data_io import DataIOWrapper,lower_config
            conf=lower_config(get_cfg_defaults(inference=True))['test'];wrapper=DataIOWrapper(adapter.model,conf)
        else:
            spec=importlib.util.spec_from_file_location('official_minima_roma_io',Path(p['source_root'])/'MINIMA/src/utils/data_io_roma.py')
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            conf=dict(img0_resize=640,img1_resize=640,df=8,padding=False,coarse_scale=.125)
            wrapper=mod.DataIOWrapper(adapter.model,conf)
        with torch.inference_mode():
            for item in ds:
                data=cpu_input(item);prediction=adapter.predict(data)
                seed=stable_pair_seed(item['pair_id'],cfg['seed']);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                # Only compatibility shim for the official wrapper's removed np.float alias;
                # not used by adapter or any timed inference call. Restore immediately.
                old=np.__dict__.get('float');np.float=float
                try:r=wrapper.from_cv_imgs(data['optical'],np.repeat(data['sar'][:,:,None],3,2))
                finally:
                    if old is None:del np.float
                    else:np.float=old
                aa=valid_correspondences(r['mkpts0'],r['mkpts1'],r['mconf'],data['optical_mask'],data['sar_mask'])[:3]
                diff={key:float(np.max(np.abs(prediction[key]-value))) if len(value) and prediction[key].shape==value.shape else 0. if prediction[key].shape==value.shape else None for key,value in zip(['source','target','score'],aa)}
                records.append(dict(method=name,pair_id=item['pair_id'],official_matches=len(r['mconf']),adapter_raw_matches=prediction['raw_matches'],max_abs_difference=diff,
                    coordinate_tolerance_px=3.1e-5,score_tolerance=1e-6,
                    passed=all(d is not None and d<=(1e-6 if key=='score' else 3.1e-5) for key,d in diff.items())))
        del wrapper,adapter;gc.collect();torch.cuda.empty_cache()
    (out/'official_io_equivalence.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in records),records=records,
        train_pairs_only=True,GT_not_used=True,np_float_compatibility_only_in_diagnostic=True,
        tolerance_note='Initial1e-5pixel check retained separately; MINIMA official np.float scale isfloat64 while adapter usesfloat32, causing1ULP(1.5259e-5px) in256 coordinates. Use2ULP at256(3.1e-5px), no matching/inference/threshold changes.'),indent=2));print(json.dumps(records,indent=2))
    assert all(r['passed'] for r in records)


if __name__=='__main__':main()
