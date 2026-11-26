"""Fixed96 validation layered diagnoses at lastcheckpoint; GT remains posthoc."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from sglm.t05 import UniformB0
from sglm.t02.data import AffinePairs256
from sglm.diagnostics import diagnose_pair,aggregate_diagnostics,bn_snapshot,bn_unchanged
from sglm.experiment_io import write_csv,flatten_diagnostic


def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    p=json.loads(Path('configs/t08_protocol_v1.json').read_text());ds=AffinePairs256('/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0',p['validation_plan'],split='val',preload=True)
    subsets=json.loads(Path('reports/t05_frozen_v1/subsets.json').read_text());indices=subsets['diagnostic_indices'];assert len(indices)==96
    out=Path('reports/t08_diagnostics_v1');out.mkdir(exist_ok=True);summaries=[];flat=[]
    with torch.inference_mode():
        for group in p['groups']:
            path=Path('runs')/('t08_'+group+'_v1')/'inference_last.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
            for k in [256,1024]:
                m=UniformB0(saved['config'],k).cuda().eval();m.load_state_dict(saved['model'],strict=True);before=bn_snapshot(m);records=[]
                for i in indices:
                    r=diagnose_pair(m,ds[i],saved['config'],dict(coarse_correctness_thresholds_px=[3,5,8],near_collinear_singular_ratio=.01));r['group']=group;r['K']=k;records.append(r);flat.append({**flatten_diagnostic(r),'group':group,'K':k})
                assert bn_unchanged(before,m);summary=dict(group=group,K=k,**aggregate_diagnostics(records),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest());summaries.append(summary)
                (out/f'{group}_K{k}.json').write_text(json.dumps(dict(summary=summary,pairs=records),separators=(',',':')));print(group,k,'diagnosed',flush=True);del m
    write_csv(out/'layered_diagnostics.csv',flat);(out/'summary.json').write_text(json.dumps(dict(rows=summaries,subset='same96 predeclaredT05validation images, not all952',GT_posthoc=True),indent=2))


if __name__=='__main__':main()
