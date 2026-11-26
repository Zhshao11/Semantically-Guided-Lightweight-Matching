"""Predeclared96pair K256 endpoint diagnostics for each pairedseed/group."""
import argparse,hashlib,json
from pathlib import Path
import torch
from sglm.t05 import UniformB0
from sglm.t02.data import AffinePairs256
from sglm.diagnostics import diagnose_pair,aggregate_diagnostics,bn_snapshot,bn_unchanged
from sglm.experiment_io import write_csv,flatten_diagnostic


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();index=json.loads(Path('reports/t09_audit_v1/run_index.json').read_text());p=index['protocol'];assert a.seed in p['seeds'];torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    ds=AffinePairs256('/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0',p['training']['validation_plan'],split='val',preload=True);ids=json.loads(Path('reports/t05_frozen_v1/subsets.json').read_text())['diagnostic_indices'];assert len(ids)==96
    out=Path('reports/t09_diagnostics_v1');out.mkdir(exist_ok=True);summaries=[];flat=[]
    with torch.inference_mode():
        for slot in [r for r in index['runs'] if r['seed']==a.seed]:
            group=slot['group'];cp=Path(slot['primary_checkpoint']);saved=torch.load(cp,map_location='cpu',weights_only=False);model=UniformB0(saved['config'],256).cuda().eval();model.load_state_dict(saved['model'],strict=True);before=bn_snapshot(model);records=[]
            if slot['reused_T08']:
                # OriginalT08same96K256diagnostics are exactlyreusable for unchangedcheckpoint.
                prior=json.loads(Path(f'reports/t08_diagnostics_v1/{group}_K256.json').read_text());records=prior['pairs'];assert prior['summary']['checkpoint_sha256']==hashlib.sha256(cp.read_bytes()).hexdigest()
            else:
                for i in ids:records.append(diagnose_pair(model,ds[i],saved['config'],dict(coarse_correctness_thresholds_px=[3,5,8],near_collinear_singular_ratio=.01)))
            assert bn_unchanged(before,model);summary=dict(seed=a.seed,group=group,**aggregate_diagnostics(records),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),reused_T08=slot['reused_T08']);summaries.append(summary)
            for record in records:flat.append({**flatten_diagnostic(record),'seed':a.seed,'group':group})
            (out/f'seed{a.seed}_{group}_K256.json').write_text(json.dumps(dict(summary=summary,pairs=records),separators=(',',':')));del model
        (out/f'seed{a.seed}_summary.json').write_text(json.dumps(dict(rows=summaries,images=96,GT_posthoc=True),indent=2));write_csv(out/f'seed{a.seed}_layered.csv',flat)


if __name__=='__main__':main()
