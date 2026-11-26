"""T07 S vs A_cap global top-K; SC secondary, same frozen independent evaluator."""
import argparse
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from sglm.t02.data import AffinePairs256
from sglm.t07_semantic import load_t07_model as load_selection_model
from sglm.model import coarse_xy
from sglm.diagnostics import diagnose_pair,aggregate_diagnostics,bn_snapshot,bn_unchanged
from sglm.evaluation import summarize_evaluations
from sglm.geometry import transform_points,valid_pixel_centers
from sglm.experiment_io import write_csv,flatten_diagnostic


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',default='reports/t07_selection_v1');ap.add_argument('--modes',nargs='+',choices=['A_cap','S','SC'],default=['A_cap','S','SC']);a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);p=json.loads(Path('configs/t07_protocol_v1.json').read_text())
    saved=torch.load(p['matcher_checkpoint'],map_location='cpu',weights_only=False);cfg=saved['config']
    assert hashlib.sha256(Path(p['matcher_checkpoint']).read_bytes()).hexdigest()==p['matcher_sha256']
    ds=AffinePairs256(a.root,p['validation_plan'],split='val',preload=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    diagnostic=dict(coarse_correctness_thresholds_px=[3,5,8],near_collinear_singular_ratio=.01)
    summaries=[];flat=[]
    # Fixed baselines are reused from their completed952 runs, never reselected.
    previous=json.loads(Path('reports/t06_selection_v1/summary.json').read_text())
    for row in previous['rows']:
        if row['mode'] in ['U','A','AC']:summaries.append(dict(**row,reused_from='T06 identical frozen model/plan'))
    for reuse_k in p['budgets']:
        for reuse_mode in ['A_cap','S','SC']:
            saved_raw=out/f'{reuse_mode}_K{reuse_k}.json'
            if reuse_mode not in a.modes and saved_raw.exists():
                prior=json.loads(saved_raw.read_text());summaries.append(prior['summary'])
                for r in prior['pairs']:flat.append(dict(mode=reuse_mode,K=reuse_k,selected_raw_correct5=r['selected_queries']['raw_correct5_fraction'],**flatten_diagnostic(r)))
    with torch.inference_mode():
        for k in p['budgets']:
            old={r['pair_id']:r for r in json.loads(Path(f'reports/t05_results_v1/K{k}_validation.json').read_text())['pairs']}
            for mode in a.modes:
                model=load_selection_model(saved,mode,k);before=bn_snapshot(model);records=[];reproduced=True
                for index,item in enumerate(ds):
                    # Ordinary forward contains score->select->gather->Kx1024 correlation.
                    args=[item[key][None].cuda() for key in ['optical','sar_aug','optical_valid_mask','sar_aug_valid_mask']]
                    with torch.autocast('cuda',dtype=torch.bfloat16):output=model(*args)
                    q=output['query_indices'];source=coarse_xy(q,32).cpu().numpy();choice=output['row_logits'][0].argmax(-1).cpu().numpy();has=choice<1024
                    target=coarse_xy(torch.as_tensor(choice.clip(0,1023),device='cuda'),32).cpu().numpy()
                    truth=transform_points(source,item['A_forward'].numpy());visible=valid_pixel_centers(truth,256,256)
                    error=np.linalg.norm(target-truth,axis=1);correct=has&visible&(error<5)
                    # Saved quantities are evaluation metrics, never scoring training labels.
                    r=diagnose_pair(model,item,cfg,diagnostic);r['mode']=mode;r['K']=k
                    r['selected_queries']=dict(count=len(q),raw_correct5_fraction=float(correct.mean()),GT_visible_fraction=float(visible.mean()),
                        raw_truth_in_window_fraction=float((has&visible&(np.abs(target-truth).max(1)<=6+1e-6)).mean()),
                        predicted_no_match_fraction=float((~has).mean()),posthoc_only=True)
                    r['coarse']['available_optical_queries']=r['coarse']['valid_optical_queries'];r['coarse']['valid_optical_queries']=k
                    if mode=='U':reproduced&=r['evaluation']==old[item['pair_id']]['evaluation'] and r['geometry']['estimation']['matrix']==old[item['pair_id']]['estimation']['matrix']
                    records.append(r);flat.append(dict(mode=mode,K=k,selected_raw_correct5=r['selected_queries']['raw_correct5_fraction'],**flatten_diagnostic(r)))
                    if (index+1)%200==0:print('validated',mode,k,index+1,flush=True)
                assert bn_unchanged(before,model)
                if mode=='U':assert reproduced,'Uniform baseline must exactly reproduceT05'
                agg=aggregate_diagnostics(records);errors=[r['evaluation']['rmse_px'] for r in records if r['evaluation']['solved']]
                summary=dict(mode=mode,K=k,**agg['registration'],RMSE_p90_px=float(np.percentile(errors,90)),
                    selected_query_correct5=float(np.mean([r['selected_queries']['raw_correct5_fraction'] for r in records])),
                    selected_query_raw_window_hit=float(np.mean([r['selected_queries']['raw_truth_in_window_fraction'] for r in records])),
                    filtered_match_correct5=agg['mean_coarse_correct_fraction_5px'],actual_refinement_window_hit=agg['mean_truth_in_window_fraction'],
                    inlier_hull_fraction=agg['mean_inlier_source_hull_fraction'],mean_matches=agg['mean_predicted_matches'],
                    head_parameters=sum(t.numel() for t in model.appearance_head.parameters()) if mode!='U' else 0,
                    BN_unchanged=True,original_thresholds_RANSAC_unchanged=True,T05_U_exact=reproduced if mode=='U' else None)
                (out/f'{mode}_K{k}.json').write_text(json.dumps(dict(summary=summary,pairs=records),separators=(',',':')))
                summaries.append(summary);print(json.dumps(summary),flush=True);del model
    write_csv(out/'selection_table.csv',summaries);write_csv(out/'layered_diagnostics.csv',flat)
    (out/'summary.json').write_text(json.dumps(dict(rows=summaries,SC='secondary coverage comparison; S vs A_cap is primary',validation_training_labels=False,
        matcher_sha256=p['matcher_sha256'],inference_source_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in ['sglm/t06_appearance.py','sglm/t07_semantic.py']}),indent=2))


if __name__=='__main__':main()
