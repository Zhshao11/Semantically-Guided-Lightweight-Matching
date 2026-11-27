"""Frozen real-data full/subset regression and train-source smoke before validation."""
import argparse,json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from sglm.sparse_refinement_experiment import OUT,PRIVATE,KEYS,setup,measure_pair
from sglm.t02.data import AffinePairs256
from sglm.t05 import UniformB0
from sglm.diagnostics import InferenceStateAudit
from sglm.experiment_io import write_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    p,cfg,model=setup();ds=AffinePairs256(a.root,OUT/'smoke_plan.json',split='train',preload=False)
    base=UniformB0(cfg).cuda().eval();base.load_state_dict(model.state_dict(),strict=True)
    audit=InferenceStateAudit(model);rows=[];checks=[];actual_calls=[]
    hook=model.local.register_forward_pre_hook(lambda m,args:actual_calls.append(len(args[2])))
    with torch.inference_mode():
        for i,item in enumerate(ds):
            args=[item[k][None].cuda() for k in KEYS]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(*args);packet=model.candidate_packet(output,0)
                full_local=model.local(output['optical_fine'],output['sar_fine'],packet['source'],packet['target'],args[2],args[3])
                with torch.autocast('cuda',enabled=False):
                    full_fp32=model.local(output['optical_fine'],output['sar_fine'],packet['source'],packet['target'],args[2],args[3])
                model.refinement_mode='all';model.refinement_budget=None
                full=model.refine(output,args);original=base.predict(*args)[0]
                assert all(torch.equal(full[k],original[k]) for k in ('source','target','score','coarse_target'))
                for variant in p['variants']:
                    model.refinement_mode=variant['mode'];model.refinement_budget=variant['budget']
                    selected=model.select_packet(packet,(256,256));idx=selected['original_candidate_idx']
                    local=model.local(output['optical_fine'],output['sar_fine'],selected['source'],selected['target'],args[2],args[3])
                    difference=float((local['target']-full_local['target'][idx]).abs().max()) if len(idx) else 0.
                    heat_difference=float((local['heatmap']-full_local['heatmap'][idx]).abs().max()) if len(idx) else 0.
                    assert difference<=p['regression']['subset_atol_px'] and heat_difference<=p['regression']['heatmap_atol']
                    with torch.autocast('cuda',enabled=False):
                        fp32=model.local(output['optical_fine'],output['sar_fine'],selected['source'],selected['target'],args[2],args[3])
                    fp32_difference=float((fp32['target']-full_fp32['target'][idx]).abs().max()) if len(idx) else 0.
                    fp32_heat_difference=float((fp32['heatmap']-full_fp32['heatmap'][idx]).abs().max()) if len(idx) else 0.
                    assert fp32_difference<=p['regression']['fp32_subset_atol_px'] and fp32_heat_difference<=p['regression']['fp32_heatmap_atol']
                    if 'valid' in local:assert torch.equal(local['valid'],full_local['valid'][idx])
                    pred=model.refine(output,args)
                    assert actual_calls[-1]==len(idx)==pred['attempted_refinements']
                    assert len(idx)==(len(packet['source']) if variant['mode']=='all' else min(variant['budget'],len(packet['source'])))
                    r,_=measure_pair(item,pred,cfg,variant,ds.samples[i]['verified512_parent'])
                    rows.append({k:v for k,v in r.items() if not isinstance(v,dict)})
                    checks.append(dict(pair_id=item['pair_id'],variant=variant['name'],all_prediction_bitwise_equal=True,
                        subset_max_coordinate_difference_px=difference,subset_max_heatmap_difference=heat_difference,
                        fp32_coordinate_difference_px=fp32_difference,fp32_heatmap_difference=fp32_heat_difference,
                        selected=len(idx),actual_local_count=actual_calls[-1]))
                # Over-budget must bypass every ordering even in confidence/coverage modes.
                for mode in ['global_conf','coverage_conf']:
                    model.refinement_mode=mode;model.refinement_budget=2048
                    pred=model.refine(output,args)
                    assert all(torch.equal(full[k],pred[k]) for k in ('source','target','score','coarse_target'))
            print('smoke/regression',i+1,len(ds),flush=True)
    hook.remove();state=audit.finish()
    write_csv(OUT/'regression_checks.csv',checks);write_csv(OUT/'smoke_metrics.csv',rows)
    result=dict(passed=True,train_pairs=len(ds),variant_runs=len(rows),no_training=True,
        all_original_predictions_bitwise_equal=True,overbudget_original_order_and_outputs_equal=True,
        actual_local_counts_equal_selection=True,subset_vs_full_max_coordinate_difference_px=max(r['subset_max_coordinate_difference_px'] for r in checks),
        subset_vs_full_max_heatmap_difference=max(r['subset_max_heatmap_difference'] for r in checks),
        fp32_subset_max_coordinate_difference_px=max(r['fp32_coordinate_difference_px'] for r in checks),
        fp32_subset_max_heatmap_difference=max(r['fp32_heatmap_difference'] for r in checks),
        tolerance=p['regression'],state_audit=state)
    (OUT/'regression.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
