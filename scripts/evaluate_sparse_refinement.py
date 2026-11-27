"""All952 fixed validation: same dense coarse proposals, refinement budget only."""
import argparse,json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from sglm.sparse_refinement_experiment import OUT,PRIVATE,KEYS,setup,measure_pair,flat_record
from sglm.t02.data import AffinePairs256
from sglm.diagnostics import InferenceStateAudit
from sglm.evaluation import summarize_evaluations
from sglm.experiment_io import write_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    p,cfg,model=setup();assert json.loads((OUT/'regression.json').read_text())['passed']
    ds=AffinePairs256(a.root,p['frozen']+'/validation_plan.json',split='val',preload=True);assert len(ds)==952
    ref={r['pair_id']:r for r in json.loads(open('reports/t05_results_v1/K1024_validation.json').read())['pairs']}
    parents={r['pair_id']:r['verified_parent_pair_id'] for r in json.loads(open(p['frozen']+'/subsets.json').read())['records']}
    audit=InferenceStateAudit(model);summaries=[];flat=[];reproduction=[]
    raw_root=PRIVATE/'predictions';raw_root.mkdir(exist_ok=True)
    with torch.inference_mode():
        for v in p['variants']:
            model.refinement_mode=v['mode'];model.refinement_budget=v['budget'];records=[]
            target_dir=raw_root/v['name'];target_dir.mkdir(exist_ok=True)
            for i,item in enumerate(ds):
                args=[item[k][None].cuda() for k in KEYS]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(*args);packet=model.candidate_packet(output,0)
                    pred=model.refine(output,args)
                r,raw=measure_pair(item,pred,cfg,v,parents[item['pair_id']],coarse=packet)
                assert pred['attempted_refinements']==(len(packet['source']) if v['mode']=='all' else min(v['budget'],len(packet['source'])))
                if v['mode']=='all':
                    old=ref[item['pair_id']]
                    assert r['estimation']['matrix']==old['estimation']['matrix']
                    assert r['evaluation']==old['evaluation']
                    assert r['num_selected']==old['attempted_refinements']
                    reproduction.append(dict(pair_id=item['pair_id'],matrix_exact=True,evaluation_exact=True,candidate_count_exact=True))
                np.savez_compressed(target_dir/f'{i:04d}.npz',**raw)
                records.append(r);flat.append(flat_record(r))
                if (i+1)%100==0:
                    print(v['name'],i+1,len(ds),flush=True)
                    (OUT/'status.json').write_text(json.dumps(dict(phase='accuracy',variant=v['name'],done=i+1)))
            errors=[r['global_probe_error'] for r in records if r['evaluation']['solved']]
            s=dict(variant=v['name'],mode=v['mode'],budget=v['budget'],**summarize_evaluations([r['evaluation'] for r in records]),
                solved_p90_rmse_px=float(np.percentile(errors,90)) if errors else None,
                registration_failures5=sum(not r['registration_success'] for r in records),
                mean_num_coarse_candidates=float(np.mean([r['num_coarse_candidates'] for r in records])),
                mean_num_selected=float(np.mean([r['num_selected'] for r in records])),
                mean_num_refined_valid=float(np.mean([r['num_refined_valid'] for r in records])),
                mean_num_ransac_inliers=float(np.mean([r['num_ransac_inliers'] for r in records])))
            for key in ['coarse_correct_fraction5','window_hit_fraction','window_inside_refine_error_px']:
                values=[r[key] for r in records if r[key] is not None];s['mean_'+key]=float(np.mean(values)) if values else None
            for key in ['coverage_opt','coverage_sar','inlier_coverage_opt','inlier_coverage_sar']:
                s['mean_'+key+'_hull_fraction']=float(np.mean([r[key]['hull_area_fraction'] for r in records]))
            summaries.append(s);(PRIVATE/f'{v["name"]}_validation.json').write_text(json.dumps(dict(summary=s,pairs=records),indent=2))
            write_csv(OUT/'accuracy_summary.csv',summaries);write_csv(OUT/'per_pair_metrics.csv',flat)
            if v['mode']=='all':
                (OUT/'full_reproduction.json').write_text(json.dumps(dict(passed=True,pairs=952,reference='T05K1024 epoch18',checks=reproduction),indent=2))
    (OUT/'state_audit.json').write_text(json.dumps(audit.finish(),indent=2))
    (OUT/'accuracy_summary.json').write_text(json.dumps(summaries,indent=2))
    (OUT/'status.json').write_text(json.dumps(dict(phase='accuracy_completed',pairs=952,variants=6)))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
