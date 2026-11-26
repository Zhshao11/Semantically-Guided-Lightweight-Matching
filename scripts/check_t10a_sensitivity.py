"""Post-primary sensitivity only; saved candidates unchanged, no rule tuning/training."""
import json
from pathlib import Path
import numpy as np
from sglm.t10_signal import paired_parent_summary
from sglm.experiment_io import write_csv


def main():
    p=json.loads(Path('configs/t10a_protocol_v1.json').read_text());plan=json.loads(Path(p['samples']['plan']).read_text())['samples'];d=dict(np.load('runs/t10a_candidates_v1/relations.npz'));parents=sorted({r['verified512_parent'] for r in plan});lookup={name:i for i,name in enumerate(parents)};codes=np.array([lookup[r['verified512_parent']] for r in plan]);n=len(d['pair_index']);rows=[];classes=json.loads(Path('reports/t07_teacher_v1/teacher_freeze.json').read_text())['target_classes'];j=2
    valid=d['random_id'][:,j]>=0;distance_gap=np.abs(d['random_distance'][:,j]-d['hard_distance'][:,None]);weight_gap=np.abs(d['random_weight'][:,j]-d['hard_weight'][:,None]);tight=valid&(distance_gap<=2)&(weight_gap<=.02)
    specifications=[('classmatched_tight_2px_weight0.02',tight,np.ones(n,bool)),('exclude24px_savedtop5_classmatched',valid&(d['random_distance'][:,j]>24),d['hard_distance']>24),('exclude32px_savedtop5_classmatched',valid&(d['random_distance'][:,j]>32),d['hard_distance']>32),('rank1_only_classmatched',valid,d['rank']==0)]
    for label,allowed,relation in specifications:
        count=allowed.sum(1);mean=np.full(n,np.nan);np.divide(np.where(allowed,d['random_semantic'][:,j],0).sum(1),count,out=mean,where=count>0);delta=np.where(relation,d['hard_semantic']-mean,np.nan)
        for c,name in [(-1,'all'),*enumerate(classes)]:
            values=np.where(d['query_class']==c,delta,np.nan) if c>=0 else delta;stats,_,_,_=paired_parent_summary(d['pair_index'],d['query_index'],values,codes,20000,p['seed']);rows.append(dict(sensitivity=label,query_pseudoclass=name,**stats))
    out=Path('reports/t10_results_v1');write_csv(out/'post_primary_sensitivity.csv',rows)
    notes=dict(timing='added after inspecting primaryresults; clearlyposthoc, not preregisteredgate',fixed_candidates_unchanged=True,primaryanalysis_unchanged=True,training_started=False,tighter_matching='withinoriginalclassmatched randompool5 draws only, no reselection; partialcoverage explicitly counted',exclusion_sensitivity='filter existinghardtop5 andrandomsets; not reextracting top5undernewradius',multiple_class_CIs='exploratory percentileintervals, no multiplicitycorrection')
    (out/'post_primary_sensitivity_notes.json').write_text(json.dumps(notes,indent=2));print([r for r in rows if r['query_pseudoclass']=='all'])


if __name__=='__main__':main()
