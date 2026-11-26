"""Descriptive attenuation on the exact same hardrelation cohort at all controllevels."""
import json
from pathlib import Path
import numpy as np
from sglm.t10_signal import paired_parent_summary
from sglm.experiment_io import write_csv


def main():
    d=dict(np.load('runs/t10a_candidates_v1/relations.npz'));plan=json.loads(Path('reports/t10_setup_v1/plan.json').read_text())['samples'];parents=sorted({r['verified512_parent'] for r in plan});lookup={name:i for i,name in enumerate(parents)};codes=np.array([lookup[r['verified512_parent']] for r in plan]);valid=d['random_id']>=0;common=valid.any(2).all(1);rows=[]
    for j,name in enumerate(['uncontrolled','distance_weight','distance_weight_class']):
        count=valid[:,j].sum(1);mean=np.full(len(count),np.nan);np.divide(np.where(valid[:,j],d['random_semantic'][:,j],0).sum(1),count,out=mean,where=count>0);delta=np.where(common,d['hard_semantic']-mean,np.nan);stats,_,_,_=paired_parent_summary(d['pair_index'],d['query_index'],delta,codes,20000,20261126);rows.append(dict(control=name,**stats,cohort='samehardrelations withall3controls'))
    write_csv('reports/t10_results_v1/common_cohort_controls.csv',rows);Path('reports/t10_results_v1/common_cohort_notes.json').write_text(json.dumps(dict(added_after_primary=True,candidate_sets_unchanged=True,primary_results_unchanged=True,attenuation_is_descriptive_not_causal_explained_variance=True,rows=rows),indent=2));print(rows)


if __name__=='__main__':main()
