"""Fixed mask coverage, class mass, entropy and spatial-continuity audit for all labels."""
import json
from pathlib import Path
import numpy as np
from sglm.experiment_io import write_csv


def main():
    names=json.loads(Path('reports/t07_teacher_v1/teacher_freeze.json').read_text())['target_classes'];rows=[];summary=[]
    for version,root in [('MS','runs/t07_semantic_cache_v1'),('T08-shuffle','runs/t08_shuffle_v1'),('T09-shuffle','runs/t09_shuffle_parent_v1')]:
        root=Path(root);target=np.load(root/'conditional7.npy',mmap_mode='r');weight=np.load(root/'weight.npy',mmap_mode='r');accepted=0;entropy_sum=0.;weighted_entropy=0.;sumweight=0.;classes=np.zeros(7,np.int64);soft=np.zeros(7);weighted=np.zeros(7);same_neighbors=0;neighbor_pairs=0
        for start in range(0,7912,64):
            p=np.array(target[start:start+64],dtype=np.float64);p/=p.sum(1,keepdims=True);w=np.array(weight[start:start+64]);mask=w>0;hard=p.argmax(1);e=-(p*np.log(np.maximum(p,1e-12))).sum(1);accepted+=int(mask.sum());entropy_sum+=float(e[mask].sum());weighted_entropy+=float((e*w).sum());sumweight+=float(w.sum())
            for c in range(7):classes[c]+=int((mask&(hard==c)).sum());soft[c]+=float(p[:,c][mask].sum());weighted[c]+=float((p[:,c]*w).sum())
            for a,b,ma,mb in [(hard[:,1:],hard[:,:-1],mask[:,1:],mask[:,:-1]),(hard[:,:,1:],hard[:,:,:-1],mask[:,:,1:],mask[:,:,:-1])]:
                valid=ma&mb;neighbor_pairs+=int(valid.sum());same_neighbors+=int(((a==b)&valid).sum())
        for c,name in enumerate(names):rows.append(dict(version=version,class_name=name,accepted_dominant_count=int(classes[c]),accepted_dominant_fraction=float(classes[c]/accepted),accepted_softclass_fraction=float(soft[c]/accepted),weighted_class_mass_per_grid=float(weighted[c]/(7912*1024))))
        summary.append(dict(version=version,accepted_grids=accepted,total_grids=7912*1024,coverage=accepted/(7912*1024),ignored_grids=7912*1024-accepted,entropy_mean_nats=entropy_sum/accepted,weighted_entropy_mean_nats=weighted_entropy/sumweight,mean_absolute_weight=sumweight/(7912*1024),accepted_neighbor_same_class_fraction=same_neighbors/neighbor_pairs,neighbor_pairs=neighbor_pairs))
    out=Path('reports/t09_audit_v1');write_csv(out/'label_class_distribution.csv',rows);write_csv(out/'label_coverage_entropy.csv',summary);(out/'label_fields.json').write_text(json.dumps(dict(rows=summary,class_distribution=rows,semantic_GT=False,spatial_note='cellwise shuffle destroys spatial continuity as well as content alignment, not a perfect correctness-only contrast'),indent=2));print(summary)


if __name__=='__main__':main()
