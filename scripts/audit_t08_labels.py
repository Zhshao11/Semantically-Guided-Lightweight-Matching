"""Pseudo-label coverage/class/stability audit; no OS semantic accuracy computation."""
import json
from pathlib import Path
import numpy as np
from sglm.experiment_io import write_csv


def main():
    cache=Path('runs/t07_semantic_cache_v1');p=np.load(cache/'conditional7.npy',mmap_mode='r');w=np.load(cache/'weight.npy',mmap_mode='r');a=np.load(cache/'retained_mass.npy',mmap_mode='r');names=json.loads(Path('reports/t07_teacher_v1/teacher_freeze.json').read_text())['target_classes']
    totals=np.zeros(7,np.int64);accepted=np.zeros(7,np.int64);weighted=np.zeros(7);stable=np.zeros(7,np.int64);mass=np.zeros(7);nactive=0;nall=7912*1024;nonzero_images=0;entropy_sum=0.;nlowmass=0
    for start in range(0,len(p),64):
        prob=np.array(p[start:start+64],dtype=np.float32);prob/=prob.sum(1,keepdims=True);weight=np.array(w[start:start+64]);hard=prob.argmax(1);mask=weight>0;agree=np.zeros_like(hard,dtype=np.int16);count=np.zeros_like(hard,dtype=np.int16)
        for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
            shifted=np.roll(np.roll(hard,dy,1),dx,2);valid=np.ones_like(mask)
            if dy==1:valid[:,0]=False
            if dy==-1:valid[:,-1]=False
            if dx==1:valid[:,:,0]=False
            if dx==-1:valid[:,:,-1]=False
            agree+=(hard==shifted)&valid;count+=valid
        stable_mask=agree>=.75*count
        nactive+=int(mask.sum());nonzero_images+=int(mask.any((1,2)).sum());nlowmass+=int((np.array(a[start:start+64])<.8).sum())
        entropy=-(prob*np.log(np.maximum(prob,1e-12))).sum(1)/np.log(7);entropy_sum+=float(entropy[mask].sum())
        for c in range(7):
            selected=mask&(hard==c);totals[c]+=int((hard==c).sum());accepted[c]+=int(selected.sum());stable[c]+=int((selected&stable_mask).sum());weighted[c]+=float((weight*prob[:,c]).astype(np.float64).sum());mass[c]+=float(prob[:,c].astype(np.float64).sum())
    rows=[dict(class_name=name,all_grid_dominant_count=int(totals[c]),accepted_dominant_count=int(accepted[c]),accepted_dominant_fraction=int(accepted[c])/max(nactive,1),weighted_softclass_mass_per_grid=weighted[c]/nall,all_softclass_mass_per_grid=mass[c]/nall,accepted_neighbor_stable_fraction=stable[c]/max(int(accepted[c]),1)) for c,name in enumerate(names)]
    out=Path('reports/t08_preflight_v1');write_csv(out/'pseudo_label_classes.csv',rows)
    result=dict(training_pairs=7912,total_grids=nall,positive_weight_grids=nactive,ignored_grids=nall-nactive,positive_weight_fraction=nactive/nall,images_with_positive_weight=nonzero_images,images_without_weight=7912-nonzero_images,retained_mass_below_point8_grids=nlowmass,mean_conditional_entropy_on_weighted_grids=entropy_sum/nactive,accepted_neighbor_stable_fraction=int(stable.sum())/nactive,stability_definition='same dominant class in at least75% of available4neighbors; diagnostic only, not semanticGT or a new trainingmask',classes=rows,OS_semantic_GT=False,OS_semantic_accuracy=None,teacher_probabilities_are_external_soft_pseudolabels=True,training_mask_unchanged_from_T07=True,labels_hardcoded_for_matching_suitability=False)
    (out/'pseudo_label_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
