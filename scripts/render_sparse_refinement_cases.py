"""Fixed-order post-hoc cases, never used to choose budgets, thresholds or rules."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sglm.sparse_refinement_experiment import OUT,PRIVATE,setup
from sglm.t02.data import AffinePairs256
from sglm.geometry import transform_points


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    p=json.loads(Path('configs/sparse_refinement_v1.json').read_text());cfg=json.loads((Path(p['frozen'])/'candidate_config.json').read_text())
    ds=AffinePairs256(a.root,p['frozen']+'/validation_plan.json',split='val',preload=False)
    versions=['A0_all','A2_prefix128','B0_global128','B1_coverage128']
    records={v:json.loads((PRIVATE/f'{v}_validation.json').read_text())['pairs'] for v in versions}
    sets=[('full_success_prefix_failure',[i for i in range(len(ds)) if records['A0_all'][i]['registration_success'] and not records['A2_prefix128'][i]['registration_success']],['A0_all','A2_prefix128']),
          ('global_failure_coverage_success',[i for i in range(len(ds)) if not records['B0_global128'][i]['registration_success'] and records['B1_coverage128'][i]['registration_success']],['B0_global128','B1_coverage128'])]
    target=PRIVATE/'failure_cases';target.mkdir(exist_ok=True);manifest=[]
    for label,indices,vs in sets:
        for index in indices[:p['cases']['per_type']]:
            item=ds[index];opt=item['optical'].numpy().transpose(1,2,0);sar=item['sar_aug'][0].numpy()
            fig,axes=plt.subplots(len(vs),3,figsize=(13,4.6*len(vs)))
            for row,v in enumerate(vs):
                r=records[v][index];raw=dict(np.load(PRIVATE/'predictions'/v/f'{index:04d}.npz'))
                mask=np.asarray(raw['selected_candidate_idx'],int);inliers=np.asarray(r['estimation']['inlier_indices'],int)
                for col,(image,allpts,selected,final) in enumerate([(opt,raw['coarse_source'],raw['coarse_source'][mask],raw['source'][inliers]),
                    (sar,raw['coarse_target_all'],raw['coarse_target_all'][mask],raw['target'][inliers])]):
                    ax=axes[row,col];ax.imshow(image,cmap='gray',vmin=0,vmax=1)
                    ax.scatter(*allpts.T,s=6,c='silver',label='all coarse')
                    ax.scatter(*selected.T,s=12,c='orange',label='selected',alpha=.7)
                    if len(final):ax.scatter(*final.T,s=15,c='lime',marker='+',label='final inlier')
                    ax.set(xlim=(-.5,255.5),ylim=(255.5,-.5),title=f'{v}: '+('Optical' if col==0 else 'SAR augmented'))
                    for b in [64,128,192]:ax.axvline(b,color='white',alpha=.2);ax.axhline(b,color='white',alpha=.2)
                    ax.legend(fontsize=6,loc='lower right');ax.tick_params(labelsize=7)
                ax=axes[row,2];ax.set_facecolor('#202530')
                if r['estimation']['matrix'] is not None:
                    predicted=transform_points(raw['probes'],r['estimation']['matrix']);truth=raw['probe_truth'];diff=predicted-truth
                    ax.scatter(*truth.T,s=4,c='lime',label='reference probe')
                    ax.quiver(truth[:,0],truth[:,1],diff[:,0],diff[:,1],angles='xy',scale_units='xy',scale=1,color='tomato',width=.003)
                co=r['coverage_opt'];cs=r['coverage_sar'];io=r['inlier_coverage_opt'];it=r['inlier_coverage_sar']
                ax.set(xlim=(-.5,255.5),ylim=(255.5,-.5),title=f'Independent probes: RMSE={r["global_probe_error"]:.3f}px; SR5={r["registration_success"]}')
                ax.set_xlabel(f'N/M/valid/inlier={r["num_coarse_candidates"]}/{r["num_selected"]}/{r["num_refined_valid"]}/{r["num_ransac_inliers"]}\n'
                    f'hull opt/SAR={co["hull_area_fraction"]:.2f}/{cs["hull_area_fraction"]:.2f}; inlier={io["hull_area_fraction"]:.2f}/{it["hull_area_fraction"]:.2f}\n'
                    f'inlier rank opt/SAR={io["rank"]}/{it["rank"]}; near-collinear={io["near_collinear"]}/{it["near_collinear"]}',fontsize=8)
            fig.suptitle(f'{label}: {item["pair_id"]}\nFirst qualifying pairs in original frozen order. Probes/affine are synthetic references; not human semantic GT.',fontsize=10)
            fig.tight_layout(rect=(0,0,1,.95));name=f'{label}_{index:04d}.png';fig.savefig(target/name,dpi=130);plt.close(fig)
            manifest.append(dict(type=label,pair_index=index,pair_id=item['pair_id'],file=name,variants=vs))
    notes=dict(selection=p['cases'],cases=manifest,eligible_counts={label:len(indices) for label,indices,_ in sets},
        arrows='nominal affine reference probe -> predicted transformed probe; length is residual in pixels',
        teacher_or_GT_used_in_prediction=False,original_satellite_scene_known=False)
    (target/'manifest.json').write_text(json.dumps(notes,indent=2));(OUT/'failure_cases_manifest.json').write_text(json.dumps(notes,indent=2))
    print(json.dumps(notes,indent=2))


if __name__=='__main__':main()
