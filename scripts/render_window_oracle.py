"""Fixed six original diagnostic inputs; figures are private, never selection/filtering."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch,Polygon
from sglm.t02.data import AffinePairs256
from sglm.window_oracle import convex_hull


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    p=json.loads(Path('configs/window_oracle_v1.json').read_text());out=Path(p['output']);cache=Path(p['cache']);private=cache.parent/'visualizations';private.mkdir(exist_ok=True)
    ds=AffinePairs256(a.root,p['frozen']+'/validation_plan.json',split='val',preload=False)
    indices=json.loads(Path(p['frozen']+'/subsets.json').read_text())['diagnostic_indices'][:6];result=json.loads((out/'results.json').read_text());byindex={r['index']:r for r in result['pairs']};manifest=[]
    for index in indices:
        item=ds[index];c=np.load(cache/f'{index:04d}.npz');r=byindex[index];mask=c['frozen_fit_mask'];selected=np.flatnonzero(mask);hit=c['hit'];positions=c['window_positions'];valid=c['window_token_valid']
        optical=item['optical'].numpy().transpose(1,2,0);sar=item['sar_aug'][0].numpy();fig,axs=plt.subplots(1,3,figsize=(13.2,4.5),gridspec_kw={'width_ratios':[1,1,1]})
        axs[0].imshow(optical);axs[1].imshow(sar,cmap='gray',vmin=0,vmax=1)
        axs[1].contourf(~item['sar_aug_valid_mask'].numpy(),levels=[.5,1.5],colors=['red'],alpha=.18)
        shown=selected[np.unique(np.linspace(0,len(selected)-1,min(10,len(selected))).astype(int))] if len(selected) else selected
        for k in shown:
            src=c['source'][k];dst=c['target'][k];truth=c['nominal_truth'][k];color='#21a354' if hit[k] else '#ef8a17'
            axs[0].plot(*src,'o',color=color,markersize=3);axs[1].plot(*dst,'x',color=color,markersize=4);axs[1].plot(*truth,'+',color='cyan',markersize=5)
            fig.add_artist(ConnectionPatch(src,dst,coordsA='data',coordsB='data',axesA=axs[0],axesB=axs[1],color=color,alpha=.35,linewidth=.6))
        axs[0].set_title('Optical: actual fitted sources');axs[1].set_title('SAR augmented: predicted x / nominal +')
        if len(selected):
            k=int(selected[0]);center=c['coarse_center'][k];pos=positions[k];v=valid[k];hull=convex_hull(pos[v])
            axs[2].imshow(sar,cmap='gray',vmin=0,vmax=1);axs[2].scatter(pos[~v,0],pos[~v,1],marker='x',color='red',s=20,label='invalid sampled token')
            axs[2].scatter(pos[v,0],pos[v,1],color='royalblue',s=14,label='valid sampled token')
            if len(hull)>=3:axs[2].add_patch(Polygon(hull,closed=True,fill=False,edgecolor='lime',linewidth=1.4))
            axs[2].scatter(*c['target'][k],marker='x',color='yellow',s=55,label='original prediction');axs[2].scatter(*c['nominal_truth'][k],marker='+',color='cyan',s=75,label='nominal GT')
            axs[2].set_xlim(center[0]-9,center[0]+9);axs[2].set_ylim(center[1]+9,center[1]-9)
            axs[2].set_title(f'First fit candidate (order fixed), hit={bool(hit[k])}')
            axs[2].legend(fontsize=6,loc='upper left')
        for ax in axs[:2]:ax.set_xlim(-.5,255.5);ax.set_ylim(255.5,-.5)
        for ax in axs:ax.set_aspect('equal');ax.tick_params(labelsize=7)
        base=r['branches']['original_prediction']['evaluation']['rmse_px'];oracle=r['branches']['inside_actual_support_nominalGT_target_only']['evaluation']['rmse_px']
        fig.suptitle(f"{item['pair_id']} | fixed fit={len(selected)}, hit={int(hit[mask].sum())} | probeRMSE A={base:.3f}, B={oracle:.3f}px",fontsize=10)
        fig.text(.02,.015,'Green lines: inside actual valid-token hull; orange: not replaceable. Display subsampling does not affect fitting. GT is nominal.',fontsize=8)
        fig.tight_layout(rect=(0,.035,1,.94));path=private/f'case_{index:04d}.png';fig.savefig(path,dpi=150);plt.close(fig)
        manifest.append(dict(index=index,pair_id=item['pair_id'],path=str(path),display_candidate_selection='10uniformpositions in originalfit order; zoom firstfit candidate',candidate_selection_does_not_read_outcome_orGT=True))
    (out/'visualization_manifest.json').write_text(json.dumps(dict(fixed_indices=indices,selection='first6 originalT05diagnostic_indices; no result-based choice',cases=manifest,private_images=True),indent=2))
    # Public synthetic boundary/convex support diagrams contain no dataset pixels.
    fig,axes=plt.subplots(1,3,figsize=(10,3))
    examples=[(np.array([[0,0],[2,0],[0,2]]),np.array([1.8,1.8]),'Inside bbox, outside convex hull'),
        (np.array([[0,0],[2,0],[0,2]]),np.array([.7,.8]),'Continuous expectation, inside hull'),
        (np.array([[1.5,1.5],[9.5,1.5],[9.5,9.5],[1.5,9.5]]),np.array([0,3.5]),'Image-border valid-token support')]
    for ax,(points,truth,title) in zip(axes,examples):
        hull=convex_hull(points);ax.add_patch(Polygon(hull,closed=True,alpha=.15,color='blue'));ax.scatter(points[:,0],points[:,1],s=25);ax.scatter(*truth,marker='x',color='red',s=50);ax.set_title(title,fontsize=8);ax.set_aspect('equal');ax.grid(alpha=.2);ax.invert_yaxis()
    fig.tight_layout();fig.savefig(out/'support_geometry_tests.png',dpi=160);plt.close(fig)
    print('Rendered fixed6 diagnostic cases',flush=True)


if __name__=='__main__':main()
