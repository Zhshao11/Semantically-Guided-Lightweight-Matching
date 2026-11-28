"""Frozen-map evidence and pooled/error plots, no matcher inference or strategy selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    p=json.loads(Path('configs/heatmap_readout_v1.json').read_text());out=Path(p['output']);private=Path(p['private']);source=json.loads(Path(p['source_freeze']).read_text());cfg=source['config'];side=2*cfg['model']['local_radius_fine_cells']+1
    subsets=json.loads(Path('reports/t05_frozen_v1/subsets.json').read_text());indices=subsets['diagnostic_indices'][:6];manifest=[]
    for index in indices:
        c=np.load(Path(p['source_candidates'])/f'{index:04d}.npz');r=np.load(private/f'{index:04d}.npz');raw_indices=np.flatnonzero(c['frozen_fit_mask']);assert len(raw_indices)
        raw_index=int(raw_indices[0]);prob=c['local_heatmap'][raw_index].reshape(side,side);valid=c['window_token_valid'][raw_index].reshape(side,side);center=c['coarse_center'][raw_index]
        nominal=(r['nominal_truth'][0]-center)/2+side//2;peak=int(r['peak_index'][0]);pr,pc=peak//side,peak%side;fig,axs=plt.subplots(1,3,figsize=(11,3.8))
        for ax,g,color in zip(axs,['A','B','C'],['#1f77b4','#ff7f0e','#2ca02c']):
            im=ax.imshow(np.ma.masked_where(~valid,prob),cmap='viridis',vmin=0,vmax=float(prob.max()),origin='upper')
            for y,x in zip(*np.where(~valid)):ax.plot(x,y,'x',color='red',markersize=5)
            xy=(r[g][0]-center)/2+side//2;ax.scatter(*xy,c=color,s=70,marker='o',edgecolors='white',label=g+' readout')
            if -.5<=nominal[0]<=side-.5 and -.5<=nominal[1]<=side-.5:ax.scatter(*nominal,c='red',s=110,marker='*',label='nominal GT')
            if g=='C':
                x0=max(pc-1,0)-.5;y0=max(pr-1,0)-.5;x1=min(pc+1,side-1)+.5;y1=min(pr+1,side-1)+.5
                ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='white',linewidth=1.5))
            ax.set_title(f"{g}: error={float(r[g+'_error'][0]):.3f}px",fontsize=10);ax.set_xlim(-.5,side-.5);ax.set_ylim(side-.5,-.5);ax.set_xticks(range(side));ax.set_yticks(range(side));ax.legend(fontsize=7,loc='lower left')
        fig.suptitle(f"{str(c['pair_id'])}, first fit ID={int(r['candidate_id'][0])}, actual-support hit={bool(r['hit'][0])}",fontsize=10)
        fig.text(.02,.015,'All panels show the SAME original probabilities. White box: fixed peak3x3 fine-grid region; 1gridstep=2originalpx. No new pixels/windows.',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.92));path=private/f'heatmap_case_{index:04d}.png';fig.savefig(path,dpi=150);plt.close(fig)
        manifest.append(dict(index=index,pair_id=str(c['pair_id']),candidate_id=int(r['candidate_id'][0]),selection='firstfitcandidate in frozen originalorder, noGT/error selection',path=str(path),nominal_GT_in_display=bool(-.5<=nominal[0]<=side-.5 and -.5<=nominal[1]<=side-.5)))
    (out/'heatmap_visualization_manifest.json').write_text(json.dumps(dict(fixed_indices=indices,cases=manifest,private=True,no_network=True),indent=2))
    errors={g:[] for g in ['A','B','C']};mass=[];ratios=[]
    for index in range(952):
        r=np.load(private/f'{index:04d}.npz');hit=r['hit']
        for g in errors:errors[g].extend(r[g+'_error'][hit])
        mass.extend(r['truth_mass'][hit]);ratios.extend(r['secondary_ratio'][hit])
    fig,axs=plt.subplots(1,3,figsize=(12,3.5))
    for g,label in [('A','A full expectation'),('B','B max peak'),('C','C peak3x3 expectation')]:
        x=np.sort(errors[g]);axs[0].plot(x,np.arange(1,len(x)+1)/len(x),label=label)
    axs[0].set(xlabel='Correspondence error (original px)',ylabel='Cumulative fraction',xlim=(0,17));axs[0].legend(fontsize=7);axs[0].grid(alpha=.2)
    axs[1].hist(mass,bins=np.linspace(0,1,26),color='#5c8db8',edgecolor='white');axs[1].set(xlabel='Original probability mass near nominal GT',ylabel='Inside-support correspondences')
    axs[2].hist(ratios,bins=np.linspace(0,1,26),color='#b4804e',edgecolor='white');axs[2].set(xlabel='Secondary/main peak probability ratio',ylabel='Inside-support correspondences')
    fig.suptitle('Same45974 nominal-GT-inside actual-support correspondences; correlated within952 pairs',fontsize=10);fig.tight_layout();fig.savefig(out/'readout_error_and_mass.png',dpi=160);plt.close(fig)
    records=json.loads((out/'pair_results.json').read_text());a=np.array([r['branches']['A']['evaluation']['rmse_px'] for r in records]);c=np.array([r['branches']['C']['evaluation']['rmse_px'] for r in records]);colors=np.where((a>=5)&(c<5),'#2ca02c',np.where((a<5)&(c>=5),'#d62728','#999999'))
    fig,ax=plt.subplots(figsize=(5,4.5));ax.scatter(a,c,c=colors,s=8,alpha=.6);hi=max(a.max(),c.max());lo=min(a.min(),c.min())*.8;ax.plot([lo,hi],[lo,hi],color='black',linewidth=.7);ax.axvline(5,color='blue',linestyle='--',linewidth=.7);ax.axhline(5,color='blue',linestyle='--',linewidth=.7)
    ax.set_xscale('log');ax.set_yscale('log');ax.set(xlabel='A probe RMSE (px, log)',ylabel='C probe RMSE (px, log)',title='Green: rescued5px; red: lost5px; all952 pairs');fig.tight_layout();fig.savefig(out/'paired_C_vs_A.png',dpi=160);plt.close(fig)
    print('Rendered fixed6 heatmap cases and same45974 distributions',flush=True)


if __name__=='__main__':main()
