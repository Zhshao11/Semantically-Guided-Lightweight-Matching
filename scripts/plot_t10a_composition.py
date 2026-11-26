"""Render saved class-composition tables; no pixel imagery or new selections."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path('reports/t10_common_support_v1');names=['Background','Building','Road','Water','Barren','Forest','Agricultural']
    rows=list(csv.DictReader((out/'joint_class_results.csv').open()))
    primary=[r for r in rows if r['control']=='distance_weight']
    matrices=[]
    for field in ['hard_mean','random_mean','delta']:
        matrices.append(np.array([float(r[field]) if r[field] else np.nan for r in primary]).reshape(7,7))
    fig,axes=plt.subplots(1,3,figsize=(15,5.4))
    for ax,m,title in zip(axes,matrices,['Hard: mean cosine','Matched random: mean cosine','Within-cell difference (hard - random)']):
        limit=np.nanmax(np.abs(m)) if title.startswith('Within') else 1
        im=ax.imshow(m,cmap='RdBu_r' if title.startswith('Within') else 'viridis',vmin=-limit if title.startswith('Within') else 0,vmax=limit)
        ax.set_xticks(range(7),names,rotation=65,ha='right');ax.set_yticks(range(7),names);ax.set_title(title,fontsize=10)
        ax.set_xlabel('Candidate pseudo-class');ax.set_ylabel('Query pseudo-class');fig.colorbar(im,ax=ax,fraction=.045)
    fig.suptitle('Frozen common support; teacher pseudo-classes are not ground truth. Blank = undefined.',fontsize=11)
    fig.tight_layout();fig.savefig(out/'joint_class_means.png',dpi=160);plt.close(fig)
    rows=list(csv.DictReader((out/'class_composition.csv').open()))
    fig,ax=plt.subplots(figsize=(9,4.8));x=np.arange(7)
    for j,group in enumerate(['hard','uncontrolled','distance_weight','distance_weight_class']):
        r=[r for r in rows if r['group']==group and r['query_pseudoclass']=='ALL']
        ax.bar(x+(j-1.5)*.2,[100*float(r['weighted_share']) for r in r],.2,label=group)
    ax.set_xticks(x,names,rotation=30,ha='right');ax.set_ylabel('Hierarchical candidate pseudo-class mass (%)')
    ax.set_title('Same frozen queries and hard slots; random repeats split each slot weight')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'candidate_composition.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
