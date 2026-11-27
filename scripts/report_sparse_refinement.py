"""Summarize the preregistered accuracy/latency variants and paired outcomes."""
import csv,json
from pathlib import Path
import numpy as np
from sglm.sparse_refinement_experiment import OUT,PRIVATE
from sglm.experiment_io import write_csv


def read(name):return list(csv.DictReader((OUT/name).open()))


def main():
    accuracy=json.loads((OUT/'accuracy_summary.json').read_text());latency=read('latency_summary.csv');summary=[]
    stages=read('stage_diagnostics.csv');stage_summary=[]
    for a in accuracy:
        t=next(r for r in latency if r['variant']==a['variant']);summary.append(dict(**a,**{k:v for k,v in t.items() if k!='variant'}))
        group=[r for r in stages if r['variant']==a['variant']]
        stage_summary.append(dict(variant=a['variant'],**{key:float(np.median([float(r[key]) for r in group])) for key in ['encode_ms','coarse_ms','selection_gather_ms','window_extraction_ms','refinement_remainder_ms','local_including_windows_ms','copy_ms','geometry_ms']}))
    write_csv(OUT/'summary.csv',summary);write_csv(OUT/'stage_summary.csv',stage_summary)
    pairs={r['variant']:json.loads((PRIVATE/f'{r["variant"]}_validation.json').read_text())['pairs'] for r in accuracy}
    changes=[]
    for reference,candidate in [('A0_all',r['variant']) for r in accuracy[1:]]+[('B0_global128','B1_coverage128')]:
        a=np.array([r['registration_success'] for r in pairs[reference]]);b=np.array([r['registration_success'] for r in pairs[candidate]])
        changes.append(dict(reference=reference,candidate=candidate,failure_to_success=int((~a&b).sum()),success_to_failure=int((a&~b).sum()),
                            both_success=int((a&b).sum()),both_failure=int((~a&~b).sum()),net_success_change=int(b.sum()-a.sum())))
    write_csv(OUT/'paired_outcome_changes.csv',changes)
    deltas=read('latency_paired_differences.csv');pd=[]
    for name in pairs:
        g=[r for r in deltas if r['variant']==name];values=[float(r['paired_delta_ms']) for r in g]
        pd.append(dict(variant=name,mean_pair_round_delta_ms=float(np.mean(values)),median_pair_round_delta_ms=float(np.median(values)),
             round_mean_delta_ms=json.dumps([float(np.mean([float(r['paired_delta_ms']) for r in g if int(r['round'])==i])) for i in range(3)])))
    write_csv(OUT/'latency_paired_summary.csv',pd)
    ref=read('original_reference_paired.csv');rr=[]
    for name in ['A0_all','B0_global128']:
        for boundary in ['network','complete']:
            g=[r for r in ref if r['variant']==name and r['boundary']==boundary]
            rr.append(dict(variant=name,boundary=boundary,mean_paired_delta_ms=float(np.mean([float(r['paired_delta_ms']) for r in g])),
                median_paired_delta_ms=float(np.median([float(r['paired_delta_ms']) for r in g]))))
    write_csv(OUT/'original_reference_paired_summary.csv',rr)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    offsets={'A0_all':(-28,-17),'A1_prefix256':(4,4),'A2_prefix128':(4,4),'A3_prefix64':(4,4),'B0_global128':(4,8),'B1_coverage128':(-28,-17)}
    for a in accuracy:
        t=next(r for r in latency if r['variant']==a['variant'])
        for ax,xkey in zip(axes,['network_median_ms','complete_median_ms']):
            ax.scatter(float(t[xkey]),100*a['success_rate@5px'],label=a['variant']);ax.annotate(a['variant'].split('_')[0],(float(t[xkey]),100*a['success_rate@5px']),fontsize=8,xytext=offsets[a['variant']],textcoords='offset points')
    for ax,title in zip(axes,['Network only','Network + copy + affine RANSAC']):
        ax.set(xlabel='No-hook pooled median latency (ms)',ylabel='SR@5px on952 fixed pairs (%)',title=title);ax.grid(alpha=.3);ax.margins(x=.13,y=.12)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('Same frozen B0, dense coarse1024; refinement budget only. Timing across24 pairs,3 rounds.',fontsize=10)
    fig.tight_layout(rect=(0,.12,1,1));fig.savefig(OUT/'accuracy_latency.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
