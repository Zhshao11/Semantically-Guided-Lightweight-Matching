"""Measured reference table and explicitly proposed next-development budgets."""
import csv,json
from pathlib import Path
import numpy as np
from sglm.experiment_io import write_csv


def main():
    p=json.loads(Path('configs/external_references_v1.json').read_text());out=Path(p['output'])
    assert json.loads((out/'acceptance.json').read_text())['passed']
    timed={r['method']:r for r in csv.DictReader((out/'latency_summary.csv').open())};rows=[]
    for name in p['methods']:
        accuracy=json.loads((out/f'{name}_accuracy.json').read_text());t=timed[name]
        mem=json.loads((out/f'{name}_memory.json').read_text())
        rows.append(dict(method=name,SR3=accuracy['success_rate@3px'],SR5=accuracy['success_rate@5px'],
            success3_count=round(accuracy['success_rate@3px']*952),success5_count=round(accuracy['success_rate@5px']*952),
            solved_pairs=accuracy['solved_pairs'],solver_failures=accuracy['failed_pairs'],registration_failures5=accuracy['registration_failures5'],
            solved_median_rmse_px=accuracy['solved_median_rmse_px'],solved_p90_rmse_px=accuracy['solved_p90_rmse_px'],
            latency_mean_ms=float(t['latency_mean_ms']),latency_median_ms=float(t['latency_median_ms']),latency_p95_ms=float(t['latency_p95_ms']),
            round_medians_ms=t['round_medians_ms'],round_median_std_ms=float(t['round_median_std_ms']),
            peak_allocated_MiB=mem['peak_allocated_MiB'],cross_method_timing_peak_MiB=float(t['peak_allocated_MiB']),actual_parameters=accuracy['actual_parameters_total'],
            mean_valid_matches=accuracy['mean_valid_matches'],native_precision=p['method_protocols'][name]['precision'],
            internal_size='256' if name=='T05' else '640' if name=='XoFTR' else '640 outer,560 coarse,864 upsample',
            training='OS7912 dedup train' if name=='T05' else 'KAIST pretrain + MegaDepth; zero-shot SAR' if name=='XoFTR' else 'MegaDepth/MD-syn + RoMa/DINO pretraining; zero-shot SAR',
            overlap_status='known OS train; unchanged val' if name=='T05' else 'OS not declared; exact per-image overlap unverified',
            protocol_note='same source inputs/evaluator/affine/timing boundary; native effective size and precision differ'))
    write_csv(out/'external_comparison.csv',rows)
    base=rows[0];accuracy_floor=dict(SR3=base['SR3'],SR5=base['SR5'],success3_min_count=base['success3_count'],success5_min_count=base['success5_count'],denominator=952)
    budget=dict(version='development-budget-v1-after-external-measurement',based_on_measured_T05=base,
        accuracy_floor=accuracy_floor,
        acceptance=dict(latency_median_ms_max=base['latency_median_ms'],latency_p95_ms_max=base['latency_p95_ms'],peak_allocated_MiB_max=base['peak_allocated_MiB']),
        development_targets=dict(latency_median_ms_max=.8*base['latency_median_ms'],paired_latency_ratio_max=.8,
            latency_p95_ms_max=base['latency_p95_ms'],peak_allocated_MiB_max=.8*base['peak_allocated_MiB']),
        target_basis='Accuracy/memory/no-regression ceilings equal measured T05; meaningful engineering stretch chosen AFTER measurements as20% improvement, not a measured optimum or achieved result. Recheck paired24 on same idle5090 over3rounds; keep full952 accuracy gate.',
        same_protocol_required=True,not_claiming_domain_general_SOTA=True,no_new_training_started=True)
    (out/'development_budget.json').write_text(json.dumps(budget,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    for r in rows:
        for ax,key in zip(axes,['latency_median_ms','peak_allocated_MiB']):
            ax.scatter(r[key],r['SR5']*100);ax.annotate(r['method'],(r[key],r['SR5']*100),xytext=(5,5),textcoords='offset points',fontsize=9)
    for ax,label in zip(axes,['End-to-end median latency (ms)','Peak PyTorch allocated memory (MiB)']):
        ax.set_xscale('log');ax.set(xlabel=label,ylabel='SR@5px (%)',ylim=(-3,65));ax.grid(alpha=.3);ax.margins(x=.3)
    fig.suptitle('Native pretrained configurations on identical952 OS pairs; internal sizes/precision differ',fontsize=10)
    fig.tight_layout();fig.savefig(out/'external_accuracy_cost.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(rows=rows,budget=budget),indent=2))


if __name__=='__main__':main()
