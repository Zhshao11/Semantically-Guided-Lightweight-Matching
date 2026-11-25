"""Export completed observations without modifying runs or choosing a new model."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sglm.experiment_io import complete_jsonl,flatten_diagnostic,write_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='append',required=True)
    ap.add_argument('--diagnostics',action='append',default=[]);ap.add_argument('--output',required=True)
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    reports=[];train_rows=[];val_rows=[];pair_rows=[]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for run_path in a.run:
        run=Path(run_path);name=run.name
        epochs=complete_jsonl(run/'epoch_summaries.jsonl');vals=complete_jsonl(run/'validation_summaries.jsonl')
        status=json.loads((run/'status.json').read_text());env=json.loads((run/'environment.json').read_text())
        audits=[]
        for val in vals:
            detail_path=run/f"validation_epoch_{val['epoch']:03d}.json"
            detail=json.loads(detail_path.read_text());pairs=detail['pairs']
            if len(pairs)!=val['image_pairs']:raise ValueError('Validation denominator mismatch')
            if not detail['model_eval'] or not detail['BN_buffers_unchanged']:raise ValueError('Invalid inference state')
            audits.append(dict(epoch=val['epoch'],pairs=len(pairs),model_eval=True,BN_buffers_unchanged=True))
            for record in pairs:
                pair_rows.append(dict(run=name,epoch=val['epoch'],pair_id=record['pair_id'],
                    **{k:v for k,v in record['evaluation'].items() if not isinstance(v,(dict,list))},
                    **{'success@'+k+'px':v for k,v in record['evaluation']['success'].items()},
                    geometry_status=record['estimation']['status']))
        train_rows.extend(dict(run=name,**r) for r in epochs)
        val_rows.extend(dict(run=name,**{k:v for k,v in r.items() if not isinstance(v,dict)}) for r in vals)
        reports.append(dict(run=name,status=status,environment=env,train=epochs,validation=vals,audits=audits))
        if epochs:axes[0].plot([r['epoch'] for r in epochs],[r['total'] for r in epochs],label=name)
        if vals:
            axes[1].plot([r['epoch'] for r in vals],[r['success_rate@5px'] for r in vals],label=name)
            axes[2].plot([r['epoch'] for r in vals],[r['failed_pairs'] for r in vals],label=name)
    for ax,title in zip(axes,['Mean training loss','Validation SR@5px (failures included)','Validation solver failures']):
        ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'observation_curves.png',dpi=160);plt.close(fig)
    write_csv(out/'epochs.csv',train_rows);write_csv(out/'validations.csv',val_rows);write_csv(out/'validation_pairs.csv',pair_rows)
    layered=[]
    for path in a.diagnostics:
        value=json.loads(Path(path).read_text())
        records=value if isinstance(value,list) else value.get('pairs',value.get('records',[]))
        layered.extend(dict(source_file=str(path),**flatten_diagnostic(r)) for r in records)
    write_csv(out/'layered_diagnostics.csv',layered)
    snapshot=dict(observed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),runs=reports,
        reliability_labels_generated=False,test_used=False,layered_rows=len(layered),
        note='Completed records only; no checkpoint selection or training control performed')
    (out/'summary.json').write_text(json.dumps(snapshot,indent=2))
    print(json.dumps(dict(runs=len(reports),validation_rows=len(val_rows),pair_rows=len(pair_rows),layered_rows=len(layered))))


if __name__=='__main__':main()
