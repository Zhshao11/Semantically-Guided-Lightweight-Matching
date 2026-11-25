"""Read-only current training snapshots, with paired augmentation-stream checks."""
import argparse
import json
import time
from pathlib import Path


def json_lines(path):
    if not path.exists(): return []
    records=[]
    for line in path.read_text().splitlines():
        try:records.append(json.loads(line))
        except json.JSONDecodeError:break  # A live writer may be completing its last line.
    return records


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    runs={}
    for label,name in [('main','t03_main_lr1e3_v1'),('lr_control','t03_control_lr2e4_v1')]:
        p=Path('runs')/name
        status=json.loads((p/'status.json').read_text())
        env=json.loads((p/'environment.json').read_text()) if (p/'environment.json').exists() else None
        epochs=json_lines(p/'epoch_summaries.jsonl');validations=json_lines(p/'validation_summaries.jsonl')
        runs[label]=dict(run_directory=str(p),status=status,environment=env,epoch_summaries=epochs,
            validation_summaries=validations,checkpoint_exists={n:(p/n).exists() for n in ('initial.pt','last.pt','best.pt')})
    common=min(len(runs['main']['epoch_summaries']),len(runs['lr_control']['epoch_summaries']))
    agreement=[]
    for i in range(common):
        x,y=runs['main']['epoch_summaries'][i],runs['lr_control']['epoch_summaries'][i]
        equal=x['sample_and_augmentation_stream_sha256']==y['sample_and_augmentation_stream_sha256']
        if not equal:raise AssertionError('LR-control presentation stream differs')
        agreement.append(dict(epoch=x['epoch'],matched=True,sha256=x['sample_and_augmentation_stream_sha256']))
    result=dict(snapshot_unix=time.time(),runs=runs,shared_epoch_stream_checks=agreement,
        initialization_states_equal=(runs['main']['environment']['initial_state_sha256']==runs['lr_control']['environment']['initial_state_sha256']),
        total_budget_epochs=20,total_budget_steps=39560,budget_is_observation_not_convergence=True)
    (out/'live_snapshot.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
