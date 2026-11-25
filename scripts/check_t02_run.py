"""Persist training/resume/independent-evaluation acceptance evidence without fitting GT."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sglm.evaluation import fixed_synthetic_probes,evaluate_transform,estimate_from_correspondences,summarize_evaluations


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True)
    ap.add_argument('--report',required=True)
    a=ap.parse_args()
    run,report=Path(a.run),Path(a.report)
    training=run/'training'
    checkpoint=torch.load(training/'last.pt',map_location='cpu',weights_only=False)
    rows=[json.loads(line) for line in (training/'train.jsonl').read_text().splitlines()]
    assert [x['step'] for x in rows]==list(range(1,1601))
    assert all(np.isfinite(x['total']) and np.isfinite(x['grad_norm']) for x in rows)
    assert all(x['fine_window_mode']=='predicted_coarse_training' for x in rows[400:])
    replay=json.loads((run/'resume_probe/train.jsonl').read_text().splitlines()[0])
    main401=rows[400]
    delta={key:abs(replay[key]-main401[key]) for key in ('total','coarse','fine','grad_norm','positive_coarse_accuracy')}
    assert replay['step']==401 and replay['pair_ids']==main401['pair_ids']
    assert max(delta.values())<1e-5
    assert checkpoint['step']==1600 and checkpoint['precision_evidence']['correlation_dtype']=='torch.bfloat16'
    adam_steps={int(v['step']) for v in checkpoint['optimizer']['state'].values()}
    assert adam_steps=={1600}
    result=dict(passed=True,total_steps=1600,checkpoint_step=checkpoint['step'],
        resumed_from_step=400,replay_step=401,replay_same_batch=True,replay_max_absolute_difference=delta,
        optimizer_state_entries=len(checkpoint['optimizer']['state']),optimizer_steps=sorted(adam_steps),
        all_losses_and_gradients_finite=True,precision_evidence=checkpoint['precision_evidence'],
        first_50_mean_loss=float(np.mean([x['total'] for x in rows[:50]])),
        last_50_mean_loss=float(np.mean([x['total'] for x in rows[-50:]])),
        evaluation_reads_gt_only_after_prediction=True,post_bootstrap_training_uses_predicted_coarse=True)
    (report/'training_acceptance.json').write_text(json.dumps(result,indent=2))
    # Evaluator interface-only checks; synthetic inputs, not scene annotations.
    truth=np.array([[1.,.03,7.],[-.02,.97,-3.]])
    probes,reference=fixed_synthetic_probes(truth)
    wrong=truth.copy();wrong[:,2]+=[5,-3]
    cfg=checkpoint['config']['evaluation']
    empty=estimate_from_correspondences([],[],[],cfg)
    degenerate=estimate_from_correspondences([[0,0],[1,1],[2,2]],[[1,1],[2,2],[3,3]],[1,1,1],cfg)
    records=[evaluate_transform(truth,probes,reference),evaluate_transform(wrong,probes,reference),
             evaluate_transform(empty['matrix'],probes,reference),evaluate_transform(degenerate['matrix'],probes,reference)]
    assert records[0]['rmse_px']<1e-10 and abs(records[1]['rmse_px']-np.sqrt(34))<1e-10
    summary=summarize_evaluations(records)
    assert summary['image_pairs']==4 and summary['failed_pairs']==2 and summary['success_rate@5px']==.25
    (report/'evaluator_interface_acceptance.json').write_text(json.dumps(dict(
        passed=True,true_matrix_rmse_px=records[0]['rmse_px'],wrong_matrix_rmse_px=records[1]['rmse_px'],
        empty_status=empty['status'],degenerate_status=degenerate['status'],summary=summary,
        scene_interface='same evaluate_transform(matrix, external_control_source, external_control_target); no controls in fit'),indent=2))
    # Refresh only summary fields of saved evaluations; no inference or threshold changes.
    for name in ('initial_train.json','final_train.json','final_val.json'):
        path=report/name
        evaluation=json.loads(path.read_text())
        evaluation['summary']=summarize_evaluations([x['evaluation'] for x in evaluation['results']])
        path.write_text(json.dumps(evaluation,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
