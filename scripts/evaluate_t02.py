"""Inference only uses image tensors/masks; evaluation GT enters the independent evaluator."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from sglm.t02.data import AffinePairs256
from sglm.t02.model import B0Full
from sglm.evaluation import estimate_from_correspondences,fixed_synthetic_probes,evaluate_transform,summarize_evaluations


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--plan',required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--split',choices=['train','val'],default='val')
    ap.add_argument('--output',required=True)
    ap.add_argument('--device',default='cuda')
    a=ap.parse_args()
    torch.set_num_threads(4)
    checkpoint=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    config=checkpoint['config']
    if hashlib.sha256(Path(a.plan).read_bytes()).hexdigest()!=checkpoint['plan_sha256']:
        raise ValueError('Evaluation plan differs from training plan')
    model=B0Full(config).to(a.device).eval()
    model.load_state_dict(checkpoint['model'])
    dataset=AffinePairs256(a.root,a.plan,a.split)
    results=[]
    for item in dataset:
        images=[item[k][None].to(a.device) for k in ('optical','sar_aug','optical_valid_mask','sar_aug_valid_mask')]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            prediction=model.predict(*images)[0]
        # A_forward is deliberately unavailable to model.predict.
        source,target,scores=[prediction[k].float().cpu().numpy() for k in ('source','target','score')]
        estimated=estimate_from_correspondences(source,target,scores,config['evaluation'],seed=config['seed'])
        probes,reference=fixed_synthetic_probes(item['A_forward'].numpy(),grid=config['evaluation']['probe_grid'],
            optical_mask=item['optical_valid_mask'].numpy(),sar_aug_mask=item['sar_aug_valid_mask'].numpy())
        evaluated=evaluate_transform(estimated['matrix'],probes,reference,config['evaluation']['success_thresholds_px'],status=estimated['status'])
        results.append(dict(pair_id=item['pair_id'],estimation=estimated,evaluation=evaluated,
            local_window_source=prediction['local_window_source'],predicted_source=source.tolist(),predicted_target=target.tolist(),
            A_forward=item['A_forward'].tolist(),probe_source_xy=probes.tolist(),probe_target_xy=reference.tolist()))
    summary=summarize_evaluations([r['evaluation'] for r in results],config['evaluation']['success_thresholds_px'])
    report=dict(protocol=config['protocol'],split=a.split,checkpoint_step=checkpoint['step'],summary=summary,
        inference_reads_ground_truth=False,fine_windows='predicted_coarse_only',evaluation_probe_independent_of_fitting=True,
        thresholds_predeclared=config['evaluation'],results=results)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(split=a.split,step=checkpoint['step'],summary=summary),indent=2),flush=True)


if __name__=='__main__':
    main()
