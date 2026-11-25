"""Prepared B0 timing protocol. Defaults to dry-run; no formal numbers on shared GPUs."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from sglm.experiment_io import require_exclusive_gpu,fixed_timing_plan


def query(arguments):
    return subprocess.run(['nvidia-smi',*arguments],check=True,capture_output=True,text=True).stdout


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--plan',required=True)
    ap.add_argument('--root',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--split',choices=['train','val'],default='val');ap.add_argument('--pairs',type=int,default=16)
    ap.add_argument('--gpu-uuid');ap.add_argument('--execute',action='store_true')
    ap.add_argument('--warmup',type=int,default=5);ap.add_argument('--repeats',type=int,default=20)
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if not a.execute:
        result=dict(mode='dry-run',latency_measured=False,arguments=vars(a),
            protocol='batch1 BF16 eval; fixed saved transforms; exclusive GPU UUID; synchronize each stage',
            stages=['data_read_decode_warp','host_to_device','optical_encode','sar_encode','coarse_correlation_and_decode','local_refinement','correspondence_to_cpu','affine_RANSAC'],
            end_to_end='separate uninstrumented predict + CPU RANSAC; excludes data/transfer',
            required='Stable candidate checkpoint, pinned comparison adapters/protocol, unused physical GPU')
        (out/'timing_protocol.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return
    if a.warmup<5 or a.repeats<20 or a.pairs<1:raise ValueError('Need >=5 warmup, >=20 repeats and positive pairs')
    def guard():
        known=query(['--query-gpu=uuid','--format=csv,noheader']).split()
        processes=query(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'])
        require_exclusive_gpu(a.gpu_uuid or '',processes,os.getpid(),known)
    guard()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid
    import numpy as np
    import torch
    from sglm.t02.data import AffinePairs256
    from sglm.t02.model import B0Full
    from sglm.evaluation import estimate_from_correspondences
    torch.set_num_threads(4);torch.manual_seed(20261125)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=saved['config']
    model=B0Full(cfg).cuda().eval();model.load_state_dict(saved['model'])
    fixed_plan=fixed_timing_plan(json.loads(Path(a.plan).read_text()),cfg,a.split)
    dataset=AffinePairs256(a.root,fixed_plan,split=a.split,preload=False)
    if a.pairs>len(dataset):raise ValueError('Not enough fixed pairs')
    def timed(fn):
        torch.cuda.synchronize();begun=time.perf_counter();value=fn();torch.cuda.synchronize()
        return value,(time.perf_counter()-begun)*1000
    records=[]
    def estimate(prediction):
        arrays=[prediction[k].float().cpu().numpy() for k in ('source','target','score')]
        return estimate_from_correspondences(*arrays,cfg['evaluation'],seed=cfg['seed'])
    with torch.inference_mode():
        for i in range(a.pairs):
            for repeat in range(-a.warmup,a.repeats):
                guard()  # Recheck between measurements; abort if another process arrives.
                item,read_ms=timed(lambda:dataset[i])
                args,transfer_ms=timed(lambda:[item[k][None].cuda() for k in ('optical','sar_aug','optical_valid_mask','sar_aug_valid_mask')])
                encode=[];started=[]
                def pre(*unused):torch.cuda.synchronize();started.append(time.perf_counter())
                def post(*unused):torch.cuda.synchronize();encode.append((time.perf_counter()-started.pop())*1000)
                h1=model.backbone.register_forward_pre_hook(pre);h2=model.backbone.register_forward_hook(post)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output,forward_ms=timed(lambda:model(*args))
                    h1.remove();h2.remove()
                    (source,target,score),decode_ms=timed(lambda:model.coarse_matches(output,0))
                    refined,fine_ms=timed(lambda:model.local(output['optical_fine'],output['sar_fine'],source,target,args[2],args[3]))
                    valid=refined.get('valid',torch.ones(len(source),device='cuda',dtype=torch.bool))
                    pred=dict(source=source[valid],target=refined['target'][valid],score=score[valid])
                    arrays,cpu_ms=timed(lambda:[pred[k].float().cpu().numpy() for k in ('source','target','score')])
                    fit,ransac_ms=timed(lambda:estimate_from_correspondences(*arrays,cfg['evaluation'],seed=cfg['seed']))
                    _,e2e_ms=timed(lambda:estimate(model.predict(*args)[0]))
                if repeat>=0:
                    records.append(dict(pair_id=item['pair_id'],repeat=repeat,data_read_decode_warp_ms=read_ms,
                        host_to_device_ms=transfer_ms,optical_encode_ms=encode[0],sar_encode_ms=encode[1],
                        coarse_correlation_and_decode_ms=max(0,forward_ms-sum(encode))+decode_ms,
                        local_refinement_ms=fine_ms,correspondence_to_cpu_ms=cpu_ms,affine_RANSAC_ms=ransac_ms,
                        end_to_end_prediction_RANSAC_ms=e2e_ms,coarse_matches=len(source),fine_matches=len(arrays[0]),
                        inliers=fit['inliers'],solver_status=fit['status']))
    guard()
    metrics={k:dict(median=float(np.median([r[k] for r in records])),p90=float(np.percentile([r[k] for r in records],90))) for k in records[0] if k.endswith('_ms')}
    report=dict(mode='executed',records=records,metrics=metrics,gpu_uuid=a.gpu_uuid,
        device=query(['--query-gpu=uuid,name,driver_version','--format=csv,noheader']),torch=torch.__version__,
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest(),
        precision='BF16 autocast; logits/geometry FP32 or FP64 per existing implementation',batch_size=1,
        warmup_per_pair=a.warmup,repeats_per_pair=a.repeats,eval=True,
        caveat='Stage synchronization adds overhead; end-to-end uses separate uninstrumented forward. Repeats are correlated; not independent scenes.',
        original_config=cfg)
    (out/'timing_results.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
