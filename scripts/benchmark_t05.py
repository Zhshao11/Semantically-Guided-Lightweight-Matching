"""Exclusive-GPU, no-hook end-to-end timings for the frozen uniform-budget screen."""
import argparse
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import random
import subprocess
import time
from sglm.experiment_io import require_exclusive_gpu,write_csv


def query(args):return subprocess.run(['nvidia-smi',*args],check=True,capture_output=True,text=True).stdout


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--gpu-uuid',required=True)
    ap.add_argument('--frozen',default='reports/t05_frozen_v1');ap.add_argument('--output',default='reports/t05_results_v1/timing')
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    def guard():
        require_exclusive_gpu(a.gpu_uuid,query(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),
            os.getpid(),query(['--query-gpu=uuid','--format=csv,noheader']).split())
    guard();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid
    import numpy as np
    import torch
    from sglm.t02.data import AffinePairs256
    from sglm.t05 import UniformB0
    from sglm.evaluation import estimate_from_correspondences
    frozen=Path(a.frozen);freeze=json.loads((frozen/'freeze.json').read_text());protocol=freeze['protocol'];timing=protocol['timing']
    assert json.loads(Path('reports/t05_results_v1/candidate_reproduction.json').read_text())['passed']
    assert hashlib.sha256(Path(freeze['frozen_checkpoint']).read_bytes()).hexdigest()==freeze['checkpoint_sha256']
    assert hashlib.sha256((frozen/'subsets.json').read_bytes()).hexdigest()==freeze['subsets_sha256']
    saved=torch.load(freeze['frozen_checkpoint'],map_location='cpu',weights_only=False);cfg=saved['config']
    torch.set_num_threads(timing['threads']);torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    model=UniformB0(cfg).cuda().eval();model.load_state_dict(saved['model'],strict=True)
    assert all(not m._forward_hooks and not m._forward_pre_hooks and not m.training for m in model.modules())
    dataset=AffinePairs256(a.root,frozen/'validation_plan.json',split='val',preload=False)
    subset=json.loads((frozen/'subsets.json').read_text())['timing_indices'];assert len(subset)==24
    rng=random.Random(protocol['seed']);records=[];stages=[];guards=[]
    hardware_before=query(['--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem','--format=csv,noheader'])
    def timed(fn):
        torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
        return value,(time.perf_counter()-start)*1000
    def predict_and_fit(args):
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict(*args)[0]
        source,target,score=[p[key].float().cpu().numpy() for key in ('source','target','score')]
        fit=estimate_from_correspondences(source,target,score,cfg['evaluation'],seed=cfg['seed'])
        return dict(attempted_refinements=p['attempted_refinements'],final_correspondences=len(source),inliers=fit['inliers'],solver_status=fit['status'])
    def staged(args):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features,encode_ms=timed(lambda:model.encode(*args))
            output,correlation_ms=timed(lambda:model.correlate(features))
            (source,target,score),decode_ms=timed(lambda:model.coarse_matches(output,0))
            local,refine_ms=timed(lambda:model.local(output['optical_fine'],output['sar_fine'],source,target,args[2],args[3]))
            valid=local.get('valid',torch.ones(len(source),device='cuda',dtype=torch.bool))
            arrays,copy_ms=timed(lambda:[v[valid].float().cpu().numpy() for v in (source,local['target'],score)])
        fit,ransac_ms=timed(lambda:estimate_from_correspondences(*arrays,cfg['evaluation'],seed=cfg['seed']))
        return dict(encode_ms=encode_ms,coarse_ms=correlation_ms+decode_ms,refinement_ms=refine_ms,
            coordinate_copy_ms=copy_ms,ransac_ms=ransac_ms,attempted_refinements=len(source),
            final_correspondences=len(arrays[0]),inliers=fit['inliers'],solver_status=fit['status'])
    with torch.inference_mode():
        for ordinal,index in enumerate(subset):
            guard();guards.append(dict(pair_index=index,before=True,time_unix=time.time()))
            item=dataset[index];args=[item[key][None].cuda() for key in ('optical','sar_aug','optical_valid_mask','sar_aug_valid_mask')]
            torch.cuda.synchronize()
            for k in protocol['budgets']:
                model.k=k
                for _ in range(timing['warmup_per_pair_per_budget']):predict_and_fit(args)
            # Primary result: no hooks, no stage timers, no GT probes. One sync at each boundary.
            for repeat in range(timing['repeats_per_pair_per_budget']):
                order=list(protocol['budgets']);rng.shuffle(order)
                for k in order:
                    model.k=k;torch.cuda.reset_peak_memory_stats()
                    workload,elapsed=timed(lambda:predict_and_fit(args))
                    records.append(dict(pair_id=item['pair_id'],K=k,repeat=repeat,end_to_end_ms=elapsed,
                        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,**workload))
            for repeat in range(timing['repeats_per_pair_per_budget']):
                order=list(protocol['budgets']);rng.shuffle(order)
                for k in order:
                    model.k=k;stages.append(dict(pair_id=item['pair_id'],K=k,repeat=repeat,**staged(args)))
            guard();guards.append(dict(pair_index=index,after=True,time_unix=time.time()))
            print('timed pair',ordinal+1,len(subset),item['pair_id'],flush=True)
            del args
    guard()
    summary=[]
    for k in protocol['budgets']:
        group=[r for r in records if r['K']==k];parts=[r for r in stages if r['K']==k]
        # Each original pair has equal repeated weight. Also export medians per original pair.
        per_pair=[dict(pair_id=uid,median_ms=float(np.median([r['end_to_end_ms'] for r in group if r['pair_id']==uid]))) for uid in sorted({r['pair_id'] for r in group})]
        summary.append(dict(K=k,pairs=len(per_pair),repeats_per_pair=timing['repeats_per_pair_per_budget'],
            end_to_end_median_ms=float(np.median([r['end_to_end_ms'] for r in group])),
            end_to_end_p90_ms=float(np.percentile([r['end_to_end_ms'] for r in group],90)),
            mean_pair_median_ms=float(np.mean([r['median_ms'] for r in per_pair])),
            peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in group),
            peak_reserved_MiB=max(r['peak_reserved_MiB'] for r in group),
            mean_attempted_refinements=float(np.mean([r['attempted_refinements'] for r in group])),
            stages_median_ms={key:float(np.median([r[key] for r in parts])) for key in ['encode_ms','coarse_ms','refinement_ms','coordinate_copy_ms','ransac_ms']},per_pair=per_pair))
    report=dict(protocol=protocol['protocol'],checkpoint_sha256=freeze['checkpoint_sha256'],subsets_sha256=freeze['subsets_sha256'],
        timing_protocol=timing,gpu_uuid=a.gpu_uuid,pid=os.getpid(),torch=torch.__version__,cuda=torch.version.cuda,
        torch_device_name=torch.cuda.get_device_name(),cudnn_benchmark=False,cudnn_deterministic=True,TF32=False,
        no_hooks_confirmed=True,other_process_guards=guards,hardware_before=hardware_before,
        hardware_after=query(['--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem','--format=csv,noheader']),
        summary=summary,records=records,stage_records=stages,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['sglm/t05.py','scripts/benchmark_t05.py']},
        note='Timing repetitions are correlated within 24 fixed image pairs; allocated peak is primary, reserved memory retains caching across interleaved budgets. Stage medians are not summed as end-to-end latency.')
    (out/'timing.json').write_text(json.dumps(report,indent=2));write_csv(out/'end_to_end.csv',records);write_csv(out/'stages.csv',stages)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
