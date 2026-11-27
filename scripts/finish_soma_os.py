"""Automatic final-only test and idle5090 timing after validation-selected training."""
import argparse,hashlib,json,os,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from sglm.soma_training import setup,SOMAModel,load_model_state,validate,predict,atomic_json
from sglm.t03.data import FullAffinePairs256
from sglm.external_reference import cpu_input
from sglm.evaluation import estimate_from_correspondences
from sglm.experiment_io import require_exclusive_gpu,write_csv


def main():
    import subprocess
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    config,dsconfig,plan=setup();out=Path(config['reports']);run=Path(config['output'])
    assert json.loads((run/'training_finished.json').read_text())['completed']
    torch.cuda.set_device(0);model=SOMAModel(config).eval();saved=load_model_state(model,run/'best.pt',config)
    sha=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest();result={}
    for split in ['val','test']:
        destination=out/f'final_{split}.json'
        if destination.exists():
            existing=json.loads(destination.read_text());assert existing['checkpoint_sha256']==sha;result[split]=existing['summary'];continue
        ds=FullAffinePairs256(a.root,plan,split,dsconfig,preload=False)
        atomic_json(run/'status.json',dict(phase='final_evaluation',split=split,pairs_done=0))
        r=validate(model,ds,config,dsconfig['evaluation'],progress=lambda done,total:atomic_json(run/'status.json',dict(phase='final_evaluation',split=split,pairs_done=done,total=total)))
        r.update(checkpoint_sha256=sha,selected_epoch=saved['epoch'],test_used_for_selection=False,
                 metric_protocol='uniform1024predicted inverse-flow correspondences -> originalfullaffineRANSAC -> independent256pixelprobes; dense field error not calledsameSR')
        atomic_json(destination,r);result[split]=r['summary']
    # Permit later rerun of timing without rerunning the final test pass.
    def guard():
        def query(args):return subprocess.run(['nvidia-smi',*args],check=True,capture_output=True,text=True).stdout
        require_exclusive_gpu(config['timing']['gpu_uuid'],query(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),
            os.getpid(),query(['--query-gpu=uuid','--format=csv,noheader']).split())
    guard();ds=FullAffinePairs256(a.root,plan,'val',dsconfig,preload=False)
    subset=json.loads(Path('reports/t05_frozen_v1/subsets.json').read_text())['timing_indices'];records=[]
    def network_from_cpu(data):
        # CPUuint8 start: match this boundary to the external reference task.
        item=dict(optical=torch.from_numpy(data['optical']).permute(2,0,1).float()/255,
            sar_aug=torch.from_numpy(data['sar'])[None].float()/255,
            optical_valid_mask=torch.from_numpy(data['optical_mask']),sar_aug_valid_mask=torch.from_numpy(data['sar_mask']))
        aa=predict(model,item,config)
        fit=estimate_from_correspondences(*aa,dsconfig['evaluation'],seed=config['evaluation']['solver_seed'])
        return dict(matches=len(aa[0]),inliers=fit['inliers'],fit_status=fit['status'])
    with torch.inference_mode():
        for rd in range(config['timing']['rounds']):
            for i in subset:
                guard();data=cpu_input(ds[i])
                for _ in range(config['timing']['warmup']):network_from_cpu(data)
                for repeat in range(config['timing']['repeats']):
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
                    workload=network_from_cpu(data);torch.cuda.synchronize();elapsed=(time.perf_counter()-begin)*1000
                    records.append(dict(round=rd,pair_index=i,pair_id=data['pair_id'],repeat=repeat,latency_ms=elapsed,
                        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,**workload))
                guard()
    write_csv(out/'latency_runs.csv',records);t=[r['latency_ms'] for r in records]
    timing=dict(latency_mean_ms=float(np.mean(t)),latency_median_ms=float(np.median(t)),latency_p95_ms=float(np.percentile(t,95)),
        peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in records),gpu_uuid=config['timing']['gpu_uuid'],pairs=24,repeats=15,
        no_hooks=True,no_feature_cache=True,boundary=config['timing']['boundary'],checkpoint_sha256=sha)
    atomic_json(out/'timing_summary.json',timing)
    rows=[]
    for split,metrics in result.items():rows.append(dict(split=split,selected_epoch=saved['epoch'],checkpoint_sha256=sha,
        SR3=metrics['success_rate@3px'],SR5=metrics['success_rate@5px'],pairs=metrics['image_pairs'],solver_failures=metrics['failed_pairs'],
        rmse_median_px=metrics['solved_median_rmse_px'],rmse_p90_px=metrics['solved_p90_rmse_px'],
        end_to_end_median_ms=timing['latency_median_ms'],peak_allocated_MiB=timing['peak_allocated_MiB']))
    write_csv(out/'final_summary.csv',rows);atomic_json(run/'status.json',dict(phase='completed',selected_epoch=saved['epoch'],summary=rows))
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
