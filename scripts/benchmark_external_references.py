"""One idle5090, common CPU-input-to-affine boundary, official native inference."""
import argparse,gc,hashlib,json,os,subprocess,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from sglm.experiment_io import require_exclusive_gpu,write_csv


def query(args):return subprocess.run(['nvidia-smi',*args],check=True,capture_output=True,text=True).stdout


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    p=json.loads(Path('configs/external_references_v1.json').read_text());spec=p['timing'];out=Path(p['output'])
    uuid=spec['gpu_uuid']
    def guard():
        return require_exclusive_gpu(uuid,query(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),os.getpid(),query(['--query-gpu=uuid','--format=csv,noheader']).split())
    guard();os.environ['CUDA_VISIBLE_DEVICES']=uuid
    import numpy as np
    import torch
    from sglm.external_reference import OfficialReference,cpu_input
    from sglm.t02.data import AffinePairs256
    from sglm.evaluation import estimate_from_correspondences
    cfg=json.loads(Path(p['frozen']+'/candidate_config.json').read_text())
    freeze=json.loads((out/'freeze.json').read_text())
    for path,digest in freeze['input_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    for name in p['methods']:assert json.loads((out/f'{name}_accuracy.json').read_text())['image_pairs']==952
    torch.set_num_threads(spec['threads']);torch.manual_seed(cfg['seed']);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    ds=AffinePairs256(a.root,p['frozen']+'/validation_plan.json',split='val',preload=False)
    subsets=json.loads(Path(p['frozen']+'/subsets.json').read_text());indices=subsets['timing_indices'];assert len(indices)==24
    # CPU image preparation and affine synthesis excluded, identically for all methods.
    inputs=[(index,cpu_input(ds[index])) for index in indices]
    hwargs=['--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,power.draw,clocks.sm','--format=csv,noheader']
    before=query(hwargs);records=[];guards=[];initial_allocated=[]
    def timed(fn):
        torch.cuda.synchronize();begin=time.perf_counter();value=fn();torch.cuda.synchronize();return value,(time.perf_counter()-begin)*1000
    with torch.inference_mode():
        for rd in range(spec['rounds']):
            order=p['methods'][rd:]+p['methods'][:rd]
            for name in order:
                guard();gc.collect();torch.cuda.empty_cache();initial_allocated.append(dict(round=rd,method=name,before_model_MiB=torch.cuda.memory_allocated()/2**20))
                adapter=OfficialReference(name,p,cfg)
                assert not any(m.training or m._forward_hooks or m._forward_pre_hooks for m in adapter.model.modules())
                def complete(data):
                    pred=adapter.predict(data)
                    fit=estimate_from_correspondences(pred['source'],pred['target'],pred['score'],cfg['evaluation'],seed=cfg['seed'])
                    return dict(raw_matches=pred['raw_matches'],valid_matches=pred['valid_matches'],fit_status=fit['status'],inliers=fit['inliers'])
                for index,data in inputs:
                    guard();guards.append(dict(round=rd,method=name,pair_index=index,before=True))
                    for _ in range(spec['warmup_per_pair']):complete(data)
                    for rep in range(spec['repeats_per_pair_per_round']):
                        torch.cuda.reset_peak_memory_stats();work,ms=timed(lambda:complete(data))
                        records.append(dict(method=name,round=rd,pair_index=index,pair_id=data['pair_id'],repeat=rep,latency_ms=ms,
                            peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,**work))
                    guard();guards.append(dict(round=rd,method=name,pair_index=index,after=True))
                del adapter,complete;gc.collect();torch.cuda.empty_cache();guard()
                print('timed',name,'round',rd+1,'rows',len(records),flush=True)
                write_csv(out/'latency_runs.csv',records)
    summaries=[];paired=[]
    for name in p['methods']:
        group=[r for r in records if r['method']==name];t=[r['latency_ms'] for r in group]
        medians=[float(np.median([r['latency_ms'] for r in group if r['round']==rd])) for rd in range(spec['rounds'])]
        pair_medians=[float(np.median([r['latency_ms'] for r in group if r['pair_index']==i])) for i in indices]
        summaries.append(dict(method=name,pairs=24,repeats_per_pair=spec['rounds']*spec['repeats_per_pair_per_round'],latency_mean_ms=float(np.mean(t)),
            latency_median_ms=float(np.median(t)),latency_p95_ms=float(np.percentile(t,95)),mean_pair_median_ms=float(np.mean(pair_medians)),
            round_medians_ms=json.dumps(medians),round_median_std_ms=float(np.std(medians,ddof=1)),
            peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in group),peak_reserved_MiB=max(r['peak_reserved_MiB'] for r in group)))
        for rd in range(spec['rounds']):
            for i in indices:
                v=np.median([r['latency_ms'] for r in group if r['pair_index']==i and r['round']==rd])
                base=np.median([r['latency_ms'] for r in records if r['method']=='T05' and r['pair_index']==i and r['round']==rd])
                paired.append(dict(method=name,round=rd,pair_index=i,method_median_ms=float(v),T05_median_ms=float(base),paired_delta_ms=float(v-base),ratio=float(v/base)))
    write_csv(out/'latency_summary.csv',summaries);write_csv(out/'latency_paired.csv',paired)
    (out/'timing_environment.json').write_text(json.dumps(dict(protocol=spec,hardware_before=before,hardware_after=query(hwargs),
        torch=torch.__version__,cuda=torch.version.cuda,pid=os.getpid(),exclusive_guards=guards,initial_allocated=initial_allocated,
        no_hooks=True,no_feature_cache=True,only_one_model_at_a_time=True,
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['sglm/external_reference.py','scripts/benchmark_external_references.py']},
        note='Same CPU-array-to-affine boundary, physical GPU, batch and supplied pairs; native internal precision/resolution differ. Memory is full PyTorch peak allocated including registered AND unregistered DINO. Reserved includes caching. Repeats correlated within24 pairs.'),indent=2))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
