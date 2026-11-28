"""One fresh process/method/round: exclusive RTX5090, no-hook end-to-end and full peaks."""
import argparse,hashlib,json,os,resource,subprocess,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from sglm.experiment_io import require_exclusive_gpu,write_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--method',required=True);ap.add_argument('--round',type=int,required=True);a=ap.parse_args()
    p=json.loads(Path('configs/sar_references_v1.json').read_text());out=Path(p['output']);spec=p['timing'];uuid=spec['gpu_uuid']
    def query(args):return subprocess.run(['nvidia-smi',*args],check=True,capture_output=True,text=True).stdout
    def guard():return require_exclusive_gpu(uuid,query(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),os.getpid(),query(['--query-gpu=uuid','--format=csv,noheader']).split())
    guard();os.environ['CUDA_VISIBLE_DEVICES']=uuid
    import numpy as np,torch
    from sglm.sar_references import SARReference
    from sglm.external_reference import cpu_input
    from sglm.t02.data import AffinePairs256
    from sglm.evaluation import estimate_from_correspondences
    freeze=json.loads((out/'freeze.json').read_text())
    for path,digest in freeze['input_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    legacy=a.method in ['MINIMA_RoMa','XoFTR']
    native=a.method=='GDROS_WHU_native32_LSR';name='GDROS_WHU' if native else a.method
    if legacy:
        reuse=json.loads((out/f'{name}_legacy_reuse.json').read_text());assert reuse['passed']
        assert hashlib.sha256(Path(reuse['full952_accuracy_path']).read_bytes()).hexdigest()==reuse['full952_accuracy_sha256']
    else:assert json.loads((out/f'{name}_accuracy.json').read_text())['image_pairs']==952
    torch.set_num_threads(spec['threads']);torch.manual_seed(20261125);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads(Path(p['frozen']+'/candidate_config.json').read_text())
    if legacy:
        from sglm.external_reference import OfficialReference
        previous=json.loads(Path('configs/external_references_v1.json').read_text());model=OfficialReference(name,previous,cfg)
    else:model=SARReference(name,p,cfg)
    assert not any(m.training or m._forward_hooks or m._forward_pre_hooks for m in model.model.modules())
    ds=AffinePairs256(a.root,p['frozen']+'/validation_plan.json',split='val',preload=False);indices=json.loads(Path(p['frozen']+'/subsets.json').read_text())['timing_indices'];assert len(indices)==24
    inputs=[(i,cpu_input(ds[i])) for i in indices]
    def complete(data):
        if native:
            field=model.predict_native_lsr(data);assert field.shape==(2,512,512)
            return dict(raw_matches=None,valid_matches=None,fit_status='native_dense_LSR_no_extra_RANSAC')
        pred=model.predict(data);fit=estimate_from_correspondences(pred['source'],pred['target'],pred['score'],cfg['evaluation'],seed=cfg['seed'])
        return dict(raw_matches=pred['raw_matches'],valid_matches=pred['valid_matches'],fit_status=fit['status'],inliers=fit['inliers'])
    hargs=['--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,power.draw,clocks.sm','--format=csv,noheader'];before=query(hargs);rows=[];guards=0
    for i,data in inputs:
        guard();guards+=1
        for _ in range(spec['warmup']):complete(data)
        for rep in range(spec['repeats']):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();result=complete(data);torch.cuda.synchronize();elapsed=(time.perf_counter()-begin)*1000
            rows.append(dict(method=a.method,round=a.round,pair_index=i,pair_id=data['pair_id'],repeat=rep,latency_ms=elapsed,
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,**result))
        guard();guards+=1
    path=out/f'{a.method}_timing_round{a.round}';write_csv(str(path)+'.csv',rows)
    (Path(str(path)+'.json')).write_text(json.dumps(dict(method=a.method,round=a.round,records=len(rows),pairs=24,median_ms=float(np.median([r['latency_ms'] for r in rows])),
        peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in rows),peak_reserved_MiB=max(r['peak_reserved_MiB'] for r in rows),process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        exclusive_guards=guards,fresh_single_model_process=True,pid=os.getpid(),torch=torch.__version__,cuda=torch.version.cuda,hardware_before=before,hardware_after=query(hargs),
        no_hooks=True,no_feature_cache=True,GT_used=False,protocol='native32+LSR full CPUdense output' if native else 'commonCPU256uint8->predictedfullaffine',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(a.method,'round',a.round,'median',float(np.median([r['latency_ms'] for r in rows])),flush=True)


if __name__=='__main__':main()
