"""Unified fresh-process24-pair T05/FHReg timing, identical outer boundary and memory policy."""
import argparse,hashlib,json,os,subprocess,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from sglm.experiment_io import require_exclusive_gpu,write_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--method',required=True,choices=['T05_Full','FHReg_GUSO']);ap.add_argument('--round',type=int,required=True);a=ap.parse_args()
    p=json.loads(Path('configs/fhreg_result_audit_v1.json').read_text());out=Path(p['output']);spec=p['timing'];uuid=spec['gpu_uuid']
    def query(args):return subprocess.run(['nvidia-smi',*args],check=True,capture_output=True,text=True).stdout
    def guard():return require_exclusive_gpu(uuid,query(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),os.getpid(),query(['--query-gpu=uuid','--format=csv,noheader']).split())
    guard();os.environ['CUDA_VISIBLE_DEVICES']=uuid
    import numpy as np,torch,cv2
    from sglm.sar_references import SARReference
    from sglm.external_reference import cpu_input
    from sglm.t02.data import AffinePairs256
    from sglm.evaluation import estimate_from_correspondences
    freeze=json.loads((out/'audit_freeze.json').read_text())
    for f,digest in freeze['inputs_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
    assert json.loads((out/'coordinate_roundtrip.json').read_text())['passed']
    assert json.loads((out/'evidence_manifest.json').read_text())['passed']
    protocol=json.loads(Path(p['source_protocol']).read_text());cfg=json.loads(Path(protocol['frozen']+'/candidate_config.json').read_text())
    torch.set_num_threads(spec['threads']);torch.manual_seed(cfg['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    model=SARReference(a.method,protocol,cfg);assert not any(m.training or m._forward_hooks or m._forward_pre_hooks for m in model.model.modules())
    ds=AffinePairs256(a.root,protocol['frozen']+'/validation_plan.json',split='val',preload=False);indices=json.loads(Path(protocol['frozen']+'/subsets.json').read_text())['timing_indices'];assert len(indices)==24
    inputs=[(i,cpu_input(ds[i])) for i in indices];hashes=[]
    for index,data in inputs:
        h=hashlib.sha256()
        for key in ['optical','sar','optical_mask','sar_mask']:h.update(key.encode());h.update(data[key].tobytes())
        hashes.append(dict(pair_index=index,pair_id=data['pair_id'],CPU_input_sha256=h.hexdigest()))
    def complete(data):
        pred=model.predict(data)
        fit=estimate_from_correspondences(pred['source'],pred['target'],pred['score'],cfg['evaluation'],seed=cfg['seed'])
        return dict(raw_matches=pred['raw_matches'],valid_matches=pred['valid_matches'],fit_status=fit['status'],inliers=fit['inliers'])
    hardware=['--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,power.draw,clocks.sm','--format=csv,noheader'];before=query(hardware);records=[];guards=0
    with torch.inference_mode():
        for index,data in inputs:
            guard();guards+=1
            for _ in range(spec['warmup_per_pair']):complete(data)
            for rep in range(spec['repeats_per_pair_per_round']):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();work=complete(data);torch.cuda.synchronize();elapsed_ms=(time.perf_counter()-begin)*1000
                records.append(dict(method=a.method,round=a.round,pair_index=index,pair_id=data['pair_id'],repeat=rep,latency_ms=elapsed_ms,
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,**work))
            guard();guards+=1
    write_csv(out/f'{a.method}_timing_round{a.round}.csv',records)
    summary=dict(method=a.method,round=a.round,pairs=24,runs=len(records),median_ms=float(np.median([r['latency_ms'] for r in records])),
        peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in records),peak_reserved_MiB=max(r['peak_reserved_MiB'] for r in records),
        exclusive_guards=guards,pid=os.getpid(),gpu_uuid=uuid,fresh_single_model_process=True,no_hooks=True,no_feature_cache=True,GT_used=False,
        sameCPUinput_hashes=hashes,warmup_per_pair=spec['warmup_per_pair'],repeats_per_pair=spec['repeats_per_pair_per_round'],cuda_sync_before_after=True,
        boundary=spec['boundary'],include_RANSAC=True,include_common_filter=True,precision='BF16' if a.method=='T05_Full' else 'FP32',
        torch=torch.__version__,cuda=torch.version.cuda,opencv=cv2.__version__,torch_threads=torch.get_num_threads(),opencv_native_threads=cv2.getNumThreads(),
        hardware_before=before,hardware_after=query(hardware),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/f'{a.method}_timing_round{a.round}.json').write_text(json.dumps(summary,indent=2));print(a.method,a.round,summary['median_ms'],flush=True)


if __name__=='__main__':main()
