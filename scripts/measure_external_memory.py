"""Fresh process per method: authoritative inference peak without cross-method workspaces."""
import argparse,gc,json,os,subprocess,weakref
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from sglm.experiment_io import require_exclusive_gpu


def query(args):return subprocess.run(['nvidia-smi',*args],check=True,capture_output=True,text=True).stdout


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--method',required=True);a=ap.parse_args()
    p=json.loads(Path('configs/external_references_v1.json').read_text());out=Path(p['output']);spec=p['timing'];uuid=spec['gpu_uuid']
    def guard():require_exclusive_gpu(uuid,query(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),os.getpid(),query(['--query-gpu=uuid','--format=csv,noheader']).split())
    guard();os.environ['CUDA_VISIBLE_DEVICES']=uuid
    import numpy as np
    import torch
    from sglm.external_reference import OfficialReference,cpu_input
    from sglm.t02.data import AffinePairs256
    from sglm.evaluation import estimate_from_correspondences
    cfg=json.loads(Path(p['frozen']+'/candidate_config.json').read_text())
    torch.set_num_threads(spec['threads']);torch.manual_seed(cfg['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    initial=torch.cuda.memory_allocated()/2**20;assert initial==0
    adapter=OfficialReference(a.method,p,cfg);ref=weakref.ref(adapter.model)
    dino_ref=weakref.ref(adapter.model.encoder.dinov2_vitl14[0]) if a.method=='MINIMA_RoMa' else None
    ds=AffinePairs256(a.root,p['frozen']+'/validation_plan.json',split='val',preload=False)
    subset=json.loads(Path(p['frozen']+'/subsets.json').read_text())['timing_indices'];rows=[]
    def run(data):
        pred=adapter.predict(data);fit=estimate_from_correspondences(pred['source'],pred['target'],pred['score'],cfg['evaluation'],seed=cfg['seed'])
        return dict(valid_matches=pred['valid_matches'],fit_status=fit['status'],inliers=fit['inliers'])
    with torch.inference_mode():
        for i in subset:
            guard();data=cpu_input(ds[i])
            for _ in range(spec['warmup_per_pair']):run(data)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();work=run(data);torch.cuda.synchronize()
            rows.append(dict(pair_index=i,pair_id=data['pair_id'],peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,**work));guard()
    del adapter,run;gc.collect();torch.cuda.empty_cache()
    model_collected=ref() is None;dino_collected=dino_ref is None or dino_ref() is None
    after_delete=torch.cuda.memory_allocated()/2**20
    # Forensic diagnostic only AFTER all measurements: distinguish persistent
    # library workspaces from live parameters. Not a timed-path optimization.
    clear=getattr(torch._C,'_cuda_clearCublasWorkspaces',None)
    if clear is not None:clear();torch.cuda.empty_cache()
    after_clear=torch.cuda.memory_allocated()/2**20
    result=dict(method=a.method,fresh_process=True,pid=os.getpid(),gpu_uuid=uuid,initial_allocated_MiB=initial,
        pairs=24,rows=rows,peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in rows),
        peak_reserved_MiB=max(r['peak_reserved_MiB'] for r in rows),model_collected=model_collected,dino_collected=dino_collected,
        after_model_delete_MiB=after_delete,after_diagnostic_clear_cublas_MiB=after_clear,
        diagnostic_clear_available=clear is not None,
        note='One new process per method; same24 inputs, native inference and postprocessing;3warmups then1peak sample/pair. No initialization peak, diskIO or GT. Residual workspaces investigated only after all peak measurements; latency uses original unmodified3round run.')
    assert model_collected and dino_collected
    (out/f'{a.method}_memory.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
