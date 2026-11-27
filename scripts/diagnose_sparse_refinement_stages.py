"""Warm, interleaved synchronized stage diagnosis; never a formal latency result."""
import argparse,json,os,random,subprocess,time,shutil
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from sglm.experiment_io import require_exclusive_gpu,write_csv


def query(args):return subprocess.run(['nvidia-smi',*args],check=True,capture_output=True,text=True).stdout


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--gpu-uuid',required=True);a=ap.parse_args()
    def guard():require_exclusive_gpu(a.gpu_uuid,query(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),os.getpid(),query(['--query-gpu=uuid','--format=csv,noheader']).split())
    guard();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid
    import torch
    import sglm.t02.model as original_module
    from sglm.sparse_refinement_experiment import OUT,KEYS,setup,arrays,sha
    from sglm.t02.data import AffinePairs256
    from sglm.evaluation import estimate_from_correspondences
    p,cfg,model=setup();sample_fn=original_module.sample_windows
    ds=AffinePairs256(a.root,p['frozen']+'/validation_plan.json',split='val',preload=False)
    subset=json.loads(open(p['frozen']+'/subsets.json').read())['timing_indices'];rng=random.Random(p['timing']['seed']+2)
    def timed(fn):
        torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize();return value,(time.perf_counter()-start)*1000
    def staged(args):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features,encode_ms=timed(lambda:model.encode(*args));output,corr_ms=timed(lambda:model.correlate(features))
            packet,decode_ms=timed(lambda:model.candidate_packet(output,0));sel,select_ms=timed(lambda:model.select_packet(packet,(256,256)))
            window_times=[]
            def tracked(*aa,**kw):
                result,ms=timed(lambda:sample_fn(*aa,**kw));window_times.append(ms);return result
            original_module.sample_windows=tracked
            try:pred,local_ms=timed(lambda:model.refine_packet(output,args,sel,0))
            finally:original_module.sample_windows=sample_fn
        aa,copy_ms=timed(lambda:arrays(pred));fit,geo_ms=timed(lambda:estimate_from_correspondences(*aa,cfg['evaluation'],seed=cfg['seed']))
        return dict(encode_ms=encode_ms,coarse_ms=corr_ms+decode_ms,selection_gather_ms=select_ms,
            window_extraction_ms=sum(window_times),refinement_remainder_ms=local_ms-sum(window_times),
            local_including_windows_ms=local_ms,copy_ms=copy_ms,geometry_ms=geo_ms,num_selected=len(sel['source']))
    rows=[]
    with torch.inference_mode():
        for i in subset:
            guard();item=ds[i];args=[item[k][None].cuda() for k in KEYS]
            order=list(p['variants']);rng.shuffle(order)
            for v in order:
                model.refinement_mode=v['mode'];model.refinement_budget=v['budget']
                for _ in range(5):staged(args)
            for repeat in range(3):
                order=list(p['variants']);rng.shuffle(order)
                for v in order:
                    model.refinement_mode=v['mode'];model.refinement_budget=v['budget']
                    rows.append(dict(pair_id=item['pair_id'],variant=v['name'],repeat=repeat,**staged(args)))
            guard();del args
    assert original_module.sample_windows is sample_fn
    if not (OUT/'stage_diagnostics_initial.csv').exists():shutil.copy2(OUT/'stage_diagnostics.csv',OUT/'stage_diagnostics_initial.csv')
    write_csv(OUT/'stage_diagnostics.csv',rows)
    (OUT/'stage_diagnostics_environment.json').write_text(json.dumps(dict(gpu_uuid=a.gpu_uuid,warmup_per_variant_per_pair=5,
        repeats=3,interleaved=True,seed=p['timing']['seed']+2,source_sha256=sha('scripts/diagnose_sparse_refinement_stages.py'),
        reason='Initial stage pass used fixed variant order without explicit stage warmup and showed first-variant startup contamination; retain initialCSV, redo only diagnostic timing; primary no-hook timings unchanged.',
        warning='Synchronized instrumentation includes CPU and synchronization overhead; do not sum or compare it as formal end-to-end latency.'),indent=2))


if __name__=='__main__':main()
