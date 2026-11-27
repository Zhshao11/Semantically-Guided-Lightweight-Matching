"""No-hook paired calibration against the genuinely unchanged T05 inference path."""
import argparse,json,os,random,subprocess,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from sglm.experiment_io import require_exclusive_gpu,write_csv


def query(args):return subprocess.run(['nvidia-smi',*args],check=True,capture_output=True,text=True).stdout


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--gpu-uuid',required=True);a=ap.parse_args()
    def guard():
        require_exclusive_gpu(a.gpu_uuid,query(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),
            os.getpid(),query(['--query-gpu=uuid','--format=csv,noheader']).split())
    guard();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid
    import numpy as np
    import torch
    from sglm.sparse_refinement_experiment import setup,OUT,KEYS,arrays,sha
    from sglm.t05 import UniformB0
    from sglm.t02.data import AffinePairs256
    from sglm.evaluation import estimate_from_correspondences
    p,cfg,model=setup();base=UniformB0(cfg).cuda().eval();base.load_state_dict(model.state_dict(),strict=True);base.requires_grad_(False)
    assert json.loads((OUT/'full_reproduction.json').read_text())['passed']
    for m in list(base.modules())+list(model.modules()):assert not m.training and not m._forward_hooks and not m._forward_pre_hooks
    ds=AffinePairs256(a.root,p['frozen']+'/validation_plan.json',split='val',preload=False)
    subset=json.loads(open(p['frozen']+'/subsets.json').read())['timing_indices'];spec=p['timing'];rng=random.Random(spec['seed']+1)
    names=['original_T05','A0_all','B0_global128'];records=[];guards=[]
    def network(args,name):
        current=base if name=='original_T05' else model
        model.refinement_mode='global_conf' if name=='B0_global128' else 'all';model.refinement_budget=128 if name=='B0_global128' else None
        with torch.autocast('cuda',dtype=torch.bfloat16):return current.predict(*args)[0]
    def complete(args,name):
        pred=network(args,name);return estimate_from_correspondences(*arrays(pred),cfg['evaluation'],seed=cfg['seed'])
    def timed(fn):
        torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
        return value,(time.perf_counter()-start)*1000
    with torch.inference_mode():
        for rd in range(spec['rounds']):
            for i in subset:
                guard();guards.append(dict(round=rd,pair_index=i,before=True));item=ds[i];args=[item[k][None].cuda() for k in KEYS]
                order=list(names);rng.shuffle(order)
                for name in order:
                    for _ in range(spec['warmup']):complete(args,name)
                for rep in range(spec['repeats_per_round']):
                    order=list(names);rng.shuffle(order)
                    for name in order:
                        boundaries=['network','complete'];rng.shuffle(boundaries)
                        for boundary in boundaries:
                            torch.cuda.reset_peak_memory_stats()
                            value,ms=timed(lambda:network(args,name) if boundary=='network' else complete(args,name))
                            peak=torch.cuda.max_memory_allocated()/2**20;del value
                            records.append(dict(round=rd,pair_index=i,pair_id=item['pair_id'],repeat=rep,variant=name,boundary=boundary,latency_ms=ms,peak_allocated_MiB=peak))
                guard();guards.append(dict(round=rd,pair_index=i,after=True));del args
            print('original calibration round',rd+1,flush=True)
    summary=[];paired=[]
    for name in names:
        r=dict(variant=name)
        for boundary in ['network','complete']:
            group=[x for x in records if x['variant']==name and x['boundary']==boundary];t=[x['latency_ms'] for x in group]
            r.update({boundary+'_mean_ms':float(np.mean(t)),boundary+'_median_ms':float(np.median(t)),boundary+'_p95_ms':float(np.percentile(t,95)),
                boundary+'_round_medians_ms':json.dumps([float(np.median([x['latency_ms'] for x in group if x['round']==rd])) for rd in range(3)])})
        summary.append(r)
    for rd in range(spec['rounds']):
        for i in subset:
            for boundary in ['network','complete']:
                g=[x for x in records if x['round']==rd and x['pair_index']==i and x['boundary']==boundary]
                baseline=np.median([x['latency_ms'] for x in g if x['variant']=='original_T05'])
                for name in names[1:]:
                    v=np.median([x['latency_ms'] for x in g if x['variant']==name])
                    paired.append(dict(round=rd,pair_index=i,boundary=boundary,variant=name,original_median_ms=float(baseline),variant_median_ms=float(v),paired_delta_ms=float(v-baseline)))
    write_csv(OUT/'original_reference_runs.csv',records);write_csv(OUT/'original_reference_summary.csv',summary);write_csv(OUT/'original_reference_paired.csv',paired)
    (OUT/'original_reference_environment.json').write_text(json.dumps(dict(gpu_uuid=a.gpu_uuid,protocol=spec,guards=guards,
        source_sha256=sha('scripts/calibrate_sparse_refinement_reference.py'),
        note='Both identical-weight modules are resident throughout this calibration; use primary six-way benchmark for comparable single-model peak memory. Three-way order is interleaved; no hooks or caches.'),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
