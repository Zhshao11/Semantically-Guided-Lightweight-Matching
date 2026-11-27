"""Exclusive no-hook timing; fresh processes for authoritative memory peaks."""
import argparse,gc,hashlib,json,os,subprocess,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from sglm.experiment_io import require_exclusive_gpu,write_csv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--memory-group',choices=['original_refiner','local_cross_attention']);a=ap.parse_args()
    p=json.loads(Path('configs/local_interaction_v1.json').read_text());spec=p['timing'];out=Path('reports/local_interaction_v1');uuid=spec['gpu_uuid']
    def query(args):return subprocess.run(['nvidia-smi',*args],check=True,capture_output=True,text=True).stdout
    def guard():return require_exclusive_gpu(uuid,query(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),os.getpid(),query(['--query-gpu=uuid','--format=csv,noheader']).split())
    guard();os.environ['CUDA_VISIBLE_DEVICES']=uuid
    import numpy as np
    import torch
    from sglm.local_interaction import build_local_group
    from sglm.t02.data import AffinePairs256
    from sglm.external_reference import cpu_input
    from sglm.evaluation import estimate_from_correspondences
    cfg=torch.load(p['checkpoint'],map_location='cpu',weights_only=False)['config']
    acceptance=json.loads((out/'evaluation_acceptance.json').read_text());assert acceptance['passed']
    torch.set_num_threads(spec['threads']);torch.manual_seed(cfg['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    ds=AffinePairs256(a.root,'reports/t05_frozen_v1/validation_plan.json',split='val',preload=False)
    indices=json.loads(Path('reports/t05_frozen_v1/subsets.json').read_text())['timing_indices'];assert len(indices)==24
    inputs=[(i,cpu_input(ds[i])) for i in indices]
    names=[a.memory_group] if a.memory_group else p['groups'];models={}
    for name in names:
        path=Path('runs/local_interaction_v1')/f'{name}_last.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==acceptance['checkpoint_sha256'][name]
        models[name]=build_local_group(cfg,torch.load(path,map_location='cpu',weights_only=False)['model'],name)
        assert not any(m.training or m._forward_hooks or m._forward_pre_hooks for m in models[name].modules())
    @torch.inference_mode()
    def complete(name,data):
        args=[torch.from_numpy(data['optical'].copy()).permute(2,0,1)[None].float().cuda()/255,
            torch.from_numpy(data['sar'].copy())[None,None].float().cuda()/255,
            torch.from_numpy(data['optical_mask'])[None].cuda(),torch.from_numpy(data['sar_mask'])[None].cuda()]
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=models[name].predict(*args)[0]
        arrays=[pred[k].float().cpu().numpy() for k in ['source','target','score']]
        fit=estimate_from_correspondences(*arrays,cfg['evaluation'],seed=cfg['seed'])
        return dict(refinements=pred['attempted_refinements'],matches=len(arrays[0]),fit_status=fit['status'])
    hardware_args=['--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,power.draw,clocks.sm','--format=csv,noheader']
    before=query(hardware_args);rows=[];guards=0
    rng=np.random.default_rng(p['seed'])
    rounds=1 if a.memory_group else spec['rounds'];repeats=1 if a.memory_group else spec['repeats']
    for rd in range(rounds):
        for i,data in inputs:
            guard();guards+=1
            for name in names:
                for _ in range(spec['warmup']):complete(name,data)
            for rep in range(repeats):
                for name in rng.permutation(names):
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();result=complete(name,data);torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                    rows.append(dict(group=name,round=rd,pair_index=i,pair_id=data['pair_id'],repeat=rep,latency_ms=elapsed,
                        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,**result))
            guard();guards+=1
        print('round',rd+1,'rows',len(rows),flush=True)
    for index,_ in inputs:
        counts={(r['refinements'],r['matches']) for r in rows if r['pair_index']==index}
        assert len(counts)==1,('Timing changed correspondence budgets',index,counts)
    suffix=f'memory_{a.memory_group}' if a.memory_group else 'latency'
    write_csv(out/f'{suffix}_runs.csv',rows)
    summaries=[]
    for name in names:
        group=[r for r in rows if r['group']==name];times=[r['latency_ms'] for r in group]
        summaries.append(dict(group=name,pairs=24,repeats_per_pair=rounds*repeats,median_ms=float(np.median(times)),p90_ms=float(np.percentile(times,90)),
            round_medians_ms=[float(np.median([r['latency_ms'] for r in group if r['round']==rd])) for rd in range(rounds)],
            peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in group),peak_reserved_MiB=max(r['peak_reserved_MiB'] for r in group)))
    (out/f'{suffix}_summary.json').write_text(json.dumps(dict(summaries=summaries,exclusive_guards=guards,protocol=spec,
        fresh_single_model_process=bool(a.memory_group),pid=os.getpid(),torch=torch.__version__,cuda=torch.version.cuda,hardware_before=before,hardware_after=query(hardware_args),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),no_hooks=True,no_feature_cache=True,GT_used=False),indent=2))
    if not a.memory_group:
        paired=[]
        for i,_ in inputs:
            control=[r['latency_ms'] for r in rows if r['pair_index']==i and r['group']=='original_refiner'];experiment=[r['latency_ms'] for r in rows if r['pair_index']==i and r['group']=='local_cross_attention']
            paired.append(dict(pair_index=i,control_median_ms=float(np.median(control)),interaction_median_ms=float(np.median(experiment)),delta_ms=float(np.median(experiment)-np.median(control))))
        write_csv(out/'latency_paired.csv',paired)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
