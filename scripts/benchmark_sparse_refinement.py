"""Exclusive no-hook end-to-end timing; synchronized diagnostic pass is separate."""
import argparse,json,os,random,subprocess,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from sglm.experiment_io import require_exclusive_gpu,write_csv


def query(args):return subprocess.run(['nvidia-smi',*args],check=True,capture_output=True,text=True).stdout


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--gpu-uuid',required=True);a=ap.parse_args()
    def guard():
        return require_exclusive_gpu(a.gpu_uuid,query(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),
            os.getpid(),query(['--query-gpu=uuid','--format=csv,noheader']).split())
    guard();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid
    import numpy as np
    import torch
    import sglm.t02.model as original_module
    from sglm.sparse_refinement_experiment import OUT,PRIVATE,KEYS,setup,arrays,sha
    from sglm.t02.data import AffinePairs256
    from sglm.evaluation import estimate_from_correspondences
    p,cfg,model=setup();assert json.loads((OUT/'full_reproduction.json').read_text())['passed']
    assert json.loads((OUT/'regression.json').read_text())['passed']
    for m in model.modules():assert not m.training and not m._forward_hooks and not m._forward_pre_hooks
    sample_fn=original_module.sample_windows
    ds=AffinePairs256(a.root,p['frozen']+'/validation_plan.json',split='val',preload=False)
    subset=json.loads(open(p['frozen']+'/subsets.json').read())['timing_indices'];assert len(subset)==24
    variants=p['variants'];spec=p['timing'];rng=random.Random(spec['seed']);records=[];stage_rows=[];guards=[]
    hw_args=['--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,power.draw,clocks.sm','--format=csv,noheader']
    before=query(hw_args)
    def timed(fn):
        torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
        return value,(time.perf_counter()-start)*1000
    def network(args):
        with torch.autocast('cuda',dtype=torch.bfloat16):return model.predict(*args)[0]
    def complete(args):
        pred=network(args);aa=arrays(pred)
        fit=estimate_from_correspondences(*aa,cfg['evaluation'],seed=cfg['seed'])
        return dict(num_selected=pred['attempted_refinements'],num_refined_valid=len(aa[0]),inliers=fit['inliers'],solver_status=fit['status'])
    with torch.inference_mode():
        # Primary measurement: no feature cache, no hooks, no GT, no per-call writes.
        for round_id in range(spec['rounds']):
            for ordinal,index in enumerate(subset):
                guard();guards.append(dict(phase='formal',round=round_id,pair_index=index,before=True))
                item=ds[index];args=[item[k][None].cuda() for k in KEYS];torch.cuda.synchronize()
                order=list(variants);rng.shuffle(order)
                for v in order:
                    model.refinement_mode=v['mode'];model.refinement_budget=v['budget']
                    for _ in range(spec['warmup']):complete(args)
                for repeat in range(spec['repeats_per_round']):
                    order=list(variants);rng.shuffle(order)
                    for v in order:
                        assert original_module.sample_windows is sample_fn
                        model.refinement_mode=v['mode'];model.refinement_budget=v['budget']
                        boundaries=['network','complete'];rng.shuffle(boundaries)
                        for boundary in boundaries:
                            torch.cuda.reset_peak_memory_stats()
                            if boundary=='network':
                                pred,elapsed=timed(lambda:network(args))
                                workload=dict(num_selected=pred['attempted_refinements'],num_refined_valid=len(pred['source']))
                                peak=torch.cuda.max_memory_allocated()/2**20;reserved=torch.cuda.max_memory_reserved()/2**20
                                del pred
                            else:
                                workload,elapsed=timed(lambda:complete(args))
                                peak=torch.cuda.max_memory_allocated()/2**20;reserved=torch.cuda.max_memory_reserved()/2**20
                            records.append(dict(pair_id=item['pair_id'],pair_index=index,round=round_id,repeat=repeat,
                                variant=v['name'],boundary=boundary,latency_ms=elapsed,peak_allocated_MiB=peak,peak_reserved_MiB=reserved,**workload))
                guard();guards.append(dict(phase='formal',round=round_id,pair_index=index,after=True))
                del args
            print('formal round finished',round_id+1,flush=True)
        # No formal times are collected below. Temporarily instrument ONLY sample_windows
        # for synchronized attribution; restore original function in finally.
        for index in subset:
            guard();item=ds[index];args=[item[k][None].cuda() for k in KEYS]
            for v in variants:
                model.refinement_mode=v['mode'];model.refinement_budget=v['budget']
                for repeat in range(3):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        features,encode_ms=timed(lambda:model.encode(*args))
                        output,corr_ms=timed(lambda:model.correlate(features))
                        packet,decode_ms=timed(lambda:model.candidate_packet(output,0))
                        selected,select_ms=timed(lambda:model.select_packet(packet,args[0].shape[-2:]))
                        window_times=[]
                        def tracked(*aa,**kw):
                            result,ms=timed(lambda:sample_fn(*aa,**kw));window_times.append(ms);return result
                        original_module.sample_windows=tracked
                        try:pred,local_ms=timed(lambda:model.refine_packet(output,args,selected,0))
                        finally:original_module.sample_windows=sample_fn
                    aa,copy_ms=timed(lambda:arrays(pred))
                    fit,geometry_ms=timed(lambda:estimate_from_correspondences(*aa,cfg['evaluation'],seed=cfg['seed']))
                    stage_rows.append(dict(pair_id=item['pair_id'],variant=v['name'],repeat=repeat,encode_ms=encode_ms,
                        coarse_ms=corr_ms+decode_ms,selection_gather_ms=select_ms,window_extraction_ms=sum(window_times),
                        refinement_remainder_ms=max(0,local_ms-sum(window_times)),local_including_windows_ms=local_ms,
                        copy_ms=copy_ms,geometry_ms=geometry_ms,num_selected=len(selected['source']),num_refined_valid=len(aa[0])))
            guard();del args,features,output,packet,selected,pred,aa
    guard();assert original_module.sample_windows is sample_fn
    summary=[]
    for v in variants:
        r=dict(variant=v['name'])
        for boundary in ['network','complete']:
            group=[x for x in records if x['variant']==v['name'] and x['boundary']==boundary]
            times=np.array([x['latency_ms'] for x in group]);round_medians=[float(np.median([x['latency_ms'] for x in group if x['round']==j])) for j in range(spec['rounds'])]
            r.update({boundary+'_mean_ms':float(times.mean()),boundary+'_median_ms':float(np.median(times)),
                boundary+'_p95_ms':float(np.percentile(times,95)),boundary+'_round_medians_ms':json.dumps(round_medians),
                boundary+'_round_median_std_ms':float(np.std(round_medians,ddof=1)),
                boundary+'_peak_allocated_MiB':max(x['peak_allocated_MiB'] for x in group)})
        summary.append(r)
    write_csv(OUT/'latency_runs.csv',records);write_csv(OUT/'latency_summary.csv',summary);write_csv(OUT/'stage_diagnostics.csv',stage_rows)
    paired=[]
    for round_id in range(spec['rounds']):
        for index in subset:
            baseline=[r['latency_ms'] for r in records if r['variant']=='A0_all' and r['boundary']=='complete' and r['round']==round_id and r['pair_index']==index]
            for v in variants:
                vv=[r['latency_ms'] for r in records if r['variant']==v['name'] and r['boundary']=='complete' and r['round']==round_id and r['pair_index']==index]
                paired.append(dict(round=round_id,pair_index=index,variant=v['name'],baseline_median_ms=float(np.median(baseline)),
                    variant_median_ms=float(np.median(vv)),paired_delta_ms=float(np.median(vv)-np.median(baseline))))
    write_csv(OUT/'latency_paired_differences.csv',paired)
    (OUT/'timing_environment.json').write_text(json.dumps(dict(gpu_uuid=a.gpu_uuid,pid=os.getpid(),
        torch=torch.__version__,cuda=torch.version.cuda,protocol=spec,hardware_before=before,hardware_after=query(hw_args),
        no_hooks=True,no_feature_cache=True,all_formal_calls_original_sample_windows=True,exclusive_guards=guards,
        source_sha256={x:sha(x) for x in ['scripts/benchmark_sparse_refinement.py','sglm/sparse_refinement.py','sglm/t02/model.py','sglm/model.py']},
        stage_warning='separate synchronized diagnostic pass; local remainder includes instrumentation overhead; never sum stages as formal end-to-end latency'),indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
