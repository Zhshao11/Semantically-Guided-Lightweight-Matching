"""Encoder profiler, predeclared BN-fold gate and separate repeated no-hook timing."""
import argparse
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from sglm.t02.data import AffinePairs256
from sglm.t06_runtime import load_selection_model
from sglm.inference_fusion import folded_inference_copy
from sglm.geometry import transform_points
from sglm.evaluation import estimate_from_correspondences
from sglm.experiment_io import require_exclusive_gpu,write_csv


def fit_prediction(model,args,precision,cfg):
    torch.backends.cudnn.allow_tf32=precision=='BF16'
    with torch.autocast('cuda',enabled=precision=='BF16',dtype=torch.bfloat16):p=model.predict(*args)[0]
    arrays=[p[k].float().cpu().numpy() for k in ['source','target','score']]
    fit=estimate_from_correspondences(*arrays,cfg['evaluation'],seed=cfg['seed'])
    return p,fit


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['profile','foldcheck','timing'],required=True)
    ap.add_argument('--root',required=True);ap.add_argument('--output',default='reports/t06_efficiency_v1');ap.add_argument('--gpu-uuid')
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);p=json.loads(Path('configs/t06_protocol_v1.json').read_text())
    def guard():
        def query(args):return subprocess.run(['nvidia-smi',*args],check=True,capture_output=True,text=True).stdout
        require_exclusive_gpu(a.gpu_uuid or '',query(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),os.getpid(),query(['--query-gpu=uuid','--format=csv,noheader']).split())
    if a.phase=='timing':guard();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_uuid
    torch.set_num_threads(4);torch.manual_seed(p['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    saved=torch.load(p['matcher_checkpoint'],map_location='cpu',weights_only=False);cfg=saved['config']
    ds=AffinePairs256(a.root,p['validation_plan'],split='val',preload=False);subset=json.loads(Path('reports/t05_frozen_v1/subsets.json').read_text())['timing_indices']
    def arguments(i):
        item=ds[i];return item,[item[k][None].cuda() for k in ['optical','sar_aug','optical_valid_mask','sar_aug_valid_mask']]
    if a.phase=='profile':
        model=load_selection_model(saved,'U',256);counts={};handles=[]
        import torch.nn.functional as functional
        real_interpolate=functional.interpolate;interpolate_calls=[0]
        def count_interpolate(*args,**kwargs):interpolate_calls[0]+=1;return real_interpolate(*args,**kwargs)
        for name,module in model.backbone.named_modules():
            if isinstance(module,(torch.nn.Conv2d,torch.nn.BatchNorm2d)):
                def count(m,args,out,key=name):counts[key]=counts.get(key,0)+1
                handles.append(module.register_forward_hook(count))
        items=[arguments(subset[i]) for i in [0,7,15,23]]
        functional.interpolate=count_interpolate
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for _,args in items:model.encode(*args)
        functional.interpolate=real_interpolate
        for h in handles:h.remove()
        counts_per_pair={key:value/4 for key,value in counts.items()}
        from torch.profiler import profile,ProfilerActivity
        with torch.inference_mode(),profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
            for _,args in items:
                with torch.autocast('cuda',dtype=torch.bfloat16):model.encode(*args)
            torch.cuda.synchronize()
        rows=[]
        for event in prof.key_averages():
            rows.append(dict(operator=event.key,calls=event.count,calls_per_pair=event.count/4,
                cpu_self_ms=event.self_cpu_time_total/1000/4,cpu_inclusive_ms=event.cpu_time_total/1000/4,
                gpu_inclusive_ms=getattr(event,'device_time_total',0)/1000/4))
        kernels={}
        for event in prof.events():
            if str(event.device_type)=='DeviceType.CUDA':
                entry=kernels.setdefault(event.name,dict(kernel=event.name,calls=0,total_ms=0.))
                entry['calls']+=1;entry['total_ms']+=(event.time_range.end-event.time_range.start)/1000
        trace=Path('runs/t06_efficiency_v1');trace.mkdir(exist_ok=True);prof.export_chrome_trace(str(trace/'encoder_trace.json'))
        write_csv(out/'encoder_operators.csv',rows)
        write_csv(out/'encoder_GPU_kernels.csv',kernels.values())
        result=dict(profile_pairs=[item['pair_id'] for item,_ in items],module_calls_per_pair=counts_per_pair,
            Conv2d_calls_per_pair=sum(v for key,v in counts_per_pair.items() if isinstance(dict(model.backbone.named_modules())[key],torch.nn.Conv2d)),
            BatchNorm_calls_per_pair=sum(v for key,v in counts_per_pair.items() if isinstance(dict(model.backbone.named_modules())[key],torch.nn.BatchNorm2d)),
            Python_interpolate_calls_per_pair=interpolate_calls[0]/4,GPU_kernel_events_per_pair=sum(v['calls'] for v in kernels.values())/4,
            GPU_kernel_active_sum_ms_per_pair=sum(v['total_ms'] for v in kernels.values())/4,
            operators=rows,ATen_count_note='AMP may create nested dispatch events (upsample8 events vs4 Python calls); do not sum nested inclusive durations',profiler_used_only_for_analysis=True,formal_timing=False,trace_path=str(trace/'encoder_trace.json'),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
        (out/'encoder_profile.json').write_text(json.dumps(result,indent=2));return
    modes=['U','A','AC','AC-cap']
    if a.phase=='foldcheck':
        records=[];yy,xx=np.mgrid[7.5:256:16,7.5:256:16];probes=np.stack([xx.ravel(),yy.ravel()],1)
        with torch.inference_mode():
            for precision in ['FP32','BF16']:
                torch.backends.cudnn.allow_tf32=precision=='BF16'
                tol=p['fold_tolerances'][precision]
                for k in p['budgets']:
                    for mode in modes:
                        original=load_selection_model(saved,mode,k);fold=folded_inference_copy(original)
                        for index in subset:
                            item,args=arguments(index)
                            with torch.autocast('cuda',enabled=precision=='BF16',dtype=torch.bfloat16):
                                fo=original.encode(*args);ff=fold.encode(*args)
                            delta=max(float((fo[key].float()-ff[key].float()).abs().max()) for key in ['optical_descriptors','sar_descriptors'])
                            po,eo=fit_prediction(original,args,precision,cfg);pf,ef=fit_prediction(fold,args,precision,cfg)
                            def pairs(pred):return {tuple(x) for x in torch.cat([pred['source'],pred['coarse_target']],1).cpu().numpy().tolist()}
                            so,sf=pairs(po),pairs(pf);jaccard=len(so&sf)/max(len(so|sf),1) if so or sf else 1.
                            ot={tuple(s):t for s,t in zip(po['source'].cpu().numpy(),po['target'].cpu().numpy())};ft={tuple(s):t for s,t in zip(pf['source'].cpu().numpy(),pf['target'].cpu().numpy())}
                            common=ot.keys()&ft.keys();errors=[np.linalg.norm(ot[key]-ft[key]) for key in common];fine95=float(np.percentile(errors,95)) if errors else 0. if not ot and not ft else None
                            if eo['matrix'] is not None and ef['matrix'] is not None:matrix_delta=float(np.sqrt(np.mean(np.sum((transform_points(probes,eo['matrix'])-transform_points(probes,ef['matrix']))**2,1))))
                            else:matrix_delta=0. if eo['matrix'] is None and ef['matrix'] is None and eo['status']==ef['status'] else None
                            passed=delta<=tol['descriptor_max_abs'] and jaccard>=tol['coarse_pair_Jaccard_min'] and fine95 is not None and fine95<=tol['common_refinement_p95_px'] and matrix_delta is not None and matrix_delta<=tol['matrix_probe_RMSE_px']
                            records.append(dict(precision=precision,K=k,mode=mode,pair_id=item['pair_id'],descriptor_max_abs=delta,coarse_Jaccard=jaccard,
                                common_refinement_p95_px=fine95,matrix_probe_RMSE_px=matrix_delta,passed=passed,
                                original_parameters=sum(t.numel() for t in original.parameters()),folded_parameters=sum(t.numel() for t in fold.parameters())))
                        print('foldchecked',precision,k,mode,flush=True);del original,fold
        gates={precision:all(r['passed'] for r in records if r['precision']==precision) for precision in ['FP32','BF16']}
        (out/'fold_consistency.json').write_text(json.dumps(dict(tolerances=p['fold_tolerances'],records=records,global_gate=gates,
            original_checkpoint_replaced=False,modal_BN_folded_independently=True,cudnn_TF32_policy={'FP32':False,'BF16':True},policy='No tolerance relaxation; family-wide gate before optimizing any custom baseline'),indent=2))
        write_csv(out/'fold_consistency.csv',records);return
    gates=json.loads((out/'fold_consistency.json').read_text())['global_gate'];models={};configs=[]
    for k in p['budgets']:
        for mode in modes:
            for precision in ['BF16','FP32']:
                if precision=='FP32' and not gates['FP32']:continue
                original=load_selection_model(saved,mode,k);name=f'{mode}_K{k}_{precision}_original';models[name]=original;configs.append(dict(name=name,mode=mode,K=k,precision=precision,folded=False))
                if gates[precision]:
                    name=f'{mode}_K{k}_{precision}_folded';models[name]=folded_inference_copy(original);configs.append(dict(name=name,mode=mode,K=k,precision=precision,folded=True))
    assert all(not m._forward_hooks and not m._forward_pre_hooks for model in models.values() for m in model.modules())
    rng=random.Random(p['seed']);records=[]
    def measured(model,args,precision):
        torch.backends.cudnn.allow_tf32=precision=='BF16'
        torch.cuda.synchronize();t=time.perf_counter();pred,fit=fit_prediction(model,args,precision,cfg);torch.cuda.synchronize()
        elapsed=(time.perf_counter()-t)*1000
        return elapsed,len(pred['source']),fit['inliers'],fit['status']
    with torch.inference_mode():
        for round_id in range(p['efficiency']['rounds']):
            for ordinal,index in enumerate(subset):
                guard();item,args=arguments(index)
                for conf in configs:
                    for _ in range(p['efficiency']['warmup']):fit_prediction(models[conf['name']],args,conf['precision'],cfg)
                for repeat in range(p['efficiency']['repeats_per_pair_round']):
                    order=configs.copy();rng.shuffle(order)
                    for conf in order:
                        elapsed,matches,inliers,status=measured(models[conf['name']],args,conf['precision'])
                        records.append(dict(**conf,pair_id=item['pair_id'],round=round_id,repeat=repeat,ms=elapsed,matches=matches,inliers=inliers,status=status))
                guard();print('timed',round_id+1,ordinal+1,24,flush=True)
    summaries=[];paired=[]
    for conf in configs:
        g=[r for r in records if r['name']==conf['name']];medians=[]
        for uid in sorted({r['pair_id'] for r in g}):
            for rnd in range(p['efficiency']['rounds']):medians.append(dict(pair_id=uid,round=rnd,median_ms=float(np.median([r['ms'] for r in g if r['pair_id']==uid and r['round']==rnd]))))
        summaries.append(dict(**conf,median_ms=float(np.median([r['ms'] for r in g])),p90_ms=float(np.percentile([r['ms'] for r in g],90)),mean_pair_round_median_ms=float(np.mean([r['median_ms'] for r in medians])),per_pair_round=medians))
    for r in summaries:
        if not r['folded']:continue
        ref=next(v for v in summaries if v['mode']==r['mode'] and v['K']==r['K'] and v['precision']==r['precision'] and not v['folded'])
        baseline={(v['pair_id'],v['round']):v['median_ms'] for v in ref['per_pair_round']}
        for v in r['per_pair_round']:paired.append(dict(mode=r['mode'],K=r['K'],precision=r['precision'],pair_id=v['pair_id'],round=v['round'],delta_fold_minus_original_ms=v['median_ms']-baseline[(v['pair_id'],v['round'])]))
    write_csv(out/'nohook_repeats.csv',records);write_csv(out/'fold_paired_deltas.csv',paired)
    (out/'timing_summary.json').write_text(json.dumps(dict(configs=configs,summary=summaries,fold_gate=gates,rounds=3,pairs=24,no_hooks=True,
        formal_gpu_uuid=a.gpu_uuid,profiler_used_in_timing=False,all_custom_baselines_same_gate=True,cudnn_TF32_policy={'FP32':False,'BF16':True},excludes='I/O,warp,H2D; includes score/selection/encode/matching/refinement/CPUcopy/RANSAC'),indent=2))


if __name__=='__main__':main()
