"""Independent cache/coordinate/geometry replay of all952 frozen-map comparisons."""
import csv,hashlib,json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np,torch
from sglm.heatmap_readout import read_frozen_heatmap
from sglm.geometry import transform_points
from sglm.evaluation import estimate_from_correspondences,evaluate_transform


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=json.loads(Path('configs/heatmap_readout_v1.json').read_text());out=Path(p['output']);freeze=json.loads((out/'freeze.json').read_text());source=json.loads(Path(p['source_freeze']).read_text());cfg=source['config']
    for f,h in freeze['input_sha256'].items():assert sha(f)==h,f
    records=json.loads((out/'pair_results.json').read_text());assert len(records)==952
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    totals=0;hits=0;success={g:{'3':0,'5':0} for g in ['A','B','C']};max_C_quantization_error=0.
    for meta,row in zip(source['pairs'],records):
        assert meta['pair_id']==row['pair_id'] and sha(meta['path'])==meta['sha256'];c=np.load(meta['path']);r=np.load(Path(p['private'])/f"{meta['index']:04d}.npz");mask=c['frozen_fit_mask'];N=len(c['source']);totals+=int(mask.sum());hits+=int(c['hit'][mask].sum())
        for key,expected in [('candidate_id',c['candidate_id'][mask]),('source',c['source'][mask]),('confidence',c['confidence'][mask]),('original_fit_mask',mask),('hit',c['hit'][mask])]:assert np.array_equal(r[key],expected),(meta['pair_id'],key)
        assert np.array_equal(c['nominal_truth'],transform_points(c['source'],c['A_forward']))
        args=[torch.from_numpy(c[key]).cuda() for key in ['local_heatmap','window_token_valid','coarse_center','window_positions']]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):new=read_frozen_heatmap(*args)
        assert torch.equal(new['A'],torch.from_numpy(c['target']).cuda())
        # Independent row/column selection, not the implementation's neighborhood helper.
        h=c['local_heatmap'][mask];valid=c['window_token_valid'][mask];side=int(h.shape[1]**.5);peak=np.where(valid,h,-np.inf).argmax(-1)
        assert np.array_equal(peak,r['peak_index'])
        locations=np.arange(side*side);local=valid&(np.abs(locations[None]//side-peak[:,None]//side)<=1)&(np.abs(locations[None]%side-peak[:,None]%side)<=1)
        assert np.array_equal(local,r['C_local_mask']) and (local&~valid).sum()==0
        assert np.array_equal(r['B'],c['window_positions'][mask][np.arange(len(peak)),peak])
        p64=h.astype(float)*local;mass=p64.sum(-1);conditional=p64/mass[:,None];delta=(c['window_positions'][mask]-c['coarse_center'][mask,None])/2
        mathematical=c['coarse_center'][mask]+2*(conditional[:,:,None]*delta).sum(1)
        if len(mathematical):max_C_quantization_error=max(max_C_quantization_error,float(np.abs(mathematical-r['C']).max()))
        for g in ['A','B','C']:
            assert np.array_equal(new[g].float().cpu().numpy()[mask],r[g])
            fit=estimate_from_correspondences(r['source'],r[g],r['confidence'],cfg['evaluation'],seed=cfg['seed'],include_inlier_indices=True)
            ev=evaluate_transform(fit['matrix'],c['probes'],c['probe_truth'],cfg['evaluation']['success_thresholds_px'],status=fit['status'])
            assert fit==row['branches'][g]['estimation'] and ev==row['branches'][g]['evaluation']
            for threshold in ['3','5']:success[g][threshold]+=int(ev['success'][threshold])
            errors=np.linalg.norm(r[g]-r['nominal_truth'],axis=1);assert np.array_equal(errors,r[g+'_error'])
    assert totals==122881 and hits==45974
    assert sum(1 for _ in csv.DictReader((Path(p['private'])/'all45974_inside_correspondences.csv').open()))==45974
    assert '161 passed' in (out/'regression_tests.txt').read_text()
    result=dict(passed=True,pairs=952,candidates=totals,inside_hit_count=hits,success_counts=success,originalA_bitwise_reconstructed=True,
        allthree_readouts_on_all_original_candidates=True,all952_geometry_and_probe_results_replayed=True,peak_and_localmask_independently_checked=True,
        no_original_heatmap_or_filter_change=True,source_confidence_ID_order_mask_unchanged=True,GTnotin_readout_API=True,no_model_or_training=True,no_test=True,
        max_C_BF16_coordinate_difference_from_FP64_conditional_expectation_px=max_C_quantization_error,
        decision='retain_A; no extra onlinecost experiment or training',source_sha256={str(f):sha(f) for f in [Path('sglm/heatmap_readout.py'),*sorted(Path('scripts').glob('*heatmap_readout*.py')),Path('tests/test_heatmap_readout.py')]})
    (out/'final_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
