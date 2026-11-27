"""Frozen-protocol setup and post-prediction diagnostics, outside timed prediction."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from .sparse_refinement import SparseRefinementB0
from .diagnostics import coverage
from .geometry import transform_points
from .evaluation import estimate_from_correspondences,fixed_synthetic_probes,evaluate_transform


OUT=Path('reports/sparse_refinement_v1')
PRIVATE=Path('runs/sparse_refinement_v1')
KEYS=('optical','sar_aug','optical_valid_mask','sar_aug_valid_mask')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    p=json.loads(Path('configs/sparse_refinement_v1.json').read_text());freeze=json.loads((OUT/'freeze.json').read_text())
    assert sha(freeze['checkpoint'])==freeze['checkpoint_sha256']==p['checkpoint_sha256']
    for path,digest in freeze['inputs_sha256'].items():assert sha(path)==digest,path
    saved=torch.load(freeze['checkpoint'],map_location='cpu',weights_only=False);cfg=saved['config']
    torch.set_num_threads(p['timing']['threads']);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    model=SparseRefinementB0(cfg).cuda().eval();model.load_state_dict(saved['model'],strict=True);model.requires_grad_(False)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    return p,cfg,model


def arrays(pred):return [pred[k].float().cpu().numpy() for k in ('source','target','score')]


def measure_pair(item,pred,cfg,variant,parent,coarse=None):
    source,target,score=arrays(pred)
    fit=estimate_from_correspondences(source,target,score,cfg['evaluation'],seed=cfg['seed'],include_inlier_indices=True)
    # Independent probes and all GT diagnostics happen ONLY after inference/fit.
    probes,truth=fixed_synthetic_probes(item['A_forward'].numpy(),grid=cfg['evaluation']['probe_grid'],
        optical_mask=item['optical_valid_mask'],sar_aug_mask=item['sar_aug_valid_mask'])
    ev=evaluate_transform(fit['matrix'],probes,truth,cfg['evaluation']['success_thresholds_px'],status=fit['status'])
    indices=np.asarray(fit['inlier_indices'],int)
    cov=dict(coverage_opt=coverage(source),coverage_sar=coverage(target),
             inlier_coverage_opt=coverage(source[indices]),inlier_coverage_sar=coverage(target[indices]))
    before=pred['coarse_target'].float().cpu().numpy();nominal=transform_points(source,item['A_forward'].numpy())
    err=np.linalg.norm(before-nominal,axis=1);after=np.linalg.norm(target-nominal,axis=1)
    inside=(np.abs(before-nominal)<=6+1e-6).all(1)
    r=dict(pair_id=item['pair_id'],parent_id=parent,variant=variant['name'],mode=variant['mode'],budget=variant['budget'],
        num_coarse_candidates=pred['num_coarse_candidates'],num_selected=pred['attempted_refinements'],
        num_refined_valid=len(source),num_ransac_inliers=fit['inliers'],fit_success=fit['success'],
        failure_reason='' if fit['success'] else fit['status'],global_probe_error=ev['rmse_px'],
        registration_success=ev['success']['5'],success3=ev['success']['3'],registration_failure_reason='' if ev['success']['5'] else fit['status'] if not fit['success'] else 'probe_RMSE_ge5px',
        coarse_correct_fraction5=float((err<5).mean()) if len(err) else None,
        window_hit_fraction=float(inside.mean()) if len(err) else None,
        window_inside_refine_error_px=float(after[inside].mean()) if inside.any() else None,
        window_outside_refine_error_px=float(after[~inside].mean()) if (~inside).any() else None,
        evaluation=ev,estimation=fit,**cov)
    raw=dict(source=source,target=target,score=score,coarse_target=before,
        original_candidate_idx=pred['original_candidate_idx'].cpu().numpy(),selected_candidate_idx=pred['selected_candidate_idx'].cpu().numpy(),
        probes=probes,probe_truth=truth,A_forward=item['A_forward'].numpy())
    if coarse is not None:raw.update(coarse_source=coarse['source'].float().cpu().numpy(),coarse_target_all=coarse['target'].float().cpu().numpy())
    return r,raw


def flat_record(r):
    row={k:v for k,v in r.items() if k not in ['evaluation','estimation','coverage_opt','coverage_sar','inlier_coverage_opt','inlier_coverage_sar']}
    for key in ['coverage_opt','coverage_sar','inlier_coverage_opt','inlier_coverage_sar']:
        for field,value in r[key].items():row[key+'.'+field]=json.dumps(value) if isinstance(value,list) else value
    return row
