"""Independent affine evaluator: NumPy + existing geometry only; no model/Torch imports.

Fit uses only predicted correspondences. Evaluation points are supplied separately.
Failed estimates always remain in the image-pair denominator.
"""
import cv2
import numpy as np
from .geometry import estimate_affine,as_homogeneous,transform_points,valid_pixel_centers


def estimate_from_correspondences(source,target,scores,config,seed=0,include_inlier_indices=False):
    source,target=np.asarray(source,dtype=float).reshape(-1,2),np.asarray(target,dtype=float).reshape(-1,2)
    scores=np.asarray(scores,dtype=float)
    if source.shape!=target.shape or scores.shape!=(len(source),):
        raise ValueError('Expected predicted Nx2/Nx2/N correspondences')
    def failure(reason):
        result=dict(success=False,status=reason,matrix=None,inliers=0,matches=len(source))
        if include_inlier_indices:
            result['inlier_indices']=[]
        return result
    if not len(source):
        return failure('empty_matches')
    valid=np.isfinite(source).all(1)&np.isfinite(target).all(1)&np.isfinite(scores)&(scores>=config['minimum_match_score'])
    if valid.sum()<3:
        return failure('insufficient_matches')
    if np.linalg.matrix_rank(source[valid]-source[valid].mean(0))<2:
        return failure('degenerate_matches')
    cv2.setRNGSeed(int(seed)%(2**31-1))
    matrix,inliers=estimate_affine(source,target,scores,threshold=config['ransac_threshold_px'],
        min_reliability=config['minimum_match_score'],max_iterations=config['ransac_max_iterations'],confidence=config['ransac_confidence'])
    if matrix is None or inliers.sum()<config['ransac_min_inliers']:
        return failure('ransac_failed_or_insufficient_inliers')
    result=dict(success=True,status='estimated',matrix=matrix.tolist(),inliers=int(inliers.sum()),matches=len(source))
    if include_inlier_indices:
        result['inlier_indices']=np.flatnonzero(inliers).tolist()
    return result


def _mask_at(points,mask):
    if mask is None:
        return np.ones(len(points),dtype=bool)
    mask=np.asarray(mask,dtype=np.float32)
    sampled=cv2.remap(mask,points[:,0].astype(np.float32)[:,None],points[:,1].astype(np.float32)[:,None],
        interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return sampled[:,0]>.999


def fixed_synthetic_probes(A_forward,width=256,height=256,grid=(16,16),optical_mask=None,sar_aug_mask=None):
    # Centers of a fixed 16x16 evaluation lattice, not the model's 32x32 coarse lattice.
    y,x=np.meshgrid((np.arange(grid[0])+.5)*height/grid[0]-.5,
                     (np.arange(grid[1])+.5)*width/grid[1]-.5,indexing='ij')
    source=np.stack([x.ravel(),y.ravel()],-1)
    target=transform_points(source,A_forward)
    valid=valid_pixel_centers(target,width,height)&_mask_at(source,optical_mask)&_mask_at(target,sar_aug_mask)
    return source[valid],target[valid]


def evaluate_transform(matrix,source_evaluation_points,target_evaluation_points,thresholds=(1,3,5),status=None):
    source,target=np.asarray(source_evaluation_points,float),np.asarray(target_evaluation_points,float)
    if source.ndim!=2 or source.shape[1:]!=(2,) or source.shape!=target.shape or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Evaluation references must be finite paired Nx2 points')
    if matrix is None:
        return dict(solved=False,status=status or 'estimation_failed',evaluation_points=len(source),rmse_px=None,
                    success={str(t):False for t in thresholds})
    try:
        as_homogeneous(matrix)
    except ValueError:
        return dict(solved=False,status='invalid_matrix',evaluation_points=len(source),rmse_px=None,success={str(t):False for t in thresholds})
    if not len(source):
        return dict(solved=False,status='no_valid_evaluation_points',evaluation_points=0,rmse_px=None,success={str(t):False for t in thresholds})
    errors=np.linalg.norm(transform_points(source,matrix)-target,axis=1)
    rmse=float(np.sqrt(np.mean(errors**2)))
    return dict(solved=True,status='evaluated',evaluation_points=len(source),rmse_px=rmse,
                median_px=float(np.median(errors)),success={str(t):bool(rmse<t) for t in thresholds})


def summarize_evaluations(records,thresholds=(1,3,5)):
    if not records:
        raise ValueError('No image-pair evaluations')
    valid=[r['rmse_px'] for r in records if r['solved'] and r['rmse_px'] is not None]
    result=dict(image_pairs=len(records),solved_pairs=len(valid),failed_pairs=len(records)-len(valid),
        failure_rate=(len(records)-len(valid))/len(records),
        solved_mean_rmse_px=float(np.mean(valid)) if valid else None,
        solved_median_rmse_px=float(np.median(valid)) if valid else None)
    for threshold in thresholds:
        successful=[r['rmse_px'] for r in records if r['success'][str(threshold)]]
        result[f'success_rate@{threshold}px']=len(successful)/len(records)
        result[f'registration_failure_rate@{threshold}px']=1-len(successful)/len(records)
        result[f'successful_mean_rmse@{threshold}px']=float(np.mean(successful)) if successful else None
    return result
