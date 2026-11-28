"""Post-hoc ideal localization within the unchanged refiner's continuous support.

Convex-hull logic is NumPy float64, independent of prediction and robust fitting.
Torch imports are local to sampling/mask introspection; model code is untouched.
"""
import hashlib
import numpy as np


def convex_hull(points):
    points=np.asarray(points,np.float64).reshape(-1,2)
    if not np.isfinite(points).all():raise ValueError('Nonfinite support')
    ordered=sorted(set(map(tuple,points.tolist())))
    if len(ordered)<2:return np.asarray(ordered,np.float64).reshape(-1,2)
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower=[];upper=[]
    for p in ordered:
        while len(lower)>1 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    for p in reversed(ordered):
        while len(upper)>1 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1]+upper[:-1],np.float64).reshape(-1,2)


def inside_convex_support(point,valid_points,tolerance_px=1e-6):
    if tolerance_px<0:raise ValueError('Negative numerical tolerance')
    point=np.asarray(point,np.float64);hull=convex_hull(valid_points)
    if not len(hull) or not np.isfinite(point).all():return False
    if len(hull)==1:return bool(np.linalg.norm(point-hull[0])<=tolerance_px)
    edge=np.roll(hull,-1,axis=0)-hull
    if len(hull)==2:
        vector=hull[1]-hull[0];length=np.linalg.norm(vector);projection=np.dot(point-hull[0],vector)/length
        distance=abs(np.cross(vector,point-hull[0]))/length
        return bool(distance<=tolerance_px and -tolerance_px<=projection<=length+tolerance_px)
    # CCW polygon; distances have pixel units, so tolerance never scales with area.
    signed=(edge[:,0]*(point[1]-hull[:,1])-edge[:,1]*(point[0]-hull[:,0]))/np.linalg.norm(edge,axis=1)
    return bool((signed>=-tolerance_px).all())


def support_hits(truth,positions,valid_tokens,source_valid,target_valid,tolerance_px=1e-6):
    truth=np.asarray(truth,np.float64).reshape(-1,2);positions=np.asarray(positions,np.float64)
    valid_tokens=np.asarray(valid_tokens,bool);source_valid=np.asarray(source_valid,bool);target_valid=np.asarray(target_valid,bool)
    if positions.shape!=(len(truth),valid_tokens.shape[1],2) or valid_tokens.shape[0]!=len(truth):raise ValueError('Misaligned windows')
    return np.asarray([bool(source_valid[i] and target_valid[i] and inside_convex_support(truth[i],positions[i,valid_tokens[i]],tolerance_px)) for i in range(len(truth))],bool)


def sample_actual_support(optical_fine,sar_fine,source,centers,optical_mask,sar_mask,radius):
    import torch
    from .model import sample_windows
    with torch.no_grad():
        _,_,source_valid=sample_windows(optical_fine.float(),source.float(),0,optical_mask)
        _,delta,valid=sample_windows(sar_fine.float(),centers.float(),radius,sar_mask)
        # Inverse of original sample_windows pixel->feature positions; not a guessed radius.
        feature=(centers.float()[:,None]+.5)/2-.5+delta[None]
        physical=2*(feature+.5)-.5
        readout_positions=centers.float()[:,None]+2*delta[None]
        assert torch.equal(physical,readout_positions)
    return physical,valid,source_valid[:,0]


def mask_valid_at(points,mask):
    import torch
    import torch.nn.functional as F
    xy=torch.as_tensor(points,dtype=torch.float32,device=mask.device).reshape(-1,2)
    if not len(xy):return np.zeros(0,bool)
    h,w=mask.shape[-2:];bounds=torch.isfinite(xy).all(-1)&(xy>=0).all(-1)&(xy<=xy.new_tensor([w-1,h-1])).all(-1)
    grid=(2*(xy+.5)/xy.new_tensor([w,h])-1)[None,None]
    sampled=F.grid_sample(mask.reshape(1,1,h,w).float(),grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0,0,0]
    return (bounds&(sampled>.999)).cpu().numpy()


def frozen_fit_mask(source,target,confidence,minimum_score):
    source=np.asarray(source);target=np.asarray(target);confidence=np.asarray(confidence)
    return np.isfinite(source).all(1)&np.isfinite(target).all(1)&np.isfinite(confidence)&(confidence>=minimum_score)


def replace_targets(target,nominal_truth,hit):
    target=np.asarray(target);truth=np.asarray(nominal_truth,np.float64);hit=np.asarray(hit,bool)
    if target.shape!=truth.shape or hit.shape!=(len(target),):raise ValueError('Misaligned oracle replacement')
    replaced=target.astype(np.float64).copy();replaced[hit]=truth[hit]
    assert np.array_equal(replaced[~hit],target[~hit]) and np.array_equal(replaced[hit],truth[hit])
    return replaced


def coverage(points):
    from .diagnostics import coverage as existing_coverage
    points=np.asarray(points,float).reshape(-1,2);result=existing_coverage(points)
    centered=points-points.mean(0) if len(points) else points
    if len(points)>=3:
        values=np.maximum(np.linalg.eigvalsh(centered.T@centered/len(points)),0)
        ratio=float(values[0]/values[1]) if values[1]>0 else 0.
    else:values=np.array([0.,0.]);ratio=None
    return dict(**result,less_than_three_points=len(points)<3,covariance_eigenvalues_px2=values.tolist(),minor_major_eigenvalue_ratio=ratio)


def array_digest(*arrays):
    h=hashlib.sha256()
    for array in arrays:
        a=np.ascontiguousarray(array);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
