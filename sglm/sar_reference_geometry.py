"""Pure NumPy/OpenCV geometry for external dense-flow interfaces; no model imports."""
import cv2
import numpy as np


def restore_resized_coordinates(points,original_hw,native_hw):
    h,w=original_hw;nh,nw=native_hw
    return (np.asarray(points,dtype=np.float32)+.5)*np.array([w/nw,h/nh],np.float32)-.5


def dense_correspondences(flow,original_hw=(256,256)):
    flow=np.asarray(flow,dtype=np.float32)
    if flow.ndim!=3 or flow.shape[0]!=2:raise ValueError('Expected optical-to-SAR2xHxW flow')
    h,w=flow.shape[1:];yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij');grid=np.stack([xx,yy],-1).astype(np.float32)
    source=restore_resized_coordinates(grid.reshape(-1,2),original_hw,(h,w))
    target=restore_resized_coordinates((grid+flow.transpose(1,2,0)).reshape(-1,2),original_hw,(h,w))
    return source,target,np.ones(len(source),dtype=np.float32)


def filter_correspondences(source,target,score,optical_mask,sar_mask,chunk_size=16384):
    # OpenCV remap destination dimensions must be <32767. No selection/truncation.
    if chunk_size<=0 or chunk_size>=32767:raise ValueError('Invalid remap chunk size')
    source=np.asarray(source,np.float32).reshape(-1,2);target=np.asarray(target,np.float32).reshape(-1,2);score=np.asarray(score,np.float32).reshape(-1)
    if source.shape!=target.shape or len(source)!=len(score):raise ValueError('Misaligned external output')
    valid=np.isfinite(source).all(1)&np.isfinite(target).all(1)&np.isfinite(score)
    for xy,mask in [(source,optical_mask),(target,sar_mask)]:
        h,w=mask.shape;valid&=(xy[:,0]>=0)&(xy[:,0]<=w-1)&(xy[:,1]>=0)&(xy[:,1]<=h-1)
        for i in range(0,len(xy),chunk_size):
            safe=np.where(np.isfinite(xy[i:i+chunk_size]),xy[i:i+chunk_size],-1000).astype(np.float32)
            sampled=cv2.remap(mask.astype(np.float32),safe[:,0,None],safe[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)[:,0]
            valid[i:i+len(safe)]&=sampled>.999
    return source[valid],target[valid],score[valid],valid



def sample_dense_targets(flow,source_probes,original_hw=(256,256)):
    flow=np.asarray(flow,np.float32);h,w=flow.shape[1:];oh,ow=original_hw
    source=np.asarray(source_probes,np.float32).reshape(-1,2)
    native=(source+.5)*np.array([w/ow,h/oh],np.float32)-.5
    delta=np.stack([cv2.remap(flow[c],native[:,0,None],native[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[:,0] for c in range(2)],-1)
    return restore_resized_coordinates(native+delta,original_hw,(h,w))


def evaluate_dense_probes(flow,source,truth,thresholds=(1,3,5)):
    predicted=sample_dense_targets(flow,source);truth=np.asarray(truth,float)
    if not len(predicted) or not np.isfinite(predicted).all():return dict(solved=False,status='nonfinite_or_empty_dense_prediction',rmse_px=None,evaluation_points=len(predicted),success={str(t):False for t in thresholds})
    errors=np.linalg.norm(predicted-truth,axis=1);rmse=float(np.sqrt(np.mean(errors**2)))
    return dict(solved=True,status='dense_probes',evaluation_points=len(predicted),rmse_px=rmse,median_px=float(np.median(errors)),success={str(t):bool(rmse<t) for t in thresholds})
