"""LoveDA native IDs: discard ground-truth 0, never discard predicted 0."""
import numpy as np


def loveda_confusion(labels,predictions):
    labels=np.asarray(labels);predictions=np.asarray(predictions)
    if labels.shape!=predictions.shape or labels.ndim!=2:raise ValueError('Expected paired native HW label maps')
    if not np.isin(labels,np.arange(8)).all() or not np.isin(predictions,np.arange(8)).all():
        raise ValueError('LoveDA native labels/predictions must be 0..7; verify mapping before evaluation')
    valid=labels!=0
    return np.bincount((labels[valid].astype(int)*8+predictions[valid].astype(int)).ravel(),minlength=64).reshape(8,8)


def loveda_metrics(confusion,names=None):
    c=np.asarray(confusion,dtype=np.int64)
    if c.shape!=(8,8) or c[0].any():raise ValueError('GT Ignore row must be excluded')
    names=names or ['Ignore','Background','Building','Road','Water','Barren','Forest','Agricultural']
    total=int(c.sum());classes=[]
    for i in range(1,8):
        tp=int(c[i,i]);gt=int(c[i].sum());pred=int(c[:,i].sum());union=gt+pred-tp
        classes.append(dict(id=i,name=names[i],GT_pixels=gt,predicted_pixels=pred,
            IoU=tp/union if union else None,precision=tp/pred if pred else None,recall=tp/gt if gt else None))
    defined=[r['IoU'] for r in classes if r['IoU'] is not None]
    return dict(valid_GT_pixels=total,overall_accuracy=float(np.trace(c)/total) if total else None,
        mean_IoU=float(np.mean(defined)) if defined else None,classes_in_mIoU=len(defined),per_class=classes,
        GT_background_fraction=float(c[1].sum()/total) if total else None,
        predicted_background_fraction_on_valid_GT=float(c[:,1].sum()/total) if total else None,
        predicted_Ignore_fraction_on_valid_GT=float(c[:,0].sum()/total) if total else None,
        confusion=c.tolist(),definition='Ignore GT=0 excluded; predictions=0 remain false negatives on GT1..7; no class shifting')
