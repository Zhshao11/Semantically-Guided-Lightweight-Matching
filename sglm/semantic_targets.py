"""Standalone semantic distillation preparation, independent of matching labels/models."""
import math
import torch
import torch.nn.functional as F


def aggregate_semantic_targets(probabilities,pixel_valid_mask,class_mapping,factor=8,
        min_valid_fraction=1.,min_retained_mass=.8,min_confidence=.6):
    """Masked area means of source probabilities, then explicit class grouping.

    Input B,S,H,W, mask B,H,W, mapping T,S with nonnegative weights and column
    sums <=1. Excluded source classes (e.g. Ignore) remain lost probability mass;
    conditional normalization can never bypass the retained-mass mask.
    No hard argmax votes and no matching-correctness supervision are generated.
    """
    p=probabilities.float()
    valid=pixel_valid_mask.bool()
    mapping=torch.as_tensor(class_mapping,device=p.device,dtype=torch.float32)
    if p.ndim!=4 or valid.shape!=(p.shape[0],p.shape[2],p.shape[3]): raise ValueError('Expected BSHW and BHW')
    if not isinstance(factor,int) or factor<1 or p.shape[-1]%factor or p.shape[-2]%factor:
        raise ValueError('Spatial dimensions must be divisible by a positive factor')
    if mapping.ndim!=2 or mapping.shape[1]!=p.shape[1] or not torch.isfinite(mapping).all() or (mapping<0).any() or (mapping.sum(0)>1+1e-6).any():
        raise ValueError('Class mapping must be nonnegative TxS with each source mass counted at most once')
    if any(not 0<=v<=1 for v in (min_valid_fraction,min_retained_mass,min_confidence)):
        raise ValueError('Mask thresholds must lie in [0,1]')
    finite=torch.isfinite(p).all(1)
    if ((p<0)&torch.isfinite(p)&valid[:,None]).any(): raise ValueError('Negative source probabilities')
    clean=torch.where(torch.isfinite(p),p,0)
    sums=clean.sum(1)
    valid=valid&finite&(sums>0)
    clean=clean/sums[:,None].clamp_min(1e-12)
    clean=torch.where(valid[:,None],clean,0)
    fraction=F.avg_pool2d(valid[:,None].float(),factor,factor)
    source_mean=F.avg_pool2d(clean,factor,factor)/fraction.clamp_min(1e-12)
    grouped=torch.einsum('ts,bshw->bthw',mapping,source_mean)
    retained=grouped.sum(1)
    conditional=grouped/retained[:,None].clamp_min(1e-12)
    confidence=conditional.max(1).values
    mask=(fraction[:,0]>=min_valid_fraction)&(fraction[:,0]>0)&(retained>=min_retained_mass)&(confidence>=min_confidence)
    return dict(probabilities=conditional,mask=mask,valid_fraction=fraction[:,0],
        retained_mass=retained,excluded_class_mass=(source_mean.sum(1)-retained).clamp_min(0),confidence=confidence)


def teacher_logits_to_native_probabilities(logits,native_hw):
    if logits.ndim!=4 or min(native_hw)<1:raise ValueError('Expected BCHW logits and native height/width')
    # Explicit policy: resize logits first, softmax on native optical grid, then area aggregate.
    return F.interpolate(logits.float(),size=tuple(native_hw),mode='bilinear',align_corners=False).softmax(1)


def masked_semantic_kl(student_logits,targets,mask):
    if student_logits.shape!=targets.shape or mask.shape!=targets.shape[:1]+targets.shape[2:]:
        raise ValueError('Student/teacher/mask shape mismatch')
    if not torch.isfinite(student_logits).all():raise ValueError('Nonfinite student logits')
    teacher=targets.detach().float()
    mask=mask.bool()
    if not mask.any():return student_logits.sum()*0
    if not torch.isfinite(teacher).all() or (teacher<0).any() or not torch.allclose(teacher.sum(1)[mask],torch.ones_like(teacher.sum(1)[mask]),atol=1e-5):
        raise ValueError('Valid soft targets must be finite distributions')
    per_pixel=(teacher*(teacher.clamp_min(1e-12).log()-student_logits.float().log_softmax(1))).sum(1)
    return per_pixel[mask].mean()


def semantic_supervision_fields(aggregation):
    """Seven-class conditional target, retained mass and entropy-based weight.

    A high conditional confidence cannot recover probability excluded as Ignore.
    The pre-existing coverage/mass/confidence mask is applied independently.
    """
    p=aggregation['probabilities'];a=aggregation['retained_mass']
    if p.shape[1]!=7:raise ValueError('Expected seven land-cover classes without Ignore')
    entropy=-(p*p.clamp_min(1e-12).log()).sum(1)/math.log(7)
    reliability=(1-entropy).clamp(0,1)
    weight=aggregation['mask'].float()*a*reliability
    return dict(conditional_probabilities=p,valid_class_mass=a,distribution_reliability=reliability,
        supervision_weight=weight,effective_class_mass=p*weight[:,None])


def weighted_semantic_kl(student_logits,targets,weight):
    """Detached teacher/weights; divide by grid cells, preserving absolute reliability.

    Normalizing by sum(weight) would cancel a uniformly weak teacher's weight.
    This interface is prepared for later semantic heads; B0/A use no semantic loss.
    """
    if student_logits.shape!=targets.shape or weight.shape!=targets.shape[:1]+targets.shape[2:]:
        raise ValueError('Student/teacher/weight shape mismatch')
    teacher=targets.detach().float();weight=weight.detach().float()
    if not torch.isfinite(student_logits).all() or not torch.isfinite(weight).all() or (weight<0).any():
        raise ValueError('Finite logits and nonnegative finite weights required')
    active=weight>0
    if not active.any():return student_logits.sum()*0
    if not torch.isfinite(teacher).all() or (teacher<0).any() or not torch.allclose(teacher.sum(1)[active],torch.ones_like(weight[active]),atol=1e-5):
        raise ValueError('Active semantic targets must be distributions')
    per_cell=(teacher*(teacher.clamp_min(1e-12).log()-student_logits.float().log_softmax(1))).sum(1)
    return (per_cell*weight).mean()
