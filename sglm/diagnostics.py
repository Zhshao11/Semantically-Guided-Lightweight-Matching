"""Post-hoc layered matching diagnostics and inference-state auditing."""
import hashlib
import json
import cv2
import numpy as np
import torch
from torch import nn
from .model import ModalBN
from .geometry import transform_points,valid_pixel_centers
from .evaluation import estimate_from_correspondences,fixed_synthetic_probes,evaluate_transform,summarize_evaluations


def bn_snapshot(model):
    return {name:t.detach().cpu().clone() for name,t in model.named_buffers()
            if name.endswith(('running_mean','running_var','num_batches_tracked'))}


def bn_unchanged(before,model):
    after=bn_snapshot(model)
    return before.keys()==after.keys() and all(torch.equal(before[k],after[k]) for k in before)


class InferenceStateAudit:
    def __init__(self,model):
        self.model=model
        self.before=bn_snapshot(model)
        self.handles=[]
        self.active={}
        self.branch_counts={0:0,1:0}
        self.backbone_calls=[]
        self.adapter_calls={0:0,1:0}
        def backbone_hook(module,args):
            modality=int(args[1]);self.backbone_calls.append(modality);self.backbone_modality=modality
        self.handles.append(model.backbone.register_forward_pre_hook(backbone_hook))
        for name,module in model.named_modules():
            if isinstance(module,ModalBN):
                def modal_hook(mod,args,key=name):
                    self.active[key]=int(args[1])
                    if int(args[1])!=self.backbone_modality:
                        raise AssertionError('Modal BN differs from active backbone modality')
                self.handles.append(module.register_forward_pre_hook(modal_hook))
                for branch,norm in enumerate(module.norms):
                    def norm_hook(mod,args,key=name,index=branch):
                        if mod.training or self.active[key]!=index:
                            raise AssertionError('Incorrect BN branch or BN training during evaluation')
                        self.branch_counts[index]+=1
                    self.handles.append(norm.register_forward_pre_hook(norm_hook))
        for index,adapter in enumerate(model.backbone.adapters):
            def adapter_hook(module,args,branch=index):
                if self.backbone_modality!=branch or module.training:
                    raise AssertionError('Incorrect adapter modality/eval state')
                self.adapter_calls[branch]+=1
            self.handles.append(adapter[1].register_forward_pre_hook(adapter_hook))

    def check(self):
        if any(m.training for m in self.model.modules()):
            raise AssertionError('Evaluation requires model.eval() for every module')

    def finish(self):
        self.check()
        unchanged=bn_unchanged(self.before,self.model)
        if not unchanged:
            raise AssertionError('BN running statistics changed during evaluation')
        if self.backbone_calls!=[m for _ in range(len(self.backbone_calls)//2) for m in (0,1)]:
            raise AssertionError('Optical/SAR backbone order mismatch')
        for handle in self.handles: handle.remove()
        return dict(all_modules_eval=True,BN_buffers_unchanged=True,
            modal_bn_branch_calls=self.branch_counts,adapter_bn_branch_calls=self.adapter_calls,
            expected_backbone_order='optical=0 then SAR=1',image_pairs_seen=len(self.backbone_calls)//2)


def errors_summary(errors):
    errors=np.asarray(errors,float)
    return dict(count=len(errors),mean_px=float(errors.mean()) if len(errors) else None,
        median_px=float(np.median(errors)) if len(errors) else None,
        rmse_px=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        p90_px=float(np.percentile(errors,90)) if len(errors) else None)


def coverage(points,size=256,near_collinear_ratio=.01):
    points=np.asarray(points,float).reshape(-1,2)
    if not len(points):
        return dict(count=0,rank=0,near_collinear=True,hull_area_fraction=0.,grid_4x4_occupied=0)
    normalized=points/(size-1)
    centered=normalized-normalized.mean(0)
    sv=np.linalg.svd(centered,compute_uv=False)
    ratio=float(sv[1]/sv[0]) if len(sv)>1 and sv[0]>0 else 0.
    cells=np.clip((normalized*4).astype(int),0,3)
    return dict(count=len(points),rank=int(np.linalg.matrix_rank(centered)),
        minor_major_singular_ratio=ratio,near_collinear=ratio<near_collinear_ratio,
        bbox_span_fraction_xy=np.ptp(normalized,axis=0).tolist(),
        hull_area_fraction=float(cv2.contourArea(cv2.convexHull(normalized.astype(np.float32)))) if len(points)>=3 else 0.,
        grid_4x4_occupied=len(np.unique(cells,axis=0)))


def refinement_comparison(before,after,mask):
    b,a=before[mask],after[mask]
    return dict(before=errors_summary(b),after=errors_summary(a),
        mean_delta_px=float((a-b).mean()) if len(a) else None,
        worsened_fraction=float((a>b+1e-6).mean()) if len(a) else None)


@torch.no_grad()
def diagnose_pair(model,item,config,diagnostic_config,device='cuda'):
    if any(m.training for m in model.modules()):
        raise AssertionError('model.eval() required')
    args=[item[k][None].to(device) for k in ('optical','sar_aug','optical_valid_mask','sar_aug_valid_mask')]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        output=model(*args)
        source,coarse,score=model.coarse_matches(output,0)
        local=model.local(output['optical_fine'],output['sar_fine'],source,coarse,args[2],args[3])
    valid=local.get('valid',torch.ones(len(source),dtype=torch.bool,device=source.device))
    raw_count=len(source)
    source,coarse,refined,score=[x[valid].float().cpu().numpy() for x in (source,coarse,local['target'],score)]
    estimate=estimate_from_correspondences(source,refined,score,config['evaluation'],seed=config['seed'],include_inlier_indices=True)
    # All GT usage below this point is post-hoc, after prediction and geometric fitting.
    A=item['A_forward'].numpy()
    truth=transform_points(source,A)
    visible=valid_pixel_centers(truth,256,256)
    if len(source):
        mask=cv2.remap(item['sar_aug_valid_mask'].numpy().astype(np.float32),truth[:,0].astype(np.float32)[:,None],
            truth[:,1].astype(np.float32)[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[:,0]>.999
        visible&=mask
    before=np.linalg.norm(coarse-truth,axis=1);after=np.linalg.norm(refined-truth,axis=1)
    inside=visible&((truth-coarse).max(axis=1)<=6+1e-6)&((coarse-truth).max(axis=1)<=6+1e-6) if len(source) else np.zeros(0,bool)
    probes,reference=fixed_synthetic_probes(A,grid=config['evaluation']['probe_grid'],
        optical_mask=item['optical_valid_mask'],sar_aug_mask=item['sar_aug_valid_mask'])
    evaluation=evaluate_transform(estimate['matrix'],probes,reference,config['evaluation']['success_thresholds_px'],status=estimate['status'])
    thresholds=diagnostic_config['coarse_correctness_thresholds_px']
    correctness={str(t):dict(correct_count=int((visible&(before<t)).sum()),
        fraction_of_predictions=float((visible&(before<t)).mean()) if len(source) else None,
        fraction_of_GT_visible=float((before[visible]<t).mean()) if visible.any() else None) for t in thresholds}
    indices=np.asarray(estimate.get('inlier_indices',[]),int)
    coarse_metrics=dict(valid_optical_queries=int(output['optical_coarse_mask'].sum()),
        valid_SAR_candidates=int(output['sar_coarse_mask'].sum()),predictions_before_local_mask=raw_count,
        predicted_correspondences=len(source),GT_visible_predictions=int(visible.sum()),
        errors_all=errors_summary(before),errors_GT_visible=errors_summary(before[visible]),correctness=correctness)
    fine_metrics=dict(all_correspondences=refinement_comparison(before,after,np.ones(len(source),bool)),
        truth_inside_window=refinement_comparison(before,after,inside),
        truth_outside_window=refinement_comparison(before,after,~inside))
    return dict(pair_id=item['pair_id'],coarse=coarse_metrics,
        local_window=dict(radius_fine_cells=3,stride_px=2,truth_inside_count=int(inside.sum()),
            truth_inside_fraction=float(inside.mean()) if len(source) else None,
            truth_inside_among_GT_visible=float(inside[visible].mean()) if visible.any() else None,
            definition='GT visible and |GT-coarse_center| <= 6px per axis; same 7x7 support as B0'),
        refinement=fine_metrics,geometry=dict(estimation=estimate,
            input_source_coverage=coverage(source,near_collinear_ratio=diagnostic_config['near_collinear_singular_ratio']),
            input_target_coverage=coverage(refined,near_collinear_ratio=diagnostic_config['near_collinear_singular_ratio']),
            inlier_source_coverage=coverage(source[indices],near_collinear_ratio=diagnostic_config['near_collinear_singular_ratio']),
            inlier_target_coverage=coverage(refined[indices],near_collinear_ratio=diagnostic_config['near_collinear_singular_ratio'])),
        evaluation=evaluation,predictions=dict(source=source.tolist(),coarse_target=coarse.tolist(),target=refined.tolist(),scores=score.tolist()),
        A_forward=A.tolist(),GT_used_only_posthoc=True)


def aggregate_diagnostics(records):
    def average(path):
        values=[]
        for record in records:
            value=record
            for key in path: value=value[key]
            if value is not None: values.append(value)
        return float(np.mean(values)) if values else None
    return dict(registration=summarize_evaluations([r['evaluation'] for r in records]),
        mean_predicted_matches=average(['coarse','predicted_correspondences']),
        mean_coarse_correct_fraction_5px=average(['coarse','correctness','5','fraction_of_predictions']),
        mean_truth_in_window_fraction=average(['local_window','truth_inside_fraction']),
        mean_refine_delta_inside_px=average(['refinement','truth_inside_window','mean_delta_px']),
        mean_refine_delta_outside_px=average(['refinement','truth_outside_window','mean_delta_px']),
        mean_inlier_source_hull_fraction=average(['geometry','inlier_source_coverage','hull_area_fraction']))
