"""Fixed-budget selection immediately before the unchanged B0 local operator."""
from numbers import Integral
import torch
from .t05 import UniformB0


MODES=('all','prefix','global_conf','coverage_conf')
CANDIDATE_FIELDS=('source','target','score','batch_index','valid_mask','original_candidate_idx')


@torch.no_grad()
def select_refinement_indices(xy_opt_px,confidence,image_hw,budget,mode,grid_hw=(4,4)):
    """Return original indices only. All limited modes preserve original output order.

    Confidence ties use the original index. Coverage uses row-major round-robin
    quotas (equal water filling); it never changes coordinates or accesses GT.
    """
    if mode not in MODES:raise ValueError('Unknown refinement selection mode')
    if budget is not None and (isinstance(budget,bool) or not isinstance(budget,Integral) or budget<0):
        raise ValueError('budget must be a nonnegative integer or None')
    n=len(confidence)
    if xy_opt_px.shape!=(n,2) or confidence.shape!=(n,):raise ValueError('Expected Nx2 and N')
    if xy_opt_px.device!=confidence.device:raise ValueError('Candidate fields must share device')
    if len(image_hw)!=2 or min(image_hw)<=0 or len(grid_hw)!=2 or min(grid_hw)<=0:
        raise ValueError('Positive (H,W) and grid (rows,cols) required')
    idx=torch.arange(n,device=confidence.device,dtype=torch.long)
    # This bypass MUST precede sorting or quota computations, including budget0/all.
    if mode=='all' or budget is None or budget>=n:return idx
    if budget==0 or n==0:return idx[:0]
    if mode=='prefix':return idx[:budget]
    ranked=torch.argsort(confidence,descending=True,stable=True)
    if mode=='global_conf':return ranked[:budget].sort().values
    rows,cols=grid_hw;h,w=image_hw
    gx=torch.floor(cols*xy_opt_px[:,0]/w).long().clamp(0,cols-1)
    gy=torch.floor(rows*xy_opt_px[:,1]/h).long().clamp(0,rows-1)
    cells=gy*cols+gx
    capacity=torch.bincount(cells,minlength=rows*cols).cpu().tolist()
    quotas=[0]*len(capacity);remaining=budget
    while remaining:
        for cell,cap in enumerate(capacity):
            if quotas[cell]<cap:
                quotas[cell]+=1;remaining-=1
                if remaining==0:break
    chosen=[ranked[cells[ranked]==cell][:q] for cell,q in enumerate(quotas) if q]
    return torch.cat(chosen).sort().values


def gather_candidate_fields(candidates,selected_idx,candidate_fields):
    """Gather explicitly declared leading candidate axes; leave all other data alone."""
    if selected_idx.dtype!=torch.long or selected_idx.ndim!=1:raise ValueError('LongTensor[K] required')
    if not candidate_fields:raise ValueError('Declare candidate fields explicitly')
    sizes={len(candidates[name]) for name in candidate_fields}
    if len(sizes)!=1:raise ValueError('Declared candidate lengths differ')
    result=dict(candidates)
    for name in candidate_fields:
        result[name]=candidates[name].index_select(0,selected_idx)
    return result


class SparseRefinementB0(UniformB0):
    """Same parameters, encoder, dense coarse matching and local refinement as T05.

    Budget is per image pair, AFTER coarse matching, unlike T05's query budget.
    """
    def __init__(self,config,budget=None,mode='all'):
        super().__init__(config,k=1024)
        self.refinement_budget=budget;self.refinement_mode=mode

    def candidate_packet(self,output,index):
        source,target,score=self.coarse_matches(output,index)
        return dict(source=source,target=target,score=score,
            batch_index=torch.full((len(source),),index,device=source.device,dtype=torch.long),
            valid_mask=torch.ones(len(source),device=source.device,dtype=torch.bool),
            original_candidate_idx=torch.arange(len(source),device=source.device))

    def select_packet(self,candidates,image_hw):
        selected=select_refinement_indices(candidates['source'],candidates['score'],image_hw,
            self.refinement_budget,self.refinement_mode)
        return gather_candidate_fields(candidates,selected,CANDIDATE_FIELDS)

    def refine_packet(self,output,args,selected,index):
        source,target,score=[selected[k] for k in ('source','target','score')]
        # FIRST candidate-dependent sampling happens inside this unchanged local call.
        local=self.local(output['optical_fine'][index:index+1],output['sar_fine'][index:index+1],
            source,target,args[2][index:index+1],args[3][index:index+1])
        valid=local.get('valid',torch.ones(len(source),device=source.device,dtype=torch.bool))
        return dict(source=source[valid],target=local['target'][valid],score=score[valid],
            coarse_target=target[valid],original_candidate_idx=selected['original_candidate_idx'][valid],
            batch_index=selected['batch_index'][valid],valid_mask=selected['valid_mask'][valid],
            attempted_refinements=len(source),selected_candidate_idx=selected['original_candidate_idx'],
            local_window_source='predicted_coarse_only')

    def refine(self,output,args,index=0):
        candidates=self.candidate_packet(output,index)
        selected=self.select_packet(candidates,args[0].shape[-2:])
        result=self.refine_packet(output,args,selected,index)
        result['num_coarse_candidates']=len(candidates['source'])
        return result
