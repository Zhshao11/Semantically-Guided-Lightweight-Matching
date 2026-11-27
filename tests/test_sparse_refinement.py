import numpy as np
import pytest
import torch
from sglm.sparse_refinement import select_refinement_indices as select,gather_candidate_fields,SparseRefinementB0,CANDIDATE_FIELDS
from sglm.t02.data import load_t02_config
from sglm.t05 import UniformB0
from sglm.evaluation import estimate_from_correspondences


def test_bypass_original_order_empty_zero_and_invalid_budget():
    xy=torch.rand(5,2);c=torch.tensor([0.,4,2,1,3])
    for mode in ['all','prefix','global_conf','coverage_conf']:
        for b in [None,5,12]:assert torch.equal(select(xy,c,(10,20),b,mode),torch.arange(5))
        assert select(xy[:0],c[:0],(10,20),0,mode).numel()==0
    assert len(select(xy,c,(10,20),0,'all'))==5 # explicit all bypass has precedence
    for mode in ['prefix','global_conf','coverage_conf']:assert len(select(xy,c,(10,20),0,mode))==0
    for b in [-1,1.2,True]:
        with pytest.raises(ValueError):select(xy,c,(10,20),b,'prefix')
    with pytest.raises(ValueError):select(xy,c,(10,20),2,'unknown')


def test_confidence_ties_original_index_and_sorted_output():
    xy=torch.zeros(8,2);c=torch.tensor([1.,2,2,0,2,0,1,1])
    assert select(xy,c,(10,20),2,'global_conf').tolist()==[1,2]
    assert select(xy,c,(10,20),3,'coverage_conf').tolist()==[1,2,4]


def test_coverage_non_square_boundaries_reallocation_and_unique_indices():
    # H40,W80 => x20 boundary is column1, y10 boundary row1.
    xy=torch.tensor([[0.,0],[19.999,0],[20.,0],[79.999,39.999],[80.,40.],[0.,10.]])
    score=torch.tensor([1.,2,3,4,5,6])
    idx=select(xy,score,(40,80),4,'coverage_conf')
    assert idx.tolist()==[1,2,4,5]
    for b in range(7):
        i=select(xy,score,(40,80),b,'coverage_conf')
        assert len(i)==min(b,6) and len(i.unique())==len(i)
    # First cell exhausted at1; other cell receives all remaining water.
    xy=torch.tensor([[0.,0]]+[[30.,0]]*9);score=torch.arange(10).float()
    assert select(xy,score,(40,80),5,'coverage_conf').tolist()==[0,6,7,8,9]


def test_uniform_fields_only_explicit_schema_is_gathered():
    packet=dict(source=torch.arange(12).reshape(6,2),score=torch.arange(6),batch=torch.tensor([0,0,0,1,1,1]),
                fine_feature=torch.randn(6,4),original_candidate_idx=torch.arange(6))
    chosen=torch.tensor([1,4]);out=gather_candidate_fields(packet,chosen,('source','score','batch','original_candidate_idx'))
    assert torch.equal(out['source'],packet['source'][chosen]) and out['score'].tolist()==[1,4]
    assert out['batch'].tolist()==[0,1] and out['fine_feature'] is packet['fine_feature']
    assert out['original_candidate_idx'].tolist()==[1,4]
    with pytest.raises(ValueError):gather_candidate_fields(packet,chosen.float(),('source',))


def test_waterfill_ten_nonempty_cells_and_fixed_remainders():
    xy=[]
    for cell in range(10):xy.extend([[cell%4*64+10.,cell//4*64+10.]]*20)
    xy=torch.tensor(xy);scores=torch.ones(200)
    indices=select(xy,scores,(256,256),128,'coverage_conf')
    assert torch.bincount(indices//20,minlength=10).tolist()==[13]*8+[12]*2
    assert indices.tolist()==sorted(indices.tolist())


def test_filter_preserves_original_ids_when_local_valid_mask_rejects_windows():
    cfg=load_t02_config();m=SparseRefinementB0(cfg,mode='all').eval();args=inputs(1)
    args[3].zero_();args[3][:,0:70,0:70]=True
    with torch.inference_mode():
        output=m(*args);xy=torch.tensor([[11.5,11.5],[155.5,155.5],[35.5,35.5]])
        packet=dict(source=xy,target=xy,score=torch.tensor([.1,.2,.3]),batch_index=torch.zeros(3,dtype=torch.long),
            valid_mask=torch.ones(3,dtype=torch.bool),original_candidate_idx=torch.arange(3))
        pred=m.refine_packet(output,args,packet,0)
        assert pred['attempted_refinements']==3 and pred['original_candidate_idx'].tolist()==[0,2]
        assert torch.equal(pred['source'],xy[[0,2]]) and torch.equal(pred['score'],packet['score'][[0,2]])


def inputs(batch=2):
    return [torch.rand(batch,3,256,256),torch.rand(batch,1,256,256),torch.ones(batch,256,256,dtype=torch.bool),torch.ones(batch,256,256,dtype=torch.bool)]


def test_original_full_and_subsets_local_are_identical_no_cross_candidate_dependency():
    torch.set_num_threads(2);torch.manual_seed(8);cfg=load_t02_config()
    base=UniformB0(cfg).eval();model=SparseRefinementB0(cfg).eval();model.load_state_dict(base.state_dict(),strict=True)
    args=inputs();calls=[];handle=model.local.register_forward_pre_hook(lambda m,a:calls.append(len(a[2])))
    with torch.inference_mode():
        output=model(*args)
        # Explicit predicted-style centers ensure many candidates even random untrained logits no-match.
        source=torch.tensor([[3.5,3.5],[11.5,19.5],[251.5,251.5],[83.5,147.5],[107.5,211.5]])
        packet=dict(source=source,target=source+torch.tensor([2.,-2.]),score=torch.tensor([.4,.7,.1,.8,.6]),
            batch_index=torch.zeros(5,dtype=torch.long),valid_mask=torch.ones(5,dtype=torch.bool),original_candidate_idx=torch.arange(5))
        full=model.refine_packet(output,args,packet,0)
        for mode in ['prefix','global_conf','coverage_conf']:
            model.refinement_budget=3;model.refinement_mode=mode;selected=model.select_packet(packet,(256,256))
            sub=model.refine_packet(output,args,selected,0);lookup=full['original_candidate_idx']
            loc=torch.nonzero(lookup[:,None]==sub['original_candidate_idx'][None],as_tuple=False)[:,0]
            assert torch.allclose(sub['target'],full['target'][loc],atol=1e-4,rtol=0)
            assert calls[-1]==3
        model.refinement_mode='all';model.refinement_budget=None
        original=base.predict(*args);actual=model.predict(*args)
        for a,b in zip(original,actual):
            for key in ('source','target','score','coarse_target'):assert torch.equal(a[key],b[key])
    handle.remove()


def test_pairwise_budget_and_zero_failure_without_fake_correspondences():
    torch.set_num_threads(2);cfg=load_t02_config();m=SparseRefinementB0(cfg,budget=2,mode='prefix').eval()
    args=inputs(2)
    with torch.inference_mode():
        output=m(*args)
        def matches(out,index):
            n=5 if index==0 else 1
            xy=torch.tensor([[3.5,3.5],[11.5,11.5],[19.5,19.5],[27.5,27.5],[35.5,35.5]])[:n]
            return xy,xy.clone(),torch.ones(n)
        m.coarse_matches=matches
        pred=[m.refine(output,args,i) for i in range(2)]
        assert [r['attempted_refinements'] for r in pred]==[2,1]
        assert pred[1]['batch_index'].tolist()==[1]
        m.refinement_budget=0
        empty=m.refine(output,args,0)
        assert empty['source'].shape==(0,2) and empty['attempted_refinements']==0
        a=[empty[k].numpy() for k in ['source','target','score']]
        fit=estimate_from_correspondences(*a,cfg['evaluation'])
        assert not fit['success'] and fit['status']=='empty_matches'
