import copy
import torch
from sglm.local_interaction import LocalCrossCorrelation,fixed_supervision
from sglm.t02.model import LocalCorrelation


def test_zero_initialized_interaction_exact_original_and_small_parameter_count():
    torch.set_num_threads(2);torch.manual_seed(5)
    base=LocalCorrelation();new=LocalCrossCorrelation();new.load_state_dict(base.state_dict(),strict=False)
    features=[torch.randn(1,16,128,128) for _ in range(2)]
    source=torch.tensor([[3.5,3.5],[19.5,27.5],[251.5,251.5]])
    mask=torch.ones(1,256,256,dtype=torch.bool)
    a=base(*features,source,source,mask,mask);b=new(*features,source,source,mask,mask)
    for key in ['target','heatmap','valid']:assert torch.equal(a[key],b[key])
    assert sum(p.numel() for p in new.parameters())-sum(p.numel() for p in base.parameters())==2048


def test_empty_invalid_windows_masks_and_subsets_no_cross_candidate_dependency():
    torch.set_num_threads(2);torch.manual_seed(6);m=LocalCrossCorrelation().eval()
    torch.nn.init.xavier_uniform_(m.cross_out.weight)
    feat=[torch.randn(1,16,128,128) for _ in range(2)];xy=torch.tensor([[3.5,3.5],[59.5,83.5],[251.5,251.5]])
    mask=torch.ones(1,256,256,dtype=torch.bool)
    full=m(*feat,xy,xy,mask,mask);part=m(*feat,xy[[0,2]],xy[[0,2]],mask,mask)
    assert torch.allclose(part['target'],full['target'][[0,2]],atol=1e-4,rtol=0)
    empty=m(*feat,xy[:0],xy[:0],mask,mask);assert empty['target'].shape==(0,2)
    invalid=m(*feat,xy,xy,mask,torch.zeros_like(mask))
    assert not invalid['valid'].any() and torch.isfinite(invalid['heatmap']).all()


def test_invalidSAR_tokens_cannot_affect_interaction_or_readout():
    torch.set_num_threads(2);torch.manual_seed(7);m=LocalCrossCorrelation().eval();torch.nn.init.xavier_uniform_(m.cross_out.weight)
    optical=torch.randn(1,16,128,128);sar=torch.randn(1,16,128,128);xy=torch.tensor([[83.5,83.5]])
    om=torch.ones(1,256,256,dtype=torch.bool);sm=torch.ones_like(om);sm[:,:,88:]=False
    a=m(optical,sar,xy,xy,om,sm)
    other=sar.clone();other[:,:,:,44:46]+=1000 # inside sampled window, but only invalid SAR tokens
    b=m(optical,other,xy,xy,om,sm)
    assert torch.equal(a['target'],b['target'])
    assert torch.all(a['target']>=xy-6) and torch.all(a['target']<=xy+6)


def test_truth_supervision_is_independent_of_refiner_and_excludes_outside_mask():
    source=torch.tensor([[3.5,3.5],[83.5,83.5],[251.5,251.5]])
    A=torch.tensor([[1.,0,8],[0,1,0]]);truth=source+torch.tensor([8.,0])
    mask=torch.ones(256,256,dtype=torch.bool);mask[80:88,88:96]=False
    expected,visible,inside=fixed_supervision(source,truth,A,mask,torch.ones_like(mask))
    assert torch.equal(expected,truth) and visible.tolist()==[True,False,False] and torch.equal(inside,visible)
    _,_,far=fixed_supervision(source,truth+20,A,torch.ones_like(mask),torch.ones_like(mask));assert not far.any()


def test_trainable_interaction_receives_gradients_without_input_feature_gradients():
    torch.set_num_threads(2);m=LocalCrossCorrelation();f=[torch.randn(1,16,128,128) for _ in range(2)]
    xy=torch.tensor([[51.5,59.5],[91.5,83.5]]);mask=torch.ones(1,256,256,dtype=torch.bool)
    p=m(*f,xy,xy,mask,mask);(p['target']-(xy+1)).square().mean().backward()
    assert m.cross_out.weight.grad is not None and m.cross_out.weight.grad.abs().sum()>0
    assert all(t.grad is None for t in f)


def test_replaced_refiner_is_also_eval_after_checkpoint_loading():
    from sglm.local_interaction import build_local_group
    from sglm.t05 import UniformB0
    cfg={'model':{'local_radius_fine_cells':3,'coarse_min_probability':.01,'mutual_nearest':True}}
    state=UniformB0(cfg,k=256).state_dict()
    for group in ['original_refiner','local_cross_attention']:
        m=build_local_group(cfg,state,group,device='cpu')
        assert not any(module.training for module in m.modules())
