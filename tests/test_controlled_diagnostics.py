import numpy as np
import torch
from sglm.evaluation import estimate_from_correspondences
from sglm.diagnostics import InferenceStateAudit,coverage,refinement_comparison
from sglm.t02.data import load_t02_config
from sglm.t02.model import B0Full


def test_optional_actual_inlier_indices_preserve_fit():
    config=load_t02_config()['evaluation']
    rng=np.random.default_rng(3)
    source=rng.uniform(0,255,(30,2));target=source+[7,-4];target[:5]+=100
    base=estimate_from_correspondences(source,target,np.ones(30),config,seed=5)
    detail=estimate_from_correspondences(source,target,np.ones(30),config,seed=5,include_inlier_indices=True)
    assert detail.pop('inlier_indices')==list(range(5,30))
    assert detail==base


def test_BN_observer_correct_modality_and_no_updates():
    torch.set_num_threads(2)
    model=B0Full(load_t02_config()).eval()
    audit=InferenceStateAudit(model);audit.check()
    mask=torch.ones(1,256,256,dtype=torch.bool)
    with torch.no_grad(): model(torch.rand(1,3,256,256),torch.rand(1,1,256,256),mask,mask)
    result=audit.finish()
    assert result['BN_buffers_unchanged'] and result['image_pairs_seen']==1
    assert result['adapter_bn_branch_calls']=={0:1,1:1}
    assert result['modal_bn_branch_calls'][0]==result['modal_bn_branch_calls'][1]


def test_layered_statistics_preserve_same_correspondence_subsets():
    before=np.array([1.,2.,30.]);after=np.array([2.,1.,28.])
    inside=np.array([True,True,False])
    result=refinement_comparison(before,after,inside)
    assert result['before']['count']==result['after']['count']==2
    assert result['mean_delta_px']==0 and result['worsened_fraction']==.5
    assert coverage([[0,0],[1,1],[2,2]])['near_collinear']
    assert coverage([[0,0],[255,0],[0,255],[255,255]])['hull_area_fraction']==1.
