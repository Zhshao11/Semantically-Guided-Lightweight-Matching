import numpy as np
import torch
from sglm.t10_signal import sar_candidate_to_nominal_optical,sample_nominal_semantics,semantic_cosine,negative_pool,choose_controls,paired_parent_summary,hierarchical_relation_weights


def test_inverse_affine_direction_and_semantic_grid_centers():
    A=np.array([[1.,0,16],[0,1,-8]])
    assert np.array_equal(sar_candidate_to_nominal_optical(np.array([[27.5,11.5]]),A),[[11.5,19.5]])
    p=torch.zeros(7,32,32);p[0]=1;p[:,2,1]=0;p[3,2,1]=1;w=torch.ones(32,32)
    sampled,_,valid=sample_nominal_semantics(p,w,np.array([[11.5,19.5],[3.5,3.5],[0.,0.]]))
    assert valid.tolist()==[True,True,False] and sampled[0,3]==1 and sampled[1,0]==1


def test_bilinear_sampling_rejects_any_unreliable_support_and_preserves_softness():
    p=torch.zeros(7,32,32);p[0]=1;p[:,0,1]=0;p[1,0,1]=1;w=torch.ones(32,32)
    sampled,_,valid=sample_nominal_semantics(p,w,np.array([[7.5,3.5]]));assert valid[0] and torch.allclose(sampled[0,:2],torch.tensor([.5,.5]))
    w[0,1]=0;_,_,valid=sample_nominal_semantics(p,w,np.array([[7.5,3.5]]));assert not valid[0]


def test_control_sampling_excludes_truth_hard_candidates_and_matches_strata():
    xy=np.stack([np.arange(12)*8,np.zeros(12)],1);pool,d=negative_pool(np.array([[0.,0.]]),xy,np.ones(12,bool),16)
    assert not pool[0,:3].any() and pool[0,3:].all()
    w=np.full(12,.55);classes=np.zeros(12,int);classes[5]=1;hard=np.array([3,7]);choices=choose_controls(pool[0],hard,d[0],w,classes,3,np.random.default_rng(3),5)
    for selected in choices:
        assert not any(i in hard for i in selected if i>=0)
    assert choices[1][0]==-1 # only24px candidate lies in16-32bin andisalreadyhard.


def test_parent_summary_does_not_treat_many_slots_or_children_as_independent():
    pairs=np.array([0]*100+[1]+[2]);queries=np.array([0]*100+[1]+[0]);values=np.array([1.]*100+[0.,0.]);parents=np.array([0,0,1])
    stats,pm,_,_=paired_parent_summary(pairs,queries,values,parents,100,seed=2)
    assert stats['mean']==.25 and np.array_equal(pm,[1,0,0])
    weights=hierarchical_relation_weights(pairs,queries,parents);assert np.isclose(weights.sum(),1) and np.isclose((weights*values).sum(),.25)


def test_soft_cosine_is_not_hard_class_equality():
    q=np.array([.7,.3]);a=np.array([[.7,.3],[.51,.49],[.1,.9]])
    score=semantic_cosine(q,a);assert score[0]>score[1]>score[2] and np.isclose(score[0],1)
