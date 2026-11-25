import pytest
import torch
from sglm.semantic_targets import aggregate_semantic_targets,teacher_logits_to_native_probabilities,masked_semantic_kl


def test_area_average_uses_soft_mass_not_hard_votes():
    p=torch.tensor([[[[.9,.4],[.9,.4]],[[.1,.6],[.1,.6]]]])
    result=aggregate_semantic_targets(p,torch.ones(1,2,2,dtype=torch.bool),torch.eye(2),factor=2,min_retained_mass=0,min_confidence=0)
    assert torch.allclose(result['probabilities'].flatten(),torch.tensor([.65,.35]))
    assert result['mask'].item()


def test_ignore_mass_cannot_be_hidden_by_renormalization():
    p=torch.tensor([.95,.05,0.]).reshape(1,3,1,1)
    mapping=torch.tensor([[0,1,0],[0,0,1]])
    result=aggregate_semantic_targets(p,torch.ones(1,1,1,dtype=torch.bool),mapping,factor=1)
    assert result['probabilities'][0,0,0,0]==1
    assert result['retained_mass'].item()==pytest.approx(.05)
    assert not result['mask'].item()


def test_masked_pixels_and_padding_do_not_dilute_valid_class():
    p=torch.tensor([[[[1.,0.],[0.,0.]],[[0.,1.],[1.,1.]]]])
    mask=torch.tensor([[[True,False],[False,False]]])
    result=aggregate_semantic_targets(p,mask,torch.eye(2),factor=2,min_valid_fraction=.25,min_retained_mass=0,min_confidence=0)
    assert torch.equal(result['probabilities'].flatten(),torch.tensor([1.,0.]))
    assert result['valid_fraction'].item()==.25 and result['mask'].item()
    strict=aggregate_semantic_targets(p,mask,torch.eye(2),factor=2)
    assert not strict['mask'].item()


def test_empty_or_nonfinite_cells_are_zero_and_invalid():
    p=torch.ones(1,2,2,2)*.5;p[0,0,0,0]=float('nan')
    out=aggregate_semantic_targets(p,torch.ones(1,2,2,dtype=torch.bool),torch.eye(2),factor=2,min_confidence=0)
    assert not out['mask'].item() and torch.isfinite(out['probabilities']).all()
    empty=aggregate_semantic_targets(torch.zeros_like(p),torch.zeros(1,2,2,dtype=torch.bool),torch.eye(2),factor=2)
    assert not empty['mask'].item() and empty['probabilities'].sum()==0


def test_grouping_conserves_mass_and_rejects_double_count():
    p=torch.tensor([.2,.3,.5]).reshape(1,3,1,1)
    grouped=aggregate_semantic_targets(p,torch.ones(1,1,1,dtype=torch.bool),[[1,1,0],[0,0,1]],factor=1,min_confidence=0)
    assert torch.equal(grouped['probabilities'].flatten(),torch.tensor([.5,.5]))
    assert grouped['retained_mass'].item()==1
    with pytest.raises(ValueError):aggregate_semantic_targets(p,torch.ones(1,1,1,dtype=torch.bool),[[1,1,1],[0,0,1]],factor=1)


def test_logits_resize_normalization_and_divisibility():
    p=teacher_logits_to_native_probabilities(torch.randn(1,8,4,4),(16,16))
    assert p.shape==(1,8,16,16) and torch.allclose(p.sum(1),torch.ones(1,16,16),atol=1e-6)
    with pytest.raises(ValueError):aggregate_semantic_targets(p[:,:,:15],torch.ones(1,15,16,dtype=torch.bool),torch.eye(8),factor=8)


def test_masked_KL_has_no_teacher_gradient_or_invalid_cell_gradient():
    student=torch.randn(1,2,2,2,requires_grad=True)
    teacher=torch.full_like(student,.5,requires_grad=True)
    mask=torch.tensor([[[True,False],[False,False]]])
    loss=masked_semantic_kl(student,teacher,mask);loss.backward()
    assert teacher.grad is None
    assert torch.equal(student.grad[0,:,1],torch.zeros(2,2)) and torch.equal(student.grad[0,:,0,1],torch.zeros(2))
    assert student.grad[0,:,0,0].abs().sum()>0


def test_all_ignored_KL_is_finite_zero():
    student=torch.randn(1,7,2,2,requires_grad=True)
    loss=masked_semantic_kl(student,torch.zeros_like(student),torch.zeros(1,2,2,dtype=torch.bool))
    loss.backward();assert loss.item()==0 and torch.equal(student.grad,torch.zeros_like(student))
