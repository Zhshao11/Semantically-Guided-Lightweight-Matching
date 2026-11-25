import torch
from sglm.semantic_targets import aggregate_semantic_targets,semantic_supervision_fields,weighted_semantic_kl


def test_ignore_mass_and_uncertainty_both_reduce_weight():
    mapping=torch.cat([torch.zeros(7,1),torch.eye(7)],1);mask=torch.ones(1,1,1,dtype=torch.bool)
    source=torch.tensor([.95,.05,0,0,0,0,0,0]).reshape(1,8,1,1)
    fields=semantic_supervision_fields(aggregate_semantic_targets(source,mask,mapping,factor=1))
    assert fields['conditional_probabilities'][0,0,0,0]==1 and fields['supervision_weight'].item()==0
    uniform=torch.tensor([0.]+[1/7]*7).reshape(1,8,1,1)
    fields=semantic_supervision_fields(aggregate_semantic_targets(uniform,mask,mapping,factor=1,min_confidence=0))
    assert fields['valid_class_mass'].item()>.99 and fields['supervision_weight'].item()<1e-6


def test_effective_class_masses_sum_to_valid_weight():
    mapping=torch.cat([torch.zeros(7,1),torch.eye(7)],1);source=torch.tensor([.1,.8,.1,0,0,0,0,0]).reshape(1,8,1,1)
    fields=semantic_supervision_fields(aggregate_semantic_targets(source,torch.ones(1,1,1,dtype=torch.bool),mapping,factor=1))
    assert 0<fields['supervision_weight'].item()<.9
    assert torch.allclose(fields['effective_class_mass'].sum(1),fields['supervision_weight'])


def test_weighted_KL_keeps_absolute_weight_and_stops_teacher_gradients():
    student=torch.zeros(1,2,1,2,requires_grad=True);teacher=torch.tensor([[[[1.,1.]],[[0.,0.]]]],requires_grad=True)
    weight=torch.tensor([[[.5,0.]]],requires_grad=True)
    loss=weighted_semantic_kl(student,teacher,weight);loss.backward()
    assert torch.allclose(loss,torch.tensor(.25)*torch.log(torch.tensor(2.)))
    assert teacher.grad is None and weight.grad is None and student.grad[0,:,0,1].abs().sum()==0
