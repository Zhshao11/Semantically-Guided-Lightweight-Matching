import json
from pathlib import Path
import pytest
import torch
from sglm.t07_semantic import SemanticScore
from sglm.t06_appearance import AppearanceHead,AppearanceB0
from sglm.semantic_targets import weighted_semantic_kl


def test_t07_gradient_and_total_capacity():
    h=SemanticScore();assert sum(p.numel() for p in h.parameters())==7977
    a=AppearanceHead(29);assert abs(sum(p.numel() for p in a.parameters())-7977)/7977<.02
    fo=torch.randn(2,64,32,32,requires_grad=True);sar=torch.randn(2,64,requires_grad=True)
    fields=h.fields(fo,sar);fields['logits'].mean().backward()
    assert fo.grad is None and sar.grad is None and all(p.grad is None for p in h.semantic.parameters())
    assert h.embedding.weight.grad is not None and h.residual[0].weight.grad is not None
    h.zero_grad(set_to_none=True)
    f=h.fields(fo,sar);target=torch.softmax(torch.randn_like(f['semantic_logits']),1)
    weighted_semantic_kl(f['semantic_logits'],target,torch.ones(2,32,32)).backward()
    assert h.semantic[0].weight.grad is not None and h.optical.weight.grad is None and fo.grad is None


@pytest.mark.parametrize('selection',['A','AC'])
def test_semantic_student_needs_no_teacher_or_GT_and_correlates_after_select(monkeypatch,selection):
    cfg=json.loads(Path('configs/t02_b0_full_v1.json').read_text());model=AppearanceB0(cfg,256,selection,SemanticScore()).eval()
    feats=dict(optical_coarse=torch.randn(1,64,32,32),sar_coarse=torch.randn(1,64,32,32),optical_descriptors=torch.randn(1,1024,32),sar_descriptors=torch.randn(1,1024,32),optical_coarse_mask=torch.ones(1,32,32,dtype=torch.bool),sar_coarse_mask=torch.ones(1,32,32,dtype=torch.bool))
    real=torch.bmm;calls=[]
    def spy(a,b):calls.append(tuple(a.shape));return real(a,b)
    monkeypatch.setattr(torch,'bmm',spy)
    o=model.correlate(feats);assert calls==[(1,256,32)] and len(o['query_indices'].unique())==256


def test_class_and_image_abstention_preserves_probability_mass_and_inputs():
    from sglm.t07_semantic import abstain_semantic_weight
    target=torch.full((2,7,2,2),.05);target[:,1]=.7;target[0,:,0,0]=.05;target[0,4,0,0]=.7
    original=target.clone();weight=torch.full((2,2,2),.8)
    out=abstain_semantic_weight(target,weight,[4],torch.tensor([False,True]))
    assert out[0,0,0]==0 and torch.all(out[0,1:]==.8) and not out[1].any()
    assert torch.equal(target,original) and torch.all(weight==.8)
