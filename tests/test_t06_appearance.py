import json
import pytest
from pathlib import Path
import torch
from sglm.model import JointScore,select_queries
from sglm.t06_appearance import AppearanceHead,AppearanceB0,coverage_queries,reliability_targets


def test_head_detaches_backbone_and_capacity_control_is_close():
    for width,count in [(16,3489),(29,7831)]:
        h=AppearanceHead(width);assert sum(p.numel() for p in h.parameters())==count
    assert sum(p.numel() for p in JointScore().parameters())==7977
    fo=torch.randn(2,64,32,32,requires_grad=True);fs=torch.randn(2,64,requires_grad=True)
    h=AppearanceHead();h(fo,fs).mean().backward()
    assert fo.grad is None and fs.grad is None and h.optical.weight.grad is not None


def test_vectorized_coverage_matches_fixed_quota_rule():
    torch.manual_seed(2);score=torch.rand(32,32);valid=torch.ones(32,32,dtype=torch.bool)
    for k in [256,512]:
        q=coverage_queries(score,valid,k)
        assert torch.equal(q,select_queries(score,valid,k,suppression=0).sort().values)
        cells=(q//32//8)*4+(q%32//8);counts=torch.bincount(cells,minlength=16)
        assert counts.min()>=2 and counts.max()<=32 and len(q.unique())==k


def test_geometric_label_annulus_and_ambiguity_are_separate():
    p=json.loads(Path('configs/t06_protocol_v1.json').read_text())['labels']
    logits=torch.full((1,1024,1025),-10.);logits[0,torch.arange(1024),torch.arange(1024)]=100
    out=dict(row_logits=logits,optical_coarse_mask=torch.ones(1,32,32,dtype=torch.bool),sar_coarse_mask=torch.ones(1,32,32,dtype=torch.bool))
    mask=torch.ones(1,256,256,dtype=torch.bool);A=torch.tensor([[[1.,0,0],[0,1.,0]]])
    correct=reliability_targets(out,A,mask,p);assert correct['geometric_correctness'][0,500]==1 and correct['ambiguity_margin'][0,500]==110
    A[0,0,2]=6;band=reliability_targets(out,A,mask,p);assert band['geometric_correctness'][0,500]==-1
    A[0,0,2]=16;wrong=reliability_targets(out,A,mask,p);assert wrong['geometric_correctness'][0,500]==0 and wrong['ambiguity_margin'][0,500]<0
    assert not wrong['is_supervisable'][0,1023]


@pytest.mark.parametrize('mode',['A','AC'])
def test_scoring_precedes_only_sparse_correlation(monkeypatch,mode):
    cfg=json.loads(Path('configs/t02_b0_full_v1.json').read_text());m=AppearanceB0(cfg,256,mode,AppearanceHead()).eval()
    feats=dict(optical_coarse=torch.randn(1,64,32,32),sar_coarse=torch.randn(1,64,32,32),optical_descriptors=torch.randn(1,1024,32),sar_descriptors=torch.randn(1,1024,32),
        optical_coarse_mask=torch.ones(1,32,32,dtype=torch.bool),sar_coarse_mask=torch.ones(1,32,32,dtype=torch.bool))
    calls=[];real=torch.bmm
    def spy(a,b):calls.append((a.shape[1],b.shape[-1]));return real(a,b)
    monkeypatch.setattr(torch,'bmm',spy);o=m.correlate(feats)
    assert calls==[(256,1024)] and o['coarse_correlation'].shape==(1,256,1024)
