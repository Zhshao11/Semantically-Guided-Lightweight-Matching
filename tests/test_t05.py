import json
from pathlib import Path
import pytest
import torch
from sglm.t02.model import B0Full
from sglm.t05 import UniformB0,uniform_queries


@pytest.mark.parametrize('k',[1024,512,256,128])
def test_uniform_queries_cover_fixed_grid_without_duplicates(k):
    q=uniform_queries(k)
    assert len(q)==k and len(q.unique())==k and q.min()>=0 and q.max()<1024
    assert torch.equal(q,uniform_queries(k))
    assert len((q//32//8).unique())==4 and len((q%32//8).unique())==4


def test_gather_reduces_bmm_and_full_logits_are_identical(monkeypatch):
    torch.set_num_threads(2);torch.manual_seed(8)
    cfg=json.loads(Path('configs/t02_b0_full_v1.json').read_text())
    original=B0Full(cfg).eval();sparse=UniformB0(cfg).eval();sparse.load_state_dict(original.state_dict(),strict=True)
    args=(torch.rand(1,3,256,256),torch.rand(1,1,256,256),torch.ones(1,256,256,dtype=torch.bool),torch.ones(1,256,256,dtype=torch.bool))
    with torch.no_grad():
        base=original(*args);full=sparse(*args)
        assert torch.equal(base['row_logits'],full['row_logits']) and torch.equal(base['reverse_logits'],full['reverse_logits'])
        features=sparse.encode(*args);calls=[];real=torch.bmm
        def observe(a,b):calls.append((tuple(a.shape),tuple(b.shape)));return real(a,b)
        monkeypatch.setattr(torch,'bmm',observe);sparse.k=128;out=sparse.correlate(features)
        assert calls==[((1,128,32),(1,32,1024))] and out['coarse_correlation'].shape==(1,128,1024)
        assert sum(p.numel() for p in sparse.parameters())==37219


def test_sparse_mutual_indices_decode_selected_global_source():
    cfg=json.loads(Path('configs/t02_b0_full_v1.json').read_text());model=UniformB0(cfg,128)
    row=torch.full((1,2,1025),-100.);row[0,0,9]=10;row[0,1,17]=10
    reverse=torch.full((1,1024,3),-100.);reverse[0,9,0]=10;reverse[0,17,1]=10
    out=dict(row_logits=row,reverse_logits=reverse,query_indices=torch.tensor([21,97]),optical_coarse_mask=torch.ones(1,32,32,dtype=torch.bool))
    source,target,score=model.coarse_matches(out,0)
    assert torch.equal(source,torch.tensor([[171.5,3.5],[11.5,27.5]]))
    assert torch.equal(target,torch.tensor([[75.5,3.5],[139.5,3.5]])) and len(score)==2
