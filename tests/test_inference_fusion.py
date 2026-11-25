import json
from pathlib import Path
import pytest
import torch
from sglm.t05 import UniformB0
from sglm.inference_fusion import folded_inference_copy


def test_folding_uses_two_distinct_BN_branches_and_preserves_original():
    torch.set_num_threads(2);torch.manual_seed(3)
    cfg=json.loads(Path('configs/t02_b0_full_v1.json').read_text());m=UniformB0(cfg,256).eval()
    for module in m.modules():
        if isinstance(module,torch.nn.BatchNorm2d):module.running_mean.copy_(torch.randn_like(module.running_mean)*.1)
    original={k:v.clone() for k,v in m.state_dict().items()};f=folded_inference_copy(m)
    args=(torch.rand(1,3,256,256),torch.rand(1,1,256,256),torch.ones(1,256,256,dtype=torch.bool),torch.ones(1,256,256,dtype=torch.bool))
    with torch.no_grad():a=m.encode(*args);b=f.encode(*args)
    for key in ['optical_descriptors','sar_descriptors']:assert torch.allclose(a[key],b[key],atol=2e-5,rtol=2e-5)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in original.items())
    assert not any(isinstance(x,torch.nn.BatchNorm2d) for x in f.backbone.modules())
    with pytest.raises(ValueError):folded_inference_copy(m.train())
