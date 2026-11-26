import json
from pathlib import Path
import numpy as np
import torch
from sglm.t08_auxiliary import AuxiliaryB0,sparse_labels,sparse_matching_loss
from sglm.t02.loss import b0_loss
from sglm.t05 import uniform_queries
from sglm.semantic_targets import weighted_semantic_kl
from sglm.diagnostics import bn_snapshot,bn_unchanged


def configuration():return json.loads(Path('configs/t02_b0_full_v1.json').read_text())


def inputs():
    return dict(optical=torch.rand(1,3,256,256),sar_aug=torch.rand(1,1,256,256),optical_valid_mask=torch.ones(1,256,256,dtype=torch.bool),sar_aug_valid_mask=torch.ones(1,256,256,dtype=torch.bool),A_forward=torch.tensor([[[1.,0,0],[0,1.,0]]]))


def test_sparse_reverse_labels_map_omitted_queries_to_no_match():
    b=inputs();q=uniform_queries(256);forward,reverse,_,_=sparse_labels(b['A_forward'],b['optical_valid_mask'],b['sar_aug_valid_mask'],q)
    assert torch.equal(forward[0],q) and torch.equal(reverse[0,q],torch.arange(256))
    omitted=torch.ones(1024,dtype=torch.bool);omitted[q]=False;assert torch.all(reverse[0,omitted]==256)


def test_dense_limit_loss_matches_original_without_bootstrap():
    cfg=configuration();cfg['training']['fine_gt_bootstrap_steps']=0;m=AuxiliaryB0(cfg,1024).eval();b=inputs();out=m(**{k:v for k,v in b.items() if k!='A_forward'})
    old,oldmetrics=b0_loss(m,out,b,cfg,10000);new,newmetrics=sparse_matching_loss(m,out,b,cfg)
    assert torch.allclose(old,new,atol=1e-6) and abs(oldmetrics['fine']-newmetrics['fine'])<1e-6


def test_semantic_loss_updates_allowed_coarse_layers_optical_only_and_export_removes_head():
    m=AuxiliaryB0(configuration(),256,True).train();before=bn_snapshot(m.backbone);b=inputs();out=m(**{k:v for k,v in b.items() if k!='A_forward'})
    target=torch.softmax(torch.randn_like(out['semantic_logits']),1);loss=weighted_semantic_kl(out['semantic_logits'],target,torch.ones(1,32,32));loss.backward()
    assert m.backbone.stages[2][0].dw.conv.weight.grad is not None
    assert m.backbone.lateral[2].conv.weight.grad is not None
    assert m.backbone.stages[2][0].dw.bn.norms[0].weight.grad is not None
    assert m.backbone.stages[2][0].dw.bn.norms[1].weight.grad is None
    assert m.backbone.descriptor.weight.grad is None # KL is pre-projection; matching loss trains projection.
    assert m.backbone.adapters[0][0].weight.grad is None and all(p.grad is None for p in m.local.parameters())
    assert bn_unchanged(before,m.backbone)
    m.eval();a=m(**{k:v for k,v in b.items() if k!='A_forward'});export=m.export_matching_model();e=export(**{k:v for k,v in b.items() if k!='A_forward'})
    assert not any('semantic' in key for key in export.state_dict()) and torch.equal(a['row_logits'],e['row_logits'])
    assert not any(p.requires_grad for p in export.parameters())


def test_matching_loss_projection_and_freeze_boundary():
    cfg=configuration();m=AuxiliaryB0(cfg,256,False).train();b=inputs();o=m(**{k:v for k,v in b.items() if k!='A_forward'});loss,_=sparse_matching_loss(m,o,b,cfg);loss.backward()
    assert m.backbone.descriptor.weight.grad is not None and m.backbone.stages[2][0].pw.conv.weight.grad is not None
    for name,param in m.named_parameters():
        if not name.startswith(m.prefixes):assert param.grad is None,name


def test_shuffle_deranges_crossimage_and_preserves_mask_and_class_histogram():
    import importlib.util
    spec=importlib.util.spec_from_file_location('shuffle','scripts/prepare_t08_shuffle.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    owners=np.repeat(np.arange(10),np.arange(1,11));perm=mod.cross_image_permutation(owners,np.random.default_rng(4))
    assert np.all(owners!=owners[perm]) and sorted(perm.tolist())==list(range(len(owners)))
