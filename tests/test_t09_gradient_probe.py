"""Measuring gradients must not alter the optimizer step or BN updates."""
import copy,json
from pathlib import Path
import torch
from sglm.t08_auxiliary import AuxiliaryB0,sparse_matching_loss
from sglm.semantic_targets import weighted_semantic_kl


def test_gradient_probe_is_observational_for_optimizer_and_BN():
    torch.manual_seed(9);cfg=json.loads(Path('configs/t02_b0_full_v1.json').read_text());plain=AuxiliaryB0(cfg,256,True).train();measured=copy.deepcopy(plain)
    batch=dict(optical=torch.rand(1,3,256,256),sar_aug=torch.rand(1,1,256,256),optical_valid_mask=torch.ones(1,256,256,dtype=torch.bool),sar_aug_valid_mask=torch.ones(1,256,256,dtype=torch.bool),A_forward=torch.tensor([[[1.,0,0],[0,1.,0]]]))
    target=torch.softmax(torch.randn(1,7,32,32),1);weight=torch.ones(1,32,32)
    for model,probe in [(plain,False),(measured,True)]:
        trainable=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(trainable,lr=.0002,weight_decay=.0001)
        output=model(**{k:v for k,v in batch.items() if k!='A_forward'});match,_=sparse_matching_loss(model,output,batch,cfg);kl=weighted_semantic_kl(output['semantic_logits'],target,weight)
        if probe:
            encoder=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(model.prefixes) and not n.startswith('backbone.descriptor.')]
            torch.autograd.grad(match,encoder,retain_graph=True,allow_unused=True);torch.autograd.grad(kl,encoder,retain_graph=True,allow_unused=True)
            assert all(p.grad is None for p in trainable)
        (match+kl).backward();torch.nn.utils.clip_grad_norm_(trainable,5);opt.step()
    for key,value in plain.state_dict().items():assert torch.equal(value,measured.state_dict()[key]),key
