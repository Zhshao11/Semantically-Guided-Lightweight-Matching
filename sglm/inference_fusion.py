"""Private inference copy: fold shared convolutions with each modality's own BN."""
import copy
import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from .model import ModalConv


class FoldedModalConv(nn.Module):
    def __init__(self,original):
        super().__init__();self.relu=original.relu
        self.convs=nn.ModuleList([fuse_conv_bn_eval(original.conv,original.bn.norms[i]) for i in [0,1]])
    def forward(self,x,modality):
        x=self.convs[modality](x)
        return torch.relu(x) if self.relu else x


def folded_inference_copy(model):
    if any(m.training for m in model.modules()):raise ValueError('BN folding requires eval() and frozen statistics')
    result=copy.deepcopy(model)
    def replace(parent):
        for name,child in list(parent.named_children()):
            if isinstance(child,ModalConv):setattr(parent,name,FoldedModalConv(child))
            else:replace(child)
    replace(result.backbone)
    for i,adapter in enumerate(result.backbone.adapters):
        result.backbone.adapters[i]=nn.Sequential(fuse_conv_bn_eval(adapter[0],adapter[1]),copy.deepcopy(adapter[2]))
    result.eval().requires_grad_(False)
    assert not any(isinstance(m,nn.BatchNorm2d) for m in result.backbone.modules())
    return result
