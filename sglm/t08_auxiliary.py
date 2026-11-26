"""Uniform sparse matching fine-tuning, optional optical-only semantic auxiliary loss."""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from .t05 import UniformB0
from .model import JointScore,coarse_xy
from .t02.loss import correspondence_labels


class AuxiliaryB0(UniformB0):
    def __init__(self,config,k=256,with_semantics=False,prefixes=None):
        super().__init__(config,k)
        self.config_for_export=copy.deepcopy(config)
        self.prefixes=tuple(prefixes or ['backbone.down.1.','backbone.stages.2.','backbone.lateral.2.','backbone.descriptor.'])
        if with_semantics:self.semantic_head=copy.deepcopy(JointScore().semantic)
        for name,param in self.named_parameters():param.requires_grad_(name.startswith(self.prefixes) or name.startswith('semantic_head.'))
    def train(self,mode=True):
        super().train(mode)
        # Affine params can adapt in allowed coarse layers; every running statistic stays frozen.
        self.backbone.eval();self.local.eval()
        return self
    def forward(self,optical,sar_aug,optical_valid_mask,sar_aug_valid_mask):
        out=super().forward(optical,sar_aug,optical_valid_mask,sar_aug_valid_mask)
        if hasattr(self,'semantic_head') and self.training:
            with torch.autocast(optical.device.type,enabled=False):out['semantic_logits']=self.semantic_head(out['optical_coarse'].float())
        return out
    def inference_state_dict(self):
        return {name:value for name,value in self.state_dict().items() if not name.startswith('semantic_head.')}
    def export_matching_model(self):
        result=UniformB0(self.config_for_export,self.k)
        result.load_state_dict(self.inference_state_dict(),strict=True)
        return result.to(next(self.parameters()).device).eval().requires_grad_(False)


def sparse_labels(A_forward,source_valid,target_valid,query_indices):
    forward,targets,positive=correspondence_labels(A_forward,source_valid,target_valid)
    homogeneous=torch.eye(3,device=A_forward.device)[None].repeat(len(A_forward),1,1);homogeneous[:,:2]=A_forward.float()
    reverse,_,_=correspondence_labels(torch.linalg.inv(homogeneous)[:,:2],target_valid,source_valid)
    k=len(query_indices);lookup=torch.full((1025,),k,device=query_indices.device,dtype=torch.long)
    lookup[query_indices]=torch.arange(k,device=query_indices.device)
    mapped=lookup[reverse.clamp_min(0)];mapped[reverse==-100]=-100
    return forward.index_select(1,query_indices),mapped,targets,positive


def classified_ce(logits,labels,no_match_weight):
    n=logits.shape[-1];loss=F.cross_entropy(logits.reshape(-1,n),labels.flatten(),ignore_index=-100,reduction='none')
    flat=labels.flatten();pos=(flat>=0)&(flat<n-1);neg=flat==n-1
    zero=logits[torch.isfinite(logits)].sum()*0
    return (loss[pos].mean() if pos.any() else zero)+no_match_weight*(loss[neg].mean() if neg.any() else zero)


def sparse_matching_loss(model,output,batch,config):
    """Original bidirectional balanced CE and predicted-window fine loss, sparse candidates."""
    labels,reverse,_,_=sparse_labels(batch['A_forward'],batch['optical_valid_mask'],batch['sar_aug_valid_mask'],output['query_indices'])
    cfg=config['training'];coarse=(classified_ce(output['row_logits'],labels,cfg['no_match_loss_weight'])+classified_ce(output['reverse_logits'],reverse,cfg['no_match_loss_weight']))/2
    fine=coarse.new_zeros(());count=0
    for i,A in enumerate(batch['A_forward']):
        source,centers,_=model.coarse_matches(output,i);expected=source@A[:,:2].float().T+A[:,2].float()
        inside=((expected-centers).abs()<=6).all(-1)&((expected>=0)&(expected<=255)).all(-1)
        source,expected,centers=source[inside],expected[inside],centers[inside]
        cap=cfg['fine_training_samples_per_pair']
        if len(source)>cap:
            choose=torch.linspace(0,len(source)-1,cap,device=source.device).long();source,expected,centers=source[choose],expected[choose],centers[choose]
        if len(source):
            local=model.local(output['optical_fine'][i:i+1],output['sar_fine'][i:i+1],source,centers,batch['optical_valid_mask'][i:i+1],batch['sar_aug_valid_mask'][i:i+1])
            valid=local['valid']
            if valid.any():fine+=F.smooth_l1_loss(local['target'][valid],expected[valid]);count+=int(valid.sum())
    fine=fine/len(batch['A_forward']);total=coarse+cfg['fine_loss_weight']*fine
    with torch.no_grad():
        pos=(labels>=0)&(labels<1024);accuracy=(output['row_logits'].argmax(-1)[pos]==labels[pos]).float().mean() if pos.any() else coarse.new_zeros(())
    return total,dict(coarse=float(coarse.detach()),fine=float(fine.detach()),positive_accuracy=float(accuracy),fine_training_pairs=count)
