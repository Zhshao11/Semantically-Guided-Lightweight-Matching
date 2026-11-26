"""Frozen-backbone student semantics; semantic evidence enters selection before correlation."""
import math
import torch
from torch.nn import functional as F
from .model import JointScore
from .t06_appearance import AppearanceB0,AppearanceHead


class SemanticScore(JointScore):
    """Same 7977-parameter v0.1 head, with masked SAR GAP and explicit logits."""
    def fields(self,optical,sar_mean):
        with torch.autocast(optical.device.type,enabled=False):
            optical=optical.detach().float();sar_mean=sar_mean.detach().float()
            semantic_logits=self.semantic(optical);p=semantic_logits.softmax(1);ps=p.detach()
            confidence=1+(ps*ps.clamp_min(1e-8).log()).sum(1,keepdim=True)/math.log(7)
            xo=F.relu(self.optical(optical));zs=F.relu(self.sar(sar_mean))[:,:,None,None].expand_as(xo)
            appearance=self.appearance(torch.cat([xo,zs],1))
            residual=self.residual(torch.cat([xo,zs,self.embedding(ps),confidence,appearance.sigmoid()],1))
            return dict(logits=(appearance+confidence*residual)[:,0],semantic_logits=semantic_logits,probabilities=p,confidence=confidence)
    def forward(self,optical,sar_mean):return self.fields(optical,sar_mean)['logits']


def load_t07_model(saved,mode,k,device='cuda'):
    # U/A/AC preserve exactly the T06 head and uniform rule.
    from .t06_runtime import load_selection_model
    if mode in ['U','A','AC']:return load_selection_model(saved,mode,k,device)
    if mode not in ['A_cap','S','SC']:raise ValueError('Unknown T07 mode')
    model=AppearanceB0(saved['config'],k,selection='U');model.load_state_dict(saved['model'],strict=True)
    path='runs/t06_head_capacity_v1/last.pt' if mode=='A_cap' else 'runs/t07_semantic_head_v1/last.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['matcher_sha256']=='f9ff3f377394514fce2dd72ef349d7a0a71a10d0b2d73a3bd59ce131e23efe90'
    head=AppearanceHead(state['width']) if mode=='A_cap' else SemanticScore()
    head.load_state_dict(state['head'],strict=True);model.appearance_head=head
    model.selection='AC' if mode=='SC' else 'A'
    return model.to(device).eval().requires_grad_(False)


def abstain_semantic_weight(conditional,weight,disabled_classes,image_abstention):
    """Abstain using class/image evidence; preserve original seven-class distribution."""
    if conditional.ndim!=4 or conditional.shape[1]!=7 or weight.shape!=conditional.shape[:1]+conditional.shape[2:]:raise ValueError('Expected B7HW / BHW')
    if image_abstention.shape!=(len(conditional),):raise ValueError('Expected per-image decisions')
    dominant=conditional.argmax(1);accepted=torch.ones_like(weight,dtype=torch.bool)
    for c in disabled_classes:
        if not 0<=c<7:raise ValueError('Invalid land-cover class')
        accepted&=dominant!=c
    return weight*accepted*(~image_abstention.bool())[:,None,None]
