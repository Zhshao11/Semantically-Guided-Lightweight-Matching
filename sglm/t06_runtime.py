"""Load a frozen matcher and a separately trained appearance head."""
import torch
from .t06_appearance import AppearanceB0,AppearanceHead


def load_selection_model(saved,mode,k,device='cuda'):
    model=AppearanceB0(saved['config'],k,selection='U',head=None)
    model.load_state_dict(saved['model'],strict=True)
    if mode!='U':
        path='runs/t06_head_capacity_v1/last.pt' if mode=='AC-cap' else 'runs/t06_head_v1/last.pt'
        h=torch.load(path,map_location='cpu',weights_only=False)
        assert h['matcher_sha256']=='f9ff3f377394514fce2dd72ef349d7a0a71a10d0b2d73a3bd59ce131e23efe90'
        head=AppearanceHead(h['width']);head.load_state_dict(h['head'],strict=True);model.appearance_head=head
    model.selection='AC' if mode=='AC-cap' else mode
    return model.to(device).eval().requires_grad_(False)
