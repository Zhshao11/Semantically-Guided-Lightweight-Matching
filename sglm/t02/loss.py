"""Forward affine correspondence labels and no-match supervision, outside inference."""
import torch
import torch.nn.functional as F
from ..model import coarse_xy


def correspondence_labels(A_forward,source_valid_mask,target_valid_mask):
    batch=len(A_forward)
    points=coarse_xy(torch.arange(1024,device=A_forward.device),32)
    target=torch.einsum('bij,nj->bni',A_forward[:,:,:2].float(),points)+A_forward[:,:,2].float()[:,None]
    cells=torch.floor((target+.5)/8).long()
    in_bounds=((target>=0)&(target<=255)).all(-1)&((cells>=0)&(cells<32)).all(-1)
    safe=cells.clamp(0,31)
    target_cells=F.avg_pool2d(target_valid_mask[:,None].float(),8,8)[:,0]>.999
    source_cells=F.avg_pool2d(source_valid_mask[:,None].float(),8,8)[:,0]>.999
    ids=torch.arange(batch,device=A_forward.device)[:,None]
    positive=in_bounds&target_cells[ids,safe[...,1],safe[...,0]]&source_cells.flatten(1)
    label=torch.where(positive,cells[...,1]*32+cells[...,0],1024)
    label=label.masked_fill(~source_cells.flatten(1),-100)
    return label,target,positive


def coarse_ce(logits,labels,no_match_weight):
    per_query=F.cross_entropy(logits.reshape(-1,1025),labels.flatten(),ignore_index=-100,reduction='none')
    flat=labels.flatten()
    positive=(flat>=0)&(flat<1024)
    negative=flat==1024
    # Separate means prevent a changing overlap ratio from hiding one class.
    pos_loss=per_query[positive].mean() if positive.any() else per_query.sum()*0
    neg_loss=per_query[negative].mean() if negative.any() else per_query.sum()*0
    return pos_loss+no_match_weight*neg_loss


def b0_loss(model,output,batch,config,step):
    forward=batch['A_forward'].float()
    labels,targets,positive=correspondence_labels(forward,batch['optical_valid_mask'],batch['sar_aug_valid_mask'])
    homogeneous=torch.eye(3,device=forward.device)[None].repeat(len(forward),1,1)
    homogeneous[:,:2]=forward
    reverse=torch.linalg.inv(homogeneous)[:,:2]
    reverse_labels,_,_=correspondence_labels(reverse,batch['sar_aug_valid_mask'],batch['optical_valid_mask'])
    no_match_weight=config['training']['no_match_loss_weight']
    coarse=(coarse_ce(output['row_logits'],labels,no_match_weight)+coarse_ce(output['reverse_logits'],reverse_labels,no_match_weight))/2
    fine=coarse.new_zeros(())
    fine_pairs=0
    bootstrap=step<config['training']['fine_gt_bootstrap_steps']
    for index in range(len(forward)):
        if bootstrap:
            queries=torch.where(positive[index])[0]
            source=coarse_xy(queries,32)
            expected=targets[index,queries]
            centers=torch.floor((expected+.5)/8)*8+3.5
        else:
            source,centers,_=model.coarse_matches(output,index)
            expected=source@forward[index,:,:2].T+forward[index,:,2]
            # Only train a local readout when the true coordinate lies inside its predicted window.
            inside=((expected-centers).abs()<=6).all(-1)&((expected>=0)&(expected<=255)).all(-1)
            source,expected,centers=source[inside],expected[inside],centers[inside]
        cap=config['training']['fine_training_samples_per_pair']
        if len(source)>cap:
            choose=torch.linspace(0,len(source)-1,cap,device=source.device).long()
            source,expected,centers=source[choose],expected[choose],centers[choose]
        if len(source):
            local=model.local(output['optical_fine'][index:index+1],output['sar_fine'][index:index+1],source,centers,
                batch['optical_valid_mask'][index:index+1],batch['sar_aug_valid_mask'][index:index+1])
            valid=local['valid']
            if valid.any():
                fine+=F.smooth_l1_loss(local['target'][valid],expected[valid])
                fine_pairs+=int(valid.sum())
    fine=fine/len(forward)
    total=coarse+config['training']['fine_loss_weight']*fine
    with torch.no_grad():
        prediction=output['row_logits'].argmax(-1)
        coarse_accuracy=(prediction[positive]==labels[positive]).float().mean() if positive.any() else coarse.new_zeros(())
    return total,dict(coarse=float(coarse.detach()),fine=float(fine.detach()),positive_coarse_accuracy=float(coarse_accuracy),
        fine_training_pairs=fine_pairs,fine_window_mode='GT_bootstrap_training_only' if bootstrap else 'predicted_coarse_training')
