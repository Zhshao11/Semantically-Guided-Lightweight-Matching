"""Initial training objectives; loss weights are engineering defaults, not validated results."""
import torch
import torch.nn.functional as F
from .model import coarse_xy


def warp(points, transform):
    return points @ transform[:, :2].T + transform[:, 2]


def targets_for_queries(query, width, height, transform, sar_mask):
    target = warp(coarse_xy(query, width), transform)
    cells = torch.floor((target + .5) / 8).long()
    valid = ((cells >= 0) & (cells < cells.new_tensor([width, height]))).all(-1)
    safe = cells.clamp_min(0).minimum(cells.new_tensor([width - 1, height - 1]))
    valid &= sar_mask[safe[:, 1], safe[:, 0]]
    labels = cells[:, 1] * width + cells[:, 0]
    labels = torch.where(valid, labels, width * height)
    return labels, target, valid


def matching_loss(model, output, batch, semantic_weight=1.):
    device = batch['optical'].device
    losses = {k: batch['optical'].new_zeros(()) for k in ('coarse', 'appearance', 'selection', 'fine', 'reliability', 'semantic')}
    b, _, h, w = output['scores']['p'].shape
    all_queries = torch.arange(h * w, device=device)
    for i, match in enumerate(output['matches']):
        gt = batch['transform'][i]
        mask = output['sar_coarse_mask'][i]
        labels, _, _ = targets_for_queries(match['query_indices'], w, h, gt, mask)
        if len(labels):
            losses['coarse'] += F.cross_entropy(match['logits'], labels)
        # Dense descriptor-independent supervision keeps the hard selector trainable.
        _, _, visible = targets_for_queries(all_queries, w, h, gt, mask)
        for name, loss_name in [('m', 'appearance'), ('v', 'selection')]:
            pred = output['scores'][name][i, 0].flatten().clamp(1e-6, 1 - 1e-6)
            losses[loss_name] += F.binary_cross_entropy(pred, visible.float())
        # Teacher-forced GT local windows bootstrap fine training before coarse matches exist.
        source = coarse_xy(match['query_indices'], w)
        target = warp(source, gt)
        cells = torch.floor((target + .5) / 8)
        centers = cells * 8 + 3.5
        inside = ((target >= 0) & (target <= target.new_tensor([w * 8 - 1, h * 8 - 1]))).all(-1)
        safe = cells.long().clamp_min(0).minimum(cells.new_tensor([w - 1, h - 1]).long())
        inside &= mask[safe[:, 1], safe[:, 0]]
        if inside.any():
            qsrc, qtarget, qcenter = source[inside], target[inside], centers[inside]
            fine = model.fine(output['fine_optical'][i:i+1], output['fine_sar'][i:i+1],
                              qsrc, qcenter, torch.ones(len(qsrc), device=device),
                              batch.get('optical_mask', None)[i:i+1] if 'optical_mask' in batch else None,
                              batch.get('sar_mask', None)[i:i+1] if 'sar_mask' in batch else None)
            losses['fine'] += F.smooth_l1_loss(fine['target'], qtarget)
            correct = ((fine['target'].detach() - qtarget).norm(dim=-1) < 3).float()
            losses['reliability'] += F.binary_cross_entropy(fine['reliability'].clamp(1e-6, 1 - 1e-6), correct)
        # Actual predictions provide negative as well as positive reliability examples.
        if len(match['source']):
            gt_target = warp(match['source'], gt)
            correct = ((match['target'].detach() - gt_target).norm(dim=-1) < 3).float()
            losses['reliability'] += F.binary_cross_entropy(match['reliability'].clamp(1e-6, 1 - 1e-6), correct)
    for k in losses:
        losses[k] = losses[k] / b
    if 'teacher' in batch:
        t = batch['teacher'].float().clamp_min(1e-8)
        t = t / t.sum(1, keepdim=True)
        kl = (t * (t.log() - output['scores']['p'].clamp_min(1e-8).log())).sum(1)
        mask = batch['teacher_mask'] & output['optical_coarse_mask']
        losses['semantic'] = (kl * mask).sum() / mask.sum().clamp_min(1)
    total = losses['coarse'] + .2 * (losses['appearance'] + losses['selection']) + losses['fine'] + .2 * losses['reliability'] + semantic_weight * losses['semantic']
    return total, losses
