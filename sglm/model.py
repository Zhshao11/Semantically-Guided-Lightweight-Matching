"""v0.1 lightweight optical-SAR matcher. Coordinates use pixel centers (x, y)."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class ModalBN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norms = nn.ModuleList([nn.BatchNorm2d(channels), nn.BatchNorm2d(channels)])

    def forward(self, x, modality):
        return self.norms[modality](x)


class ModalConv(nn.Module):
    def __init__(self, cin, cout, kernel=1, stride=1, groups=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=False)
        self.bn = ModalBN(cout)
        self.relu = relu

    def forward(self, x, modality):
        x = self.bn(self.conv(x), modality)
        return F.relu(x) if self.relu else x


class DSRes(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = ModalConv(c, c, 3, groups=c)
        self.pw = ModalConv(c, c, relu=False)

    def forward(self, x, modality):
        return F.relu(x + self.pw(self.dw(x, modality), modality))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleList([nn.Sequential(nn.Conv2d(c, 16, 3, 2, 1, bias=False),
                                      nn.BatchNorm2d(16), nn.ReLU()) for c in (3, 1)])
        self.stages = nn.ModuleList([nn.ModuleList([DSRes(c) for _ in range(n)])
                                    for c, n in [(16, 2), (32, 2), (64, 3)]])
        self.down = nn.ModuleList([nn.ModuleList([ModalConv(c, c, 3, 2, c), ModalConv(c, c * 2)])
                                  for c in (16, 32)])
        self.lateral = nn.ModuleList([ModalConv(c, c) for c in (16, 32, 64)])
        self.up = nn.ModuleList([ModalConv(32, 16), ModalConv(64, 32)])
        self.fuse = nn.ModuleList([DSRes(16), DSRes(32)])
        self.descriptor = nn.Conv2d(64, 32, 1)

    def forward(self, image, modality):
        x, features = self.adapters[modality](image), []
        for i, blocks in enumerate(self.stages):
            if i:
                for layer in self.down[i - 1]:
                    x = layer(x, modality)
            for block in blocks:
                x = block(x, modality)
            features.append(x)
        p = self.lateral[2](features[2], modality)
        coarse = p
        for i in (1, 0):
            p = self.fuse[i](self.lateral[i](features[i], modality) +
                             F.interpolate(self.up[i](p, modality), size=features[i].shape[-2:],
                                           mode='bilinear', align_corners=False), modality)
        desc = F.normalize(self.descriptor(coarse), dim=1).flatten(2).transpose(1, 2)
        return coarse, p, desc


class JointScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 32, 1),
                                      nn.ReLU(), nn.Conv2d(32, 7, 1))
        self.optical = nn.Conv2d(64, 16, 1)
        self.sar = nn.Linear(64, 16)
        self.appearance = nn.Sequential(nn.Conv2d(32, 32, 1), nn.ReLU(),
                                       nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.ReLU(),
                                       nn.Conv2d(32, 1, 1))
        self.embedding = nn.Conv2d(7, 8, 1)
        self.residual = nn.Sequential(nn.Conv2d(42, 32, 1), nn.ReLU(), nn.Conv2d(32, 1, 1))

    def forward(self, fo, fs, use_semantics=True):
        p = self.semantic(fo).softmax(1)
        # KL alone updates the semantic head; selection must not collapse its predictions.
        ps = p.detach()
        c = 1 + (ps * ps.clamp_min(1e-8).log()).sum(1, keepdim=True) / math.log(7)
        xo = F.relu(self.optical(fo))
        zs = F.relu(self.sar(fs.mean((2, 3))))[:, :, None, None].expand_as(xo)
        a = self.appearance(torch.cat([xo, zs], 1))
        m = a.sigmoid()
        r = self.residual(torch.cat([xo, zs, self.embedding(ps), c, m], 1))
        v = (a + c * r).sigmoid() if use_semantics else m
        return dict(p=p, c=c, m=m, v=v)


def grid_masks(h, w, device):
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    return [((y * 4 // h == gy) & (x * 4 // w == gx)).flatten() for gy in range(4) for gx in range(4)]


def masked_mean(x, mask):
    return (x * mask).sum(-1) / mask.sum(-1).clamp_min(1)


class Budget(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(178, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 3))
        self.register_buffer('mean', torch.zeros(178))
        self.register_buffer('std', torch.ones(178))
        self.register_buffer('stats_fitted', torch.tensor(False))

    def statistics(self, fo, fs, scores, mo, ms):
        b, _, h, w = fo.shape
        om, sm = mo.flatten(1).float(), ms.flatten(1).float()
        stats = [masked_mean(fo.flatten(2), om[:, None]), masked_mean(fs.flatten(2), sm[:, None])]
        for name in ('v', 'm', 'c'):
            x = scores[name].flatten(1)
            mean = masked_mean(x, om)
            std = masked_mean((x - mean[:, None]).square(), om).sqrt()
            stats.append(torch.stack([mean, std, masked_mean((x > .5).float(), om)], 1))
        values = scores['v'].flatten(1)
        grid_mean, grid_top = [], []
        for mask in grid_masks(h, w, fo.device):
            valid = om.bool() & mask[None]
            grid_mean.append(masked_mean(values, valid))
            top = values.masked_fill(~valid, -torch.inf).topk(min(8, int(mask.sum())), 1).values
            grid_top.append(torch.where(torch.isfinite(top), top, 0).sum(1) /
                            torch.isfinite(top).sum(1).clamp_min(1))
        stats.extend([torch.stack(grid_mean, 1), torch.stack(grid_top, 1),
                      masked_mean(scores['p'].flatten(2), om[:, None]),
                      torch.stack([om.mean(1), sm.mean(1)], 1)])
        return torch.cat(stats, 1)

    def forward(self, stats):
        if not bool(self.stats_fitted):
            raise RuntimeError('Fit budget normalization on training data before dynamic-budget inference.')
        return self.head((stats - self.mean) / self.std.clamp_min(1e-3))


@torch.no_grad()
def select_queries(value, valid, k, suppression=1):
    """Hard 4x4 allocation: base 2/cell, capped Hamilton apportionment, then local NMS."""
    h, w = value.shape
    masks = grid_masks(h, w, value.device)
    flat, good = value.flatten(), valid.flatten().bool()
    candidates = [torch.where(mask & good)[0] for mask in masks]
    capacity = [min(32, len(c)) for c in candidates]
    k = min(int(k), sum(capacity))
    quota = [min(2, c) for c in capacity]
    if sum(quota) > k:
        raise ValueError('Budget is below the base coverage quota.')
    utility = torch.stack([flat[c].topk(min(8, len(c))).values.mean() if len(c)
                           else flat.new_tensor(-1e4) for c in candidates])
    weights = utility.softmax(0).cpu().tolist()
    # Recompute allocation among cells with remaining capacity; deterministic tie-breaking.
    while sum(quota) < k:
        remaining = k - sum(quota)
        active = [i for i in range(16) if quota[i] < capacity[i]]
        total = sum(weights[i] for i in active)
        shares = {i: remaining * weights[i] / total for i in active}
        added = 0
        for i in active:
            n = min(capacity[i] - quota[i], int(shares[i]))
            quota[i] += n
            added += n
        if not added:
            for i in sorted(active, key=lambda j: (-(shares[j] % 1), j))[:remaining]:
                quota[i] += 1
    selected = []
    for c, q in zip(candidates, quota):
        ordered = c[torch.argsort(flat[c], descending=True, stable=True)].tolist()
        picks, skipped = [], []
        for idx in ordered:
            if len(picks) == q:
                break
            if any(max(abs(idx // w - j // w), abs(idx % w - j % w)) <= suppression for j in picks):
                skipped.append(idx)
            else:
                picks.append(idx)
        if len(picks) < q:
            picks.extend(skipped[:q - len(picks)])
        selected.extend(picks)
    return torch.tensor(selected, device=value.device, dtype=torch.long)


def coarse_xy(indices, width):
    # Cell center in input pixels. align_corners=False throughout the project.
    return torch.stack([indices % width, indices // width], -1).float() * 8 + 3.5


def sample_windows(feature, centers, radius, image_mask=None):
    """Input-pixel centers -> stride-2 feature samples; no clamp of invalid candidates."""
    _, c, h, w = feature.shape
    y, x = torch.meshgrid(torch.arange(-radius, radius + 1, device=feature.device),
                          torch.arange(-radius, radius + 1, device=feature.device), indexing='ij')
    delta = torch.stack([x, y], -1).flatten(0, 1).float()
    positions = (centers[:, None] + .5) / 2 - .5 + delta[None]
    mask = ((positions >= 0) & (positions <= positions.new_tensor([w - 1, h - 1]))).all(-1)
    grid = 2 * (positions + .5) / positions.new_tensor([w, h]) - 1
    if image_mask is not None:
        sampled_valid = F.grid_sample(image_mask[:, None].float(), grid[None], align_corners=False, mode='bilinear')[0, 0]
        mask &= sampled_valid > .999
    tokens = F.grid_sample(feature, grid[None], align_corners=False, mode='bilinear')
    return tokens[0].permute(1, 2, 0), delta, mask


class Fine(nn.Module):
    def __init__(self):
        super().__init__()
        self.wf = nn.Linear(16, 32)
        for name in ('wa', 'wc', 'wagg', 'wq', 'wy'):
            setattr(self, name, nn.Linear(32, 32, bias=False))
        self.wo = nn.Linear(2, 32, bias=False)
        self.ws = nn.Linear(2, 32, bias=False)
        self.weight = nn.Linear(32, 1, bias=False)
        self.ln0, self.lnq, self.lny, self.lnf = [nn.LayerNorm(32) for _ in range(4)]
        self.attn = nn.MultiheadAttention(32, 2, batch_first=True, dropout=0)
        self.ffn = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(.1)))
        self.reliability = nn.Sequential(nn.Linear(36, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, lo, ls, source, target, coarse_conf, optical_mask=None, sar_mask=None):
        if not len(source):
            return {'target': target, 'reliability': coarse_conf, 'heatmap': target.new_empty(0, 49)}
        x, po, om = sample_windows(lo, source, 1, optical_mask)
        y, delta, sm = sample_windows(ls, target, 3, sar_mask)
        x, y = self.wf(x), self.wf(y)
        xc = x[:, 4]
        alpha = self.weight(F.relu(self.wa(x) + self.wc(xc)[:, None] + self.wo(po)[None])).squeeze(-1)
        alpha = alpha.masked_fill(~om, -torch.inf).softmax(-1)
        q0 = self.ln0(xc + self.wagg((alpha[:, :, None] * x).sum(1)))
        yn = self.lny(y)
        attended, _ = self.attn(self.lnq(q0)[:, None], yn + self.ws(delta / 3)[None],
                                yn, key_padding_mask=~sm, need_weights=False)
        q1 = q0 + attended[:, 0]
        q2 = q1 + self.ffn(self.lnf(q1))
        logits = (F.normalize(self.wq(q2), dim=-1)[:, None] * F.normalize(self.wy(y), dim=-1)).sum(-1)
        heat = (logits / self.log_temperature.exp().clamp_min(1e-4)).masked_fill(~sm, -torch.inf).softmax(-1)
        offset = heat @ delta
        refined = target + 2 * offset
        entropy = -(heat * heat.clamp_min(1e-8).log()).sum(-1) / math.log(49)
        variance = (heat * (delta[None] - offset[:, None]).square().sum(-1)).sum(-1)
        evidence = torch.stack([coarse_conf, heat.max(-1).values, entropy, variance], -1)
        reliability = self.reliability(torch.cat([q2, evidence], -1)).sigmoid().squeeze(-1)
        return {'target': refined, 'reliability': reliability, 'heatmap': heat}


class Matcher(nn.Module):
    def __init__(self, fixed_budget=256, use_semantics=True, coarse_threshold=.05):
        super().__init__()
        self.backbone, self.score, self.budget, self.fine = Backbone(), JointScore(), Budget(), Fine()
        self.log_temperature = nn.Parameter(torch.tensor(math.log(.1)))
        self.no_match = nn.Parameter(torch.tensor(0.))
        self.fixed_budget, self.use_semantics = fixed_budget, use_semantics
        self.coarse_threshold = coarse_threshold

    def forward(self, optical, sar, optical_mask=None, sar_mask=None):
        if optical.shape[-2:] != sar.shape[-2:] or any(s % 8 for s in optical.shape[-2:]):
            raise ValueError('Images must have equal sizes divisible by 8.')
        fo, lo, do = self.backbone(optical, 0)
        fs, ls, ds = self.backbone(sar, 1)
        b, _, h, w = fo.shape
        def mask_at(mask):
            if mask is None:
                return torch.ones(b, h, w, dtype=torch.bool, device=fo.device)
            return F.avg_pool2d(mask[:, None].float(), 8, 8)[:, 0] > .999
        mo, ms = mask_at(optical_mask), mask_at(sar_mask)
        scores = self.score(fo, fs, self.use_semantics)
        stats = self.budget.statistics(fo, fs, scores, mo, ms)
        budget_logits = self.budget(stats.detach()) if self.fixed_budget is None else None
        results = []
        for i in range(b):
            k = self.fixed_budget if self.fixed_budget is not None else (128, 256, 384)[budget_logits[i].argmax().item()]
            qi = select_queries(scores['v'][i, 0], mo[i], k)
            # True Gather precedes any cross-image correlation; SAR remains full size.
            logits = do[i, qi] @ ds[i].T / self.log_temperature.exp().clamp_min(1e-4)
            logits = logits.masked_fill(~ms[i].flatten()[None], -torch.inf)
            logits = torch.cat([logits, self.no_match.expand(len(qi), 1)], 1)
            probs = logits.softmax(-1)
            conf, ti = probs.max(-1)
            keep = (ti < h * w) & (conf >= self.coarse_threshold)
            # For duplicate SAR targets retain only the row with greatest confidence.
            unique = torch.zeros_like(keep)
            for row in torch.argsort(conf, descending=True).tolist():
                if keep[row] and not ((ti[unique] == ti[row]).any()):
                    unique[row] = True
            rows = torch.where(unique)[0]
            source, target = coarse_xy(qi[rows], w), coarse_xy(ti[rows], w)
            fine = self.fine(lo[i:i+1], ls[i:i+1], source, target, conf[rows],
                             None if optical_mask is None else optical_mask[i:i+1],
                             None if sar_mask is None else sar_mask[i:i+1])
            results.append(dict(query_indices=qi, logits=logits, coarse_rows=rows, source=source,
                                coarse_target=target, coarse_confidence=conf[rows], **fine))
        return dict(matches=results, scores=scores, statistics=stats,
                    budget_logits=budget_logits, fine_optical=lo, fine_sar=ls,
                    optical_coarse_mask=mo, sar_coarse_mask=ms)
