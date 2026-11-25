"""Deterministic held-out evaluation. Never removes failed pairs from the CMR denominator."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sglm import Matcher
from sglm.data import OSPairs
from sglm.geometry import estimate_affine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--min-reliability', type=float, default=.5)
    ap.add_argument('--coarse-threshold', type=float, default=.05)
    ap.add_argument('--seed', type=int, default=20261125)
    a = ap.parse_args()
    torch.set_num_threads(4)
    cv_seed = a.seed % (2**31 - 1)
    import cv2
    cv2.setRNGSeed(cv_seed)
    ckpt = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
    model = Matcher(fixed_budget=ckpt.get('fixed_budget', 256), use_semantics=ckpt['use_semantics'],
                    coarse_threshold=a.coarse_threshold).to(a.device).eval()
    model.load_state_dict(ckpt['model'])
    ds = OSPairs(a.root, split=a.split, limit=a.limit, seed=a.seed)
    records = []
    with torch.no_grad():
        for item in ds:
            result = model(item['optical'][None].to(a.device), item['sar'][None].to(a.device),
                           item['optical_mask'][None].to(a.device), item['sar_mask'][None].to(a.device))['matches'][0]
            source, target, conf = [result[k].cpu().numpy() for k in ('source', 'target', 'reliability')]
            matrix, inliers = estimate_affine(source, target, conf, min_reliability=a.min_reliability)
            gt = item['transform'].numpy()
            expected = source @ gt[:, :2].T + gt[:, 2]
            pair_rmse = float(np.sqrt(np.mean(np.sum((target - expected)**2, axis=1)))) if len(source) else None
            # Additional fixed-grid transform error cannot be improved merely by emitting few matches.
            y, x = np.meshgrid(np.linspace(16, 239, 8), np.linspace(16, 239, 8), indexing='ij')
            grid = np.stack([x.ravel(), y.ravel()], -1)
            gt_grid = grid @ gt[:, :2].T + gt[:, 2]
            valid = ((gt_grid >= 0) & (gt_grid <= 255)).all(1)
            grid = grid[valid]
            err = None
            if matrix is not None and len(grid):
                predicted = grid @ matrix[:, :2].T + matrix[:, 2]
                expected_grid = grid @ gt[:, :2].T + gt[:, 2]
                err = float(np.sqrt(np.mean(np.sum((predicted - expected_grid)**2, 1))))
            records.append(dict(id=item['id'], matches=len(source), inliers=int(inliers.sum()),
                                correspondence_rmse_px=pair_rmse, affine_grid_rmse_px=err,
                                affine=None if matrix is None else matrix.tolist(), gt_affine=gt.tolist()))
    n = len(records)
    summary = dict(protocol='builtin-OSdataset-256 + deterministic-synthetic-affine-v1', split=a.split,
                   pairs=n, seed=a.seed, checkpoint=a.checkpoint,
                   use_semantics=ckpt['use_semantics'],
                   affine_failures=sum(r['affine'] is None for r in records))
    for label, field in [('correspondence', 'correspondence_rmse_px'), ('affine_grid', 'affine_grid_rmse_px')]:
        finite = [r[field] for r in records if r[field] is not None]
        summary[label + '_defined_pairs'] = len(finite)
        summary[label + '_mean_rmse_defined_pairs_px'] = float(np.mean(finite)) if finite else None
        for threshold in (1, 3, 5):
            summary[f'{label}_CMR@{threshold}'] = sum(r[field] is not None and r[field] < threshold for r in records) / n
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(dict(summary=summary, config=vars(a), pairs=records), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
