import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sglm import Matcher
from sglm.config import load_config
from sglm.data import OSPairs
from sglm.losses import matching_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--max-steps', type=int)
    ap.add_argument('--semantic-cache')
    ap.add_argument('--resume')
    ap.add_argument('--config')
    ap.add_argument('--manifest')
    ap.add_argument('--resolution', type=int, choices=[256, 512], default=256)
    a = ap.parse_args()
    effective = load_config(a.config)
    a.seed = effective['seed'] if a.seed is None else a.seed
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(a), indent=2))
    ds = OSPairs(a.root, limit=a.limit, seed=a.seed, semantic_cache=a.semantic_cache,
                 resolution=a.resolution, manifest=a.manifest, config_path=a.config)
    (out / 'data_geometry_config.json').write_text(json.dumps(ds.config, indent=2))
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=0)
    # Explicit visual baseline when no trained semantic teacher cache is provided.
    model = Matcher(use_semantics=bool(a.semantic_cache)).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    step, start_epoch = 0, 0
    if a.resume:
        ckpt = torch.load(a.resume, map_location=a.device, weights_only=False)
        if bool(ckpt['config'].get('semantic_cache')) != bool(a.semantic_cache):
            raise ValueError('Resume must preserve semantic supervision mode.')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step, start_epoch = ckpt['step'], ckpt['epoch'] + 1
    begun = time.time()
    model.train()
    for epoch in range(start_epoch, a.epochs):
        ds.epoch = epoch
        for batch in loader:
            batch = {k: v.to(a.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(batch['optical'], batch['sar'], batch['optical_mask'], batch['sar_mask'])
            total, losses = matching_loss(model, output, batch)
            if not torch.isfinite(total):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            total.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            record = dict(step=step, epoch=epoch, total=float(total.detach()),
                          grad_norm=float(grad), seconds=time.time() - begun,
                          **{k: float(v.detach()) for k, v in losses.items()})
            with (out / 'train.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            if step % 10 == 0 or step == 1:
                print(json.dumps(record), flush=True)
            if a.max_steps and step >= a.max_steps:
                break
        ckpt = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=vars(a),
                    epoch=epoch, step=step, use_semantics=model.use_semantics, fixed_budget=model.fixed_budget)
        torch.save(ckpt, out / 'last.pt')
        if a.max_steps and step >= a.max_steps:
            break
    print(json.dumps({'completed_steps': step, 'checkpoint': str(out / 'last.pt')}), flush=True)


if __name__ == '__main__':
    main()
