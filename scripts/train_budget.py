"""Train-only budget oracle and normalization; matcher remains frozen.
Oracle selects smallest K retaining 90% of best GT-correct coarse matches and
covering at least four macro cells, with at least six correct matches.
Pairs failing all budgets are skipped, never labeled as an artificial success.
"""
import argparse
import json
from pathlib import Path
import torch
from sglm import Matcher
from sglm.data import OSPairs


def main():
    ap = argparse.ArgumentParser()
    for key in ('root', 'checkpoint', 'output'):
        ap.add_argument('--' + key, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--epochs', type=int, default=100)
    a = ap.parse_args()
    torch.manual_seed(20261125)
    torch.set_num_threads(4)
    checkpoint = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
    model = Matcher(use_semantics=checkpoint['use_semantics']).to(a.device).eval()
    model.load_state_dict(checkpoint['model'])
    data = OSPairs(a.root, split='train', limit=a.limit)
    statistics, labels, oracle = [], [], []
    with torch.no_grad():
        for item in data:
            counts, coverage = [], []
            stats = None
            gt = item['transform'].to(a.device)
            for k in (128, 256, 384):
                model.fixed_budget = k
                output = model(item['optical'][None].to(a.device), item['sar'][None].to(a.device),
                               item['optical_mask'][None].to(a.device), item['sar_mask'][None].to(a.device))
                stats = output['statistics'][0]
                result = output['matches'][0]
                expected = result['source'] @ gt[:, :2].T + gt[:, 2]
                correct = (result['coarse_target'] - expected).norm(dim=-1) < 8
                points = result['source'][correct]
                cells = (points // 64).long()
                counts.append(int(correct.sum()))
                coverage.append(len(cells.unique(dim=0)))
            eligible = [i for i in range(3) if counts[i] >= max(6, .9 * max(counts)) and coverage[i] >= 4]
            oracle.append(dict(id=item['id'], correct=counts, coverage=coverage,
                               label=eligible[0] if eligible else None))
            if eligible:
                statistics.append(stats)
                labels.append(eligible[0])
    if not statistics:
        raise RuntimeError('No budget-oracle successes: train the fixed-budget matcher before training a budget head.')
    x, y = torch.stack(statistics), torch.tensor(labels, device=a.device)
    model.budget.mean.copy_(x.mean(0))
    model.budget.std.copy_(x.std(0, unbiased=False).clamp_min(1e-3))
    model.budget.stats_fitted.fill_(True)
    optimizer = torch.optim.AdamW(model.budget.head.parameters(), lr=1e-3)
    for _ in range(a.epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model.budget(x), y)
        loss.backward()
        optimizer.step()
    checkpoint.update(model=model.state_dict(), fixed_budget=None)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    output.with_suffix('.oracle.json').write_text(json.dumps(dict(split='train', total=len(data),
        usable=len(labels), config=vars(a), oracle=oracle), indent=2))
    print(json.dumps(dict(usable=len(labels), loss=float(loss.detach()), output=str(output))))


if __name__ == '__main__':
    main()
