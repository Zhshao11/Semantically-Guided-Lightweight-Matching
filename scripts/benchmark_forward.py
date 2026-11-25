"""End-to-end GPU forward latency including hard selection and duplicate removal.
Excludes data loading and external CPU RANSAC. No claims about MACs are inferred.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from sglm import Matcher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--output', required=True)
    ap.add_argument('--repeats', type=int, default=30)
    a = ap.parse_args()
    torch.manual_seed(20261125)
    torch.set_num_threads(4)
    model = Matcher(use_semantics=False).to(a.device).eval()
    optical = torch.rand(1, 3, 256, 256, device=a.device)
    sar = torch.rand(1, 1, 256, 256, device=a.device)
    def sync():
        if a.device.startswith('cuda'):
            torch.cuda.synchronize()
    records = []
    with torch.no_grad():
        for k in (128, 256, 384):
            model.fixed_budget = k
            for _ in range(5):
                model(optical, sar)
            times = []
            for _ in range(a.repeats):
                sync()
                begun = time.perf_counter()
                output = model(optical, sar)
                sync()
                times.append((time.perf_counter() - begun) * 1000)
            records.append(dict(k=k, median_ms=float(np.median(times)), p90_ms=float(np.percentile(times, 90)),
                                correlation_shape=list(output['matches'][0]['logits'].shape),
                                fine_matches=len(output['matches'][0]['source'])))
    report = dict(torch=torch.__version__, device=torch.cuda.get_device_name() if a.device.startswith('cuda') else a.device,
                  parameters_total=sum(p.numel() for p in model.parameters()),
                  mode='untrained, visual baseline, default threshold; fine workload depends on predictions',
                  measures='model forward incl. selection; excludes loading and RANSAC', results=records)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
