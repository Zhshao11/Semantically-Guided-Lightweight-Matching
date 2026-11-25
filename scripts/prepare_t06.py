"""Freeze train-only affine labels and paired teacher presentation plans."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sglm.t02.data import sample_transform


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='reports/t06_setup_v1');a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'training_plan.json').exists():raise FileExistsError('Do not overwrite frozen plan')
    protocol=json.loads(Path('configs/t06_protocol_v1.json').read_text())
    cfg=json.loads(Path('reports/t05_frozen_v1/candidate_config.json').read_text())
    assert hashlib.sha256(Path(protocol['matcher_checkpoint']).read_bytes()).hexdigest()==protocol['matcher_sha256']
    rows=[json.loads(line) for line in Path('reports/metadata_finalize_v1/train_256_dedup_v1.jsonl').read_text().splitlines()]
    rows=sorted(rows,key=lambda r:r['pair_id']);assert len(rows)==7912 and all(r['split']=='train' for r in rows)
    samples=[]
    for r in rows:
        seed=int.from_bytes(hashlib.sha256(f"{protocol['seed']}/{r['pair_id']}/T06-fixed-label".encode()).digest()[:8],'little')
        A,parameters=sample_transform(np.random.default_rng(seed),cfg)
        samples.append(dict(pair_id=r['pair_id'],split='train',optical_path=r['optical']['path'],sar_path=r['sar']['path'],
            optical_sha256=r['optical']['sha256'],sar_sha256=r['sar']['sha256'],A_forward=A.tolist(),parameters=parameters))
    plan=dict(config=cfg,samples=samples);(out/'training_plan.json').write_text(json.dumps(plan,separators=(',',':')))
    rng=np.random.default_rng(protocol['seed']);indices=sorted(rng.choice(7912,128,replace=False).tolist())
    # Retain old8 plus random train-only choices for blind OS semantic review.
    old=[r for r in json.loads(Path('reports/t02_b0_full_v1/fixed_plan.json').read_text())['samples'] if r['split']=='train'][:8]
    keep={r['pair_id'] for r in old};extra=[samples[i] for i in rng.permutation(7912).tolist() if samples[i]['pair_id'] not in keep]
    review=old+extra[:24];assert len(review)==32
    (out/'OS_train_review.json').write_text(json.dumps(dict(samples=review,split='train',semantic_GT=False),indent=2))
    result=dict(protocol=protocol,train_plan_sha256=hashlib.sha256((out/'training_plan.json').read_bytes()).hexdigest(),
        probe_indices=indices,validation_or_test_used_for_training_labels=False)
    (out/'setup.json').write_text(json.dumps(result,indent=2));print('Saved fixed training transforms',len(samples))


if __name__=='__main__':main()
