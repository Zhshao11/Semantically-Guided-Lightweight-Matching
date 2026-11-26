"""Freeze TRAIN-only128verifiedparents x2randomchildren before signal inspection."""
import hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=json.loads(Path('configs/t10a_protocol_v1.json').read_text());out=Path('reports/t10_setup_v1');out.mkdir(exist_ok=True)
    if (out/'plan.json').exists():raise FileExistsError('Frozen diagnostic plan already exists')
    original=json.loads(Path('reports/t06_setup_v1/training_plan.json').read_text());manifest={r['pair_id']:r for r in [json.loads(x) for x in Path('reports/metadata_finalize_v1/train_256_dedup_v1.jsonl').read_text().splitlines()]};families={}
    for i,sample in enumerate(original['samples']):
        r=manifest[sample['pair_id']]['parent_mapping'];assert r['status']=='verified';families.setdefault(r['parent_pair_id'],[]).append(i)
    assert len(families)==1978 and all(len(v)==4 for v in families.values());rng=np.random.default_rng(p['seed']);parents=sorted(families);picked=sorted(rng.choice(len(parents),128,replace=False).tolist());samples=[]
    for j in picked:
        for i in sorted(rng.choice(families[parents[j]],2,replace=False).tolist()):samples.append(dict(**original['samples'][i],original_cache_index=i,verified512_parent=parents[j],crop_xy=manifest[original['samples'][i]['pair_id']]['parent_mapping']['crop_xy']))
    assert len(samples)==256 and all('/train/' in x['pair_id'] for x in samples)
    plan=dict(config=original['config'],samples=samples,selection='seeded128parents x2randomchildren, no imagecontent/semanticquality filtering; sampling before predictions',scene_provenance='unresolved; distributed512parents only');(out/'plan.json').write_text(json.dumps(plan,separators=(',',':')))
    metadata=dict(protocol=p,plan_sha256=hashlib.sha256((out/'plan.json').read_bytes()).hexdigest(),T05_checkpoint_sha256=hashlib.sha256(Path(p['matcher_checkpoint']).read_bytes()).hexdigest(),source_plan_sha256=hashlib.sha256(Path('reports/t06_setup_v1/training_plan.json').read_bytes()).hexdigest(),pairs=256,verified512parents=128,children_per_parent=2,validation_or_test_used=False,training_started=False)
    assert metadata['T05_checkpoint_sha256']==p['matcher_sha256'];(out/'freeze.json').write_text(json.dumps(metadata,indent=2));print('Frozen256TRAINpairs,128verifiedparents,storedaffines')


if __name__=='__main__':main()
