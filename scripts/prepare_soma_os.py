"""Freeze train/validation/test roles and test affine draws before model training."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from sglm.t02.data import sample_transform


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cfg=json.loads(Path('configs/soma_os_training_v1.json').read_text());out=Path(cfg['reports']);out.mkdir(parents=True,exist_ok=True)
    if (out/'data_plan.json').exists():raise FileExistsError('Frozen SOMA plan exists')
    base=json.loads(Path('reports/t03_setup_v1/full_data_plan.json').read_text())
    val=json.loads(Path('reports/t05_frozen_v1/validation_plan.json').read_text())['samples']
    assert base['val']==val or [x['A_forward'] for x in base['val']]==[x['A_forward'] for x in val]
    reference=json.loads(Path(cfg['reference_config']).read_text());reference['seed']=cfg['seed']
    test=[];manifest='reports/data_geometry_v1/dataset_manifest.jsonl'
    entries=[json.loads(x) for x in Path(manifest).read_text().splitlines()]
    candidates=sorted([x for x in entries if x['dataset']=='OSdataset' and x['resolution']==256 and x['split']=='test' and x['usable']],key=lambda x:x['pair_id'])
    for x in candidates:
        seed=int.from_bytes(hashlib.sha256(f'{cfg["seed"]}/{x["pair_id"]}/SOMA-final-test-v1'.encode()).digest()[:8],'little')
        A,params=sample_transform(np.random.default_rng(seed),reference)
        test.append(dict(pair_id=x['pair_id'],split='test',optical_path=x['optical']['path'],sar_path=x['sar']['path'],
            optical_sha256=x['optical']['sha256'],sar_sha256=x['sar']['sha256'],A_forward=A.tolist(),parameters=params))
    plan=dict(protocol=cfg['protocol'],train=base['train'],val=val,test=test)
    assert {k:len(plan[k]) for k in ['train','val','test']}==cfg['split_counts']
    sets=[{x['pair_id'] for x in plan[k]} for k in ['train','val','test']]
    assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    (out/'data_plan.json').write_text(json.dumps(plan,indent=2))
    (out/'dataset_config.json').write_text(json.dumps(reference,indent=2))
    frozen=dict(config=cfg,plan_sha256=sha(out/'data_plan.json'),test_transforms_frozen_before_training=True,
        source_train_manifest_sha256=sha('reports/metadata_finalize_v1/train_256_dedup_v1.jsonl'),
        original_full_plan_sha256=sha('reports/t03_setup_v1/full_data_plan.json'),
        T05_validation_plan_sha256=sha('reports/t05_frozen_v1/validation_plan.json'),
        training_test_roles='onlyval selects checkpoint; test once after training; no testdata or scores used for stopping')
    (out/'freeze.json').write_text(json.dumps(frozen,indent=2));print(cfg['split_counts'])


if __name__=='__main__':main()
