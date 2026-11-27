"""Freeze official source snapshots, checkpoint hashes, data protocol and environments."""
import hashlib,json,subprocess
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=json.loads(Path('configs/external_references_v1.json').read_text());out=Path(p['output']);out.mkdir(exist_ok=True,parents=True)
    if (out/'freeze.json').exists():raise FileExistsError('External reference freeze exists')
    root=Path(p['source_root']);expected=json.loads((out/'official_source_snapshot.json').read_text())
    for rel,digest in expected['files_sha256'].items():assert sha(root/rel)==digest,rel
    weights={}
    for name in ['minima_roma.pth','weights_xoftr_640.ckpt','dinov2_vitl14_pretrain.pth']:
        path=root/'weights'/name;weights[str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
    old=json.loads(Path(p['frozen']+'/freeze.json').read_text())
    assert sha(old['frozen_checkpoint'])==old['checkpoint_sha256']
    for path,record in old['original_code_verification'].items():assert sha(path)==record['sha256']
    files=['configs/external_references_v1.json',p['frozen']+'/validation_plan.json',p['frozen']+'/subsets.json',p['frozen']+'/candidate_config.json',
           'reports/t10_setup_v1/plan.json','sglm/external_reference.py']
    (out/'freeze.json').write_text(json.dumps(dict(protocol=p,base_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        checkpoint=old['frozen_checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],input_sha256={s:sha(s) for s in files},
        official_source_snapshot_sha256=sha(out/'official_source_snapshot.json'),weights=weights,no_training=True,test_used=False),indent=2))
    print(json.dumps(weights,indent=2))


if __name__=='__main__':main()
