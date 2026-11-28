"""Freeze provenance and native smoke gates before any full validation."""
import hashlib,importlib.metadata,json,subprocess
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=json.loads(Path('configs/sar_references_v1.json').read_text());out=Path(p['output']);root=Path(p['source_root'])
    if (out/'freeze.json').exists():raise FileExistsError('Frozen run already exists')
    snapshot=json.loads((out/'official_source_snapshot.json').read_text())
    for file,digest in snapshot['files_sha256'].items():assert sha(root/file)==digest,file
    smoke={}
    for name in p['methods']:
        path=out/f'{name}_smoke.json';assert json.loads(path.read_text())['passed'];smoke[name]=sha(path)
    weights={}
    for name in ['FHReg_GUSO','GDROS_WHU']:
        spec=p[name];path=root/spec['weight'];assert sha(path)==spec['sha256']
        weights[name]=dict(path=str(path),bytes=path.stat().st_size,sha256=spec['sha256'])
    checkpoint='runs/t05_frozen_v1/checkpoint.pt';old=json.loads(Path(p['frozen']+'/freeze.json').read_text());assert sha(checkpoint)==old['checkpoint_sha256']
    paths=['configs/sar_references_v1.json',p['frozen']+'/validation_plan.json',p['frozen']+'/subsets.json',p['frozen']+'/candidate_config.json',
        'reports/t10_setup_v1/plan.json','sglm/sar_references.py','sglm/sar_reference_geometry.py','sglm/external_reference.py',
        'sglm/t05.py','sglm/t02/model.py','sglm/model.py','sglm/evaluation.py','sglm/geometry.py','scripts/evaluate_sar_references.py']
    versions={name:importlib.metadata.version(name) for name in ['torch','torchvision','timm','mamba-ssm','PyWavelets','pytorch-lightning','kornia','numpy','opencv-python']}
    record=dict(protocol=p,code_base_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),checkpoint=checkpoint,checkpoint_sha256=sha(checkpoint),
        input_sha256={s:sha(s) for s in paths},official_source_snapshot_sha256=sha(out/'official_source_snapshot.json'),weights=weights,native_smoke_sha256=smoke,
        environment=versions,dependency_environment='runs/sar_references_v1/.venv; system-site torch shared read-only; new deps installed only here',
        compatibility='FH uses PL2profilers namespace alias for native PL1import; native inference only PassThroughProfiler; no core model/forward changes; mamba built with CUDA12.8includingSM120',
        GDROS_cached_weight_origin='byte-identical copy of existing author-named official_models_reference cache; no remote checksum supplied by authors; training source follows official WHU filename and README, not independently audited original training samples',
        GDROS_OS_weight_excluded=True,no_training=True,test_used=False)
    (out/'freeze.json').write_text(json.dumps(record,indent=2));print(json.dumps(record['environment'],indent=2))


if __name__=='__main__':main()
