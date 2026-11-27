"""Finalize the reviewed protocol and input/code identities before any formal updates."""
import hashlib,json,subprocess,time
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cfg=json.loads(Path('configs/soma_os_training_v1.json').read_text());out=Path(cfg['reports']);run=Path(cfg['output'])
    if (run/'steps.jsonl').exists() or (out/'launch_freeze.json').exists():raise FileExistsError('Formal launch already frozen')
    assert json.loads((out/'preflight_micro1.json').read_text())['passed']
    assert json.loads((out/'throughput_estimate.json').read_text())['passed']
    tests=(out/'tests.txt').read_text();assert '127 passed' in tests and 'failed' not in tests
    initial=json.loads((out/'freeze.json').read_text());(out/'freeze_initial_preflight.json').write_text(json.dumps(initial,indent=2))
    initial['config']=cfg;initial['preflight_precision_decision']='FP16DDPstep13overflow archived; BF16outer preflightpassed. OriginalResNet internally uses itsFP16 autocast; noGradScaler inBF16run.'
    (out/'freeze.json').write_text(json.dumps(initial,indent=2))
    files=['configs/soma_os_training_v1.json',cfg['data_plan'],str(out/'dataset_config.json'),cfg['dinov2_weights'],
        'sglm/soma_os.py','sglm/soma_training.py',*map(str,sorted(Path('scripts').glob('*soma*.py'))),
        'tests/test_soma_os.py','reports/t05_frozen_v1/subsets.json']
    original=json.loads(Path('reports/soma_audit_v1/source_audit.json').read_text())
    for p,digest in original['source_sha256'].items():assert sha(Path(cfg['upstream_source'])/p)==digest,p
    record=dict(protocol=cfg,code_base_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),frozen_unix=time.time(),
        input_source_sha256={p:sha(p) for p in files},official_SOMA_source_unchanged=True,preflight_passed=True,
        data_test_frozen_before_training=True,no_test_selection=True,final_metrics_pending_training=True)
    (out/'launch_freeze.json').write_text(json.dumps(record,indent=2));print(record['input_source_sha256'][cfg['dinov2_weights']])


if __name__=='__main__':main()
