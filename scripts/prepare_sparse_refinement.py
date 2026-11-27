"""Freeze sparse-refinement protocol and audit historical candidate counts only."""
import argparse,hashlib,json,subprocess,shutil
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--amend-bf16-tolerance',action='store_true');a=ap.parse_args()
    out=Path('reports/sparse_refinement_v1');out.mkdir(parents=True,exist_ok=True)
    prior=None
    if (out/'freeze.json').exists():
        if not a.amend_bf16_tolerance:raise FileExistsError('Frozen experiment exists')
        assert (out/'bf16_operator_diagnosis.json').exists() and not (out/'regression.json').exists()
        assert not (out/'freeze_initial.json').exists()
        prior=sha(out/'freeze.json')
        for src,dst in [('freeze.json','freeze_initial.json'),('frozen_config.yaml','frozen_config_initial.yaml'),('regression_tests.txt','initial_regression_failure.txt')]:
            shutil.copy2(out/src,out/dst)
    p=json.loads(Path('configs/sparse_refinement_v1.json').read_text());frozen=Path(p['frozen'])
    old=json.loads((frozen/'freeze.json').read_text())
    assert sha(old['frozen_checkpoint'])==old['checkpoint_sha256']==p['checkpoint_sha256']
    assert sha(frozen/'validation_plan.json')==old['validation_plan_sha256']
    assert sha(frozen/'subsets.json')==old['subsets_sha256']
    for path,r in old['original_code_verification'].items():assert sha(path)==r['sha256']
    original=json.loads(Path('reports/t05_results_v1/K1024_validation.json').read_text())
    n=np.array([r['attempted_refinements'] for r in original['pairs']]);assert len(n)==952
    assert np.mean(n>=64)>.5,'Need freeze revised budgets BEFORE experiments'
    nstats=dict(pairs=len(n),min=int(n.min()),P10=float(np.percentile(n,10)),median=float(np.median(n)),P90=float(np.percentile(n,90)),max=int(n.max()),
                fraction_below64=float(np.mean(n<64)),fraction_below128=float(np.mean(n<128)),fraction_below256=float(np.mean(n<256)),budgets_retained=True)
    train=json.loads(Path('reports/t10_setup_v1/plan.json').read_text())
    smoke=dict(config=train['config'],samples=train['samples'][:32]);assert len(smoke['samples'])==32 and all(r['split']=='train' for r in smoke['samples'])
    (out/'smoke_plan.json').write_text(json.dumps(smoke,indent=2))
    cfg=json.loads((frozen/'candidate_config.json').read_text())
    (out/'frozen_config.yaml').write_text(json.dumps(dict(protocol=p,original_matcher_config=cfg),indent=2)+'\n')
    files=['configs/sparse_refinement_v1.json',str(frozen/'validation_plan.json'),str(frozen/'subsets.json'),str(frozen/'candidate_config.json'),
        'reports/t05_results_v1/K1024_validation.json',str(out/'smoke_plan.json'),str(out/'frozen_config.yaml'),
        'sglm/model.py','sglm/t02/model.py','sglm/t05.py','sglm/evaluation.py','sglm/geometry.py','sglm/sparse_refinement.py']
    record=dict(checkpoint=old['frozen_checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],
        base_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),inputs_sha256={s:sha(s) for s in files},
        original_candidate_count_distribution=nstats,no_training=True,test_used=False,prior_freeze_sha256=prior)
    (out/'freeze.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


if __name__=='__main__':main()
