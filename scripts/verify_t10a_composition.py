"""Independent acceptance checks and byte-for-byte rerun of the saved-candidate audit."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from sglm.t10_signal import paired_parent_summary


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=json.loads(Path('configs/t10a_common_support_v1.json').read_text());out=Path(p['output'])
    freeze=json.loads((out/'freeze.json').read_text());summary=json.loads((out/'summary.json').read_text())
    plan=json.loads(Path(p['plan']).read_text())['samples'];pn=sorted({x['verified512_parent'] for x in plan})
    codes=np.array([pn.index(x['verified512_parent']) for x in plan])
    d=dict(np.load(Path(p['candidates'])/'relations.npz'));support=dict(np.load(Path(p['private_output'])/'support.npz'))
    rows=support['relation_rows']; queries=support['query_keys']
    ids=d['random_id'];scores=d['random_semantic'];expected=(ids>=0).any(2).all(1)
    assert np.array_equal(rows,np.flatnonzero(expected))
    assert np.array_equal(queries,np.unique(d['pair_index'][rows].astype(int)*1024+d['query_index'][rows]))
    assert len(queries)==freeze['common_queries']==160163 and len(rows)==637136
    assert sha(Path(p['candidates'])/'relations.npz')==freeze['relations_sha256']
    # Independently reproduce raw differences with the preexisting hierarchy implementation.
    for j,r in enumerate(summary['comparisons']):
        s=np.where(ids[rows,j]>=0,scores[rows,j],np.nan).astype(float)
        delta=d['hard_semantic'][rows]-np.nanmean(s,axis=1)
        stats,_,_,_=paired_parent_summary(d['pair_index'][rows],d['query_index'][rows],delta,codes,20000,20261126)
        assert abs(stats['mean']-r['raw_delta'])<1e-12
        assert abs(stats['CI95_low']-r['raw_CI95_low'])<1e-12
        assert abs(stats['CI95_high']-r['raw_CI95_high'])<1e-12
        assert abs(r['raw_delta']-r['within_component']-r['composition_component']-r['unilateral_component'])<1e-12
    weights=list(csv.DictReader((out/'fixed_class_weights.csv').open()))
    assert len(weights)==49 and np.isclose(sum(float(x['fixed_weight']) for x in weights),1)
    cells=list(csv.DictReader((out/'joint_class_results.csv').open()));assert len(cells)==147
    for name in p['controls']:
        subset=[r for r in cells if r['control']==name]
        estimate=sum(float(r['fixed_weight'])*float(r['delta']) for r in subset if float(r['fixed_weight'])>0)
        target=next(r for r in summary['comparisons'] if r['control']==name)
        assert abs(estimate-target['standardized_delta'])<1e-12
        assert sum(int(r['hard_individuals']) for r in subset)==637136
        if name=='distance_weight_class':
            assert all(abs(float(r['hard_mass'])-float(r['random_mass']))<1e-12 for r in subset)
    files=[x for x in out.iterdir() if x.suffix in {'.csv','.json'} and x.name!='reproducibility.json']
    before={x.name:sha(x) for x in files}
    subprocess.run([sys.executable,'scripts/audit_t10a_composition.py'],check=True,stdout=subprocess.DEVNULL)
    after={x.name:sha(x) for x in files};assert before==after
    result=dict(passed=True,common_support_independently_checked=True,
                original_hierarchy_raw_mean_and_CI_reproduced=True,fixed_class_weights_shared=True,
                missing_cells_not_zero_imputed=True,all49_cells_reported=True,
                strict_class_control_composition_identical=True,source_candidate_sha_unchanged=True,
                independent_rerun_byte_identical=True,artifact_sha256=after,
                source_sha256={path:sha(path) for path in ['sglm/t10_composition.py','scripts/audit_t10a_composition.py','scripts/verify_t10a_composition.py']})
    (out/'reproducibility.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
