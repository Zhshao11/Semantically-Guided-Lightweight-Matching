"""Reuse prior full952 accuracy only with exact source/weight/data and2prediction regression."""
import argparse,hashlib,json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np,torch
from sglm.external_reference import OfficialReference,cpu_input
from sglm.t02.data import AffinePairs256


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--method',required=True);a=ap.parse_args()
    out=Path('reports/sar_references_v1');p=json.loads(Path('configs/external_references_v1.json').read_text());old=json.loads(Path('reports/external_references_v1/freeze.json').read_text());new=json.loads((out/'freeze.json').read_text())
    assert a.method in ['XoFTR','MINIMA_RoMa']
    for path,digest in old['input_sha256'].items():assert sha(path)==digest,path
    for path,record in old['weights'].items():assert sha(path)==record['sha256'],path
    for file in ['validation_plan.json','subsets.json','candidate_config.json']:
        path=p['frozen']+'/'+file;assert old['input_sha256'][path]==new['input_sha256'][path]
    snapshot=json.loads(Path('reports/external_references_v1/official_source_snapshot.json').read_text())
    for path,digest in snapshot['files_sha256'].items():assert sha(Path(p['source_root'])/path)==digest,path
    cfg=json.loads(Path(p['frozen']+'/candidate_config.json').read_text());torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    model=OfficialReference(a.method,p,cfg);ds=AffinePairs256(a.root,p['frozen']+'/validation_plan.json',split='val',preload=False);checks=[]
    for index in [0,1]:
        pred=model.predict(cpu_input(ds[index]));previous=np.load(Path(p['source_root'])/'predictions'/a.method/f'{index:04d}.npz')
        exact={key:np.array_equal(pred[key],previous[key]) for key in ['source','target','score']};assert all(exact.values()),(a.method,index,exact)
        checks.append(dict(pair_id=ds[index]['pair_id'],exact=exact))
    summary=Path('reports/external_references_v1')/f'{a.method}_accuracy.json';assert json.loads(summary.read_text())['image_pairs']==952
    (out/f'{a.method}_legacy_reuse.json').write_text(json.dumps(dict(passed=True,method=a.method,checks=checks,full952_accuracy_path=str(summary),full952_accuracy_sha256=sha(summary),
        original_freeze_sha256=sha('reports/external_references_v1/freeze.json'),same_sources_weights_data=True,
        note='Accuracy reuses original complete952 run, not a new952 run;2fixed predictions rechecked bitwise. Resources measured afresh this round.'),indent=2))
    print(a.method,'legacy reuse regression passed',flush=True)


if __name__=='__main__':main()
