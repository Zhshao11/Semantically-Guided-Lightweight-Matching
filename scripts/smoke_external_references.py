"""Two fixed train-source pairs: official checkpoint loading and prediction repeatability."""
import argparse,json,os,sys,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from sglm.external_reference import OfficialReference,cpu_input
from sglm.t02.data import AffinePairs256
from sglm.evaluation import estimate_from_correspondences


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--method',required=True);a=ap.parse_args()
    p=json.loads(Path('configs/external_references_v1.json').read_text());out=Path(p['output']);cfg=json.loads(Path(p['frozen']+'/candidate_config.json').read_text())
    torch.set_num_threads(p['timing']['threads']);torch.manual_seed(cfg['seed']);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    plan=json.loads(Path('reports/t10_setup_v1/plan.json').read_text());ds=AffinePairs256(a.root,dict(config=plan['config'],samples=plan['samples'][:2]),split='train',preload=False)
    start=time.perf_counter();adapter=OfficialReference(a.method,p,cfg);load_seconds=time.perf_counter()-start;rows=[]
    with torch.inference_mode():
        for item in ds:
            inputs=cpu_input(item);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            pred=adapter.predict(inputs);fit=estimate_from_correspondences(pred['source'],pred['target'],pred['score'],cfg['evaluation'],seed=cfg['seed']);torch.cuda.synchronize()
            elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/2**20
            again=adapter.predict(inputs)
            exact={k:bool(np.array_equal(pred[k],again[k])) for k in ['source','target','score']}
            assert all(exact.values()),exact
            rows.append(dict(pair_id=item['pair_id'],raw_matches=pred['raw_matches'],valid_matches=pred['valid_matches'],removed_invalid=pred['removed_invalid'],
                solver_status=fit['status'],inliers=fit['inliers'],unwarmed_seconds=elapsed,peak_allocated_MiB=peak,repeat_exact=exact))
    record=dict(passed=True,method=a.method,train_source_pairs=len(rows),rows=rows,load_seconds=load_seconds,
        parameters_registered=adapter.model_parameters,dinov2_unregistered_parameters=adapter.hidden_dino_parameters,
        total_parameters_including_hidden=adapter.model_parameters+adapter.hidden_dino_parameters,
        torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,train=False,formal_timing=False)
    (out/f'{a.method}_smoke.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
