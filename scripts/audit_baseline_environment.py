"""Read-only CPU/import checks of existing external repositories; no installation."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def command(args,cwd=None,timeout=30):
    try:
        result=subprocess.run(args,cwd=cwd,capture_output=True,text=True,timeout=timeout,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
        return dict(returncode=result.returncode,stdout=result.stdout[-6000:],stderr=result.stderr[-6000:])
    except (OSError,subprocess.TimeoutExpired) as error:return dict(error=str(error))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',action='append',default=[])
    ap.add_argument('--output',required=True);a=ap.parse_args()
    versions={}
    for package in ['torch','torchvision','numpy','opencv-python','kornia','timm','mamba-ssm','causal-conv1d','triton','yacs','einops','pytorch-lightning']:
        try:versions[package]=metadata.version(package)
        except metadata.PackageNotFoundError:versions[package]=None
    repos=[]
    probes={
        'XFeat':"import json,hashlib,torch; from pathlib import Path; from modules.xfeat import XFeat; torch.set_num_threads(2); m=XFeat().eval(); r=m.detectAndCompute(torch.zeros(1,3,256,256),top_k=128)[0]; print(json.dumps(dict(cpu_smoke=True,keypoints_shape=list(r['keypoints'].shape),descriptors_shape=list(r['descriptors'].shape),weights_sha256=hashlib.sha256(Path('weights/xfeat.pt').read_bytes()).hexdigest())))",
        'JamMa':"from src.jamma.jamma import JamMa; print('JamMa import passed')",
        'UltraMatch':"from src.ultramatch.ultramatch import UltraMatch; print('UltraMatch import passed (no checkpoint/forward)')",
    }
    for item in a.repo:
        name,path=item.split('=',1);root=Path(path)
        entry=dict(name=name,path=path,exists=root.is_dir())
        if root.is_dir():
            if (root/'source_manifest.json').exists():
                entry['pinned_snapshot']=json.loads((root/'source_manifest.json').read_text())
                checks=[hashlib.sha256((root/f['path']).read_bytes()).hexdigest()==f['sha256'] for f in entry['pinned_snapshot']['files']]
                entry['snapshot_hashes_verified']=all(checks)
                if not all(checks):raise ValueError('External snapshot content changed')
            else:
                top=command(['git','rev-parse','--show-toplevel'],root)
                if top.get('returncode')==0 and Path(top['stdout'].strip()).resolve()==root.resolve():
                    entry.update(origin=command(['git','remote','get-url','origin'],root),
                        revision=command(['git','rev-parse','HEAD'],root),
                        working_tree=command(['git','status','--porcelain','-uno'],root))
                else:entry['git_checkout']=False
            entry['probe']=command([sys.executable,'-c',probes[name]],root) if name in probes else dict(status='not_executed')
        repos.append(entry)
    result=dict(python=sys.version,platform=platform.platform(),executable=sys.executable,versions=versions,repositories=repos,
        hardware_inventory=command(['nvidia-smi','--query-gpu=index,name,driver_version,compute_cap','--format=csv,noheader']),
        CUDA_used=False,installations_performed=False,formal_benchmark=False,
        limitation='CPU import/smoke is not a trained accuracy or CUDA kernel compatibility result')
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
