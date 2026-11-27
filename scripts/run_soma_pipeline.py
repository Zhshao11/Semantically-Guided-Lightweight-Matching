"""Detached long-run supervisor: train -> finaltest -> timing, no continuous agent polling."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--gpus',default='4,5,6,7');ap.add_argument('--resume');a=ap.parse_args()
    config=json.loads(Path('configs/soma_os_training_v1.json').read_text());run=Path(config['output']);run.mkdir(parents=True,exist_ok=True)
    from sglm.soma_training import atomic_json
    env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=a.gpus;env['OMP_NUM_THREADS']='4';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    uuid=config['timing']['gpu_uuid'];world=len(a.gpus.split(','))
    launcher=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}','scripts/train_soma_os.py','--root',a.root]
    if a.resume:launcher.extend(['--resume',a.resume])
    try:
        atomic_json(run/'pipeline.json',dict(pid=os.getpid(),phase='training',started_unix=time.time(),gpus=a.gpus,world_size=world,command=launcher))
        with (run/'training_console.log').open('a') as log:subprocess.run(launcher,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        env['CUDA_VISIBLE_DEVICES']=uuid
        atomic_json(run/'pipeline.json',dict(pid=os.getpid(),phase='final_test_and_timing',training_finished=True))
        with (run/'final_console.log').open('a') as log:subprocess.run([sys.executable,'scripts/finish_soma_os.py','--root',a.root],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        atomic_json(run/'pipeline.json',dict(pid=os.getpid(),phase='completed',completed_unix=time.time(),metrics_csv=str(Path(config['reports'])/'final_summary.csv')))
    except Exception as exc:
        atomic_json(run/'pipeline.json',dict(pid=os.getpid(),phase='failed',error=repr(exc),traceback=traceback.format_exc(),
            instruction='Inspect training_console.log/final_console.log. Resume last.pt if training failed. Finaltest existingfile is reused, not used to change model.'))
        raise


if __name__=='__main__':main()
