"""Temporarily release only this project's SOMA job, time, and always resume it."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path


def descendants(root):
    table={}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            stat=(entry/'stat').read_text().split(') ',1)[1].split();table[int(entry.name)]=int(stat[1])
        except (OSError,IndexError):pass
    found={root}
    while True:
        new={pid for pid,parent in table.items() if parent in found}
        if new<=found:return found
        found|=new


def main():
    run=Path('runs/local_interaction_v1');out=Path('reports/local_interaction_v1');soma=Path('runs/soma_os_v1/training')
    assert json.loads((out/'evaluation_acceptance.json').read_text())['passed']
    pipeline=json.loads((soma/'pipeline.json').read_text());pid=pipeline['pid'];assert pipeline['phase']=='training'
    rootcmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'scripts/run_soma_pipeline.py' in rootcmd
    own=descendants(pid);inventory={}
    for child in own:
        path=Path(f'/proc/{child}');assert path.stat().st_uid==os.getuid()
        cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert 'soma' in cmd or 'torch.distributed.run' in cmd,(child,cmd)
        inventory[str(child)]=cmd
    record=dict(original_supervisor=pid,processes=inventory,pause_unix=time.time(),prior_pipeline=pipeline,
        status_before=json.loads((soma/'status.json').read_text()),checkpoints_preserved=True)
    record_path=out/'soma_timing_coordination.json';record_path.write_text(json.dumps(record,indent=2))
    checkpoint=soma/'last.pt';assert checkpoint.is_file()
    error=None
    try:
        # Stop supervisor first so it cannot dispatch final evaluation while workers exit.
        os.kill(pid,signal.SIGTERM)
        for child in own-{pid}:
            try:os.kill(child,signal.SIGTERM)
            except ProcessLookupError:pass
        deadline=time.time()+45
        while time.time()<deadline:
            active=[]
            for child in own:
                try:
                    stat=Path(f'/proc/{child}/stat').read_text().split(') ',1)[1].split()[0]
                    if stat!='Z':active.append(child)
                except FileNotFoundError:pass
            if not active:break
            time.sleep(.25)
        else:raise RuntimeError(f'Own SOMA workers did not exit: {active}; do not kill unrelated work')
        # Checkpoint is stable now. It must load before formal timing begins.
        import torch
        checkpoint=soma/'last.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert 'optimizer' in saved and saved['updates']>0
        last_logged=json.loads((soma/'steps.jsonl').read_text().splitlines()[-1])['update']
        record.update(last_logged_update_at_pause=last_logged,replayed_updates=max(0,last_logged-saved['updates']),resume_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),resume_updates=saved['updates'],
            resume_epoch=saved['epoch'],resume_next_update=saved.get('next_update'),checkpoint_keys=list(saved));del saved
        record_path.write_text(json.dumps(record,indent=2))
        env=dict(os.environ);env['PYTHONPATH']='.'
        with (run/'formal_timing.log').open('w') as log:
            for extra in [[],['--memory-group','original_refiner'],['--memory-group','local_cross_attention']]:
                subprocess.run([sys.executable,'scripts/benchmark_local_refinement.py','--root','/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0',*extra],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        record['timing_completed']=True
    except BaseException as exc:
        error=exc;record['timing_error']=repr(exc)
    finally:
        # Same original 4-GPU protocol, resuming the stable model+optimizer+scheduler checkpoint.
        env=dict(os.environ);env['PYTHONPATH']='.'
        with (soma/'supervisor_console.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'scripts/run_soma_pipeline.py','--root','/mnt/sda/zhushihao/data/vis_sar/OSDataset2.0',
                '--gpus','4,5,6,7','--resume',str(checkpoint)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record.update(resumed_supervisor_pid=proc.pid,resume_unix=time.time(),resume_command_same_4GPU=True)
        record_path.write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)
    if error:raise error


if __name__=='__main__':main()
