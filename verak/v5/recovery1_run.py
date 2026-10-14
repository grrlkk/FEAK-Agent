"""Resumable, detached collection -> short training -> fixed diagnostic."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback

from .recovery1 import ROOT, WORKTREE, prepare, gpu, build_data
from .common import read_json, write_json, INFERENCE_PYTHON
from verak.v4.dryrun_common import TRAIN_PYTHON
from verak.v4.full_run import idle


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare()
        try:
            for stage in ('collection','data','training','evaluation'):
                write_json(ROOT/'status.json',{'stage':stage,'at':time.time(),'pid':os.getpid()})
                print('stage',stage,time.time(),flush=True)
                if stage == 'collection': gpu('collect')
                elif stage == 'data': build_data()
                elif stage == 'training':
                    if (ROOT/'adapters/korean/complete.json').exists(): continue
                    hardware = idle((0,1))
                    command = [str(TRAIN_PYTHON),'-u','-m','torch.distributed.run','--standalone',
                        '--nnodes=1','--nproc-per-node=2','-m','verak.v5.recovery1_train']
                    if (ROOT/'adapters/korean/recipe.json').exists(): command.append('--resume')
                    environment = {**os.environ,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':'0,1',
                        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
                        'OMP_NUM_THREADS':'8','WANDB_DISABLED':'true'}
                    write_json(ROOT/'training_launch.json',{'command':command,'hardware':hardware,'at':time.time()})
                    with (ROOT/'train_korean.log').open('a') as log:
                        result = subprocess.run(command,cwd=WORKTREE,env=environment,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode: raise RuntimeError('Training failed; logs/checkpoints retained')
                else: gpu('infer')
            write_json(ROOT/'status.json',{'stage':'local_complete','at':time.time(),'pid':os.getpid(),
                'quality_judgment':'pending separately authorized evaluation'})
        except Exception as exc:
            write_json(ROOT/'failure.json',{'at':time.time(),'error':str(exc),'traceback':traceback.format_exc()})
            raise


def start():
    prepare()
    with (ROOT/'controller.log').open('a') as log:
        child = subprocess.Popen([sys.executable,'-u','-m','verak.v5.recovery1_run','run'],
            cwd=WORKTREE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(ROOT/'controller.json',{'pid':child.pid,'at':time.time(),'worktree':str(WORKTREE)})
    print('started',child.pid,flush=True)


def controller_signal(name):
    """Pause only this experiment's scheduler; collection children can finish."""
    if name=='resume':
        from .recovery1 import verify_data
        verify_data()
        for filename in ('rollout_controller.json','full_rollout_controller.json'):
            if not (ROOT/filename).exists(): continue
            child=read_json(ROOT/filename)['pid']
            stat=__import__('pathlib').Path('/proc')/str(child)/'stat'
            if stat.exists() and stat.read_text().split(') ',1)[1].split()[0]!='Z':
                raise RuntimeError('Collection process has not exited: '+str(child))
    pid=read_json(ROOT/'controller.json')['pid']
    command=(__import__('pathlib').Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
    if b'verak.v5.recovery1_run' not in command or b'run' not in command:
        raise RuntimeError('Recorded PID is no longer this experiment controller')
    os.kill(pid, signal.SIGSTOP if name=='pause' else signal.SIGCONT)
    write_json(ROOT/f'controller_{name}.json',{'pid':pid,'at':time.time(),
        'reason':'Review actual rollout recovery coverage before training' if name=='pause' else 'Reviewed data frozen'})
    print(name,pid,flush=True)


def start_rollouts():
    from .recovery1_rollout import prepare_rollouts
    prepare_rollouts()
    with (ROOT/'rollout_collection.log').open('a') as log:
        child=subprocess.Popen([str(INFERENCE_PYTHON),'-u','-m','verak.v5.recovery1_rollout','collect'],
            cwd=WORKTREE,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMP_NUM_THREADS':'8'},
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(ROOT/'rollout_controller.json',{'pid':child.pid,'at':time.time()})
    print('multistep collection started',child.pid,flush=True)


def start_full_rollouts():
    from .recovery1_rollout import full_design
    full_design()
    with (ROOT/'full_rollout_collection.log').open('a') as log:
        child=subprocess.Popen([str(INFERENCE_PYTHON),'-u','-m','verak.v5.recovery1_rollout','full_collect'],
            cwd=WORKTREE,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMP_NUM_THREADS':'8'},
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(ROOT/'full_rollout_controller.json',{'pid':child.pid,'at':time.time()})
    print('full training rollout collection started',child.pid,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part',choices=['start','run','pause','resume','start_rollouts','start_full_rollouts','validate_reviews','build_rollout_data'])
    part=parser.parse_args().part
    if part in ('pause','resume'): controller_signal(part)
    elif part in ('validate_reviews','build_rollout_data'):
        from . import recovery1_rollout
        getattr(recovery1_rollout,'build_data' if part=='build_rollout_data' else part)()
    else: globals()[part]()
