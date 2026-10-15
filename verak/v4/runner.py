"""Detached, low-priority CPU/API collectors; never start a GPU process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import ROOT, constrain_cpu, write_json


def launch(component, limit=None):
    from .common import collection_lock
    if component not in {'B','D'}:
        raise ValueError('Only authorized B/D collectors')
    root=ROOT/component
    with collection_lock(root/'launcher'):
        pidfile=root/'worker.json'
        if pidfile.exists():
            prior=json.loads(pidfile.read_text())
            try:
                cmdline=Path(f'/proc/{prior["pid"]}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            except FileNotFoundError:
                cmdline=''
            if 'verak.v4.runner work '+component in cmdline:
                return {'already_running':True,**prior}
        cmd=[sys.executable,'-m','verak.v4.runner','work',component]
        if limit is not None:
            cmd+=['--limit',str(limit)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        root.mkdir(parents=True,exist_ok=True)
        with (root/'worker.log').open('ab',buffering=0) as log:
            proc=subprocess.Popen(cmd,cwd=Path(__file__).resolve().parents[2],env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        value={'pid':proc.pid,'component':component,'limit':limit,'started_at':time.time(),
               'gpu_used':False,'command':cmd}
        write_json(pidfile,value)
        return value


def work(component,limit):
    constrain_cpu()
    dependency=ROOT/'design.json' if component=='B' else ROOT/'B/complete.json'
    while not dependency.exists():
        write_json(ROOT/component/'waiting.json',{'dependency':str(dependency),'at':time.time()})
        time.sleep(10)
    if component=='B':
        from .feedback import run
    else:
        from .content import run
    run(limit=limit)
    write_json(ROOT/component/'worker_done.json',{'at':time.time(),'limit':limit})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('launch','work'))
    parser.add_argument('component',choices=('B','D'))
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.command=='launch':
        print(json.dumps(launch(args.component,args.limit)))
    else:
        work(args.component,args.limit)
