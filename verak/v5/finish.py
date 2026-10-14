"""Finish the authorized V5 pilot after the already running local queue.

No training entry point. A single process-exit event replaces progress polling.
"""
import argparse
import fcntl
import os
from pathlib import Path
import select
import subprocess
import time
import traceback

from .common import ROOT, WORKTREE, INFERENCE_PYTHON, atomic_new, read_json, write_json, append_report, file_sha
from .run import process_handle


def api_access(*,refresh=False):
    """Read-only model metadata; no generation or student text is sent."""
    from .api import credentials
    import httpx
    contract=read_json(ROOT/'api_contract.json')
    if not contract.get('user_approved'):raise PermissionError('API cap not authorized')
    path=ROOT/'api_model_access.json'
    if path.exists() and not refresh:
        value=read_json(path)
        if not value['all_available']:raise RuntimeError('Prior API model access check failed')
        return value
    keys=credentials();rows={}
    with httpx.Client(timeout=30) as client:
        for kind in ('luna','claude'):
            model=contract[kind]['model'];key=keys['OPENAI_API_KEY' if kind=='luna' else 'ANTHROPIC_API_KEY']
            if not key:raise RuntimeError('Missing '+kind+' credential')
            url=('https://api.openai.com/v1/models/' if kind=='luna' else 'https://api.anthropic.com/v1/models/')+model
            headers={'Authorization':'Bearer '+key} if kind=='luna' else {'x-api-key':key,'anthropic-version':'2023-06-01'}
            response=client.get(url,headers=headers)
            metadata=response.json() if response.status_code==200 else {}
            rows[kind]={'model':model,'http_status':response.status_code,'response_id':metadata.get('id'),
                        'available':response.status_code==200 and metadata.get('id')==model}
    result={'all_available':all(r['available'] for r in rows.values()),'models':rows,
            'paid_generation_requests':0,'student_text_sent':False,'at':time.time()}
    if path.exists():
        # An explicitly requested metadata refresh can follow a credential
        # replacement. Preserve the old failed preflight; never retry a paid call.
        archive=ROOT/'api_model_access_history'/(file_sha(path)+'.json')
        if not archive.exists():atomic_new(archive,read_json(path))
        write_json(path,result)
    else:atomic_new(path,result)
    if not result['all_available']:raise RuntimeError('Pinned API model not available; see sanitized access artifact')
    return result


def follow_pid():
    matches=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=[x.decode() for x in path.read_bytes().split(b'\0') if x]
        except (OSError,UnicodeError):continue
        if len(args)>=5 and args[-3:]==['-m','verak.v5.run','follow_local']:
            matches.append(int(path.parent.name))
    if len(matches)!=1:raise RuntimeError(f'Expected one owned local queue; found {len(matches)}')
    return matches[0]


def await_local_queue():
    if (ROOT/'local_queue_complete.json').exists():return
    pid=follow_pid()
    write_json(ROOT/'finish_waiting.json',{'local_queue_pid':pid,'waiting':'local_queue_complete','event_based':True,'at':time.time()})
    try:fd=process_handle(pid)
    except ProcessLookupError:fd=None
    if fd is not None:
        try:event=select.poll();event.register(fd,select.POLLIN);event.poll()
        finally:os.close(fd)
    if not (ROOT/'local_queue_complete.json').exists():
        raise RuntimeError('Local queue ended before completion; no inference or paid-call retry')


def invoke(name,module,*args,gpu=False):
    command=[str(INFERENCE_PYTHON),'-u','-m',module,*args]
    environment={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4'}
    if not gpu:environment['CUDA_VISIBLE_DEVICES']=''
    started=time.monotonic()
    with (ROOT/f'finish_{name}.log').open('a') as log:
        process=subprocess.Popen(command,cwd=WORKTREE,env=environment,stdout=log,stderr=subprocess.STDOUT)
        write_json(ROOT/'finish_stage.json',{'stage':name,'pid':process.pid,'command':command,'at':time.time()})
        code=process.wait()
    write_json(ROOT/f'finish_{name}_exit.json',{'returncode':code,'wall_seconds':time.monotonic()-started,'at':time.time()})
    if code:raise RuntimeError(name+' failed; saved outputs are preserved without automatic retry')


def run():
    if (ROOT/'finalized.json').exists():return
    with (ROOT/'finish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic()
        write_json(ROOT/'finish_controller.json',{'pid':os.getpid(),'started_at':time.time(),
            'sequence':['wait_local','Luna_API','Luna_FEAK_repair_and_GPU_scores','Claude_judge','report'],
            'total_cap_usd':15,'training':False})
        try:
            api_access();await_local_queue()
            # Check hashes of the frozen B generation/measurement protocol.
            from .freeze import run as check_contract
            check_contract()
            invoke('luna','verak.v5.pilot','c_api')
            invoke('repair','verak.v5.run','c_repair',gpu=True)
            invoke('judge','verak.v5.judge')
            invoke('report','verak.v5.report')
            atomic_new(ROOT/'finalized.json',{'wall_seconds':time.monotonic()-started,'report':'reports/V5_PIPE1.md',
                'cost':read_json(ROOT/'api/accounting.json'),'training':False,'test_cohort_used':False,'at':time.time()})
        except Exception as exc:
            write_json(ROOT/'finish_failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'at':time.time()})
            money=read_json(ROOT/'api/accounting.json') if (ROOT/'api/accounting.json').exists() else None
            append_report('C — execution paused before completion',
                'The inference-only controller stopped: '+str(exc)+'. Saved attempts and request reservations are preserved; '
                'no automatic model retry or budget increase. Diagnostic: `verak/v5/outputs/pipe1/finish_failure.json`. '
                'Recorded API accounting: '+str(money),seconds=time.monotonic()-started)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part',choices=['access','run'])
    parser.add_argument('--refresh',action='store_true',help='Repeat read-only access metadata after a credential update; preserve prior check')
    args=parser.parse_args()
    if args.part=='access':print(api_access(refresh=args.refresh))
    else:
        if args.refresh:parser.error('--refresh is only allowed for access metadata')
        run()
