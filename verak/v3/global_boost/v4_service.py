"""Durable bounded GLOBAL data continuation, separate from the running RFT."""
import os
from pathlib import Path
import subprocess
import sys
import time

from ..common import read_json, write_json
from ..train.teacher_bulk import atomic_new, collection_lock
from .config import PHASE
from .expansion import batch_config, root_for
from .paid import BoostAPI


def run(config):
    from .qc import run as qc
    from .teacher import run as teacher
    from .v4_data import prepare
    from .v4_teacher import run_rescue
    root = root_for(config)
    with collection_lock(root/'v4/collection'):
        plan = prepare(config)
        status_path = root/'v4/status.json'
        stop = 'v4_diversified_practice_quota'
        for batch_path in plan['batch_roots']:
            batch = Path(batch_path)
            if (batch/'generation_finished.json').exists():
                continue
            cfg = batch_config(config,batch)
            # Retain funds for two Luna attempts on every yet-uncollected new
            # candidate, conservatively assuming every pending QC will pass.
            remaining = sum(read_json(Path(p)/'source_plan.json')['counts'][op]
                for p in plan['batch_roots'] if not (Path(p)/'generation_finished.json').exists()
                for op in ('G_PARA_SWAP','G_SENT_MOVE'))
            cfg[PHASE]['qc_hold_usd'] = remaining*2*.012
            write_json(status_path,{'stage':'qc','batch':str(batch),'at':time.time(),'gpu_used':False})
            write_json(root/'expansion_status.json',{'stage':'v4_collecting','batch':str(batch),'at':time.time()})
            if not (batch/'teacher_design.json').exists():
                qc(cfg,max_api_calls=100000)
            while True:
                write_json(status_path,{'stage':'teacher','batch':str(batch),'at':time.time(),'gpu_used':False})
                result = teacher(cfg,max_api_calls=100000)
                if result['stop_reason'] != 'bareun_priority_suspended':
                    break
                time.sleep(30)
            atomic_new(batch/'generation_finished.json',{'teacher_status':result,'at':time.time(),'gpu_used':False})
            if result['stop_reason']=='budget_cap':
                stop='budget_cap'
                break
        if stop!='budget_cap':
            write_json(status_path,{'stage':'Sol_GLOBAL_rescue','at':time.time(),'gpu_used':False})
            result=run_rescue(config)
            if result['stop_reason']=='budget_cap':stop='budget_cap'
        api=BoostAPI(config,0)
        account=api.accounting();api.close()
        if account['pending']:
            raise RuntimeError('Do not close collection with live paid requests')
        value={'stage':'generation_finished','stop_reason':stop,'task_version':'v4_prep',
            'budget':account,'at':time.time(),'gpu_used':False,'training':False,
            'legacy_unfinished_slots':'Archived after the authorized diversity rescope; never counted as completions.'}
        write_json(status_path,value)
        write_json(root/'expansion_status.json',value)
        return value


def launch(config):
    root=root_for(config)
    path=root/'v4/worker.json'
    if path.exists():
        value=read_json(path)
        proc=Path('/proc')/str(value['pid'])/'cmdline'
        if proc.exists() and b'v4-run' in proc.read_bytes():return value
    with (root/'v4/worker.log').open('a') as stream:
        process=subprocess.Popen([sys.executable,'-m','verak.v3.cli.global_boost','v4-run'],
            cwd=str(Path(__file__).resolve().parents[3]),stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    value={'pid':process.pid,'at':time.time(),'gpu_used':False,'training':False}
    write_json(path,value)
    return value
