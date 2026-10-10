"""Resumable CPU/API supervisor. Never starts or controls GPU services."""
import os
from pathlib import Path
import subprocess
import sys
import time

from ..common import read_json, write_json
from .config import PHASE


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def launch(config, *, wait_pid, max_api_calls):
    root = config['paths'][PHASE + '_output']
    marker = root / 'continuation_pid.json'
    if marker.exists() and alive(read_json(marker)['pid']):
        raise RuntimeError('An owned retry supervisor is already running')
    cmd = [sys.executable, '-m', 'verak.v3.cli.v2_ops_retry', 'continue', '--config', 'v2',
           '--max-api-calls', str(max_api_calls), '--paid-approval', 'Proceed', '--detached']
    if wait_pid:
        cmd += ['--wait-pid', str(wait_pid)]
    with (root / 'continuation.log').open('a') as log:
        process = subprocess.Popen(cmd, cwd=Path(__file__).parents[3], stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=os.environ.copy())
    result = {'pid': process.pid, 'wait_pid': wait_pid, 'command': cmd, 'gpu_used': False, 'started': time.time()}
    write_json(marker, result)
    return result


def run(config, *, wait_pid, max_api_calls):
    root = config['paths'][PHASE + '_output']
    def status(stage, **kwargs):
        write_json(root / 'continuation_status.json', {'stage': stage, 'pid': os.getpid(),
                   'at': time.time(), 'gpu_used': False, 'training': False, **kwargs})
    if wait_pid:
        status('waiting_for_existing_qc', wait_pid=wait_pid)
        while alive(wait_pid):
            time.sleep(10)
    def step(name):
        status(name)
        cmd = [sys.executable, '-m', 'verak.v3.cli.v2_ops_retry', name, '--config', 'v2']
        if name in {'qc', 'teacher'}:
            cmd += ['--max-api-calls', str(max_api_calls), '--paid-approval', 'Proceed']
        with (root / ('continued_' + name + '.log')).open('a') as log:
            result = subprocess.run(cmd, cwd=Path(__file__).parents[3], stdout=log, stderr=subprocess.STDOUT,
                                    env=os.environ.copy())
        if result.returncode:
            status('stage_failed', failed_stage=name, returncode=result.returncode)
            raise RuntimeError('Retry continuation stage failed: ' + name)
    step('qc')
    qc_status = read_json(root / 'qc_status.json')
    error = qc_status.get('error')
    if error and error['type'] != 'CallBudgetExceeded':
        status('stage_failed', failed_stage='qc', error=error)
        return
    decision = qc_status['qc']['operators']['G_DEL_LINK']['decision']
    if decision == 'retain':
        step('teacher')
    else:
        from .teacher import prepare
        prepare(config)  # Freeze the empty teacher design and make the skip explicit.
    from .retry_report import report
    result = report(config, final=decision != 'retain')
    status('awaiting_cpu_scoring' if decision == 'retain' else 'completed_awaiting_final_audit',
           decision=decision, report=result['report'], budget=result['budget'])
    return result
