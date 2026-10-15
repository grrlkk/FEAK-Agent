"""Detached supervisor for the approved bounded data task; no training stages."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..common import read_json, write_json
from .config import PHASE


def pid_matches(pid, text):
    try:
        value = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        return text in value
    except (OSError, UnicodeError):
        return False


def launch(config, *, qc_pid, max_api_calls, paid_approved):
    if not paid_approved:
        raise PermissionError('The v2 paid task requires explicit Proceed')
    root = config['paths'][PHASE + '_output']
    marker = root / 'continuation_pid.json'
    if marker.exists() and pid_matches(read_json(marker)['pid'], 'verak.v3.cli.v2_ops continue'):
        return read_json(marker)
    command = [sys.executable, '-m', 'verak.v3.cli.v2_ops', 'continue', '--config', 'v2',
               '--qc-pid', str(qc_pid), '--max-api-calls', str(max_api_calls),
               '--paid-approval', 'Proceed', '--detached']
    with (root / 'continuation.log').open('a') as log:
        child = subprocess.Popen(command, cwd=Path(__file__).parents[3], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    value = {'pid': child.pid, 'waiting_for_qc_pid': qc_pid, 'at': time.time(), 'training': False}
    write_json(marker, value)
    return value


def run(config, *, qc_pid, max_api_calls, paid_approved):
    if not paid_approved:
        raise PermissionError('The v2 paid task requires explicit Proceed')
    root = config['paths'][PHASE + '_output']
    with (root / '.continuation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(stage, **extra):
            value = {'stage': stage, 'at': time.time(), 'training': False, **extra}
            write_json(root / 'continuation_status.json', value)
            print(json.dumps(value), flush=True)
        status('waiting_for_existing_qc', pid=qc_pid)
        while pid_matches(qc_pid, 'verak.v3.cli.v2_ops qc'):
            time.sleep(10)
        for stage in ('qc', 'teacher', 'score', 'report'):
            status('running_' + stage)
            command = [sys.executable, '-m', 'verak.v3.cli.v2_ops', stage, '--config', 'v2']
            if stage in ('qc', 'teacher'):
                command += ['--max-api-calls', str(max_api_calls), '--paid-approval', 'Proceed']
            if stage == 'report':
                command += ['--final']
            with (root / ('continued_' + stage + '.log')).open('a') as log:
                child = subprocess.Popen(command, cwd=Path(__file__).parents[3], stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                status('running_' + stage, child_pid=child.pid)
                code = child.wait()
            if code:
                status('stage_failed', failed_stage=stage, exit_code=code)
                return
            if stage == 'qc':
                error = read_json(root / 'qc_status.json').get('error')
                if error and error['type'] != 'CallBudgetExceeded':
                    status('stage_failed', failed_stage=stage, error=error)
                    return
        status('completed_awaiting_final_audit')
