"""Durable host supervisor for bounded collection, deferred scoring and reporting."""
import os
from pathlib import Path
import subprocess
import sys
import time

from ..common import file_sha, read_json, write_json
from ..v2_ops.config import PHASE


def launch(config, *, max_api_calls):
    root = config['paths'][PHASE + '_output']
    status = root / 'continuation_status.json'
    if status.exists() and read_json(status).get('stage') not in {'complete', 'error'}:
        raise RuntimeError('Inspect the existing insertion supervisor before launching another')
    command = [sys.executable, '-m', 'verak.v3.cli.insertion_boost', 'continue', '--config', 'v2',
        '--max-api-calls', str(max_api_calls), '--paid-approval', 'Proceed', '--detached']
    with (root / 'continuation.log').open('ab') as stream:
        worker = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3],
            env=os.environ.copy(), stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
    result = {'pid': worker.pid, 'stage': 'launched', 'at': time.time(), 'command': command,
              'gpu_used': False, 'training': False}
    write_json(root / 'continuation_worker.json', result)
    return result


def run(config, *, max_api_calls):
    root = config['paths'][PHASE + '_output']
    def update(stage, **kwargs):
        write_json(root / 'continuation_status.json', {'pid': os.getpid(), 'stage': stage,
            'at': time.time(), 'gpu_used': False, 'training': False, **kwargs})
    try:
        from .teacher import run as teacher
        update('teacher')
        teacher_result = teacher(config, max_api_calls=max_api_calls, paid_approved=True)
        if teacher_result['errors'] and any(e['type'] != 'CallBudgetExceeded' for e in teacher_result['errors']):
            # Saved failed episodes are still preserved; unsent slots need a reviewable cause.
            if teacher_result['saved_attempts'] != teacher_result['planned_attempts']:
                raise RuntimeError('Teacher dispatch stopped early; inspect teacher_status.json')
        from .evaluate import run as score
        update('global_scoring', teacher=teacher_result)
        scored = score(config)
        if scored['errors']:
            raise RuntimeError('GLOBAL scoring has unresolved errors; inspect scoring_status.json')
        calibration = config['paths']['data_boost_shared'] / 'cpu_scorer/calibration.json'
        update('waiting_for_cpu_calibration')
        while not calibration.exists():
            time.sleep(10)
        from .report import report
        result = report(config, final=True)
        update('complete', report=result['report'], budget=result['budget'],
               selected_counts=result['selected_counts'], calibration_sha256=file_sha(calibration))
        return result
    except Exception as exc:
        update('error', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
