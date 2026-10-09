"""Durable data-only continuation. No GPU process or training entry point."""
import os
import subprocess
import sys
import time
from pathlib import Path

from ..common import read_json, write_json
from .config import PHASE


def launch(config):
    root = config['paths'][PHASE + '_output']
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'controller_worker.json'
    if path.exists():
        old = read_json(path)
        cmdline = Path(f"/proc/{old['pid']}/cmdline")
        if cmdline.exists() and b'verak.v3.cli.global_boost' in cmdline.read_bytes():
            return old
    with (root / 'controller.log').open('a') as stream:
        process = subprocess.Popen([sys.executable, '-m', 'verak.v3.cli.global_boost', 'continue'],
            cwd=str(Path(__file__).resolve().parents[3]), stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    row = {'pid': process.pid, 'at': time.time(), 'gpu_used': False,
           'stage': 'data_only_continuation', 'worktree': str(Path(__file__).resolve().parents[3])}
    write_json(path, row)
    return row


def run(config):
    from .qc import run as qc
    from .teacher import run as teacher
    from .measure import run as measure
    from .report import report
    root = config['paths'][PHASE + '_output']
    status = {'pid': os.getpid(), 'gpu_used': False, 'training': False}
    try:
        for name, function in [('qc', qc), ('teacher', teacher), ('measure', measure), ('report', report)]:
            write_json(root / 'status.json', {**status, 'stage': name, 'at': time.time()})
            function(config)
        write_json(root / 'status.json', {**status, 'stage': 'complete', 'at': time.time()})
    except Exception as exc:
        write_json(root / 'status.json', {**status, 'stage': 'failed', 'failed_stage': name,
            'error': {'type': type(exc).__name__, 'message': str(exc)}, 'at': time.time()})
        raise
    return read_json(root / 'status.json')
