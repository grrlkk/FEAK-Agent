"""Durable data-only continuation. No GPU process or training entry point."""
import os
import signal
import sqlite3
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


def launch_prefetch(config, *, restart=False):
    root = config['paths'][PHASE + '_output']
    path = root / 'prefetch_worker.json'
    if path.exists():
        previous = read_json(path)
        command = Path(f"/proc/{previous['pid']}/cmdline")
        if command.exists() and b'verak.v3.cli.global_boost' in command.read_bytes() and b'prefetch-loop' in command.read_bytes():
            if not restart:
                return previous
            os.kill(previous['pid'], signal.SIGTERM)
            # This watcher queues immutable local files only: it owns no model,
            # request reservation, API connection or GPU process.
            for _ in range(50):
                if not command.exists() or not command.read_bytes():
                    break
                time.sleep(.1)
            else:
                raise RuntimeError('Owned score watcher did not terminate')
    with (root / 'prefetch.log').open('a') as stream:
        process = subprocess.Popen([sys.executable, '-m', 'verak.v3.cli.global_boost', 'prefetch-loop'],
            cwd=str(Path(__file__).resolve().parents[3]), stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    value = {'pid': process.pid, 'at': time.time(), 'gpu_used': False, 'paid_calls': 0}
    write_json(path, value)
    return value


def restart(config):
    """Stop only this data controller while its ledger proves no live API send.

    Holding the SQLite writer lock prevents the next request reservation. Saved
    episode files are immutable; an interrupted unsaved episode replays its
    completed response cache, including any durable record not yet cached.
    """
    root = config['paths'][PHASE + '_output']
    previous = read_json(root / 'controller_worker.json')
    pid = previous['pid']
    proc = Path(f'/proc/{pid}')
    if not proc.exists():
        return launch(config)
    command = (proc / 'cmdline').read_bytes()
    if b'verak.v3.cli.global_boost' not in command or b'continue' not in command:
        raise RuntimeError('Refuse to stop a process that does not own this data pipeline')
    deadline = time.monotonic() + 700
    while True:
        db = sqlite3.connect(root / 'api/ledger.sqlite', timeout=10)
        try:
            db.execute('BEGIN IMMEDIATE')
            pending = db.execute("SELECT COUNT(*) FROM calls WHERE status='pending'").fetchone()[0]
            if not pending:
                os.kill(pid, signal.SIGTERM)
                for _ in range(50):
                    if not proc.exists() or (proc / 'stat').read_text().split()[2] == 'Z':
                        break
                    time.sleep(.1)
                else:
                    raise RuntimeError('Owned controller did not terminate; do not launch another')
                write_json(root / f'controller_restart_{int(time.time())}.json', {
                    'previous_pid': pid, 'no_pending_api_under_ledger_lock': True, 'at': time.time(),
                    'saved_attempts_preserved': True, 'gpu_processes_touched': False})
                break
        finally:
            db.rollback()
            db.close()
        if time.monotonic() > deadline:
            raise TimeoutError('No safe between-call restart window; leave existing controller running')
        time.sleep(.2)
    return launch(config)


def run(config):
    from .qc import run as qc
    from .teacher import run as teacher
    from .measure import run as measure
    root = config['paths'][PHASE + '_output']
    status = {'pid': os.getpid(), 'gpu_used': False, 'training': False}
    try:
        # CPU results remain internal. Only the aggregate finalizer may export
        # selection/report artifacts after the root-owned GPU reference pass.
        for name, function in [('qc', qc), ('teacher', teacher), ('measure', measure)]:
            write_json(root / 'status.json', {**status, 'stage': name, 'at': time.time()})
            function(config)
        write_json(root / 'status.json', {**status, 'stage': 'initial_cpu_provisional_finished', 'at': time.time()})
    except Exception as exc:
        write_json(root / 'status.json', {**status, 'stage': 'failed', 'failed_stage': name,
            'error': {'type': type(exc).__name__, 'message': str(exc)}, 'at': time.time()})
        raise
    return read_json(root / 'status.json')
