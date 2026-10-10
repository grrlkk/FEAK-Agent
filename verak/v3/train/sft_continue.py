"""Continue the authorized sequential SFT/evaluation stages, then shut down."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from ..common import read_json, write_json
from .sft_data import PHASE, config_for
from .teacher_bulk import collection_lock


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def run(global_pid, luna_pid):
    config = config_for()
    root = config['paths'][PHASE + '_output']
    repo = config['paths']['repo']
    status_path = root / 'continuation_status.json'
    history, children = [], []
    server = None
    def status(stage, **extra):
        record = {'stage': stage, 'at': time.time(), **extra}
        history.append(record)
        write_json(status_path, {'current': record, 'history': history})
        print(json.dumps(record), flush=True)
    def child(arguments, log):
        with (root / log).open('a', encoding='utf-8') as stream:
            process = subprocess.Popen([sys.executable, '-m', 'verak.v3.cli.sft_warm_start', *arguments],
                                       cwd=repo, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(process)
            while process.poll() is None:
                time.sleep(5)
            if process.returncode:
                raise RuntimeError(f'{arguments} failed with exit {process.returncode}; see {log}')
    with collection_lock(root / 'continuation'):
        try:
            status('waiting_for_global', pid=global_pid)
            while alive(global_pid):
                time.sleep(5)
            if not (root / 'adapters/global/complete.json').exists():
                raise RuntimeError('GLOBAL stopped before completing both epochs')
            status('training_korean')
            if not (root / 'adapters/korean/complete.json').exists():
                args = ['train', '--role', 'korean']
                if (root / 'adapters/korean/recipe.json').exists():
                    args += ['--resume']
                child(args, 'train_korean.log')
            status('waiting_for_luna', pid=luna_pid)
            while alive(luna_pid):
                time.sleep(5)
            if not (root / 'evaluation/luna_low/run_status.json').exists():
                raise RuntimeError('Luna comparison did not close cleanly')
            status('starting_vllm')
            # This process owns only the server it starts and shuts it down at the end.
            server_log = (root / 'vllm.log').open('a', encoding='utf-8')
            server = subprocess.Popen(['/home/chanwoo/anaconda3/envs/verak_vllm/bin/python', '-m',
                'verak.v3.cli.serve_policy', '--sft-root', str(root)], cwd=repo,
                stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(server)
            deadline = time.monotonic() + 900
            expected = {'kanana-policy', 'sft-global-ep1', 'sft-korean-ep1', 'sft-global-ep2', 'sft-korean-ep2'}
            while True:
                if server.poll() is not None:
                    raise RuntimeError('vLLM stopped during startup; see vllm.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8030/v1/models', timeout=5) as response:
                        models = json.load(response)
                    if expected <= {row['id'] for row in models['data']}:
                        write_json(root / 'served_models.json', models)
                        break
                except (OSError, ValueError, KeyError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('vLLM did not become ready within 15 minutes')
                time.sleep(5)
            for condition in ('base', 'epoch_1', 'epoch_2'):
                status('evaluating_' + condition)
                child(['evaluate', '--condition', condition], 'evaluate_' + condition + '.log')
                result = read_json(root / 'evaluation' / condition / 'run_status.json')
                if result['errors'] or result['stop_reason'] != 'all_requested_attempted':
                    raise RuntimeError(f'{condition} evaluation needs review before continuing')
            status('checking_real_markers')
            child(['markers', '--max-api-calls', '2000'], 'sol_markers.log')
            status('measuring_real_outputs')
            child(['measure-real'], 'real_measurements.log')
            status('writing_draft_report')
            child(['report'], 'report.log')
            status('completed_awaiting_final_audit', rft_started=False)
        except BaseException as exc:
            status('failed', error=type(exc).__name__ + ': ' + str(exc))
            raise
        finally:
            for process in reversed(children):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=20)
            if server is not None:
                server_log.close()
