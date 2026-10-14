"""Owned GPU1 dry-run subprocesses only; collectors/scorers are never controlled."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from .dryrun_common import (ROOT, ROLES, BASE, PORT, TRAIN_PYTHON, VLLM_PYTHON,
    require_gpu_approval, constrain_cpu, read_json, write_json, file_sha, reference)


def gpu1_idle():
    # Restrict the query to the assigned physical device. A busy GPU0 is expected.
    raw = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid',
                                   '--format=csv,noheader,nounits'], text=True)
    pids = [line.strip() for line in raw.splitlines() if line.strip()]
    if any(not value.isdigit() or int(value) <= 0 for value in pids):
        raise RuntimeError('Cannot verify physical GPU1 ownership from malformed PID output')
    if pids:
        raise RuntimeError('Physical GPU1 is occupied; do not displace another process: '+','.join(pids))
    return subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=index,uuid,name',
                                    '--format=csv,noheader'], text=True).strip()


def gpu1_uuid(actual_uuid):
    physical = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid',
                                       '--format=csv,noheader'], text=True).strip()
    actual = str(actual_uuid)
    if (actual in {'None', '', 'unavailable'} or not physical
            or actual.removeprefix('GPU-').lower() != physical.removeprefix('GPU-').lower()):
        raise RuntimeError('Logical CUDA0 UUID does not match physical GPU1')
    return {'logical_cuda_device': 0, 'physical_gpu': 1,
            'cuda_uuid': actual, 'nvidia_smi_gpu1_uuid': physical, 'uuid_match': True}


def train_only(*, root=ROOT):
    constrain_cpu(gpu=True)
    require_gpu_approval(root=root)
    worktree = Path(__file__).resolve().parents[2]
    for role in ROLES:
        output = root/'adapters'/role
        if (output/'failure.json').exists():
            raise RuntimeError('Preserved training failure; no automatic retry: '+role)
        if (output/'complete.json').exists():
            continue
        hardware = gpu1_idle()
        write_json(root/'status.json', {'stage': 'training_'+role, 'physical_gpu': 1,
            'hardware': hardware, 'controller_pid': os.getpid(), 'at': time.time()})
        with (root/f'train_{role}.log').open('a') as stream:
            process = subprocess.run([str(TRAIN_PYTHON), '-u', '-m', 'verak.v4.dryrun',
                'train-role', role], cwd=worktree, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            write_json(root/'training_failure.json', {'role': role, 'returncode': process.returncode,
                'log': str(root/f'train_{role}.log'), 'no_retry': True})
            raise RuntimeError('Dry-run training failed: '+role)
    write_json(root/'training_complete.json', {'status': 'complete', 'physical_gpu': 1,
        'roles': {r: reference(root/'adapters'/r/'complete.json') for r in ROLES},
        'fresh_adapters': True, 'epochs_per_role': 1, 'technical_dry_run_only': True})
    write_json(root/'status.json', {'stage': 'training_complete', 'at': time.time(), 'gpu_released': True})


def server_command(*, root=ROOT):
    adapters = []
    for role in ROLES:
        complete = read_json(root/'adapters'/role/'complete.json')
        adapter = root/'adapters'/role/'epoch_1'
        provenance = read_json(adapter/'provenance.json')
        if not complete['completed'] or file_sha(adapter/'adapter_model.safetensors') != provenance['adapter_sha256']:
            raise ValueError('Completed dry-run adapter/provenance required')
        adapters.append('dryrun-'+role+'='+str(adapter))
    return [str(VLLM_PYTHON), '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(BASE), '--served-model-name', 'kanana-dryrun', '--host', '127.0.0.1', '--port', str(PORT),
        '--enable-lora', '--max-loras', '2', '--max-cpu-loras', '4', '--max-lora-rank', '16',
        '--dtype', 'bfloat16', '--max-model-len', '8192', '--max-num-seqs', '4',
        '--gpu-memory-utilization', '.90', '--max-num-batched-tokens', '2048', '--enforce-eager',
        '--enable-prefix-caching', '--generation-config', 'vllm', '--seed', '47', '--lora-modules', *adapters]


def serve_and_evaluate(*, root=ROOT):
    import httpx
    import socket
    constrain_cpu(gpu=True)
    require_gpu_approval(root=root)
    if not (root/'evaluation/manifest.json').exists():
        raise ValueError('Freeze the approved diagnostic cohorts before occupying GPU1')
    if (root/'evaluation/complete.json').exists() or (root/'inference_failure.json').exists():
        raise ValueError('Existing evaluation success/failure is immutable; no implicit repeat')
    hardware = gpu1_idle()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', PORT))
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '1', 'VLLM_NO_USAGE_STATS': '1', 'DO_NOT_TRACK': '1'}
    command = server_command(root=root)
    process = None
    try:
        with (root/'vllm.log').open('a') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                       cwd=Path(__file__).resolve().parents[2])
            write_json(root/'server.json', {'pid': process.pid, 'physical_gpu': 1, 'hardware': hardware,
                'command': command, 'at': time.time(), 'owned_by_dryrun': True})
            deadline = time.monotonic()+900
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'Owned vLLM server exited ({process.returncode}); inspect vllm.log')
                try:
                    response = httpx.get(f'http://127.0.0.1:{PORT}/health', timeout=2)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError('Owned vLLM server did not become ready in900s')
            # CPU evaluator hides all GPUs. Only the owned vLLM subprocess uses GPU1.
            evaluator = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
            with (root/'inference.log').open('a') as output:
                run = subprocess.run([str(TRAIN_PYTHON), '-u', '-m', 'verak.v4.dryrun', 'evaluate-local'],
                    cwd=Path(__file__).resolve().parents[2], env=evaluator, stdout=output, stderr=subprocess.STDOUT)
            if run.returncode:
                raise RuntimeError(f'Local evaluator exited ({run.returncode}); inspect inference.log')
    except Exception as exc:
        write_json(root/'inference_failure.json', {'error': type(exc).__name__+': '+str(exc),
            'traceback': traceback.format_exc(), 'retried': False})
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=10)
        write_json(root/'server_stopped.json', {'owned_pid': process.pid if process else None,
            'physical_gpu': 1, 'at': time.time(), 'other_processes_untouched': True})
