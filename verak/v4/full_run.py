"""Owned two-GPU full SFT -> integrated diagnostic -> one-shot training, then stop."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

from .common import REPO, atomic_new, file_sha, read_json, write_json
from .dryrun_common import BASE, TRAIN_PYTHON, VLLM_PYTHON
from .full_data import ROOT, ROLES, verify, ref
from .full_infer import PORT
from . import orchestrator_v44 as orch

WORKTREE = Path(__file__).resolve().parents[2]
INFERENCE_PYTHON = REPO/'verak/v4/outputs/scale2/search/venv/bin/python'


def inference_runtime():
    """Validate the established local SEARCH runtime before starting training."""
    code = ('import importlib.metadata,json; import bm25s,kiwipiepy; '
        'from verak.v4.search_v43 import verify_index; '
        'print(json.dumps({"packages":{n:importlib.metadata.version(n) '
        'for n in ("bm25s","kiwipiepy","torch","transformers","peft","bitsandbytes")},'
        '"index":verify_index()}))')
    output = subprocess.check_output([str(INFERENCE_PYTHON), '-c', code], cwd=WORKTREE,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1'}, text=True)
    return {'python': str(INFERENCE_PYTHON), **json.loads(output.splitlines()[-1])}


def preflight():
    data = verify()
    contract = orch.verify()
    value = {'data_manifest': ref(ROOT/'data/manifest.json'), 'orchestrator_contract': ref(orch.FROZEN),
        'worker_training_format': 'v4.3', 'inference_environment': 'v4.4',
        'prompt_tokens': {k: contract[k] for k in ('policy_tokens', 'system_message_tokens')},
        'training_roles_in_order': ['korean', 'revision', 'orchestrator'],
        'epochs': {'korean': 2, 'revision': 2, 'orchestrator': 2, 'oneshot': 1},
        'recipe': {'fresh_pinned_base': True, 'r': 16, 'alpha': 32, 'lr': 1e-4, 'context': 8192,
                   'world_size': 2, 'micro_batch': 1, 'gradient_accumulation_per_rank': 4},
        'code_sha256': {p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob('full_*.py'))},
        'no_RFT': True, 'no_full_evaluation': True, 'no_oneshot_evaluation': True, 'test_cohort_untouched': True,
        'new_paid_API_calls': 0, 'inference_runtime': inference_runtime(),
        'data_counts': {r: {p: x['turns'] for p, x in parts.items()} for r, parts in data['roles'].items()}}
    path = ROOT/'preflight.json'
    if path.exists() and read_json(path) != value:
        raise ValueError('Saved full-training preflight differs')
    if not path.exists(): atomic_new(path, value)
    return value


def check_preflight():
    saved = read_json(ROOT/'preflight.json')
    if saved['data_manifest']['sha256'] != file_sha(ROOT/'data/manifest.json'):
        raise ValueError('Training manifest changed after preflight')
    for name, sha in saved['code_sha256'].items():
        if file_sha(Path(__file__).with_name(name)) != sha:
            raise ValueError('Reviewed training/inference code changed: '+name)
    return saved


def idle(devices):
    identities = []
    for device in devices:
        pids = subprocess.check_output(['nvidia-smi', '-i', str(device), '--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'], text=True).strip()
        if pids: raise RuntimeError(f'GPU{device} is occupied; do not displace: {pids}')
        identities.append(subprocess.check_output(['nvidia-smi', '-i', str(device),
            '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader'], text=True).strip())
    return identities


def train_role(role):
    if (ROOT/'adapters'/role/'complete.json').exists(): return
    check_preflight(); hardware = idle((0, 1))
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'OMP_NUM_THREADS': '8', 'WANDB_DISABLED': 'true'}
    command = [str(TRAIN_PYTHON), '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        '--nproc-per-node=2', '-m', 'verak.v4.full_train', role]
    if (ROOT/'adapters'/role/'recipe.json').exists():
        command += ['--resume']
    write_json(ROOT/'status.json', {'stage': 'training_'+role, 'at': time.time(), 'hardware': hardware,
        'controller_pid': os.getpid(), 'command': command})
    with (ROOT/f'train_{role}.log').open('a') as log:
        result = subprocess.run(command, cwd=WORKTREE, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        write_json(ROOT/f'failure_training_{role}.json', {'returncode': result.returncode, 'at': time.time(),
            'log': str(ROOT/f'train_{role}.log'), 'checkpoints_preserved': True})
        raise RuntimeError('Full training failed: '+role)


def infer():
    if (ROOT/'smoke/complete.json').exists(): return
    check_preflight(); hardware = idle((0, 1))
    with socket.socket() as probe: probe.bind(('127.0.0.1', PORT))
    adapters = ['v44-'+role+'='+str(ROOT/'adapters'/role/'epoch_2') for role in ROLES if role != 'oneshot']
    command = [str(VLLM_PYTHON), '-m', 'vllm.entrypoints.openai.api_server', '--model', str(BASE),
        '--served-model-name', 'kanana-v44', '--host', '127.0.0.1', '--port', str(PORT),
        '--enable-lora', '--max-loras', '4', '--max-cpu-loras', '4', '--max-lora-rank', '16',
        '--dtype', 'bfloat16', '--max-model-len', '16384', '--max-num-seqs', '1',
        '--gpu-memory-utilization', '.90', '--max-num-batched-tokens', '2048', '--enforce-eager',
        '--enable-prefix-caching', '--generation-config', 'vllm', '--seed', '47', '--lora-modules', *adapters]
    environment = {**os.environ, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '0',
        'VLLM_NO_USAGE_STATS': '1', 'DO_NOT_TRACK': '1', 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '8'}
    process = None
    write_json(ROOT/'status.json', {'stage': 'integrated_smoke', 'at': time.time(), 'hardware': hardware})
    try:
        import httpx
        with (ROOT/'vllm.log').open('a') as log:
            process = subprocess.Popen(command, cwd=WORKTREE, env=environment, stdout=log, stderr=subprocess.STDOUT)
            write_json(ROOT/'server.json', {'pid': process.pid, 'physical_gpu': 0, 'command': command, 'at': time.time()})
            deadline = time.monotonic()+900
            while time.monotonic() < deadline:
                if process.poll() is not None: raise RuntimeError('vLLM failed; inspect vllm.log')
                try:
                    if httpx.get(f'http://127.0.0.1:{PORT}/health', timeout=2).status_code == 200: break
                except httpx.HTTPError: pass
                time.sleep(2)
            else: raise RuntimeError('vLLM readiness timeout')
            with (ROOT/'inference.log').open('a') as out:
                run = subprocess.run([str(INFERENCE_PYTHON), '-u', '-m', 'verak.v4.full_infer'], cwd=WORKTREE,
                    env={**environment, 'CUDA_VISIBLE_DEVICES': '1'}, stdout=out, stderr=subprocess.STDOUT)
            if run.returncode: raise RuntimeError('Integrated smoke failed; inspect inference.log')
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=45)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
        write_json(ROOT/'server_stopped.json', {'owned_pid': process.pid if process else None, 'at': time.time()})


def run():
    check_preflight()
    try:
        for role in ('korean', 'revision', 'orchestrator'): train_role(role)
        try:
            infer()
        except Exception as exc:
            write_json(ROOT/'failure_smoke.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        train_role('oneshot')
        write_json(ROOT/'execution_complete.json', {'at': time.time(), 'roles_trained': list(ROLES),
            'smoke_complete': (ROOT/'smoke/complete.json').exists(), 'GPU_jobs_finished': True,
            'RFT_started': False, 'full_evaluation_started': False, 'oneshot_evaluated': False})
    except Exception as exc:
        write_json(ROOT/'execution_failure.json', {'at': time.time(), 'error': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        from .full_report import report
        report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'run', 'report'])
    args = parser.parse_args()
    if args.command == 'preflight':
        result = preflight(); print(json.dumps({'ready': True, 'counts': result['data_counts']}), flush=True)
    elif args.command == 'run': run()
    else:
        from .full_report import report
        report()
