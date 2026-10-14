"""Own the authorized fix1 continuation and diagnostic GPU jobs, then stop."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

from .common import REPO, atomic_new, file_sha, read_json, write_json
from .dryrun_common import BASE, REVISION, TRAIN_PYTHON, VLLM_PYTHON, tokenizer
from .full_data import ROOT as TRAIN, DRY, ref, verified
from .full_run import INFERENCE_PYTHON, inference_runtime, idle
from .fix1_taxonomy import ROOT
from .fix1_data import verify as verify_data
from .fix1_infer import PORT, CONDITIONS, SETTINGS
from .environment_v46 import VERSION, CONTEXT, OUTPUT
from . import orchestrator_v44 as orch, policy_prompts_v43 as workers

WORKTREE = Path(__file__).resolve().parents[2]
CODE = ('environment_v46.py', 'fix1_taxonomy.py', 'fix1_data.py', 'fix1_train.py',
        'fix1_infer.py', 'fix1_eval.py', 'fix1_run.py', 'environment_v45.py',
        'maps_v45.py', 'environment_v44.py', 'orchestrator_v44.py', 'prep3_maps.py', 'render_v43.py')


def adapter_paths():
    return {'fix1-orchestrator': TRAIN/'adapters/orchestrator/epoch_2',
            'fix1-old-revision': TRAIN/'adapters/revision/epoch_1',
            'fix1-old-korean': TRAIN/'adapters/korean/epoch_2',
            'fix1-new-revision': ROOT/'adapters/revision/epoch_1',
            'fix1-new-korean': ROOT/'adapters/korean/epoch_1'}


def preflight():
    data = verify_data(); contract = orch.verify(); worker = workers.verify_frozen(); tok = tokenizer()
    from .smoke45 import verify_preflight as verify_previous
    previous = verify_previous()
    prompts = {role: {k: v[k] for k in ('sha256', 'policy_tokens', 'system_message_tokens')}
               for role, v in worker['roles'].items()}
    prompts['orchestrator'] = {k: contract[k] for k in ('sha256', 'policy_tokens', 'system_message_tokens')}
    for role, text in {**workers.PROMPTS, 'orchestrator': orch.PROMPT}.items():
        assert len(tok.encode(text, add_special_tokens=False)) == prompts[role]['policy_tokens']
        assert max(prompts[role]['policy_tokens'], prompts[role]['system_message_tokens']) <= 400
    checkpoints = {}
    for name, path in adapter_paths().items():
        if name.startswith('fix1-new'):
            continue
        provenance = read_json(path/'provenance.json')
        if provenance['base_revision'] != REVISION or provenance['adapter_sha256'] != file_sha(path/'adapter_model.safetensors'):
            raise ValueError('Original checkpoint identity changed')
        checkpoints[name] = {p.name: ref(p) for p in sorted(path.iterdir()) if p.is_file()}
    manifest = read_json(DRY/'evaluation/manifest.json')
    if {name: len(c['cases']) for name, c in manifest['cohorts'].items()} != {'smoke20': 20, 'agent_dev10': 10}:
        raise ValueError('Fixed diagnostic sets differ')
    for cohort in manifest['cohorts'].values():
        for case in cohort['cases']:
            verified(case['row'])
    config = __import__('verak.v4.common', fromlist=['load_config']).load_config()
    if config['bareun'].get('host') not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Only the existing local Bareun endpoint is authorized')
    value = {'version': VERSION, 'training_manifest': ref(ROOT/'data/manifest.json'),
        'taxonomy': ref(ROOT/'taxonomy.json'), 'base_revision': REVISION,
        'prompts_unchanged': prompts, 'conditions': CONDITIONS, 'settings': SETTINGS,
        'context': CONTEXT, 'output': OUTPUT, 'old_and_new_environment': 'v4.6',
        'old_checkpoints': checkpoints, 'evaluation_manifest': ref(DRY/'evaluation/manifest.json'),
        'previous_preflight': ref(REPO/'verak/v4/outputs/smoke45/preflight.json'),
        'protected_files': previous['protected_files']+[ref(REPO/'imple/reports/V4_SMOKE2.md')],
        'code_sha256': {name: file_sha(Path(__file__).with_name(name)) for name in CODE},
        'inference_runtime': inference_runtime(), 'training': {'roles': ['korean', 'revision'],
            'epochs': 1, 'lr': 5e-5, 'r': 16, 'alpha': 32, 'context': 8192, 'world_size': 2,
            'initial': {'korean': 2, 'revision': 1}, 'orchestrator_unchanged': True},
        'data_counts': {r: data['roles'][r]['counts'] for r in ('korean', 'revision')},
        'paid_API_calls': 0, 'test_cohort_untouched': True, 'large_RFT': False}
    path = ROOT/'preflight.json'
    if path.exists() and read_json(path) != value:
        raise ValueError('Frozen fix1 preflight differs')
    if not path.exists():
        atomic_new(path, value)
    return value


def verify_preflight():
    saved = read_json(ROOT/'preflight.json')
    verify_data(); orch.verify(); workers.verify_frozen()
    for name, digest in saved['code_sha256'].items():
        if file_sha(Path(__file__).with_name(name)) != digest:
            raise ValueError('Frozen fix1 code changed: '+name)
    refs = [saved['training_manifest'], saved['taxonomy'], saved['evaluation_manifest'],
            saved['previous_preflight'], *saved['protected_files']]
    refs.extend(r for files in saved['old_checkpoints'].values() for r in files.values())
    for item in refs:
        if file_sha(item['path']) != item['sha256']:
            raise ValueError('Frozen input changed: '+item['path'])
    return saved


def train_role(role):
    if (ROOT/'adapters'/role/'complete.json').exists():
        return
    verify_preflight(); hardware = idle((0, 1))
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'OMP_NUM_THREADS': '8', 'WANDB_DISABLED': 'true'}
    command = [str(TRAIN_PYTHON), '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        '--nproc-per-node=2', '-m', 'verak.v4.fix1_train', role]
    if (ROOT/'adapters'/role/'recipe.json').exists():
        command += ['--resume']
    write_json(ROOT/'status.json', {'stage': 'training_'+role, 'at': time.time(), 'hardware': hardware,
                                  'controller_pid': os.getpid(), 'command': command})
    print('Starting continuation: '+role, flush=True)
    with (ROOT/f'train_{role}.log').open('a') as log:
        result = subprocess.run(command, cwd=WORKTREE, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Fix1 training failed; checkpoints retained: '+role)
    print('Completed continuation: '+role, flush=True)


def server_command():
    adapters = [name+'='+str(path) for name, path in adapter_paths().items()]
    return [str(VLLM_PYTHON), '-m', 'vllm.entrypoints.openai.api_server', '--model', str(BASE),
        '--served-model-name', 'kanana-fix1', '--host', '127.0.0.1', '--port', str(PORT),
        '--enable-lora', '--max-loras', '4', '--max-cpu-loras', '8', '--max-lora-rank', '16',
        '--dtype', 'bfloat16', '--max-model-len', '16384', '--max-num-seqs', '1',
        '--gpu-memory-utilization', '.90', '--max-num-batched-tokens', '2048', '--enforce-eager',
        '--enable-prefix-caching', '--generation-config', 'vllm', '--seed', '47', '--lora-modules', *adapters]


def infer():
    if (ROOT/'evaluation/complete.json').exists():
        return
    verify_preflight(); hardware = idle((0, 1))
    for name, path in adapter_paths().items():
        p = read_json(path/'provenance.json')
        if p['adapter_sha256'] != file_sha(path/'adapter_model.safetensors'):
            raise ValueError('Evaluation checkpoint hash differs: '+name)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', PORT))
    environment = {**os.environ, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '0',
        'VLLM_NO_USAGE_STATS': '1', 'DO_NOT_TRACK': '1', 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '8'}
    process = None
    write_json(ROOT/'status.json', {'stage': 'diagnostics', 'at': time.time(), 'hardware': hardware})
    print('Starting paired diagnostics: 180 deterministic + 60 sampling-preview episodes', flush=True)
    try:
        import httpx
        with (ROOT/'vllm.log').open('a') as log:
            command = server_command()
            process = subprocess.Popen(command, cwd=WORKTREE, env=environment, stdout=log, stderr=subprocess.STDOUT)
            write_json(ROOT/'server.json', {'pid': process.pid, 'physical_gpu': 0, 'command': command, 'at': time.time()})
            deadline = time.monotonic()+900
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('vLLM failed; inspect vllm.log')
                try:
                    if httpx.get(f'http://127.0.0.1:{PORT}/health', timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError('vLLM readiness timeout')
            with (ROOT/'inference.log').open('a') as out:
                result = subprocess.run([str(INFERENCE_PYTHON), '-u', '-m', 'verak.v4.fix1_eval'],
                    cwd=WORKTREE, env={**environment, 'CUDA_VISIBLE_DEVICES': '1'}, stdout=out, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('Fix1 diagnostics failed; inspect inference.log')
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=10)
        write_json(ROOT/'server_stopped.json', {'owned_pid': process.pid if process else None, 'at': time.time()})


def run():
    verify_preflight()
    try:
        for role in ('korean', 'revision'):
            train_role(role)
        infer()
        verify_preflight()
        write_json(ROOT/'gpu_complete.json', {'at': time.time(), 'hardware_idle': idle((0, 1))})
        result = subprocess.run([str(TRAIN_PYTHON), '-m', 'verak.v4.fix1_report'], cwd=WORKTREE,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4'})
        if result.returncode:
            raise RuntimeError('GPU work completed; report needs repair')
        write_json(ROOT/'status.json', {'stage': 'complete', 'at': time.time(), 'report': str(REPO/'imple/reports/V4_FIX1.md')})
        print('Fix1 complete; GPUs stopped and report written', flush=True)
    except Exception as exc:
        write_json(ROOT/'execution_failure.json', {'error': str(exc), 'traceback': traceback.format_exc(), 'at': time.time()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'verify', 'run'])
    args = parser.parse_args()
    if args.command == 'preflight':
        preflight(); print('Fix1 preflight frozen')
    elif args.command == 'verify':
        verify_preflight(); print('Fix1 frozen inputs verified')
    else:
        run()
