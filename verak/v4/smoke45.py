"""Own only the paired v4.5 inference jobs. No training or paid API entry point."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

from .common import REPO, atomic_new, file_sha, read_json, write_json
from .dryrun_common import BASE, REVISION, VLLM_PYTHON, tokenizer
from .environment_v45 import VERSION, CONTEXT, OUTPUT, SPAN_POLICY
from .full_data import ROOT as TRAIN, DRY, ref, verified
from .full_run import INFERENCE_PYTHON, inference_runtime, idle
from . import orchestrator_v44 as orch, policy_prompts_v43 as workers
from .smoke45_infer import ROOT, PORT, CONDITIONS

WORKTREE = Path(__file__).resolve().parents[2]
CODE = ('environment_v45.py', 'maps_v45.py', 'smoke45_infer.py', 'smoke45.py',
        'environment_v44.py', 'orchestrator_v44.py', 'prep3_maps.py', 'maps.py')
PROTECTED = ('imple/FEAK_AGENT_METHOD.md', 'imple/reports/V4_TRAIN.md',
             'imple/reports/V4_SCALE3.md', 'imple/reports/V4_DRYRUN.md',
             'imple/reports/V3_PHASE_8_RFT1.md', 'imple/reports/V3_ONESHOT_BASELINE.md')


def server_command():
    adapters = [f'v45-{role}-e{epoch}={TRAIN}/adapters/{role}/epoch_{epoch}'
                for role, epoch in [('korean', 2), ('revision', 2), ('orchestrator', 2), ('revision', 1)]]
    return [str(VLLM_PYTHON), '-m', 'vllm.entrypoints.openai.api_server', '--model', str(BASE),
        '--served-model-name', 'kanana-v45', '--host', '127.0.0.1', '--port', str(PORT),
        '--enable-lora', '--max-loras', '4', '--max-cpu-loras', '4', '--max-lora-rank', '16',
        '--dtype', 'bfloat16', '--max-model-len', '16384', '--max-num-seqs', '1',
        '--gpu-memory-utilization', '.90', '--max-num-batched-tokens', '2048', '--enforce-eager',
        '--enable-prefix-caching', '--generation-config', 'vllm', '--seed', '47', '--lora-modules', *adapters]


def preflight():
    contract = orch.verify(); worker = workers.verify_frozen(); tok = tokenizer()
    prompts = {role: {k: data[k] for k in ('sha256', 'policy_tokens', 'system_message_tokens')}
               for role, data in worker['roles'].items()}
    prompts['orchestrator'] = {k: contract[k] for k in ('sha256', 'policy_tokens', 'system_message_tokens')}
    for role, text in {**workers.PROMPTS, 'orchestrator': orch.PROMPT}.items():
        assert len(tok.encode(text, add_special_tokens=False)) == prompts[role]['policy_tokens']
        assert len(tok.apply_chat_template([{'role': 'system', 'content': text}], tokenize=True)) == prompts[role]['system_message_tokens']
        assert max(prompts[role]['policy_tokens'], prompts[role]['system_message_tokens']) <= 400
    checkpoints = {}
    for role, epoch in [('korean', 2), ('revision', 2), ('orchestrator', 2), ('revision', 1)]:
        directory = TRAIN/f'adapters/{role}/epoch_{epoch}'
        provenance = read_json(directory/'provenance.json')
        if (provenance['base_revision'] != REVISION or provenance['role'] != role or provenance['epoch'] != epoch
                or provenance['adapter_sha256'] != file_sha(directory/'adapter_model.safetensors')):
            raise ValueError('Saved checkpoint identity differs')
        checkpoints[f'{role}_epoch{epoch}'] = {p.name: ref(p) for p in sorted(directory.iterdir()) if p.is_file()}
    manifest = read_json(DRY/'evaluation/manifest.json')
    if {name: len(c['cases']) for name, c in manifest['cohorts'].items()} != {'smoke20': 20, 'agent_dev10': 10}:
        raise ValueError('Diagnostic cohort differs')
    for cohort in manifest['cohorts'].values():
        for case in cohort['cases']: verified(case['row'])
    value = {'version': VERSION, 'conditions': CONDITIONS, 'context': CONTEXT, 'output': OUTPUT,
        'prompts_unchanged': prompts, 'prompt_manifests': [ref(workers.FROZEN), ref(orch.FROZEN)],
        'span_matching': SPAN_POLICY, 'MAP_policy': 'training edge filter; invalid metadata dropped, unknowns null',
        'completion_a': 'FINISH and all worker stages model STOP(done or blocked with nonblank summary/issues); no protocol/controller/context/step/runtime/scorer failure',
        'completion_b': 'completion_a plus every worker stage model STOP(done); Orchestrator must FINISH',
        'checkpoints': checkpoints, 'base_revision': REVISION,
        'evaluation_manifest': ref(DRY/'evaluation/manifest.json'),
        'training_manifest_unchanged': ref(TRAIN/'data/manifest.json'),
        'protected_files': [ref(REPO/p) for p in PROTECTED],
        'code_sha256': {name: file_sha(Path(__file__).with_name(name)) for name in CODE},
        'server_command': server_command(), 'inference_runtime': inference_runtime(),
        'sampling': {'temperature': 0., 'top_p': 1., 'seed': 47},
        'no_training': True, 'no_RFT': True, 'no_full_evaluation': True, 'test_cohort_untouched': True,
        'new_paid_API_calls': 0, 'private_plans_or_records_at_inference': False}
    path = ROOT/'preflight.json'
    if path.exists() and read_json(path) != value: raise ValueError('Frozen v4.5 preflight differs')
    if not path.exists(): atomic_new(path, value)
    return value


def verify_preflight():
    saved = read_json(ROOT/'preflight.json')
    orch.verify(); workers.verify_frozen()
    for name, digest in saved['code_sha256'].items():
        if file_sha(Path(__file__).with_name(name)) != digest: raise ValueError('Frozen code changed: '+name)
    refs = [saved['evaluation_manifest'], saved['training_manifest_unchanged'],
            *saved['prompt_manifests'], *saved['protected_files']]
    refs.extend(r for files in saved['checkpoints'].values() for r in files.values())
    for reference in refs:
        if file_sha(Path(reference['path'])) != reference['sha256']:
            raise ValueError('Frozen input changed: '+reference['path'])
    if saved['conditions'] != CONDITIONS or saved['context'] != CONTEXT or saved['output'] != OUTPUT:
        raise ValueError('Paired conditions/context changed')
    return saved


def run():
    verify_preflight()
    if (ROOT/'complete.json').exists(): return
    hardware = idle((0, 1))
    with socket.socket() as probe: probe.bind(('127.0.0.1', PORT))
    environment = {**os.environ, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '0',
        'VLLM_NO_USAGE_STATS': '1', 'DO_NOT_TRACK': '1', 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '8'}
    process = None
    write_json(ROOT/'status.json', {'stage': 'paired_inference', 'at': time.time(), 'hardware': hardware})
    try:
        import httpx
        with (ROOT/'vllm.log').open('a') as log:
            command = server_command()
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
                result = subprocess.run([str(INFERENCE_PYTHON), '-u', '-m', 'verak.v4.smoke45_infer'],
                    cwd=WORKTREE, env={**environment, 'CUDA_VISIBLE_DEVICES': '1'}, stdout=out, stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError('Paired smoke failed; inspect inference.log')
        verify_preflight()
        write_json(ROOT/'status.json', {'stage': 'complete', 'at': time.time(), 'training': False, 'RFT': False})
    except Exception as exc:
        write_json(ROOT/'execution_failure.json', {'error': str(exc), 'traceback': traceback.format_exc(), 'at': time.time()})
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=45)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
        write_json(ROOT/'server_stopped.json', {'owned_pid': process.pid if process else None, 'at': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'run', 'verify'])
    args = parser.parse_args()
    if args.command == 'preflight':
        saved = preflight(); print(json.dumps({'ready': True, 'prompts': saved['prompts_unchanged'], 'conditions': CONDITIONS}))
    elif args.command == 'verify': verify_preflight(); print('Frozen v4.5 inputs verified')
    else: run()
