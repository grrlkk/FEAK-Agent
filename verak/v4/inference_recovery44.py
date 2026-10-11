"""Recheck only SEARCH dependency failures, after all authorized training ends.

No prompt, model, environment, gate, teacher data or original smoke file changes.
The established Wiki venv supplies the missing local search packages. Its core
model libraries must match the training interpreter before GPU execution.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

from .common import atomic_new, file_sha, load_config, read_json, write_json, REPO
from .full_data import ROOT, DRY, ref, verified
from .full_infer import PORT, LocalBackend, run_case, summarize
from .full_run import WORKTREE, check_preflight, idle
from .dryrun_common import tokenizer, local_analysis_environment
from .environment_v43 import BoundaryParagraphs
from . import orchestrator_v44 as orch

RECOVERY = ROOT / 'smoke_search_runtime_recheck'
SEARCH_PYTHON = REPO / 'verak/v4/outputs/scale2/search/venv/bin/python'


def dependency_failure(attempt):
    return any(c.get('action', {}).get('error') == "ModuleNotFoundError: No module named 'bm25s'"
               for c in attempt['calls'])


def prepare():
    check_preflight(); orch.verify()
    probe_path = ROOT / 'audit/search_runtime_probe.json'
    probe = read_json(probe_path)
    if probe['training'] != probe['search'] or probe['GPU_calls'] or probe['API_calls']:
        raise ValueError('Search runtime did not pass the CPU-only dependency probe')
    original = read_json(DRY / 'evaluation/manifest.json')
    from .common import safe_id
    cohorts = {}
    for name, cohort in original['cohorts'].items():
        entries = []
        for case in cohort['cases']:
            path = ROOT/'smoke'/name/'attempts'/(safe_id(case['case_id'])+'.json')
            attempt = read_json(path)
            entries.append({'case_id': case['case_id'], 'row': case['row'], 'original': ref(path),
                            'retry': dependency_failure(attempt)})
        cohorts[name] = entries
    plan = {'reason': 'First inference interpreter lacked bm25s; reuse the existing local Wiki venv.',
        'core_model_libraries_identical': True, 'runtime_probe': ref(probe_path),
        'initial_smoke': ref(ROOT/'smoke/complete.json'), 'evaluation_manifest': ref(DRY/'evaluation/manifest.json'),
        'orchestrator_contract': ref(orch.FROZEN), 'code_sha256': file_sha(Path(__file__)),
        'search_python': str(SEARCH_PYTHON), 'server_command': read_json(ROOT/'server.json')['command'],
        'cohorts': cohorts, 'retry_count': sum(c['retry'] for cs in cohorts.values() for c in cs),
        'training_changed': False, 'inference_contract_changed': False, 'new_paid_calls': 0,
        'test_cohort_untouched': True, 'retry_context_limits_alone': False}
    path = RECOVERY/'plan.json'
    if path.exists():
        if read_json(path) != plan: raise ValueError('Dependency recheck plan changed')
    else: atomic_new(path, plan)
    return plan


def evaluate():
    plan = prepare(); local_analysis_environment(); tok = tokenizer(); backend = LocalBackend()
    config = load_config(); config['scorer']['gpu'] = 0
    config['paths']['output'] = RECOVERY/'scorer'
    scorer = None
    def score(question, text):
        nonlocal scorer
        if scorer is None:
            from verak.v3.score.kanana import KananaScorer
            scorer = KananaScorer(config, cache_enabled=False)
            initial = read_json(ROOT/'smoke/scorer_identity.json')
            if scorer.fingerprint != initial['fingerprint']:
                raise ValueError('Dependency repair changed the frozen scorer fingerprint')
            write_json(RECOVERY/'scorer_identity.json', {'physical_gpu': 1, 'logical_gpu': 0,
                'fingerprint': scorer.fingerprint, 'identical_to_initial': True})
        return scorer.score(question, text, use_cache=False).to_dict()
    result = {'cohorts': {}, 'retry_count': plan['retry_count'], 'initial_smoke': plan['initial_smoke'],
        'plan': ref(RECOVERY/'plan.json'), 'quality_results': False, 'new_paid_calls': 0,
        'test_cohort_untouched': True, 'aggregation': 'Original unaffected cases plus dependency-rechecked cases; denominator remains 20+10.'}
    try:
        for name, cases in plan['cohorts'].items():
            attempts = []
            for case in cases:
                if not case['retry']:
                    attempts.append(verified(case['original']))
                    continue
                saved = verified(case['row'])
                row = {k: saved[k] for k in ('source_id', 'genre', 'question', 'text', 'profile',
                    'layout', 'normalization', 'paragraphs', 'environment_version') if k in saved}
                attempts.append(run_case(row, case_id=case['case_id'], output=RECOVERY/name,
                    backend=backend, tok=tok, analysis=BoundaryParagraphs(config, cache_dir=RECOVERY/'bareun'), score=score))
            result['cohorts'][name] = summarize(attempts, len(cases))
            result['cohorts'][name]['dependency_rechecked_cases'] = [c['case_id'] for c in cases if c['retry']]
        atomic_new(RECOVERY/'complete.json', result)
        return result
    finally:
        backend.close()
        if scorer is not None: scorer.close()


def append_report():
    from .full_report import report
    result = report()
    path = Path(result['report']['path'])
    original = path.read_text()
    archive = RECOVERY/'V4_TRAIN_initial_smoke.md'
    if not archive.exists(): archive.write_text(original)
    plan = read_json(RECOVERY/'plan.json')
    completed = read_json(RECOVERY/'complete.json') if (RECOVERY/'complete.json').exists() else None
    lines = ['', '## SEARCH runtime repair and targeted diagnostic recheck', '',
        'The first smoke used the base training interpreter, which did not contain `bm25s`. '
        'This caused 23 rejected SEARCH calls across 11 essays (9 smoke20, 2 agent_dev10). '
        'These were runtime dependency failures, not malformed model actions. All first-pass outputs and metrics above are retained.', '',
        'After one-shot training finished, only those 11 affected essays were rerun using the existing Wiki virtual environment. '
        'Torch, Transformers, PEFT, bitsandbytes, NumPy and SciPy versions **and import paths** matched the initial interpreter; '
        'a CPU-only local SEARCH probe passed. No packages were installed, no prompts or environment rules changed, '
        'and no model was retrained. Model sampling settings, adapters and the frozen scorer fingerprint were unchanged. '
        'The test cohort was not used. The 19 unaffected initial cases are reused explicitly; the denominator remains 30.', '']
    if completed:
        lines += ['|Cohort|Dependency rechecks|Final protocol complete|Orchestrator FINISH|Mean model delegations|', '|---|---:|---:|---:|---:|']
        for name, group in completed['cohorts'].items():
            lines.append(f"|{name}|{len(group['dependency_rechecked_cases'])}|{group['protocol_complete']}/{group['planned']}|{group['orchestrator_FINISH_rate']:.3f}|{group['mean_model_delegations']:.3f}|")
        lines += ['', '|Cohort / role|Valid / returned|Valid action rate|Model STOP done / blocked|STOP rate|Controller blocked|Context limits|', '|---|---:|---:|---:|---:|---:|---:|']
        for name, group in completed['cohorts'].items():
            for role, value in group['roles'].items():
                stop = '—' if value['model_STOP_rate'] is None else f"{value['model_STOP_rate']:.3f}"
                lines.append(f"|{name} / {role}|{value['valid_actions']}/{value['returned_actions']}|{value['valid_action_rate']:.3f}|{value['model_STOP_done']}/{value['model_STOP_blocked']}|{stop}|{value['controller_STOP_blocked']}|{len(value['context_limits'])}|")
        lines += ['', 'Final diagnostic details (execution only, no quality claims):', '', '```json', json.dumps(completed, ensure_ascii=False, indent=2), '```', '']
    else:
        lines += ['The dependency recheck did not complete. See the preserved failure artifact; the initial metrics remain the only complete smoke.', '']
    lines += [f"Recheck plan: `{RECOVERY/'plan.json'}`. Original and rechecked request/response logs are separate. "
        'No RFT, full evaluation, or one-shot evaluation follows this report.', '']
    path.write_text(original+'\n'.join(lines))
    result.update(smoke_runtime_recheck=completed, runtime_recheck_plan=ref(RECOVERY/'plan.json'))
    result['report'] = ref(path)
    write_json(ROOT/'report_metrics.json', result)
    write_json(RECOVERY/'report_complete.json', {'report': ref(path), 'completed': completed is not None})


def run():
    plan = prepare()
    # Wait for the existing controller; never stop or compete with its training.
    while not (ROOT/'execution_complete.json').exists():
        if (ROOT/'execution_failure.json').exists():
            raise RuntimeError('Original training controller failed; do not take its GPUs')
        time.sleep(15)
    check_preflight(); hardware = idle((0, 1))
    if (RECOVERY/'complete.json').exists(): return append_report()
    with socket.socket() as probe: probe.bind(('127.0.0.1', PORT))
    environment = {**os.environ, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '0',
        'VLLM_NO_USAGE_STATS': '1', 'DO_NOT_TRACK': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '8'}
    process = None
    try:
        import httpx
        write_json(ROOT/'status.json', {'stage': 'smoke_search_runtime_recheck', 'at': time.time(), 'hardware': hardware})
        with (RECOVERY/'vllm.log').open('a') as log:
            process = subprocess.Popen(plan['server_command'], cwd=WORKTREE, env=environment, stdout=log, stderr=subprocess.STDOUT)
            write_json(RECOVERY/'server.json', {'pid': process.pid, 'command': plan['server_command'], 'at': time.time()})
            deadline = time.monotonic()+900
            while time.monotonic() < deadline:
                if process.poll() is not None: raise RuntimeError('Recheck vLLM failed; inspect its log')
                try:
                    if httpx.get(f'http://127.0.0.1:{PORT}/health', timeout=2).status_code == 200: break
                except httpx.HTTPError: pass
                time.sleep(2)
            else: raise RuntimeError('Recheck vLLM readiness timeout')
            with (RECOVERY/'inference.log').open('a') as log:
                execution = subprocess.run([str(SEARCH_PYTHON), '-u', '-m', 'verak.v4.inference_recovery44', 'evaluate'],
                    cwd=WORKTREE, env={**environment, 'CUDA_VISIBLE_DEVICES': '1'}, stdout=log, stderr=subprocess.STDOUT)
            if execution.returncode: raise RuntimeError('Dependency recheck failed; inspect its inference log')
    except Exception as exc:
        write_json(RECOVERY/'failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=45)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
        write_json(RECOVERY/'server_stopped.json', {'owned_pid': process.pid if process else None, 'at': time.time()})
        append_report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'evaluate', 'report'])
    args = parser.parse_args()
    if args.command == 'prepare': print(json.dumps({'rechecks': prepare()['retry_count']}))
    elif args.command == 'run': run()
    elif args.command == 'evaluate': evaluate()
    else: append_report()
