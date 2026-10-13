"""Frozen Revision notice ablation; no training or paid policy calls."""
from copy import deepcopy
from collections import Counter
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from .common import (REPO, WORKTREE, INFERENCE_PYTHON, PORT, ROOT as PIPE_ROOT,
    Backend, atomic_new, read_json, ref, verified, write_json, safe_id, tokenizer,
    load_config, local_analysis_environment, check_runtime)
from .check_judge import ROOT, REPORT, freeze
from .check_outputs import require_gate, prepare_case, affected, structural_kind
from .detector import changes
from .judge import paragraphs
from .pipeline import ScopedEnvironment, stage
from verak.v4.render_v43 import assert_rendered, render
from verak.v4.scale2_projection import save_document


def retain_control_notices(value):
    """Remove cohesion messages everywhere, preserving split/ID control notices."""
    if isinstance(value, list):
        return [retain_control_notices(v) for v in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, data in value.items():
        if key in {'notices', 'marker_change_notices'} and isinstance(data, list):
            result[key] = [deepcopy(v) for v in data if isinstance(v, dict) and v.get('kind') == 'sentence_split']
        else:
            result[key] = retain_control_notices(data)
    return result


class NoticeEnvironment(ScopedEnvironment):
    def __init__(self, *args, show_notices=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_notices = show_notices
        self.last_change_notices = []

    def public_messages(self, role, steps_left=None):
        messages = super().public_messages(role, steps_left)
        payload = retain_control_notices(assert_rendered(messages))
        if self.show_notices:
            payload['notices'].extend(deepcopy(self.last_change_notices))
        return render(payload, messages[0]['content'])

    def step(self, raw, role):
        before = self.document.clone()
        result = super().step(raw, role)
        self.last_change_notices = changes(before, self.document, self.shared_map) if result.get('valid') and result.get('text_changed') else []
        for notice in self.last_change_notices:
            notice['id'] = f'T{self.steps_used}_'+notice['id']
        # Stored in both arms for analysis; visible only in the intervention arm.
        result['diagnostic_per_action_changes'] = deepcopy(self.last_change_notices)
        return result


def prepare():
    from .pilot import cases
    require_gate()
    records = []
    for case in cases():
        row = verified(case['row'])
        path = REPO/'verak/v4/outputs/fix1/evaluation/deterministic/run1/old'/case['cohort']/'attempts'/(safe_id(case['case_id'])+'.json')
        value = read_json(path)
        calls = sorted(value['calls'], key=lambda c: int(Path(c['prefix']['path']).name.split('_')[0]))
        first = next(c for c in calls if c['role'] == 'orchestrator' and c.get('action', {}).get('valid')
                     and c['action']['action'] == 'DELEGATE')
        payload = json.loads(verified(first['prefix'])['messages'][1]['content'])
        if first['action']['value']['to'] != 'revision' or payload['paragraphs'] != row['paragraphs']:
            raise ValueError('First-round saved plan is not Revision on the original C input')
        records.append({**case, 'plan_source': ref(path), 'plan_prefix': first['prefix'],
            'delegation': deepcopy(first['action']['value']), 'map': payload['map']})
    value = {'cases': records, 'conditions': ['without_notices', 'with_notices'],
        'model': ref(REPO/'verak/v4/outputs/train44/adapters/revision/epoch_1/adapter_model.safetensors'),
        'plan_origin': 'saved frozen Orchestrator epoch2, FIX1 old workers T0 run1, first DELEGATE on unchanged original',
        'temperature': 0., 'top_p': 1., 'seed': 47, 'environment': 'v4.6',
        'changed_observation_only': 'cohesion notices removed in both arms; with arm adds last action V5 phi changes. Split/ID notices kept in both.',
        'scope_and_action_budgets': 'identical saved first Revision delegation; 6 steps; unchanged v4.6 guards',
        'no_Sol_plans': True, 'new_Orchestrator_calls': 0, 'training': False, 'test_cohort': False}
    freeze(ROOT/'step2/design.json', value)
    return value


def infer():
    from verak.v4.environment_v43 import BoundaryParagraphs
    require_gate()
    if (ROOT/'step2/inference_complete.json').exists():
        return
    started = time.monotonic()
    design = read_json(ROOT/'step2/design.json')
    local_analysis_environment(); tok = tokenizer()
    analysis = BoundaryParagraphs(load_config(), cache_dir=ROOT/'bareun')
    backend = Backend()
    summaries, cases = [], []
    try:
        for case in design['cases']:
            row = verified(case['row'])
            for condition in design['conditions']:
                out = ROOT/'step2/episodes'/case['cohort']/condition/safe_id(case['case_id'])
                path = out/'episode.json'
                if path.exists():
                    result = read_json(path)
                else:
                    env = NoticeEnvironment(row, analysis, show_notices=condition == 'with_notices')
                    env.shared_map = deepcopy(case['map'])
                    initial = save_document(env.document)
                    result_stage = stage(env, 'revision', case['delegation']['items'], backend, tok, out/'delegation', number=1)
                    result = {'case_id': case['case_id'], 'cohort': case['cohort'], 'condition': condition,
                        'initial': initial, 'final': save_document(env.document), 'lineage': deepcopy(env.lineage),
                        'stage': result_stage, 'plan_prefix': case['plan_prefix'],
                        'no_protocol_error': result_stage['no_protocol_error']}
                    atomic_new(path, result)
                actions = []
                for call in result['stage']['calls']:
                    if 'action' not in call:
                        continue
                    before, after = paragraphs(call['before']), paragraphs(call['after'])
                    kind = structural_kind(call['action'], before, after)
                    if kind:
                        actions.append({'id': f'E{len(actions)+1}', 'kind': kind, 'action': call['action'],
                            'before': before, 'after': after, 'sites': affected(before, after, call['action'])})
                candidate = prepare_case({'case_id': case['case_id'], 'source': case['cohort']+'_'+condition,
                    'question': row['question'], 'original': paragraphs(result['initial']), 'final': paragraphs(result['final']),
                    'actions': actions, 'lineage': result['lineage'], 'protocol': result['no_protocol_error'], 'provenance': ref(path)})
                case_path = ROOT/'step2/cases'/candidate['source']/(safe_id(case['case_id'])+'.json')
                freeze(case_path, candidate); cases.append(ref(case_path))
                action_rows = [c['action'] for c in result['stage']['calls'] if 'action' in c]
                summaries.append({'cohort': case['cohort'], 'condition': condition, 'case_id': case['case_id'],
                    'valid_actions': sum(a['valid'] for a in action_rows), 'actions': len(action_rows),
                    'no_protocol_error': result['no_protocol_error'], 'ending': result['stage']['ending'],
                    'stop_status': result['stage']['stop_status'],
                    'notices': sum(len(a.get('diagnostic_per_action_changes', [])) for a in action_rows)})
                print('step2', len(cases), '/60', case['case_id'], condition, result['stage']['ending'], flush=True)
    finally:
        backend.close()
    freeze(ROOT/'step2/manifest.json', {'cases': cases, 'validated_setting': require_gate(),
        'design': ref(ROOT/'step2/design.json'), 'new_Luna_calls': 0, 'no_test_cohort': True})
    atomic_new(ROOT/'step2/execution_rows.json', summaries)
    completed = {'wall_seconds': time.monotonic()-started, 'episodes': len(cases), 'GPU': 0, 'new_API_calls': 0}
    atomic_new(ROOT/'step2/inference_complete.json', completed)
    lines = ['## 2. Local in-loop execution', '',
        f'Inference wall-clock: {completed["wall_seconds"]:.1f} s. Frozen Revision epoch1, temperature 0; 60 runs on C30.', '',
        'Both arms reuse the exact same first-round frozen Orchestrator epoch2 plan and MAP from saved '
        'FIX1 old-worker run1. All 30 plans were Revision delegations on the unchanged original C essay. '
        'The prompts, full document/profile, scope, six-step budget and guards are identical. '
        'Only cohesion-change observations differ: the with-notices arm sees changes from the immediately '
        'preceding action; the without arm removes cohesion notices throughout the observation/journal. '
        'Sentence-split/new-ID control notices remain in both arms. No Sol plan, new Orchestrator generation or new teacher call.', '',
        '| Cohort | Condition | Protocol / essays | STOP done | STOP blocked | Valid actions / actions | Phi notices generated |',
        '|---|---|---:|---:|---:|---:|---:|']
    for cohort in ('smoke20', 'agent_dev10'):
        for condition in design['conditions']:
            rs = [r for r in summaries if r['cohort'] == cohort and r['condition'] == condition]
            lines.append(f'| {cohort} | {condition} | {sum(r["no_protocol_error"] for r in rs)}/{len(rs)} | '
                f'{sum(r["ending"]=="STOP" and r["stop_status"]=="done" for r in rs)} | '
                f'{sum(r["ending"]=="STOP" and r["stop_status"]=="blocked" for r in rs)} | '
                f'{sum(r["valid_actions"] for r in rs)}/{sum(r["actions"] for r in rs)} | {sum(r["notices"] for r in rs)} |')
    with REPORT.open('a') as f:
        f.write('\n'+'\n'.join(lines)+'\n')


def gpu(part):
    from .run import server_command
    from verak.v4.full_run import idle
    require_gate()
    marker = ROOT/part/'inference_complete.json'
    if marker.exists():
        return
    check_runtime()
    hardware = idle((0, 1))
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', PORT))
    environment = {**os.environ, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '0',
        'VLLM_NO_USAGE_STATS': '1', 'DO_NOT_TRACK': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '8'}
    started = time.monotonic(); process = None
    try:
        import httpx
        with (ROOT/part/'vllm.log').open('a') as log:
            process = subprocess.Popen(server_command(), cwd=WORKTREE, env=environment, stdout=log, stderr=subprocess.STDOUT)
            write_json(ROOT/part/'server.json', {'pid': process.pid, 'hardware': hardware, 'training': False})
            deadline = time.monotonic()+900
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('VLLM failed; see saved log')
                try:
                    if httpx.get(f'http://127.0.0.1:{PORT}/health', timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError('VLLM readiness timed out')
            module = 'verak.v5.check_local' if part == 'step2' else 'verak.v5.check_a4'
            with (ROOT/part/'inference.log').open('a') as log:
                completed = subprocess.run([str(INFERENCE_PYTHON), '-u', '-m', module, 'infer'], cwd=WORKTREE,
                    env={**environment, 'CUDA_VISIBLE_DEVICES': '1'}, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError('Inference failed; see '+str(ROOT/part/'inference.log'))
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=10)
        write_json(ROOT/part/'server_stopped.json', {'pid': process.pid if process else None,
            'wall_seconds': time.monotonic()-started, 'training': False})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=['prepare', 'infer', 'gpu'])
    args = parser.parse_args()
    if args.part == 'prepare':
        print('Prepared', len(prepare()['cases']), 'paired plans without model calls')
    elif args.part == 'infer':
        infer()
    else:
        gpu('step2')
