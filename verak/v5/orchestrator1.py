"""Replace only the COOPERATE1 Orchestrator with Luna on the fixed C30."""
from collections import Counter
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import time

from .api import PaidAPI, BudgetStop, credentials
from .check_judge import freeze
from .common import (REPO, WORKTREE, ROOT as PIPE_ROOT, PORT, ADAPTERS,
    INFERENCE_PYTHON, ReferenceScores, tokenizer, load_config, public_row,
    local_analysis_environment, check_runtime, ref, verified, read_json,
    write_json, file_sha, safe_id, sha_text)
from .cooperate import RepeatedBackend
from verak.v4.dryrun_student import prompt_token_audit
from verak.v4.fix1_infer import run_case, SETTINGS

PROBE_ROOT = REPO/'verak/v5/outputs/orchestrator1'
JSON_PROBE_ROOT = REPO/'verak/v5/outputs/orchestrator1_json'
ROOT = REPO/'verak/v5/outputs/orchestrator1_chat'
CONTROL = REPO/'verak/v5/outputs/cooperate1'
REPORT = REPO/'reports/V5_ORCHESTRATOR1.md'
SETTING = SETTINGS[0]
MODEL = 'gpt-6-luna'


def case_dir(case):
    return ROOT/'evaluation/run1'/case['cohort']


def prepare():
    if (ROOT/'design.json').exists():
        return verify_design()
    from verak.v4 import orchestrator_v44, policy_prompts_v43
    from verak.v4.environment_v46 import CONTEXT, OUTPUT
    from .recovery1_eval import PROMPT
    orchestrator_v44.verify(); policy_prompts_v43.verify_frozen()
    previous = read_json(CONTROL/'design.json')
    cases = []
    for case in previous['cases']:
        row = public_row(verified(case['row']))
        row_path = ROOT/'public_rows'/(safe_id(case['case_id'])+'.json')
        freeze(row_path, row)
        controls = {}
        for run in (1, 2, 3):
            base = CONTROL/'evaluation'/f'run{run}'/case['cohort']
            episode = base/'attempts'/(safe_id(case['case_id'])+'.json')
            if read_json(episode)['initial']['text'] != row['text']:
                raise ValueError('Saved control input differs')
            controls[str(run)] = {'episode': ref(episode),
                'scores': ref(base/'quality_scores'/(safe_id(case['case_id'])+'.json'))}
        cases.append({'case_id':case['case_id'], 'cohort':case['cohort'],
                      'row':ref(row_path), 'controls':controls})
    if Counter(c['cohort'] for c in cases) != {'smoke20':20, 'agent_dev10':10}:
        raise ValueError('Expected the same 20+10 diagnostic cases')
    old = read_json(PIPE_ROOT/'api_contract.json')
    prior = {p.name:ledger_accounting(p) for p in (PROBE_ROOT, JSON_PROBE_ROOT)}
    spent = sum(p['confirmed_usd']+p['reserved_usd'] for p in prior.values())
    # Fresh authorization: do not inherit any previous experiment's approval.
    contract = {'user_approved':True,
        'authorization':'User approved replacing only the Orchestrator with Luna on the same 30 essays: 그래 그걸로 해보자.',
        'cap_provenance':'Agent-set conservative $3 ceiling, disclosed before execution; not a user-specified amount.',
        'scope':'ORCHESTRATOR1 Luna Orchestrator C30 and blinded final-essay comparison only; no training.',
        'allowed_providers':['luna','claude'], 'total_cap_usd':3.-spent,
        'prior_format_probe':prior,
        'combined_ceiling_usd':3.,
        'luna':deepcopy(old['luna']), 'claude':deepcopy(old['claude']),
        'documentation_checked':'2026-10-14',
        'documentation':['https://developers.openai.com/api/docs/models/gpt-6-luna',
                         'https://developers.openai.com/api/docs/guides/latest-model']}
    contract['luna'].update(model=MODEL, cap_usd=1.-spent, reasoning_effort='none', temperature=0,
                            text_format={'type':'json_object'}, endpoint='chat_completions')
    contract['claude']['cap_usd'] = 2.
    for spec in (contract['luna'], contract['claude']):
        spec.pop('expected_usd', None)
    freeze(ROOT/'api_contract.json', contract)
    code = [Path(__file__), WORKTREE/'verak/v5/api.py', WORKTREE/'verak/v5/common.py',
        WORKTREE/'verak/v5/run.py', WORKTREE/'verak/v5/cooperate.py',
        WORKTREE/'verak/v5/cooperate_env.py', WORKTREE/'verak/v4/fix1_infer.py',
        WORKTREE/'verak/v4/orchestrator_v44.py', WORKTREE/'verak/v4/maps_v45.py',
        WORKTREE/'verak/v4/content_env.py', WORKTREE/'verak/v4/render_v43.py',
        WORKTREE/'verak/v4/search_v43.py', WORKTREE/'verak/v4/policy_prompts_v43.py']
    code += sorted((WORKTREE/'verak/v4').glob('environment_v4*.py'))
    adapters = {r:ref(ADAPTERS[r]/'adapter_model.safetensors') for r in ('orchestrator','revision','korean')}
    for role, artifact in adapters.items():
        if artifact['sha256'] != previous['adapters'][role]['sha256']:
            raise ValueError('COOPERATE1 adapter changed')
    design = {'experiment':'ORCHESTRATOR1', 'cases':cases, 'setting':SETTING,
        'control_design':ref(CONTROL/'design.json'), 'adapters':adapters,
        'worker_prompts':ref(policy_prompts_v43.FROZEN),
        'orchestrator_prompt':ref(orchestrator_v44.FROZEN),
        'code':{str(p.relative_to(WORKTREE)):ref(p) for p in code},
        'api_contract':ref(ROOT/'api_contract.json'), 'api_model':MODEL,
        'context':CONTEXT, 'output':OUTPUT, 'context_basis':'Unchanged Kanana-token content admission; provider token counts logged separately, never equated.',
        'intervention':'Only Orchestrator generation uses Luna through Chat Completions with JSON-object mode for the existing single-action protocol. Same system prompt, public observations, MAP validation, tools, local SEARCH, delegation and step budgets, worker models and generation settings.',
        'format_diagnostics':[ref(p/'format_diagnostic.json') for p in (PROBE_ROOT,JSON_PROBE_ROOT)
                              if (p/'format_diagnostic.json').exists()],
        'primary_control':'Saved COOPERATE1 run1, seed47; runs2/3 describe variability, not independent essays.',
        'sampling':'Luna reasoning none, temperature0, no API seed; local workers temperature0/top_p1/seed47.',
        'evaluation':{'prompt':PROMPT, 'prompt_sha256':sha_text(PROMPT),
            'seed':731014, 'orders':2, 'primary':'First randomized order; reverse order is a sensitivity check, not an independent evaluator.',
            'rubrics':'Whole-essay preference, original-relative improvement, meaning preservation, and source support.',
            'retry':'One retry only for malformed completed JSON. No selective semantic retries.',
            'identity_check':'Original-identical variants cannot count as improvements; identical variants cannot have a strict preference.'},
        'training':False, 'test_cohort':False, 'new_episodes':30,
        'limits':['Reused diagnostic C30, one new inference run.',
                  'Prompt and observation control does not equate different model tokenizers or sampling implementations.',
                  'Luna additionally uses API JSON-object syntax control; this is not an unconstrained model-only swap.',
                  'LLM judge is not human gold; protocol completion is not writing quality.']}
    freeze(ROOT/'design.json', design)
    return design


def verify_design():
    design = read_json(ROOT/'design.json')
    artifacts = list(design['code'].values())+list(design['adapters'].values())
    artifacts += [design[k] for k in ('worker_prompts','orchestrator_prompt','api_contract')]
    for artifact in artifacts:
        if file_sha(artifact['path']) != artifact['sha256']:
            raise ValueError('Frozen artifact changed: '+artifact['path'])
    return design


class HybridBackend:
    """Forward worker requests byte-for-byte; only the Orchestrator uses API."""
    def __init__(self, local, api, case_id, *, expected_first_messages=None):
        if api.model != MODEL:
            raise ValueError('The requested Orchestrator model must stay Luna')
        self.local = local; self.api = api; self.case_id = case_id
        self.expected_first_messages = expected_first_messages
        self.orchestrator_calls = 0; self.budget_stop = False

    def generate(self, messages, role):
        if role in {'revision','korean'}:
            return self.local.generate(messages, role)
        if role != 'orchestrator':
            raise ValueError('Unknown policy role')
        if not self.orchestrator_calls and self.expected_first_messages is not None:
            if messages != self.expected_first_messages:
                raise ValueError('First Orchestrator observation differs from saved control')
        self.orchestrator_calls += 1
        identity = f'ORCHESTRATOR1:{self.case_id}:o{self.orchestrator_calls:02d}'
        with self.api.db() as db:
            previous = {r[0] for r in db.execute('SELECT fingerprint FROM calls')}
        from verak.v4.environment_v46 import OUTPUT
        try:
            response = self.api.request(messages, item_id=identity, max_output=OUTPUT[role])
        except BudgetStop:
            self.budget_stop = True
            raise
        return {**response, 'new_paid_call':response['fingerprint'] not in previous,
            'generation':{'model':self.api.model, 'provider':'openai',
                'reasoning_effort':self.api.spec['reasoning_effort'],
                'temperature':self.api.spec['temperature'], 'text_format':self.api.spec.get('text_format'),
                'endpoint':self.api.spec.get('endpoint','responses'),
                'max_output_tokens':OUTPUT[role]}}

    def audit_prompt_tokens(self, role, local_count, response):
        if role != 'orchestrator':
            return prompt_token_audit(local_count, response.get('usage'))
        usage = response.get('usage', {})
        valid = (response.get('response_model') == MODEL
                 and response.get('generation', {}).get('provider') == 'openai'
                 and type(usage.get('input_tokens')) is int and usage['input_tokens'] > 0)
        return {'status':'different_tokenizer' if valid else 'invalid_api_provenance',
            'admission_tokenizer':'frozen Kanana', 'admission_prefix_tokens':local_count,
            'provider_model':response.get('response_model'), 'provider_input_tokens':usage.get('input_tokens'),
            'counts_are_equivalent':False}


def infer():
    if (ROOT/'inference_complete.json').exists(): return
    from verak.v4.environment_v43 import BoundaryParagraphs
    design = verify_design(); local_analysis_environment(); started = time.monotonic()
    tok = tokenizer(); analysis = BoundaryParagraphs(load_config(), cache_dir=ROOT/'bareun')
    scorer = ReferenceScores(root=ROOT, fallback=PIPE_ROOT)
    api = PaidAPI('luna', root=ROOT); local = RepeatedBackend(SETTING)
    try:
        for i, case in enumerate(design['cases'], 1):
            row = public_row(verified(case['row']))
            control = verified(case['controls']['1']['episode'])
            first = next(c for c in control['calls'] if c['role']=='orchestrator')
            expected = verified(first['prefix'])['messages']
            backend = HybridBackend(local, api, case['case_id'], expected_first_messages=expected)
            result = run_case(row, case_id=case['case_id'], output=case_dir(case),
                backend=backend, tok=tok, analysis=analysis, score=scorer,
                cooperation=True, capture_states=True)
            freeze(case_dir(case)/'quality_scores'/(safe_id(case['case_id'])+'.json'), {
                'original':scorer(row['question'], row['text']),
                'final':scorer(row['question'], result['final']['text']), 'reference_GPU_only':True})
            write_json(ROOT/'progress.json', {'completed':i, 'planned':30, 'case_id':case['case_id'],
                'protocol_complete':result['complete'], 'at':time.time()})
            print(f'ORCHESTRATOR1 {i}/30 {case["case_id"]} protocol={result["complete"]}', flush=True)
            if backend.budget_stop:
                freeze(ROOT/'budget_stop.json', {'case_id':case['case_id'], 'at':time.time()})
                raise BudgetStop('Luna budget exhausted; unstarted essays retained as pending')
        freeze(ROOT/'inference_complete.json', {'episodes':30, 'wall_seconds':time.monotonic()-started,
            'training':False, 'orchestrator_model':MODEL, 'worker_models_unchanged':True})
    finally:
        api.close(); local.close(); scorer.close()


def probe():
    """Validate the first real request before allocating GPUs; replay it later."""
    design = verify_design(); case = design['cases'][0]
    control = verified(case['controls']['1']['episode'])
    first = next(c for c in control['calls'] if c['role']=='orchestrator')
    messages = verified(first['prefix'])['messages']
    api = PaidAPI('luna', root=ROOT)
    try:
        backend = HybridBackend(None, api, case['case_id'], expected_first_messages=messages)
        response = backend.generate(messages, 'orchestrator')
        action = json.loads(response['raw'])
        if action != {'action':'SCORE'}:
            raise ValueError('First response must be exactly one SCORE action')
        freeze(ROOT/'format_preflight.json', {'case_id':case['case_id'], 'action':action,
            'fingerprint':response['fingerprint'], 'input_prefix':first['prefix'],
            'same_first_request_reused_during_inference':True})
        print('Luna single-action API preflight passed; cached response will be reused.', flush=True)
    finally: api.close()


def gpu():
    if (ROOT/'inference_complete.json').exists(): return
    from verak.v4.full_run import idle
    from .run import server_command
    verify_design()
    if not all(credentials().values()):
        raise RuntimeError('A required API credential is unavailable; no model request sent')
    probe()
    hardware = idle((0,1))
    runtime = check_runtime()
    write_json(ROOT/'search_runtime_latest.json', runtime)
    if not (ROOT/'search_runtime.json').exists():
        freeze(ROOT/'search_runtime.json', runtime)
    with socket.socket() as port_probe:
        port_probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port_probe.bind(('127.0.0.1', PORT))
    environment = {**os.environ, 'CUDA_DEVICE_ORDER':'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES':'0',
        'VLLM_NO_USAGE_STATS':'1', 'DO_NOT_TRACK':'1', 'HF_HUB_OFFLINE':'1',
        'TRANSFORMERS_OFFLINE':'1', 'TOKENIZERS_PARALLELISM':'false', 'OMP_NUM_THREADS':'8'}
    process = None; started = time.monotonic()
    try:
        import httpx
        with (ROOT/'vllm.log').open('a') as log:
            process = subprocess.Popen(server_command(), cwd=WORKTREE, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            write_json(ROOT/'server.json', {'pid':process.pid, 'command':server_command(), 'hardware':hardware})
            deadline = time.monotonic()+900
            while time.monotonic() < deadline:
                if process.poll() is not None: raise RuntimeError('Owned vLLM failed; inspect log')
                try:
                    if httpx.get(f'http://127.0.0.1:{PORT}/health', timeout=2).status_code == 200: break
                except httpx.HTTPError: pass
                time.sleep(2)
            else: raise RuntimeError('vLLM readiness timeout')
            completed = subprocess.run([str(INFERENCE_PYTHON), '-u', '-m', 'verak.v5.orchestrator1', 'infer'],
                cwd=WORKTREE, env={**environment, 'CUDA_VISIBLE_DEVICES':'1'})
            if completed.returncode: raise RuntimeError('Inference incomplete; saved outputs preserved')
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=45)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
        write_json(ROOT/'server_stopped.json', {'pid':process.pid if process else None,
            'at':time.time(), 'wall_seconds':time.monotonic()-started})


def ledger_accounting(root):
    ledger = root/'api/ledger.sqlite'
    if not ledger.exists(): return {'calls':0, 'confirmed_usd':0., 'reserved_usd':0.}
    with sqlite3.connect(f'file:{ledger}?mode=ro', uri=True) as db:
        rows = db.execute('SELECT kind,status,reserved,confirmed FROM calls').fetchall()
    return {'calls':len(rows), 'confirmed_usd':sum(r[3] for r in rows), 'reserved_usd':sum(r[2] for r in rows),
        'providers':{kind:{'calls':sum(r[0]==kind for r in rows),
            'confirmed_usd':sum(r[3] for r in rows if r[0]==kind),
            'reserved_usd':sum(r[2] for r in rows if r[0]==kind)} for kind in ('luna','claude')}}


def accounting():
    runs = {root.name:ledger_accounting(root) for root in (PROBE_ROOT, JSON_PROBE_ROOT, ROOT)}
    return {'calls':sum(r['calls'] for r in runs.values()),
        'confirmed_usd':sum(r['confirmed_usd'] for r in runs.values()),
        'reserved_usd':sum(r['reserved_usd'] for r in runs.values()), 'runs':runs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=['prepare','probe','gpu','infer','judge','report'])
    part = parser.parse_args().part
    if part in {'judge','report'}:
        from . import orchestrator1_eval
        getattr(orchestrator1_eval, part)()
    else:
        globals()[part]()
