"""Three-agent integration diagnostic, saved FIX1 controls, no training."""
from collections import Counter
from copy import deepcopy
import argparse
import os
from pathlib import Path
import socket
import subprocess
import time

from .common import (REPO, WORKTREE, ROOT as PIPE_ROOT, PORT, Backend, ReferenceScores,
    ADAPTERS, INFERENCE_PYTHON, ref, verified, read_json, write_json, atomic_new,
    file_sha, safe_id, tokenizer, public_row, load_config, local_analysis_environment, check_runtime)
from .check_judge import freeze
from .cooperate_env import VERSION
from verak.v4.fix1_infer import run_case, SETTINGS
from verak.v4.scale2_projection import restore_document

ROOT = REPO/'verak/v5/outputs/cooperate1'
REPORT = REPO/'reports/V5_COOPERATE1.md'
RUNS = SETTINGS[:3]


def prepare():
    if (ROOT/'design.json').exists():
        return read_json(ROOT/'design.json')
    from verak.v4 import orchestrator_v44, policy_prompts_v43
    orchestrator_v44.verify(); policy_prompts_v43.verify_frozen()
    tok = tokenizer()
    manifest_path = REPO/'verak/v4/outputs/dryrun/evaluation/manifest.json'
    manifest = read_json(manifest_path)
    cases = []
    for cohort, group in manifest['cohorts'].items():
        for c in group['cases']:
            row = public_row(verified(c['row']))
            row_path = ROOT/'public_rows'/(safe_id(c['case_id'])+'.json')
            freeze(row_path, row)
            old = {}
            for setting in RUNS:
                p = REPO/'verak/v4/outputs/fix1/evaluation/deterministic'/f'run{setting["run"]}'/'old'/cohort/'attempts'/(safe_id(c['case_id'])+'.json')
                value = read_json(p)
                if value['case_id'] != c['case_id'] or value['initial']['text'] != row['text']:
                    raise ValueError('FIX1 baseline input differs')
                old[str(setting['run'])] = ref(p)
            base = PIPE_ROOT/'C'/cohort/safe_id(c['case_id'])
            c_outputs = {n: ref(base/n/f) for n, f in [('onepass','onepass.json'), ('luna','edit_list.json')]}
            for r in c_outputs.values():
                if restore_document(verified(r)['initial']).text != row['text']:
                    raise ValueError('Saved C input differs')
            cases.append({'case_id': c['case_id'], 'cohort': cohort, 'row': ref(row_path),
                          'baselines': old, 'C_outputs': c_outputs})
    if Counter(c['cohort'] for c in cases) != {'smoke20':20, 'agent_dev10':10}:
        raise ValueError('Expected diagnostic 20+10')
    prior = read_json(REPO/'verak/v4/outputs/fix1/preflight.json')
    adapter_refs = {r: ref(p/'adapter_model.safetensors') for r,p in ADAPTERS.items() if r != 'oneshot'}
    for role, current in adapter_refs.items():
        # Provenance is checked independently of any path alias used in serving.
        info = read_json(ADAPTERS[role]/'provenance.json')
        if info['adapter_sha256'] != current['sha256'] or info['base_revision'] != prior['base_revision']:
            raise ValueError('Frozen adapter/base differs')
    code = [Path(__file__), Path(__file__).with_name('cooperate_env.py'),
            WORKTREE/'verak/v4/fix1_infer.py']
    design = {'version':VERSION, 'cases':cases, 'settings':RUNS,
        'evaluation_manifest':ref(manifest_path), 'baseline_preflight':ref(REPO/'verak/v4/outputs/fix1/preflight.json'),
        'adapters':adapter_refs, 'base_revision':prior['base_revision'],
        'worker_prompts':ref(policy_prompts_v43.FROZEN), 'orchestrator_prompt':ref(orchestrator_v44.FROZEN),
        'prompt_tokens':{r:len(tok.encode(policy_prompts_v43.prompt_for(r),add_special_tokens=False))
                         for r in ('revision','korean')},
        'orchestrator_prompt_tokens':len(tok.encode(orchestrator_v44.PROMPT,add_special_tokens=False)),
        'code':{str(p.relative_to(WORKTREE)):ref(p) for p in code},
        'training':False, 'test_cohort':False, 'a4_handoff':False, 'new_teacher_calls':0,
        'baseline':'Reuse all 90 saved FIX1 old-worker T0 runs, preserving failures and variation.',
        'intervention':'Same three frozen adapters and system prompts; full public-input Orchestrator plans. '
            'Restore complete worker reports/current-state handoff, require final-pass report review, '
            'keep full Korean scope and stage-level STOP, give state-specific rejected-action feedback without sentence bans.',
        'budgets':{'orchestrator_requests':20, 'model_delegations':6, 'revision_steps':6, 'korean_steps':14,
                   'rejections_in_a_row':3, 'Claude_usd':5},
        'completion':'No controller/context/step/format failure; reasoned model blocked is a protocol report, not task done.',
        'comparison_limit':'Bundle of integration repairs; three T0 runs estimate nondeterminism, not independent source samples.',
        'quality':'GPU reference Q and validated full-context Claude, with saved C onepass/Luna; separate from operations.',
        'cleanup':'Separate successor corpus before any retraining; historical diagnostic inputs and metrics unchanged.'}
    freeze(ROOT/'design.json',design)
    contract=deepcopy(read_json(REPO/'verak/v5/outputs/repair1/api_contract.json'))
    for key in ('prior_Claude_confirmed_usd','prior_accounting','cumulative_cap_usd','budget_amendment'):
        contract.pop(key,None)
    contract.update(scope='V5_COOPERATE1 saved-output quality evaluation only', total_cap_usd=5,
        user_approved=True, authorization='User explicitly approved an additional maximum $5 for full-context Claude quality comparison.',
        allowed_providers=['claude'])
    contract['claude']['cap_usd']=5; contract['luna']['cap_usd']=5
    freeze(ROOT/'api_contract.json',contract)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text('# V5_COOPERATE1 — 세 에이전트 협업 안정화\n\n설계 고정. 학습 없음; smoke20 + agent_dev10; test 미사용.\n\n'
        '저장된 FIX1 old 3회와 새 통합 실행 3회를 비교한다. a4 규칙 위임은 사용하지 않는다. '
        '총괄과 작업자 system prompt·가중치·행동/문맥 예산은 유지하고 인계·관찰·종료 제어만 수정한다.\n')
    return design


def verify_design():
    design=read_json(ROOT/'design.json')
    for r in list(design['code'].values())+list(design['adapters'].values())+[design['worker_prompts'],design['orchestrator_prompt']]:
        if file_sha(r['path']) != r['sha256']: raise ValueError('Frozen file changed: '+r['path'])
    return design


class RepeatedBackend(Backend):
    def __init__(self, setting):
        super().__init__(); self.setting=setting

    def generate(self,messages,role):
        from verak.v4.environment_v46 import OUTPUT
        started=time.monotonic()
        body={'model':'v5-'+role,'messages':messages,'temperature':0.,'top_p':1.,
              'seed':self.setting['seed'],'max_tokens':OUTPUT[role]}
        response=self.client.post(f'http://127.0.0.1:{PORT}/v1/chat/completions',json=body)
        response.raise_for_status(); data=response.json()
        return {'raw':data['choices'][0]['message']['content'],'usage':data.get('usage',{}),
                'finish_reason':data['choices'][0]['finish_reason'],'elapsed_s':time.monotonic()-started,
                'generation':{k:v for k,v in body.items() if k!='messages'}}


def output_dir(setting,case):
    return ROOT/'evaluation'/f'run{setting["run"]}'/case['cohort']


def infer():
    if (ROOT/'inference_complete.json').exists(): return
    from verak.v4.environment_v43 import BoundaryParagraphs
    started=time.monotonic(); design=verify_design(); local_analysis_environment()
    tok=tokenizer(); analysis=BoundaryParagraphs(load_config(),cache_dir=ROOT/'bareun')
    score=ReferenceScores(root=ROOT,fallback=PIPE_ROOT)
    try:
        for setting in RUNS:
            backend=RepeatedBackend(setting)
            try:
                for i,case in enumerate(design['cases'],1):
                    row=public_row(verified(case['row']))
                    result=run_case(row,case_id=case['case_id'],output=output_dir(setting,case),
                        backend=backend,tok=tok,analysis=analysis,score=score,cooperation=True,capture_states=True)
                    freeze(output_dir(setting,case)/'quality_scores'/(safe_id(case['case_id'])+'.json'), {
                        'original':score(row['question'],row['text']),
                        'final':score(row['question'],result['final']['text']), 'reference_GPU_only':True})
                    print(f'cooperate run{setting["run"]} {i}/30 {case["case_id"]} protocol={result["complete"]}',flush=True)
            finally: backend.close()
        # Saved comparisons receive this same reference scorer; no CPU values.
        for case in design['cases']:
            row=verified(case['row']); values={}
            for run,reference in case['baselines'].items():
                saved=verified(reference)
                values['old_run'+run]=score(row['question'],saved['final']['text'])
            for label,reference in case['C_outputs'].items():
                values[label]=score(row['question'],restore_document(verified(reference)['final']).text)
            freeze(ROOT/'comparison_scores'/(safe_id(case['case_id'])+'.json'),values)
        freeze(ROOT/'inference_complete.json',{'new_episodes':90,'saved_baselines':90,'saved_C_outputs':60,
            'wall_seconds':time.monotonic()-started,'new_API_calls':0,'training':False})
    finally: score.close()


def gpu():
    if (ROOT/'inference_complete.json').exists(): return
    from verak.v4.full_run import idle
    from .run import server_command
    verify_design(); hardware=idle((0,1)); freeze(ROOT/'search_runtime.json',check_runtime())
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',PORT))
    environment={**os.environ,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':'0',
        'VLLM_NO_USAGE_STATS':'1','DO_NOT_TRACK':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'8'}
    process=None;started=time.monotonic()
    try:
        import httpx
        with (ROOT/'vllm.log').open('a') as log:
            process=subprocess.Popen(server_command(),cwd=WORKTREE,env=environment,stdout=log,stderr=subprocess.STDOUT)
            write_json(ROOT/'server.json',{'pid':process.pid,'hardware':hardware,'training':False,'command':server_command()})
            deadline=time.monotonic()+900
            while time.monotonic()<deadline:
                if process.poll() is not None: raise RuntimeError('vLLM failed; see vllm.log')
                try:
                    if httpx.get(f'http://127.0.0.1:{PORT}/health',timeout=2).status_code==200: break
                except httpx.HTTPError: pass
                time.sleep(2)
            else: raise RuntimeError('vLLM readiness timeout')
            done=subprocess.run([str(INFERENCE_PYTHON),'-u','-m','verak.v5.cooperate','infer'],
                cwd=WORKTREE,env={**environment,'CUDA_VISIBLE_DEVICES':'1'})
            if done.returncode: raise RuntimeError('Integration inference failed; saved outputs retained')
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=45)
            except subprocess.TimeoutExpired: process.kill();process.wait(timeout=10)
        write_json(ROOT/'server_stopped.json',{'pid':process.pid if process else None,
            'wall_seconds':time.monotonic()-started,'training':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part',choices=['prepare','gpu','infer','report','judge'])
    part=parser.parse_args().part
    if part in {'report','judge'}:
        from . import cooperate_eval
        getattr(cooperate_eval,part)()
    else: globals()[part]()
