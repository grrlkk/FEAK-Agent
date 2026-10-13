"""Small Korean continuation on observed student rejections; no paid collection."""
from collections import Counter
from copy import deepcopy
import argparse
import gzip
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import time

from .common import (REPO, WORKTREE, PORT, ADAPTERS, INFERENCE_PYTHON, Backend,
    ReferenceScores, tokenizer, read_json, write_json, atomic_new, ref, verified,
    file_sha, safe_id, public_row, load_config, local_analysis_environment)
from .cooperate_env import CooperationEnvironment
from verak.v4.content_env import dumps
from verak.v4.fix1_data import load, encode, key, choose_diverse
from verak.v4.render_v43 import assert_rendered

ROOT = REPO/'verak/v5/outputs/recovery1'
CLEAN = REPO/'verak/v5/outputs/cooperate1/lconn_cleanup/sft_without_bad_conn'
SAMPLES = 384
REPLAY = 384


def prepare():
    if (ROOT/'design.json').exists():
        return read_json(ROOT/'design.json')
    old = read_json(REPO/'verak/v4/outputs/train44/data/manifest.json')
    rows = load(CLEAN/'korean.train.jsonl.gz')
    forbidden = set(old['shared_validation_groups'])
    excluded = set(old['test_exclusion']['source_ids'])
    for row in rows:
        assert row['partition'] == 'train' and row['source_group'] not in forbidden
        assert row['source_id'] not in excluded
    pools = {'EDIT': [], 'STOP': []}
    for i, row in enumerate(rows):
        payload = assert_rendered(row['messages'][:-1])
        target = json.loads(row['messages'][-1]['content'])
        if (payload['steps_left'] < 4 or row['prefix_tokens'] > 5800
                or payload.get('last_result', {}).get('controller_stop')):
            continue
        if target['action'] == 'STOP' and target.get('status') != 'done':
            continue
        if target['action'] in pools:
            pools[target['action']].append({**row, 'recovery_source_index': i})
    selected = []
    for action in ('EDIT', 'STOP'):
        selected += choose_diverse(pools[action], SAMPLES//2)
    selected.sort(key=key)
    coop = read_json(REPO/'verak/v5/outputs/cooperate1/design.json')
    cases = []
    for case in coop['cases']:
        baselines = {str(n): ref(REPO/'verak/v5/outputs/cooperate1/evaluation'/f'run{n}'/
                      case['cohort']/'attempts'/(safe_id(case['case_id'])+'.json')) for n in (1,2,3)}
        cases.append({**case, 'cooperate_baselines': baselines})
    design = {'version': 'v5_recovery1_20261013', 'created_at': time.time(),
        'training_source': ref(CLEAN/'korean.train.jsonl.gz'),
        'validation_source': ref(CLEAN/'korean.validation.jsonl.gz'),
        'source_manifest': ref(REPO/'verak/v4/outputs/train44/data/manifest.json'),
        'selected_indices': [s['recovery_source_index'] for s in selected],
        'selected_counts': dict(Counter(json.loads(s['messages'][-1]['content'])['action'] for s in selected)),
        'cases': cases, 'initial_adapter': ref(ADAPTERS['korean']/'adapter_model.safetensors'),
        'recipe': {'epochs': 1, 'learning_rate': 3e-5, 'replay_rows': REPLAY,
                   'recovery_rows': REPLAY, 'seed': 71, 'roles': ['korean']},
        'collection': 'Frozen cleaned training observations; at most two actual student rejections per seed. '
            'Reuse the accepted teacher next action only while the document remains exactly unchanged. '
            'Do not label any accepted student edit or controller-forced STOP as a correction.',
        'limits': 'This targets invalid-action recovery and stopping, not recovery from accepted semantic errors. '
            'Teacher labels are inherited, not independently rejudged. This is a diagnostic training pilot, not RFT success filtering.',
        'evaluation': 'Same 30 diagnostic inputs, one T0 run with only Korean weights changed; '
            'paired cooperate1 run1 primary and saved three-run variation reported. No test evaluation.',
        'new_paid_collection_calls': 0, 'test_cohort_untouched': True,
        'quality_evaluation': 'Whole-essay anonymous original/control/new comparison; separate source-grounding audit. '
            'External judge needs a new separately approved cap; old API authorization is not reused.'}
    atomic_new(ROOT/'design.json', design)
    return design


def restore_seed(sample, analysis):
    """Rebuild a training snapshot, preserving public history and actual text."""
    from verak.v3.corrupt.document import Document, Paragraph, Unit
    from verak.v4.environment_v43 import document_profile, ENVIRONMENT_VERSION
    p = assert_rendered(sample['messages'][:-1])
    rendering = p['rendering']; leading = dict(rendering['leading'])
    doc = Document([Paragraph(x['id'], [Unit(s['id'], s['text'], [], leading[s['id']])
                   for s in x['sentences']]) for x in p['paragraphs']],
                   list(rendering['gaps']), rendering['tail'], 'recovery1_refresh')
    analysis.refresh(doc, {x.pid for x in doc.paragraphs})
    row = {'source_id': sample['source_id'], 'genre': 'training_snapshot', 'question': p['question'],
           'text': doc.text, 'profile': document_profile(doc).to_dict(), 'layout': doc.snapshot(),
           'environment_version': ENVIRONMENT_VERSION,
           'normalization': {'lineage': {u.sid: {'parent_sid': u.sid, 'initial': True} for u in doc.units}}}
    env = CooperationEnvironment(row, analysis)
    env.start_korean(p.get('assigned_tasks', []))
    env.steps_used = env.step_limit-p['steps_left']
    env.actions = {role: [{k: deepcopy(v) for k,v in a.items() if k not in ('role','turn')}
                         for a in p['work_journal'] if a['role'] == role]
                   for role in ('revision','korean')}
    env.handoff = deepcopy(p.get('revision_handoff'))
    env.notices = deepcopy(p.get('notices', []))
    env.last_result = deepcopy(p.get('last_result', {}))
    env.relations = {r['source']: {k:v for k,v in r.items() if k != 'source'}
                     for r in p.get('declared_relations', [])}
    env.lineage.update(deepcopy(p.get('sentence_lineage', {})))
    env.worker_reports = deepcopy(p.get('cooperation',{}).get('previous_worker_reports',[]))
    env.search.history = deepcopy(p.get('search_history', []))
    env.search.successful_sourced_inserts = p.get('successful_sourced_inserts', 0)
    env.search.citations = {c['sentence_id']: {k:deepcopy(v) for k,v in c.items()
        if k not in ('sentence_id','text','citation_text')} for c in p.get('source_citations', [])}
    if 'shared_map' in p:
        env.shared_map = deepcopy(p['shared_map'])
    # The resumed stage has no recoverable prior undo stack. Never invent one.
    env.undo = []; env.citation_undo = []
    env.rejected_in_a_row = 0
    for a in reversed(env.actions['korean']):
        if a['valid']: break
        env.rejected_in_a_row += 1
    if env.rejected_in_a_row >= 2:
        raise ValueError('Seed already at controller stop boundary')
    expected = [(s['id'],s['text']) for para in p['paragraphs'] for s in para['sentences']]
    assert [(u.sid,u.text) for u in env.document.units] == expected
    return env


def collect():
    if (ROOT/'collection_complete.json').exists(): return
    from verak.v4.environment_v43 import BoundaryParagraphs
    from verak.v4.dryrun_student import prompt_token_audit
    design = prepare(); rows = load(verified_path(design['training_source']))
    local_analysis_environment(); tok = tokenizer()
    analysis = BoundaryParagraphs(load_config(), cache_dir=ROOT/'bareun')
    backend = Backend(); counts = Counter(); started = time.monotonic()
    try:
        for number, index in enumerate(design['selected_indices'], 1):
            path = ROOT/'collection'/f'{index:05d}.json'
            if path.exists(): continue
            sample = rows[index]; target = json.loads(sample['messages'][-1]['content'])
            value = {'source_index': index, 'source_id': sample['source_id'], 'target': target, 'calls': []}
            try:
                env = restore_seed(sample, analysis)
                # An inherited label is eligible only if the actual current environment accepts it.
                check = restore_seed(sample, analysis)
                accepted = check.step(dumps(target), 'korean')
                if not accepted['valid']: raise ValueError('Inherited teacher target rejected: '+accepted.get('error',''))
                original = env.document.text
                for step in range(2):
                    if env.closed or env.remaining <= 1: break
                    messages = env.public_messages('korean')
                    prefix = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                    if len(prefix)+1024 > 8192:
                        value['ending'] = 'training_context_limit'; break
                    response = backend.generate(messages, 'korean')
                    if prompt_token_audit(len(prefix), response.get('usage'))['status'] != 'match':
                        raise ValueError('Student serving prefix differs')
                    try:
                        proposed = json.loads(response['raw'])
                    except (ValueError, TypeError):
                        proposed = None
                    if isinstance(proposed, dict) and proposed.get('action') == 'UNDO':
                        value.update(ending='unrestorable_undo_history', unapplied_response=response)
                        break
                    action = env.step(response['raw'], 'korean')
                    call = {'messages': messages, 'response': response, 'action': deepcopy(action)}
                    if not action['valid'] and not env.closed and env.remaining:
                        assert env.document.text == original
                        call['recovery_messages'] = env.public_messages('korean')
                    value['calls'].append(call)
                    if action['valid']:
                        value['ending'] = 'accepted_student_action'; break
                value['status'] = 'completed'
            except ValueError as exc:
                value.update(status='excluded', error=str(exc))
            atomic_new(path, value)
            counts[value['status']] += 1
            if number % 8 == 0:
                print(f'recovery collection {number}/{len(design["selected_indices"])} {dict(counts)}', flush=True)
        atomic_new(ROOT/'collection_complete.json', {'seconds': time.monotonic()-started,
            'cases': len(design['selected_indices']), 'new_paid_calls': 0})
    finally: backend.close()


def verified_path(reference):
    if file_sha(reference['path']) != reference['sha256']:
        raise ValueError('Frozen data changed: '+reference['path'])
    return reference['path']


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for row in rows: stream.write(dumps(row)+'\n')


def build_data():
    if (ROOT/'data/manifest.json').exists(): return verify_data()
    if not (ROOT/'collection_complete.json').exists(): raise RuntimeError('Collection incomplete')
    design = prepare(); raw = load(verified_path(design['training_source'])); tok = tokenizer()
    corrections = []; audit = Counter(); seen = set()
    for index in design['selected_indices']:
        value = read_json(ROOT/'collection'/f'{index:05d}.json')
        audit['seeds_'+value['status']] += 1
        for call in value['calls']:
            if call['action']['valid']:
                audit['accepted_student_'+call['action']['action']] += 1; continue
            audit['actual_rejections'] += 1
            for name in ('messages','recovery_messages'):
                if name not in call: continue
                messages = call[name]; identity = dumps([messages,value['target']])
                if identity in seen: continue
                try:
                    row = encode(raw[index], messages, value['target'], tok,
                        {'kind': 'actual_student_'+name, 'source_index': index,
                         'target_evidence': 'unchanged document and accepted original teacher next action',
                         'collection': ref(ROOT/'collection'/f'{index:05d}.json')})
                except OverflowError:
                    audit['context_excluded'] += 1; continue
                row['recovery1'] = row.pop('fix1'); corrections.append(row); seen.add(identity)
    if len(corrections) < 16:
        raise RuntimeError(f'Only {len(corrections)} unique corrective states: insufficient for pilot')
    corrections.sort(key=key)
    # Fixed bounded mixture; this explicitly records repeated corrective examples.
    corrective = [deepcopy(corrections[i % len(corrections)]) for i in range(REPLAY)]
    pools = {a: [s for s in raw if json.loads(s['messages'][-1]['content'])['action'] == a]
             for a in ('EDIT','STOP')}
    replay = choose_diverse(pools['EDIT'], 3*REPLAY//4)+choose_diverse(pools['STOP'], REPLAY//4)
    training = corrective+deepcopy(replay)
    random.Random(71).shuffle(training)
    validation = load(verified_path(design['validation_source']))
    assert not ({s['source_group'] for s in training} & {s['source_group'] for s in validation})
    for part, rows in [('train',training),('validation',validation)]:
        save_rows(ROOT/f'data/korean.{part}.jsonl.gz', rows)
    manifest = {'roles': {'korean': {part: {**ref(ROOT/f'data/korean.{part}.jsonl.gz'), 'turns': len(rows)}
        for part,rows in [('train',training),('validation',validation)]}},
        'source_manifest': design['source_manifest'], 'clean_training_source': design['training_source'],
        'clean_validation_source': design['validation_source'], 'design': ref(ROOT/'design.json'),
        'unique_corrective_states': len(corrections), 'sampled_corrective_rows': len(corrective),
        'sampled_replay_rows': len(replay), 'audit': dict(audit),
        'actions': dict(Counter(json.loads(s['messages'][-1]['content'])['action'] for s in training)),
        'train_sources': len({s['source_group'] for s in training}), 'new_paid_calls': 0}
    atomic_new(ROOT/'data/manifest.json', manifest)
    print(dumps({k:v for k,v in manifest.items() if k not in ('roles',)}), flush=True)
    return manifest


def verify_data():
    m = read_json(ROOT/'data/manifest.json')
    for item in [m['design'],m['source_manifest'],m['clean_training_source'],m['clean_validation_source'],
                 *m['roles']['korean'].values()]: verified_path(item)
    for name in ('rollout_design','full_rollout_design','teacher_review','teacher_validation'):
        if m.get(name): verified_path(m[name])
    return m


def infer():
    if (ROOT/'inference_complete.json').exists(): return
    from verak.v4.environment_v43 import BoundaryParagraphs
    from verak.v4.fix1_infer import run_case
    design = prepare(); local_analysis_environment(); tok = tokenizer()
    analysis = BoundaryParagraphs(load_config(),cache_dir=ROOT/'bareun')
    score = ReferenceScores(root=ROOT,fallback=REPO/'verak/v5/outputs/cooperate1')
    backend = Backend(); started = time.monotonic()
    try:
        for number, case in enumerate(design['cases'],1):
            row = public_row(verified(case['row'])); output = ROOT/'evaluation/run1'/case['cohort']
            result = run_case(row,case_id=case['case_id'],output=output,backend=backend,tok=tok,
                              analysis=analysis,score=score,cooperation=True,capture_states=True)
            path = output/'quality_scores'/(safe_id(case['case_id'])+'.json')
            if not path.exists():
                atomic_new(path, {'original':score(row['question'],row['text']),
                    'final':score(row['question'],result['final']['text']),'reference_GPU_only':True})
            print(f'recovery evaluation {number}/30 {case["case_id"]} protocol={result["complete"]}',flush=True)
        atomic_new(ROOT/'inference_complete.json', {'episodes':30,'seconds':time.monotonic()-started})
    finally: backend.close(); score.close()


def gpu(part):
    from verak.v4.full_run import idle
    from .run import server_command
    marker = 'collection_complete.json' if part == 'collect' else 'inference_complete.json'
    if (ROOT/marker).exists(): return
    prepare(); hardware = idle((0,1))
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); sock.bind(('127.0.0.1',PORT))
    command = server_command()
    if part == 'infer':
        adapter = ROOT/'adapters/korean/epoch_1'
        if not (ROOT/'adapters/korean/complete.json').exists(): raise RuntimeError('Training incomplete')
        command = [('v5-korean='+str(adapter)) if x.startswith('v5-korean=') else x for x in command]
    environment = {**os.environ,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':'0',
        'VLLM_NO_USAGE_STATS':'1','DO_NOT_TRACK':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'8'}
    process = None
    try:
        import httpx
        with (ROOT/f'vllm_{part}.log').open('a') as log:
            process = subprocess.Popen(command,cwd=WORKTREE,env=environment,stdout=log,stderr=subprocess.STDOUT)
            write_json(ROOT/f'server_{part}.json',{'pid':process.pid,'hardware':hardware,'command':command})
            deadline = time.monotonic()+900
            while time.monotonic()<deadline:
                if process.poll() is not None: raise RuntimeError('vLLM startup failed')
                try:
                    if httpx.get(f'http://127.0.0.1:{PORT}/health',timeout=2).status_code == 200: break
                except httpx.HTTPError: pass
                time.sleep(2)
            else: raise RuntimeError('vLLM startup timeout')
            child = subprocess.run([str(INFERENCE_PYTHON),'-u','-m','verak.v5.recovery1',part],
                cwd=WORKTREE,env={**environment,'CUDA_VISIBLE_DEVICES':'1'})
            if child.returncode: raise RuntimeError(part+' failed; saved artifacts retained')
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=45)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
        write_json(ROOT/f'server_{part}_stopped.json',{'pid':process.pid if process else None,'at':time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part',choices=['prepare','collect','build_data','infer','gpu_collect','gpu_infer'])
    part = parser.parse_args().part
    if part.startswith('gpu_'): gpu(part[4:])
    else: globals()[part]()
