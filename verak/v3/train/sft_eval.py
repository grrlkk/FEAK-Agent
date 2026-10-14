"""Same-cohort base/epoch/teacher evaluation with isolated paid-call caps."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import json
from pathlib import Path
import time

from feak_tc.runtime.openai import CallBudgetExceeded
from ..agent.backends import PolicyBackend
from ..agent.runner import run_episode, system_prompt
from ..common import file_sha, read_json, sha_text, write_json
from ..env import RevisionEnv
from ..eval.measurement import changes
from ..eval.resources import Resources
from ..phase2 import read_jsonl
from ..view_data import load_episode_examples
from .pilot import safe_id, stratified_order
from .sft_data import PHASE, REVISION, ROLES, config_for
from .teacher_bulk import BulkAPI, BulkTeacher, atomic_new, collection_lock, load_environment

CONDITIONS = ('base', 'epoch_1', 'epoch_2', 'luna_low')


def action_kind(action):
    name = action.get('action', 'PARSE_ERROR')
    args = action.get('args') or {}
    if name == 'EDIT':
        target, text = args.get('target'), args.get('new_text')
        if not isinstance(target, str) or not isinstance(text, str):
            return 'malformed_EDIT'
        if target.startswith(('before:', 'after:')):
            return 'sentence_insert'
        if ':' not in target and not text.strip():
            return 'sentence_delete'
        return 'in_sentence_EDIT'
    return name


def eval_config():
    config = config_for()
    root = config['paths'][PHASE + '_output']
    model = read_json(config['paths']['phase4_output'] / 'models.json')['luna_model']
    for suffix, model_name, ceiling in [('luna', model, 8000), ('sol', 'gpt-6.1-sol', 2000)]:
        key = PHASE + '_' + suffix
        config[key] = {'model': model_name, 'max_cost_usd': 3.,
                       'max_concurrent_requests': 4, 'phase_api_ceiling': ceiling}
        config['paths'][key + '_output'] = root / suffix
    return config


def valid_saved_luna(row, original, tokenizer, model):
    if row.get('model') != model or row.get('mode') != 'two_stage':
        return False
    if row.get('observation_setting', 'current') != 'current' or row.get('prompt_variant') != 'default':
        return False
    if row.get('initial_layout') != original['corrupted_layout']:
        return False
    for call in row.get('calls', []):
        if (call.get('reasoning_effort') != 'low' or call.get('max_output_tokens') != 1024 or
                call['messages'][0]['content'] != system_prompt(call['role'])):
            return False
        if len(tokenizer.apply_chat_template(call['messages'], tokenize=True, add_generation_prompt=True)) > 7168:
            return False
    return bool(row.get('calls'))


def prepare(config):
    root = config['paths'][PHASE + '_output']
    path = config['paths']['active_corrupt'] / 'agent_dev.jsonl'
    corpus = {r['episode_id']: r for r in read_jsonl(path)}
    ids = stratified_order(list(corpus.values()), seed=73)[:100]
    if Counter(corpus[i]['level'] for i in ids) != {level: 25 for level in ('L1', 'L2', 'L3', 'L4')}:
        raise ValueError('Evaluation requires 25 dev essays per curriculum level')
    real_ids = read_json(config['paths']['phase6_output'] / 'design.json')['real_ids']
    examples = {e.id: e for e in load_episode_examples(config, 'agent_dev')}
    if not set(real_ids) <= examples.keys() or any(corpus[i]['source_id'] not in examples for i in ids):
        raise ValueError('Evaluation may only read authorized agent_dev inputs')
    source_train = {r['source_id'] for r in read_jsonl(config['paths']['active_corrupt'] / 'agent_train.jsonl')}
    if source_train & ({corpus[i]['source_id'] for i in ids} | set(real_ids)):
        raise ValueError('Training/evaluation source overlap')
    contract = {'base_revision': REVISION, 'dev_corpus_sha256': file_sha(path), 'seed': 73,
        'dev_ids': ids, 'real_ids': real_ids, 'counts': {'L1': 25, 'L2': 25, 'L3': 25, 'L4': 25},
        'dev_unique_sources': len({corpus[i]['source_id'] for i in ids}),
        'conditions': list(CONDITIONS), 'context_limit': 8192, 'output_limit': 1024,
        'mode': 'two_stage', 'enable_check': False, 'policy_temperature': 0., 'policy_seed': 47,
        'workers': 4, 'paired_adapters': {'epoch_1': 1, 'epoch_2': 2},
        'luna_model': config[PHASE + '_luna']['model'], 'luna_cap_usd': 3., 'sol_cap_usd': 3.,
        'prompt_sha256': {role: sha_text(system_prompt(role)) for role in ROLES},
        'runtime_sha256': {str(p): file_sha(p) for p in
            [config['paths']['repo'] / 'verak/v3' / f for f in
             ('agent/runner.py', 'env/environment.py', 'env/observation.py', 'reward/total.py')]
             if p.is_file()}}
    target = root / 'evaluation_design.json'
    if target.exists():
        design = read_json(target)
        if design['contract'] != contract:
            raise ValueError('Frozen evaluation design changed')
        for entry in design['reuse'].values():
            if file_sha(Path(entry['path'])) != entry['sha256']:
                raise ValueError('Reused baseline changed')
        return design, corpus, examples
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    reuse = {}
    directories = sorted(p for p in (root.parent).glob('**/episodes') if root not in p.parents)
    candidates = []
    for episode_id in ids:
        name = safe_id(episode_id) + '.json'
        for directory in directories:
            candidate = directory / name
            if not candidate.exists():
                continue
            row = read_json(candidate)
            candidates.append({'path': str(candidate), 'model': row.get('model')})
            if valid_saved_luna(row, corpus[episode_id], tokenizer, contract['luna_model']):
                reuse[episode_id] = {'path': str(candidate), 'sha256': file_sha(candidate),
                                    'events': str(directory.parent / 'events' / (safe_id(episode_id) + '.jsonl'))}
                break
    baseline = root.parent / 'observation_test/current'
    for episode_id in real_ids:
        candidate = baseline / 'episodes' / (safe_id(episode_id) + '.json')
        row = read_json(candidate)
        if (row['model'] != contract['luna_model'] or row['observation_setting'] != 'current' or
                any(c['messages'][0]['content'] != system_prompt(c['role']) for c in row['calls'])):
            raise ValueError('Real Luna baseline is not the accepted setting (a)')
        reuse[episode_id] = {'path': str(candidate), 'sha256': file_sha(candidate),
            'events': str(baseline / 'events' / (safe_id(episode_id) + '.jsonl'))}
    design = {'contract': contract, 'reuse': reuse, 'saved_dev_candidates_checked': candidates,
              'created_unix': time.time()}
    atomic_new(target, design)
    return design, corpus, examples


class CachedPolicy(PolicyBackend):
    def generate(self, messages, *, episode_id, role, turn):
        fingerprint = sha_text(json.dumps({'messages': messages, 'episode_id': episode_id,
            'role': role, 'turn': turn, 'settings': self.settings}, sort_keys=True, ensure_ascii=False))
        cache = self.output / 'cache' / (fingerprint + '.json')
        if cache.exists():
            return {**read_json(cache), 'replayed': True}
        result = super().generate(messages, episode_id=episode_id, role=role, turn=turn)
        atomic_new(cache, result)
        return result


def execute(config, condition, *, max_api_calls=0, limit=None):
    design, corpus, examples = prepare(config)
    root = config['paths'][PHASE + '_output']
    output = root / 'evaluation' / condition
    with collection_lock(output):
        load_environment(config)
        if condition != 'luna_low':
            config = deepcopy(config)
            config['policy']['adapters'] = {} if condition == 'base' else {
                role: f'sft-{role}-ep{condition[-1]}' for role in ROLES}
            adapter_metadata = {}
            if condition != 'base':
                for role in ROLES:
                    path = root / 'adapters' / role / condition
                    metadata = read_json(path / 'provenance.json')
                    if (metadata['base_revision'] != REVISION or
                            metadata['adapter_sha256'] != file_sha(path / 'adapter_model.safetensors')):
                        raise ValueError('Evaluation checkpoint changed')
                    adapter_metadata[role] = metadata
            write_json(output / 'adapters.json', adapter_metadata)
        api = BulkAPI(config, max_api_calls, phase=PHASE + '_luna') if condition == 'luna_low' else None
        if api:
            api.settle_interrupted()
        resources = Resources(config, examples, output_key=PHASE + '_output')
        requested = design['contract']['dev_ids'] + design['contract']['real_ids']
        tasks = [i for i in requested if not (condition == 'luna_low' and i in design['reuse'])]
        if limit:
            tasks = tasks[:limit]
        stop_reason, errors, completed = 'all_requested_attempted', [], 0
        started = time.monotonic()
        def one(episode_id):
            path = output / 'episodes' / (safe_id(episode_id) + '.json')
            if path.exists():
                return read_json(path)
            state = resources.worker()
            row = corpus.get(episode_id)
            if row:
                episode = {k: row[k] for k in ('episode_id', 'source_id', 'genre', 'level', 'question',
                    'records', 'corrupted_score', 'preexisting_spell_spans')}
                episode.update(source=resources.source(row['source_id']), document=resources.corrupted(row))
            else:
                ex = examples[episode_id]
                episode = {'episode_id': episode_id, 'source_id': episode_id, 'genre': ex.genre,
                           'question': ex.question, 'document': resources.source(episode_id)}
            backend = (BulkTeacher(api, 'sft_dev', state.tokenizer) if api else
                       CachedPolicy(config, output / 'calls' / safe_id(episode_id)))
            env = RevisionEnv(config, analysis=state.analysis, scorer=resources,
                              similarity=resources.similarity, tokenizer=state.tokenizer)
            try:
                result = run_episode(env, episode, backend,
                    event_path=output / 'events' / (safe_id(episode_id) + '.jsonl'))
            finally:
                if not api:
                    backend.close()
            result.update(condition=condition, cohort='dev' if row else 'real',
                          seen_by_scorer=examples[episode['source_id']].seen_by_scorer)
            # Save generation before terminal-only measurements.
            write_json(path, result)
            try:
                acts = [a for values in result['actions_by_role'].values() for a in values]
                result['measurement'] = changes(env.corrupted, env.document, acts)
                if not row and result['completed']:
                    before = resources.score(episode['question'], env.corrupted.text)
                    after = resources.score(episode['question'], env.document.text)
                    result['quality_measurement'] = {'before': before, 'after': after,
                                                    'delta_q': after['mean'] - before['mean']}
            except Exception as exc:
                result['measurement_error'] = {'type': type(exc).__name__, 'message': str(exc)}
            write_json(path, result)
            print(json.dumps({'condition': condition, 'id': episode_id, 'completed': result['completed'],
                'steps': result['steps'], 'error': result['runtime_error'],
                'usd': api.accounting()['confirmed_usd'] if api else 0.}, ensure_ascii=False), flush=True)
            return result
        try:
            iterator = iter(tasks)
            with ThreadPoolExecutor(max_workers=4) as pool:
                active = {pool.submit(one, i): i for i in [next(iterator, None) for _ in range(4)] if i}
                stopped = False
                while active:
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        item = active.pop(future)
                        try:
                            result = future.result()
                            completed += 1
                            if (result.get('runtime_error') or {}).get('type') == 'CallBudgetExceeded':
                                stopped, stop_reason = True, 'budget_cap'
                        except Exception as exc:
                            errors.append({'id': item, 'type': type(exc).__name__, 'message': str(exc)})
                            stopped, stop_reason = True, 'worker_failure'
                        following = None if stopped else next(iterator, None)
                        if following:
                            active[pool.submit(one, following)] = following
        finally:
            resources.close()
            if api:
                api.close()
            write_json(output / 'run_status.json', {'condition': condition, 'requested': len(tasks),
                'finished_this_invocation': completed, 'stop_reason': stop_reason, 'errors': errors,
                'elapsed_s': time.monotonic() - started, 'budget': api.accounting() if api else None})


def saved_rows(config, condition):
    design, _, _ = prepare(config)
    root = config['paths'][PHASE + '_output']
    for episode_id in design['contract']['dev_ids'] + design['contract']['real_ids']:
        entry = design['reuse'].get(episode_id) if condition == 'luna_low' else None
        path = Path(entry['path']) if entry else root / 'evaluation' / condition / 'episodes' / (safe_id(episode_id) + '.json')
        if path.exists():
            row = read_json(path)
            row.update(condition=condition, cohort='dev' if episode_id in design['contract']['dev_ids'] else 'real')
            events = Path(entry['events']) if entry else path.parent.parent / 'events' / (safe_id(episode_id) + '.jsonl')
            yield row, path, events
