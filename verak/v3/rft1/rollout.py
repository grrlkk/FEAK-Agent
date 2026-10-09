"""Four frozen, resumable samples per active training essay, with unchanged v1 tools."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import json
from pathlib import Path
import time

from ..agent.backends import PolicyBackend, append_jsonl
from ..agent.runner import run_episode, system_prompt
from ..common import file_sha, read_json, sha_text, write_json
from ..env import RevisionEnv
from ..eval.resources import Resources
from ..phase2 import read_jsonl
from ..reward.total import rewards
from ..train.pilot import safe_id, stratified_order
from ..train.teacher_bulk import atomic_new, collection_lock
from ..view_data import load_episode_examples
from .config import PHASE, ROLES, REVISION, load_environment, runtime_hashes


def request_settings(episode_id, sample, *, evaluation=False):
    seed = 47 if evaluation else (83 + sample * 1000003 + int(sha_text(episode_id)[:8], 16)) % (2**31)
    return {'temperature': 0. if evaluation else .7, 'top_p': 1. if evaluation else .95,
            'seed': seed, 'max_tokens': 1024}


class SamplePolicy(PolicyBackend):
    def __init__(self, config, output, *, sample, evaluation=False):
        super().__init__(config, output)
        self.sample, self.evaluation = sample, evaluation

    def generate(self, messages, *, episode_id, role, turn):
        model = self.settings['adapters'][role]
        settings = request_settings(episode_id, self.sample, evaluation=self.evaluation)
        payload = {'model': model, 'messages': deepcopy(messages), **settings}
        identity = {'payload': payload, 'episode_id': episode_id, 'role': role, 'turn': turn,
                    'sample': self.sample, 'evaluation': self.evaluation}
        fingerprint = sha_text(json.dumps(identity, ensure_ascii=False, sort_keys=True))
        cache = self.output / 'cache' / (fingerprint + '.json')
        if cache.exists():
            return {**read_json(cache), 'replayed': True}
        started = time.monotonic()
        response = self.client.post(self.settings['base_url'] + '/chat/completions', json=payload)
        response.raise_for_status()
        value = response.json()
        result = {'episode_id': episode_id, 'role': role, 'turn': turn, 'model': model,
            'sample': self.sample, 'sampling': settings, 'messages': payload['messages'],
            'raw': value['choices'][0]['message']['content'], 'usage': value.get('usage', {}),
            'finish_reason': value['choices'][0]['finish_reason'], 'response_id': value.get('id'),
            'elapsed_s': time.monotonic()-started, 'cost': {'confirmed_usd': 0},
            'replayed': False, 'fingerprint': fingerprint}
        atomic_new(cache, result)
        append_jsonl(self.output / 'policy_calls.jsonl', result)
        return result


def prepare(config):
    root = config['paths'][PHASE + '_output']
    source = config['paths']['active_corrupt'] / 'agent_train.jsonl'
    corpus = {r['episode_id']: r for r in read_jsonl(source)}
    if len(corpus) != 1430 or any(r['split'] != 'agent_train' for r in corpus.values()):
        raise ValueError('Exactly all 1,430 active agent_train essays are required')
    if any(r['op'] in {'G_DELETE_SUPPORT', 'L_CONJ_DROP'} for e in corpus.values() for r in e['records']):
        raise ValueError('Inactive operator in RFT corpus')
    sft = config['paths']['phase7_sft_output']
    evaluation = read_json(sft / 'evaluation_design.json')
    dev = {r['episode_id']: r for r in read_jsonl(config['paths']['active_corrupt'] / 'agent_dev.jsonl')}
    excluded = {dev[i]['source_id'] for i in evaluation['contract']['dev_ids']} | set(evaluation['contract']['real_ids'])
    if excluded & {r['source_id'] for r in corpus.values()}:
        raise ValueError('Training/evaluation source leakage')
    order = stratified_order(list(corpus.values()), seed=83)
    design = {'version': 'v1', 'phase': PHASE, 'corpus': str(source), 'corpus_sha256': file_sha(source),
        'essays': 1430, 'samples_per_essay': 4, 'requested': 5720, 'base_revision': REVISION,
        'settings': config[PHASE], 'mode': 'two_stage', 'enable_check': False,
        'context': 8192, 'output_limit': 1024, 'order': order,
        'sampling_seed_rule': '83 + sample*1000003 + first8hex(SHA256(runtime episode ID)), modulo 2^31',
        'levels': dict(Counter(r['level'] for r in corpus.values())),
        'v1_runtime_sha256': runtime_hashes(config),
        'prompt_sha256': {role: sha_text(system_prompt(role)) for role in ROLES},
        'adapters': {role: {'path': str(sft / 'adapters' / role / 'epoch_2'),
            'sha256': file_sha(sft / 'adapters' / role / 'epoch_2/adapter_model.safetensors')} for role in ROLES},
        'evaluation_design_sha256': file_sha(sft / 'evaluation_design.json'),
        'sft_data_manifest_sha256': file_sha(sft / 'data/manifest.json')}
    path = root / 'rollout_design.json'
    if path.exists():
        if read_json(path) != design:
            raise ValueError('Frozen RFT rollout design changed')
    else:
        atomic_new(path, design)
    return design, corpus


def episode_path(root, sample, episode_id):
    return root / f'rollouts/sample_{sample}/episodes' / (safe_id(episode_id) + '.json')


def completed_global(env):
    """Keep an observed GLOBAL stage even if the later KOREAN context cannot fit."""
    if env.stage1 is None or not env.termination.get('global'):
        return None
    before = env.score(env.corrupted, 'terminal_reward_corrupted')
    middle = env.score(env.stage1, 'terminal_reward_stage1')
    return rewards(env.source, env.corrupted, env.stage1, env.records, genre=env.genre,
        q_corrupted=before['mean'], q_final=middle['mean'], q_stage1=middle['mean'],
        config=env.config['reward'], mode='two_stage', stage1=env.stage1,
        stage1_actions=env.actions['global'], stage2_actions=[], similarity=env.similarity,
        tau=env.config['similarity']['tau'],
        preexisting_spell_spans=env.episode.get('preexisting_spell_spans', ()))['global']


def run(config, *, limit=None):
    root = config['paths'][PHASE + '_output']
    design, corpus = prepare(config)
    load_environment(config)
    examples = {e.id: e for e in load_episode_examples(config, 'agent_train')}
    tasks = [(sample, eid) for eid in design['order'] for sample in range(1, 5)
             if not episode_path(root, sample, eid).exists()]
    if limit is not None:
        tasks = tasks[:limit]
    errors, finished, elapsed = [], 0, time.monotonic()
    with collection_lock(root / 'rollouts'):
        resources = Resources(config, examples, output_key=PHASE + '_output')
        def one(task):
            sample, eid = task
            row = corpus[eid]
            state = resources.worker()
            episode = {k: row[k] for k in ('episode_id', 'source_id', 'question', 'genre', 'level',
                       'records', 'corrupted_score', 'preexisting_spell_spans')}
            episode.update(source=resources.source(row['source_id']), document=resources.corrupted(row))
            env = RevisionEnv(config, analysis=state.analysis, scorer=resources,
                              similarity=resources.similarity, tokenizer=state.tokenizer)
            directory = root / f'rollouts/sample_{sample}'
            backend = SamplePolicy(config, directory / 'calls' / safe_id(eid), sample=sample)
            try:
                result = run_episode(env, episode, backend,
                    event_path=directory / 'events' / (safe_id(eid) + '.jsonl'))
            finally:
                backend.close()
            if not result['completed'] and env.stage1 is not None:
                try:
                    result['global_only_reward'] = completed_global(env)
                except Exception as exc:
                    result['global_reward_error'] = {'type': type(exc).__name__, 'message': str(exc)}
            result['rft1'] = {'sample': sample, 'design_sha256': file_sha(root / 'rollout_design.json'),
                             'policy': 'accepted SFT epoch2 pair'}
            atomic_new(episode_path(root, sample, eid), result)
            print(json.dumps({'sample': sample, 'episode': eid, 'completed': result['completed'],
                'steps': result['steps'], 'termination': result['termination'], 'error': result['runtime_error']}), flush=True)
            if (result.get('runtime_error') or {}).get('type') in {'ConnectError', 'ReadTimeout', 'HTTPStatusError'}:
                raise RuntimeError('Serving transport failure; preserved sample and stopping dispatch')
            return result
        try:
            iterator = iter(tasks)
            with ThreadPoolExecutor(max_workers=4) as pool:
                active = {}
                while True:
                    while len(active) < 4 and not errors:
                        task = next(iterator, None)
                        if task is None:
                            break
                        active[pool.submit(one, task)] = task
                    if not active:
                        break
                    done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                    for future in done:
                        task = active.pop(future)
                        try:
                            future.result()
                            finished += 1
                        except Exception as exc:
                            errors.append({'task': task, 'type': type(exc).__name__, 'message': str(exc)})
                    write_json(root / 'rollout_progress.json', {'finished_this_run': finished,
                        'in_flight': list(active.values()), 'errors': errors,
                        'elapsed_s': time.monotonic()-elapsed})
        finally:
            resources.close()
            saved = sum(episode_path(root, sample, eid).exists() for eid in corpus for sample in range(1, 5))
            write_json(root / 'rollout_status.json', {'requested': 5720, 'saved': saved,
                'finished_this_run': finished, 'errors': errors, 'elapsed_s': time.monotonic()-elapsed,
                'sampling_complete': saved == 5720, 'paid_calls': 0})
        if errors:
            raise RuntimeError('RFT rollout worker failed; inspect preserved errors before resume')
