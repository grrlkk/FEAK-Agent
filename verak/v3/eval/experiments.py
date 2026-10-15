"""Phase 6 dev experiments, resumable outputs, no training or hidden-answer prompts."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import random
import time

from ..common import read_json, write_json, sha_text
from ..env import RevisionEnv
from ..agent.runner import run_episode
from ..corrupt.document import Document
from ..corrupt.instance_policy import candidates
from ..corrupt.operators import apply
from ..reward.total import rewards
from .api import Phase6Teacher
from .design import prepare, REWRITE_PROMPT, CONTENT_PROMPT, CONTENT_SCHEMA
from .resources import Resources
from .measurement import align_rewrite, changes

LOCAL_OPS = ('L_CONN', 'L_CONJ', 'L_SUBJ_INSERT', 'L_REGISTER', 'L_POLARITY', 'L_SPACING')


def safe_id(value):
    return value.replace(':', '_').replace('/', '_')


def all_actions(result):
    return [a for actions in result['actions_by_role'].values() for a in actions]


def run_agent(config, resources, api, condition, row=None, example=None):
    mode = 'single' if condition == 'single' else 'two_stage'
    variant = 'check_once' if condition == 'check_once' else 'default'
    output = config['paths']['phase6_output']/condition
    item_id = row['episode_id'] if row else example.id
    path = output/'episodes'/(safe_id(item_id)+'.json')
    if path.exists() and read_json(path).get('completed'):
        return read_json(path)
    state = resources.worker()
    if row:
        episode = {k: row[k] for k in ('episode_id', 'source_id', 'genre', 'level', 'question',
                   'records', 'corrupted_score', 'preexisting_spell_spans')}
        episode.update(source=resources.source(row['source_id']), document=resources.corrupted(row))
    else:
        episode = {'episode_id': example.id, 'source_id': example.id, 'genre': example.genre,
                   'question': example.question, 'document': resources.source(example.id)}
    env = RevisionEnv(config, mode=mode, analysis=state.analysis, scorer=resources,
                      similarity=resources.similarity, tokenizer=state.tokenizer)
    result = run_episode(env, episode, Phase6Teacher(api, condition), prompt_variant=variant,
                         event_path=output/'events'/(safe_id(item_id)+'.jsonl'))
    write_json(path, result)  # Save model work before extra measurements that may fail.
    try:
        result['measurement'] = changes(env.corrupted, env.document, all_actions(result))
        if example and result['completed']:
            before = resources.score(example.question, env.corrupted.text)
            after = resources.score(example.question, env.document.text)
            result['quality_measurement'] = {'before': before, 'after': after, 'delta_q': after['mean']-before['mean']}
            assert result['reward'] is None
    except Exception as exc:
        result['measurement_error'] = {'type': type(exc).__name__, 'message': str(exc)}
    write_json(path, result)
    return result


def judge_real(config, api, result, example):
    path = config['paths']['phase6_output']/'real_judgments'/(safe_id(example.id)+'.json')
    if path.exists():
        return read_json(path)
    payload = {'question': example.question, 'original': example.text, 'final': result['final_text']}
    response = api.request([{'role': 'system', 'content': CONTENT_PROMPT},
        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
        stage='content_judge', item_id=example.id, effort='high', max_output=4096, schema=CONTENT_SCHEMA)
    judgment = json.loads(response['raw'])
    if set(judgment) != set(CONTENT_SCHEMA['required']) or any(type(judgment[k]) is not bool for k in ('new_content', 'meaning_changed')):
        raise ValueError('Invalid fabrication judgment')
    if not all(isinstance(s, str) and s in result['final_text'] for s in judgment['new_content_spans']):
        raise ValueError('Judge spans must quote the final text exactly')
    saved = {'source_id': example.id, 'genre': example.genre, 'judgment': judgment,
             'phase_call': response['phase_call'], 'verification': 'LLM-verified',
             'original_hash': sha_text(example.text), 'final_hash': sha_text(result['final_text'])}
    write_json(path, saved)
    return saved


def run_rewrite(config, resources, api, row, backend):
    condition = 'rewrite_'+backend
    output = config['paths']['phase6_output']/condition
    item_id = row['episode_id']
    path = output/'episodes'/(safe_id(item_id)+'.json')
    if path.exists() and read_json(path).get('completed'):
        return read_json(path)
    raw_path = output/'generations'/(safe_id(item_id)+'.json')
    messages = [{'role': 'system', 'content': REWRITE_PROMPT}, {'role': 'user',
        'content': '[문항]\n'+row['question']+'\n[학생 글]\n'+row['corrupted_text']}]
    started = time.monotonic()
    if raw_path.exists():
        generation = read_json(raw_path)
    elif backend == 'sol':
        generation = api.request(messages, stage=condition, item_id=item_id, max_output=8192)
        write_json(raw_path, generation)
    else:
        import httpx
        payload = {'model': config['policy']['model'], 'messages': messages,
                   'temperature': 0., 'top_p': 1., 'seed': 53, 'max_tokens': 4096}
        with httpx.Client(timeout=600) as client:
            response = client.post(config['policy']['base_url']+'/chat/completions', json=payload)
            response.raise_for_status()
            value = response.json()
        generation = {'raw': value['choices'][0]['message']['content'], 'messages': messages,
            'usage': value.get('usage', {}), 'finish_reason': value['choices'][0]['finish_reason'],
            'model': payload['model'], 'revision': config['policy']['base_revision'], 'sampling': payload,
            'elapsed_s': time.monotonic()-started, 'cost': {'confirmed_usd': 0}}
        write_json(raw_path, generation)
    final_text = generation['raw'].strip()
    # Preserve the entire answer. No LLM cleanup, best-of sampling or hidden-answer selection.
    result = {'corpus_episode_id': item_id, 'source_id': row['source_id'], 'genre': row['genre'],
        'level': row['level'], 'backend': backend, 'condition': condition, 'final_text': final_text,
        'generation': generation, 'completed': False, 'runtime_error': None,
        'cost_usd': generation.get('cost', {}).get('confirmed_usd', 0),
        'steps': {'single': 1}, 'checks': {'single': 0}, 'actions_by_role': {}}
    try:
        if not final_text:
            raise ValueError('Empty one-shot rewrite')
        state = resources.worker()
        source, corrupted = resources.source(row['source_id']), resources.corrupted(row)
        raw_doc = Document.from_profile(final_text, state.analysis.profile(final_text))
        final, alignment = align_rewrite(corrupted, raw_doc, config['phase6']['sentence_alignment_threshold'])
        score = resources.score(row['question'], final_text)
        reward = rewards(source, corrupted, final, row['records'], genre=row['genre'],
            q_corrupted=row['q_corrupted'], q_final=score['mean'], config=config['reward'], mode='single',
            actions=[{'action': 'ONE_SHOT'}], similarity=resources.similarity,
            tau=config['similarity']['tau'], preexisting_spell_spans=row['preexisting_spell_spans'])
        result.update(completed=True, reward=reward, final_score=score, alignment=alignment,
                      initial_layout=corrupted.snapshot(), final_layout=final.snapshot(),
                      measurement=changes(corrupted, final))
    except Exception as exc:
        result['runtime_error'] = {'type': type(exc).__name__, 'message': str(exc)}
    result['elapsed_s'] = time.monotonic()-started
    write_json(path, result)
    return result


def run_local(config, resources, source_ids):
    """One verified instance per applicable operator per source, each from the original."""
    output = config['paths']['phase6_output']/'local_link'
    for i, sid in enumerate(source_ids):
        path = output/'essays'/(safe_id(sid)+'.json')
        if path.exists():
            continue
        example = resources.examples[sid]
        source = resources.source(sid)
        base = resources.score(example.question, source.text)
        results = []
        for op in LOCAL_OPS:
            key = f"{config['phase6']['local_seed']}:{sid}:{op}"
            rng = random.Random(int(sha_text(key), 16))
            proposals = candidates(source, op)
            rng.shuffle(proposals)
            discarded = []
            chosen = None
            for proposal in proposals:
                try:
                    damaged, record = apply(source, proposal, resources.worker().bank)
                except ValueError as exc:
                    discarded.append(str(exc))
                    continue
                chosen = (damaged, record)
                break
            result = {'source_id': sid, 'op': op, 'genre': example.genre,
                      'proposals': len(proposals), 'discarded': discarded, 'applied': chosen is not None}
            if chosen:
                damaged, record = chosen
                score = resources.score(example.question, damaged.text)
                result.update(record=record, corrupted_text=damaged.text, score=score,
                              delta_q=score['mean']-base['mean'],
                              delta_rubric=[b-a for a, b in zip(base['expected'], score['expected'])])
            results.append(result)
        write_json(path, {'source_id': sid, 'genre': example.genre, 'source_score': base, 'operators': results})
        print({'stage': 'local_link', 'completed': i+1, 'total': len(source_ids)}, flush=True)


def execute(config, api, *, conditions=None, limit=None):
    design, rows, examples = prepare(config)
    conditions = conditions or ('two_stage', 'single', 'check_once', 'real', 'rewrite_sol', 'rewrite_kanana', 'local_link')
    resources = Resources(config, examples)
    output = config['paths']['phase6_output']
    jobs = []
    for item_id in design['paired_ids'][:limit]:
        for condition in ('two_stage', 'single', 'check_once', 'rewrite_sol'):
            if condition in conditions and (condition != 'check_once' or item_id in design['check_ids']):
                jobs.append((condition, item_id))
    if 'real' in conditions:
        jobs.extend(('real', sid) for sid in design['real_ids'][:limit])

    def task(job):
        condition, item_id = job
        if condition == 'rewrite_sol':
            result = run_rewrite(config, resources, api, rows[item_id], 'sol')
        elif condition == 'real':
            result = run_agent(config, resources, api, condition, example=examples[item_id])
            if result['completed']:
                judge_real(config, api, result, examples[item_id])
        else:
            result = run_agent(config, resources, api, condition, row=rows[item_id])
        print({'stage': condition, 'id': item_id, 'completed': result['completed'],
               'error': result.get('runtime_error'), 'episode_cost': result['cost_usd'],
               'accounting': api.accounting()}, flush=True)
        if not result['completed'] and (result.get('runtime_error') or {}).get('type') in ('CallBudgetExceeded', 'RuntimeError'):
            raise RuntimeError('API/runtime interruption: preserve outputs and inspect before resuming')
        return {'condition': condition, 'id': item_id, 'completed': result['completed']}

    errors, finished = [], []
    try:
        # Local work never creates OpenAI requests. One GPU0 generator plus one local-analysis worker.
        with ThreadPoolExecutor(max_workers=2) as local_pool, ThreadPoolExecutor(max_workers=4) as pool:
            local_jobs = []
            if 'rewrite_kanana' in conditions:
                def base_job():
                    for item_id in design['paired_ids'][:limit]:
                        result = run_rewrite(config, resources, api, rows[item_id], 'kanana')
                        print({'stage': 'rewrite_kanana', 'id': item_id, 'completed': result['completed'],
                               'error': result.get('runtime_error')}, flush=True)
                local_jobs.append(local_pool.submit(base_job))
            if 'local_link' in conditions:
                local_jobs.append(local_pool.submit(run_local, config, resources, design['local_source_ids'][:limit]))
            iterator, pending = iter(jobs), {}
            def fill():
                while len(pending) < 4 and not errors:
                    job = next(iterator, None)
                    if job is None:
                        break
                    pending[pool.submit(task, job)] = job
            fill()
            while pending:
                done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    job = pending.pop(future)
                    try:
                        finished.append(future.result())
                    except Exception as exc:
                        errors.append({'condition': job[0], 'id': job[1], 'type': type(exc).__name__, 'message': str(exc)})
                write_json(output/'progress.json', {'requested_jobs': len(jobs), 'finished': finished,
                    'errors': errors, 'accounting': api.accounting()})
                fill()
            for future in local_jobs:
                try:
                    future.result()
                except Exception as exc:
                    errors.append({'condition': 'local', 'type': type(exc).__name__, 'message': str(exc)})
    finally:
        resources.close()
        write_json(output/'run_status.json', {'requested_jobs': len(jobs), 'finished': finished,
                                            'errors': errors, 'accounting': api.accounting()})
    if errors:
        raise RuntimeError('Phase 6 paused; inspect run_status.json; completed requests are cached')
