"""Resumable stages in user-prioritized order, sharing one six-dollar ledger."""
from collections import Counter
import json

from feak_tc.runtime.openai import CallBudgetExceeded
from ..common import read_json, write_json, sha_text
from ..eval.resources import Resources
from ..env.analysis import alias_structure
from ..train.pilot import safe_id
from ..train.pilot2_data import bounded_map
from .data import PHASE, prepare
from .accounting import audit_reward
from . import graph
from .environment import AgenticEnv, PROMPTS
from .runner import run, PILOT_STAGE


def graph_path(config, item):
    return config['paths'][PHASE + '_output'] / 'graphs' / (safe_id(item) + '.json')


def extract(config, api, item, document, question):
    target = graph_path(config, item)
    if target.exists():
        value = read_json(target)
        if value['input_hash'] != sha_text(document.text) or value['question'] != question:
            raise ValueError('Graph input changed')
        if value.get('validation_stage') == 'intersection' or value['status'] == 'completed':
            return value
        # Older preflight validation stopped before the mandated intersection.
        # Preserve it and replay the same cached requests; generate only a missing
        # second extraction. Final degree validity is checked after intersection.
        archive = target.parent.parent / 'graph_preintersection_validation' / target.name
        if not archive.exists():
            write_json(archive, value)
    _, smap, pmap = graph.public_input(document)
    value = {'id': item, 'input_hash': sha_text(document.text), 'question': question,
             'sentence_ids': smap, 'paragraph_ids': pmap, 'input_layout': document.snapshot(),
             'status': 'error', 'runs': [], 'validation_stage': 'intersection'}
    messages, schema = graph.request(document, question)
    try:
        for number in (1, 2):
            response = api.request(messages, stage='graph_extract', item_id=f'{item}:run{number}',
                                   effort='low', max_output=8192, schema=schema)
            # The API schema fixes the raw shape and IDs. Intersect both raw
            # assertions first, then validate all semantic edge constraints.
            discourse = json.loads(response['raw'])
            value['runs'].append({'discourse': discourse, 'phase_call': response['phase_call']})
        discourse, counts = graph.intersection(*[r['discourse'] for r in value['runs']])
        graph.validate(discourse, list(smap.values()), list(pmap.values()))
        value.update(status='completed', discourse=discourse, intersection_counts=counts,
                     graph=graph.state(alias_structure(document.structure(), smap, pmap), discourse, pmap.values()))
    except CallBudgetExceeded:
        # Successful first requests are recovered from the API ledger on resume.
        raise
    except Exception as exc:
        value['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    write_json(target, value)
    print(json.dumps({'graph': item, 'status': value['status'], 'cost': api.accounting()['confirmed_usd']}), flush=True)
    return value


def quality(config, design, dev):
    counts, failures, intersection_counts = Counter(), [], Counter()
    for path in sorted((config['paths'][PHASE + '_output'] / 'graphs').glob('*.json')):
        value = read_json(path)
        if value['status'] == 'completed':
            for kind, stats in value['intersection_counts'].items():
                for key, n in stats.items():
                    intersection_counts[kind + '_' + key] += n
    for item in design['quality_corrupted_ids'] + design['quality_source_ids']:
        path = graph_path(config, item)
        value = read_json(path) if path.exists() else {'status': 'not_run'}
        if item in dev:
            records = [r for r in dev[item]['records'] if r['op'] == 'G_OFFTOPIC']
            counts['offtopic_total'] += len(records)
        if value['status'] != 'completed':
            failures.append({'id': item, 'status': value['status']})
            continue
        flagged = set(value['graph']['off_topic_candidate'])
        if item in dev:
            for r in records:
                counts['offtopic_evaluable'] += 1
                counts['offtopic_flagged'] += value['sentence_ids'][r['recovery_target']['inserted_id']] in flagged
        else:
            counts['source_essays'] += 1
            counts['source_sentences'] += len(value['sentence_ids'])
            counts['source_flagged'] += len(flagged)
    result = {**counts, 'intersection_counts': dict(intersection_counts), 'failures': failures,
              'false_flag_definition': 'fraction of uncorrupted source sentences flagged; source essays are not certified perfectly relevant',
              'extraction_protocol': 'two independent requests, identical prompt; no controllable API sampling seed'}
    write_json(config['paths'][PHASE + '_output'] / 'graph_quality.json', result)
    return result


def graphs(config, api, limit=None):
    design, train, dev, examples = prepare(config)
    resources = Resources(config, examples, output_key=PHASE + '_output')
    # Quality population first, then graphs needed by the 92/30 pilots.
    ids = list(dict.fromkeys(design['quality_corrupted_ids'] + design['quality_source_ids'] +
                             design['corrupted_ids'] + design['real_ids']))
    def one(item):
        row = train.get(item) or dev.get(item)
        doc = resources.corrupted(row) if row else resources.source(item)
        return extract(config, api, item, doc, row['question'] if row else examples[item].question)
    try:
        errors = bounded_map(ids[:limit] if limit else ids, one)
    finally:
        resources.close()
    result = {'requested': len(ids), 'errors': errors, 'budget': api.accounting()}
    write_json(config['paths'][PHASE + '_output'] / 'graphs_status.json', result)
    quality(config, design, dev)
    return result


def prompt_check(config, tokenizer):
    counts = {r: len(tokenizer.encode(p, add_special_tokens=False)) for r, p in PROMPTS.items()}
    counts['graph'] = len(tokenizer.encode(graph.PROMPT, add_special_tokens=False))
    write_json(config['paths'][PHASE + '_output'] / 'prompt_tokens.json', counts)
    if any(n > 400 for n in counts.values()):
        raise ValueError('Prompt exceeds 400 policy tokens: ' + str(counts))
    return counts


def pilot(config, api, cohort, limit=None):
    design, train, dev, examples = prepare(config)
    root = config['paths'][PHASE + '_output']
    resources = Resources(config, examples, output_key=PHASE + '_output')
    prompt_check(config, resources.worker().tokenizer)
    ids = design['corrupted_ids' if cohort == 'corrupted' else 'real_ids']
    def one(item):
        target = root / 'episodes' / (safe_id(item) + '.json')
        if target.exists():
            return
        path = graph_path(config, item)
        saved = read_json(path) if path.exists() else {'status': 'not_run'}
        if saved['status'] != 'completed':
            write_json(target, {'episode_id': item, 'cohort': cohort, 'completed': False, 'calls': [], 'actions': [],
                               'sequences': [], 'reward': None, 'runtime_error': {'type': 'GraphUnavailable', 'message': saved['status']}})
            return
        row = train.get(item)
        if row:
            ep = {k: row[k] for k in ('episode_id', 'source_id', 'genre', 'question', 'question_id', 'records',
                                      'corrupted_score', 'preexisting_spell_spans')}
            ep.update(document=resources.corrupted(row), source=resources.source(row['source_id']))
        else:
            ex = examples[item]
            ep = {'episode_id': item, 'source_id': item, 'genre': ex.genre, 'question': ex.question,
                  'question_id': 'Q:' + ex.question_hash, 'document': resources.source(item)}
        assert sha_text(ep['document'].text) == saved['input_hash']
        state = resources.worker()
        env = AgenticEnv(ep, saved['discourse'], analysis=state.analysis, scorer=resources)
        # Final accounting is separate from the frozen policy loop: all control
        # actions except read tools/ledgers retain the combined step cost.
        result = audit_reward(run(env, api, state.tokenizer, config, root / 'events' / (safe_id(item) + '.jsonl')))
        # Include paid incomplete/error responses too, not only successful calls.
        with api.db() as db:
            found = db.execute('SELECT confirmed,path FROM calls WHERE stage=? AND item_id LIKE ?',
                               (PILOT_STAGE + cohort, item + ':%')).fetchall()
        result['confirmed_episode_cost'] = sum(r[0] for r in found)
        result['api_attempts'] = len(found)
        result['api_request_paths'] = [r[1] for r in found if r[1]]
        write_json(target, result)
        print(json.dumps({'pilot': item, 'completed': result['completed'], 'error': result['runtime_error'],
                          'cost': api.accounting()['confirmed_usd']}), flush=True)
        if result['runtime_error'] and result['runtime_error']['type'] == 'CallBudgetExceeded':
            raise CallBudgetExceeded('Shared six-dollar cap reached')
    try:
        errors = bounded_map(ids[:limit] if limit else ids, one)
    finally:
        resources.close()
    value = {'requested': len(ids), 'errors': errors, 'budget': api.accounting()}
    write_json(root / (cohort + '_status.json'), value)
    return value
