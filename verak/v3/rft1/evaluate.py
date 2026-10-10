"""Frozen SFT comparison cohort and unchanged v1 RFT evaluation."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import json
from pathlib import Path
import time

from ..agent.runner import run_episode
from ..common import file_sha, read_json, write_json
from ..env import RevisionEnv
from ..eval.measurement import changes
from ..eval.resources import Resources
from ..phase2 import read_jsonl
from ..reward.overedit import overedit
from ..train.pilot import safe_id
from ..train.teacher_bulk import atomic_new, collection_lock
from ..view_data import load_episode_examples
from .config import PHASE, ROLES, REVISION, load_environment, runtime_hashes
from .rollout import SamplePolicy


def inputs(config):
    old = config['paths']['phase7_sft_output'] / 'evaluation_design.json'
    design = read_json(old)
    corpus_file = config['paths']['active_corrupt'] / 'agent_dev.jsonl'
    if file_sha(corpus_file) != design['contract']['dev_corpus_sha256']:
        raise ValueError('Accepted SFT evaluation corpus changed')
    corpus = {r['episode_id']: r for r in read_jsonl(corpus_file)}
    examples = {e.id: e for e in load_episode_examples(config, 'agent_dev')}
    return design, corpus, examples


def adapter_metadata(config, condition):
    root = config['paths']['repo'] / 'verak/v3/outputs' / (PHASE if condition == 'rft1' else 'oneshot_baseline')
    roles = ROLES if condition == 'rft1' else ('oneshot',)
    values = {}
    for role in roles:
        path = root / 'adapters' / role / 'epoch_1'
        metadata = read_json(path / 'provenance.json')
        if metadata['base_revision'] != REVISION or metadata['adapter_sha256'] != file_sha(path / 'adapter_model.safetensors'):
            raise ValueError('Evaluation adapter provenance changed')
        values[role] = {**metadata, 'path': str(path)}
    return values


def execute(config, *, limit=None):
    config = deepcopy(config)
    config['policy']['adapters'] = {r: f'rft1-{r}' for r in ROLES}
    root = config['paths'][PHASE + '_output']
    output = root / 'evaluation/rft1'
    old, corpus, examples = inputs(config)
    contract = {'sft_evaluation_design_sha256': file_sha(config['paths']['phase7_sft_output'] / 'evaluation_design.json'),
        'dev_ids': old['contract']['dev_ids'], 'real_ids': old['contract']['real_ids'],
        'context': 8192, 'output_limit': 1024, 'temperature': 0, 'top_p': 1, 'seed': 47,
        'mode': 'two_stage', 'environment': 'v1', 'enable_check': False, 'workers': 4,
        'runtime_sha256': runtime_hashes(config), 'adapters': adapter_metadata(config, 'rft1')}
    target = root / 'evaluation_design.json'
    if target.exists():
        if read_json(target) != contract:
            raise ValueError('Frozen RFT evaluation design changed')
    else:
        atomic_new(target, contract)
    load_environment(config)
    with collection_lock(output):
        resources = Resources(config, examples, output_key=PHASE + '_output')
        requested = contract['dev_ids'] + contract['real_ids']
        pending = [i for i in requested if not (output / 'episodes' / (safe_id(i) + '.json')).exists()]
        if limit is not None:
            pending = pending[:limit]
        started, finished, errors = time.monotonic(), 0, []

        def one(eid):
            state = resources.worker()
            row = corpus.get(eid)
            if row:
                episode = {k: row[k] for k in ('episode_id', 'source_id', 'question', 'genre', 'level',
                           'records', 'corrupted_score', 'preexisting_spell_spans')}
                episode.update(source=resources.source(row['source_id']), document=resources.corrupted(row))
            else:
                ex = examples[eid]
                episode = {'episode_id': eid, 'source_id': eid, 'question': ex.question, 'genre': ex.genre,
                           'document': resources.source(eid)}
            env = RevisionEnv(config, analysis=state.analysis, scorer=resources,
                              similarity=resources.similarity, tokenizer=state.tokenizer)
            backend = SamplePolicy(config, output / 'calls' / safe_id(eid), sample=0, evaluation=True)
            try:
                result = run_episode(env, episode, backend, event_path=output / 'events' / (safe_id(eid) + '.jsonl'))
            finally:
                backend.close()
            result.update(condition='rft1', cohort='dev' if row else 'real',
                          seen_by_scorer=examples[episode['source_id']].seen_by_scorer)
            path = output / 'episodes' / (safe_id(eid) + '.json')
            # Preserve the generated trajectory before any additional terminal measurement.
            write_json(path, result)
            try:
                actions = [a for values in result['actions_by_role'].values() for a in values]
                result['measurement'] = changes(env.corrupted, env.document, actions)
                if not row:
                    result['R_over_all_source'] = overedit(env.corrupted, env.corrupted, env.document, [])['value']
                    if result['completed']:
                        before = resources.score(episode['question'], env.corrupted.text)
                        after = resources.score(episode['question'], env.document.text)
                        result['quality_measurement'] = {'before': before, 'after': after,
                                                        'delta_q': after['mean'] - before['mean']}
            except Exception as exc:
                result['measurement_error'] = {'type': type(exc).__name__, 'message': str(exc)}
            write_json(path, result)
            print(json.dumps({'condition': 'rft1', 'id': eid, 'completed': result['completed'],
                'termination': result['termination'], 'steps': result['steps'], 'error': result['runtime_error']}), flush=True)
            if (result.get('runtime_error') or {}).get('type') in {'ConnectError', 'ReadTimeout', 'HTTPStatusError'}:
                raise RuntimeError('Serving transport failure: stop dispatch and inspect')
            return result
        try:
            iterator = iter(pending)
            with ThreadPoolExecutor(max_workers=4) as pool:
                active = {}
                while True:
                    while len(active) < 4 and not errors:
                        eid = next(iterator, None)
                        if eid is None:
                            break
                        active[pool.submit(one, eid)] = eid
                    if not active:
                        break
                    done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                    for future in done:
                        eid = active.pop(future)
                        try:
                            future.result()
                            finished += 1
                        except Exception as exc:
                            errors.append({'id': eid, 'type': type(exc).__name__, 'message': str(exc)})
                    write_json(output / 'progress.json', {'finished_this_run': finished,
                        'in_flight': list(active.values()), 'errors': errors, 'elapsed_s': time.monotonic() - started})
        finally:
            resources.close()
            saved = sum((output / 'episodes' / (safe_id(i) + '.json')).exists() for i in requested)
            write_json(output / 'status.json', {'requested': 130, 'saved': saved, 'finished_this_run': finished,
                'errors': errors, 'elapsed_s': time.monotonic() - started, 'all_attempted': saved == 130})
        if errors:
            raise RuntimeError('Evaluation worker failure; inspect preserved output')


def saved_rows(config, condition='rft1'):
    design, _, _ = inputs(config)
    root = config['paths']['repo'] / 'verak/v3/outputs' / (PHASE if condition == 'rft1' else 'oneshot_baseline')
    for eid in design['contract']['dev_ids'] + design['contract']['real_ids']:
        path = root / 'evaluation' / condition / 'episodes' / (safe_id(eid) + '.json')
        if path.exists():
            yield read_json(path), path, path.parent.parent / 'events' / (safe_id(eid) + '.jsonl')


def measure_saved_real(config, condition='rft1'):
    """Resume interrupted terminal measurements without regenerating a policy call."""
    from ..agentic.preservation import restore_final
    from .context_audit import layout_text
    phase = PHASE if condition == 'rft1' else 'oneshot_baseline'
    root = config['paths']['repo'] / 'verak/v3/outputs' / phase
    config['paths'][phase + '_output'] = root
    pending = [(row, path) for row, path, _ in saved_rows(config, condition)
               if row['cohort'] == 'real' and row['completed'] and
               (row.get('measurement_error') or any(k not in row for k in
                ('measurement', 'R_over_all_source', 'quality_measurement')))]
    if not pending:
        write_json(root / 'measurement_status.json', {'repaired': 0, 'errors': [], 'policy_calls': 0})
        return
    load_environment(config)
    _, _, examples = inputs(config)
    resources = Resources(config, examples, output_key=phase + '_output')
    errors, repaired = [], 0
    try:
        for row, path in pending:
            try:
                analysis = resources.worker().analysis
                before = restore_final(row['initial_layout'], layout_text(row['initial_layout']), analysis)
                after = restore_final(row['final_layout'], row['final_text'], analysis)
                actions = [a for group in row['actions_by_role'].values() for a in group]
                row['measurement'] = changes(before, after, actions)
                row['R_over_all_source'] = overedit(before, before, after, [])['value']
                if 'quality_measurement' not in row:
                    question = examples[row['source_id']].question
                    initial, final = resources.score(question, before.text), resources.score(question, after.text)
                    row['quality_measurement'] = {'before': initial, 'after': final, 'delta_q': final['mean'] - initial['mean']}
                if row.get('measurement_error'):
                    row.setdefault('prior_measurement_errors', []).append(row.pop('measurement_error'))
                write_json(path, row)
                repaired += 1
            except Exception as exc:
                errors.append({'id': row['corpus_episode_id'], 'type': type(exc).__name__, 'message': str(exc)})
    finally:
        resources.close()
        write_json(root / 'measurement_status.json', {'repaired': repaired, 'errors': errors, 'policy_calls': 0})
    if errors:
        raise RuntimeError('Saved real measurements still fail; inspect before paid marker checks')
