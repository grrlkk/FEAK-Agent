"""A teacher-final-text baseline, selected without mixing two attempts' outputs."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import time

from ..common import file_sha, read_json, sha_text, write_json
from ..corrupt.document import Document
from ..env.feedback import cohesion_facts
from ..eval.design import REWRITE_PROMPT
from ..eval.measurement import align_rewrite, changes
from ..eval.resources import Resources
from ..phase2 import read_jsonl
from ..reward.overedit import overedit
from ..reward.total import rewards
from ..train.formatting import encode_labels, lengths
from ..train.pilot import safe_id
from ..train.teacher_bulk import atomic_new, collection_lock
from ..train.teacher_comparison import absolute_selection
from .config import PHASE, REVISION, ROLES, load_environment
from .evaluate import adapter_metadata, inputs

ONE_SHOT = 'oneshot_baseline'


def messages_for(question, essay):
    return [{'role': 'system', 'content': REWRITE_PROMPT},
            {'role': 'user', 'content': '[문항]\n' + question + '\n[학생 글]\n' + essay}]


def choose_target(attempts, corpus):
    """Choose the higher combined R first; never fall back if its role filter fails."""
    available = [(attempt, row) for attempt, row in attempts if row.get('completed') and row.get('reward')]
    if not available:
        return None, 'no_completed_teacher_final'
    attempt, row = max(available, key=lambda pair: (pair[1]['reward']['combined']['R'], -pair[0]))
    accepted = absolute_selection(row, corpus)
    if any(r['level'] == 'GLOBAL' for r in corpus['records']) and not accepted['global']:
        return (attempt, row), 'higher_combined_attempt_fails_GLOBAL'
    if not accepted['korean']:
        return (attempt, row), 'higher_combined_attempt_fails_KOREAN'
    return (attempt, row), 'eligible'


def composition(ids, corpus):
    ids = list(ids)
    return {'episodes': len(ids), 'sources': len({corpus[i]['source_id'] for i in ids}),
        'levels': dict(Counter(corpus[i]['level'] for i in ids)),
        'operator_trajectories': dict(Counter(op for i in ids for op in {r['op'] for r in corpus[i]['records']})),
        'operator_records': dict(Counter(r['op'] for i in ids for r in corpus[i]['records']))}


def export(config):
    # C is deliberately sequenced after all of A, including the report.
    if not (config['paths'][PHASE + '_output'] / 'a_complete.json').exists():
        raise RuntimeError('Complete A before starting C')
    root = config['paths']['repo'] / 'verak/v3/outputs' / ONE_SHOT
    destination = root / 'data/manifest.json'
    if destination.exists():
        manifest = read_json(destination)
        for metadata in manifest['roles']['oneshot'].values():
            if file_sha(metadata['path']) != metadata['sha256']:
                raise ValueError('One-shot export changed')
        return manifest
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    sft_path = config['paths']['phase7_sft_output'] / 'data/manifest.json'
    sft = read_json(sft_path)
    split = sft['contract']['split']['roles']
    held = set().union(*(set(split[r]['validation_sources']) for r in ROLES))
    train_ids = set().union(*(set(split[r]['train_ids']) for r in ROLES))
    validation_ids = set().union(*(set(split[r]['validation_ids']) for r in ROLES))
    if train_ids & validation_ids:
        raise ValueError('SFT source partition unexpectedly overlaps')
    corpus = {r['episode_id']: r for r in read_jsonl(config['paths']['active_corrupt'] / 'agent_train.jsonl')}
    teacher_root = root.parent / 'teacher_bulk_two_stage'
    teacher_design = read_json(teacher_root / 'design.json')
    selected = {'train': [], 'validation': []}
    excluded, audit = [], []
    for eid in sorted(train_ids | validation_ids):
        paths, attempts = {}, []
        for attempt in (1, 2):
            reuse = teacher_design['reuse_attempt_1'].get(eid) if attempt == 1 else None
            path = Path(reuse['path']) if reuse else teacher_root / f'attempt_{attempt}/episodes' / (safe_id(eid) + '.json')
            if path.exists():
                if reuse and file_sha(path) != reuse['sha256']:
                    raise ValueError('Reused teacher changed')
                paths[attempt] = path
                attempts.append((attempt, read_json(path)))
        chosen, reason = choose_target(attempts, corpus[eid])
        source_id = corpus[eid]['source_id']
        part = 'validation' if source_id in held else 'train'
        if (eid in train_ids) != (part == 'train'):
            raise ValueError('C partition disagrees with SFT source holdout')
        record = {'episode_id': eid, 'source_id': source_id, 'partition': part, 'reason': reason,
            'attempts': [{'attempt': n, 'path': str(paths[n]), 'sha256': file_sha(paths[n]),
                          'completed': r['completed'], 'combined_R': (r.get('reward') or {}).get('combined', {}).get('R')}
                         for n, r in attempts]}
        if chosen:
            attempt, row = chosen
            record.update(chosen_attempt=attempt, teacher_path=str(paths[attempt]),
                          teacher_sha256=file_sha(paths[attempt]), combined_R=row['reward']['combined']['R'])
        audit.append(record)
        if reason != 'eligible':
            excluded.append(record)
            continue
        messages = messages_for(corpus[eid]['question'], corpus[eid]['corrupted_text'])
        full = messages + [{'role': 'assistant', 'content': row['final_text']}]
        encoded = encode_labels(tokenizer, full, last_assistant_only=True)
        prompt_length = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        if len(encoded['input_ids']) > 8192:
            record['reason'] = 'full_prompt_and_target_exceed_8192_no_truncation'
            excluded.append(record)
            continue
        selected[part].append({**record, 'messages': full, **encoded, 'role': 'oneshot',
            'prompt_tokens': prompt_length, 'total_tokens': len(encoded['input_ids']),
            'loss_tokens': sum(n != -100 for n in encoded['labels']),
            'loss_policy': 'whole final essay assistant content and EOT only; input question and essay masked'})
    if not selected['train']:
        raise RuntimeError('No quality-filtered one-shot training examples')
    result = {'base_revision': REVISION, 'sft_manifest_sha256': file_sha(sft_path),
        'teacher_design_sha256': file_sha(teacher_root / 'design.json'), 'prompt_sha256': sha_text(REWRITE_PROMPT),
        'selection': 'higher combined-R complete teacher attempt, then existing SFT GLOBAL gate if GLOBAL records and KOREAN R>=.80; no fallback',
        'tie': 'lower attempt ID', 'context': 8192, 'holdout_sources': sorted(held), 'roles': {'oneshot': {}},
        'new_teacher_calls': 0, 'SFT_train_union': composition(train_ids, corpus),
        'agent_SFT_by_role': {role: composition(split[role]['train_ids'], corpus) for role in ROLES},
        'selection_counts': {part: composition([r['episode_id'] for r in rows], corpus) for part, rows in selected.items()},
        'excluded_counts': {part: dict(Counter(r['reason'] for r in excluded if r['partition'] == part))
                            for part in ('train', 'validation')}}
    (root / 'data').mkdir(parents=True, exist_ok=True)
    for part, rows in selected.items():
        path = root / f'data/oneshot.{part}.jsonl.gz'
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        result['roles']['oneshot'][part] = {'path': str(path), 'sha256': file_sha(path), 'trajectories': len(rows),
            'turns': len(rows), 'sources': len({r['source_id'] for r in rows}),
            'tokens': sum(r['total_tokens'] for r in rows), 'target_tokens': sum(r['loss_tokens'] for r in rows),
            'lengths': lengths([r['total_tokens'] for r in rows]),
            'target_lengths': lengths([r['loss_tokens'] for r in rows])}
    write_json(root / 'selection_audit.json', audit)
    atomic_new(destination, result)
    return result


def execute(config, *, limit=None):
    import httpx
    root = config['paths']['repo'] / 'verak/v3/outputs' / ONE_SHOT
    config['paths'][ONE_SHOT + '_output'] = root
    output = root / 'evaluation/oneshot'
    old, corpus, examples = inputs(config)
    contract = {'sft_evaluation_design_sha256': file_sha(config['paths']['phase7_sft_output'] / 'evaluation_design.json'),
        'dev_ids': old['contract']['dev_ids'], 'real_ids': old['contract']['real_ids'],
        'context': 8192, 'max_output': 4096, 'output_rule': 'min(4096,8192-input_tokens); never truncate input',
        'temperature': 0, 'top_p': 1, 'seed': 47, 'prompt_sha256': sha_text(REWRITE_PROMPT),
        'alignment': 'existing Phase6 corrupted_only_hungarian_lexical, threshold0.35; source and records unseen',
        'single_reward_action': 'ONE_SHOT, the existing Phase6 combined-R convention',
        'role_rewards_and_tool_STOP': 'not applicable: no intermediate stage or action protocol',
        'adapters': adapter_metadata(config, 'oneshot')}
    design_path = root / 'evaluation_design.json'
    if design_path.exists():
        if read_json(design_path) != contract:
            raise ValueError('One-shot evaluation contract changed')
    else:
        atomic_new(design_path, contract)
    load_environment(config)
    requested = contract['dev_ids'] + contract['real_ids']
    with collection_lock(output):
        resources = Resources(config, examples, output_key=ONE_SHOT + '_output')
        errors, finished, started = [], 0, time.monotonic()

        def one(eid):
            path = output / 'episodes' / (safe_id(eid) + '.json')
            if path.exists():
                return read_json(path)
            state = resources.worker()
            row = corpus.get(eid)
            example = examples[row['source_id'] if row else eid]
            corrupted = resources.corrupted(row) if row else resources.source(eid)
            source = resources.source(example.id)
            messages = messages_for(example.question, corrupted.text)
            prompt_tokens = len(state.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
            result = {'corpus_episode_id': eid, 'source_id': example.id, 'genre': example.genre,
                'cohort': 'dev' if row else 'real', 'level': row['level'] if row else None, 'condition': 'oneshot',
                'seen_by_scorer': example.seen_by_scorer, 'completed': False, 'runtime_error': None,
                'initial_layout': corrupted.snapshot(), 'initial_text': corrupted.text,
                'steps': {'single': 0}, 'checks': {'single': 0}, 'actions_by_role': {}, 'calls': [],
                'termination': {}, 'reward': None, 'final_text': corrupted.text, 'final_layout': corrupted.snapshot()}
            generation_path = output / 'generations' / (safe_id(eid) + '.json')
            begin = time.monotonic()
            try:
                if prompt_tokens >= 8192:
                    raise ValueError('Full question and essay exceed 8192; no truncation permitted')
                payload = {'model': 'oneshot-oneshot', 'messages': messages, 'temperature': 0., 'top_p': 1.,
                           'seed': 47, 'max_tokens': min(4096, 8192 - prompt_tokens)}
                if generation_path.exists():
                    generation = read_json(generation_path)
                    if generation['sampling'] != payload:
                        raise ValueError('Cached one-shot request changed')
                else:
                    with httpx.Client(timeout=600) as client:
                        response = client.post(config['policy']['base_url'] + '/chat/completions', json=payload)
                        response.raise_for_status()
                        value = response.json()
                    generation = {'raw': value['choices'][0]['message']['content'],
                        'finish_reason': value['choices'][0]['finish_reason'], 'usage': value.get('usage', {}),
                        'response_id': value.get('id'), 'sampling': payload, 'prompt_tokens_local': prompt_tokens}
                    atomic_new(generation_path, generation)
                result['generation'] = generation
                result['steps']['single'] = 1
                text = generation['raw'].strip()
                result['final_text'] = text
                result['termination']['single'] = generation['finish_reason']
                if not text:
                    raise ValueError('Empty one-shot output')
                raw_doc = Document.from_profile(text, state.analysis.profile(text))
                final, alignment = align_rewrite(corrupted, raw_doc, .35)
                result.update(final_layout=final.snapshot(), alignment=alignment,
                              measurement=changes(corrupted, final),
                              marker_changes=cohesion_facts(corrupted.structure(), final.structure(),
                                                           {u.sid for u in final.units}))
                # As in Phase6, score the entire nonempty bounded output. An output
                # limit is reported separately, just as agent step limits are.
                score = resources.score(example.question, text)
                if row:
                    result['reward'] = rewards(source, corrupted, final, row['records'], genre=row['genre'],
                        q_corrupted=row['q_corrupted'], q_final=score['mean'], config=config['reward'], mode='single',
                        actions=[{'action': 'ONE_SHOT'}], similarity=resources.similarity,
                        tau=config['similarity']['tau'], preexisting_spell_spans=row['preexisting_spell_spans'])
                else:
                    before = resources.score(example.question, corrupted.text)
                    result['quality_measurement'] = {'before': before, 'after': score, 'delta_q': score['mean'] - before['mean']}
                    result['R_over_all_source'] = overedit(corrupted, corrupted, final, [])['value']
                result.update(completed=True, final_score=score)
            except Exception as exc:
                result['runtime_error'] = {'type': type(exc).__name__, 'message': str(exc)}
            result['elapsed_s'] = time.monotonic() - begin
            atomic_new(path, result)
            print(json.dumps({'condition': 'oneshot', 'id': eid, 'completed': result['completed'],
                              'error': result['runtime_error']}), flush=True)
            if (result.get('runtime_error') or {}).get('type') in {'ConnectError', 'ReadTimeout', 'HTTPStatusError'}:
                raise RuntimeError('One-shot serving transport failure')
            return result
        try:
            tasks = [eid for eid in requested if not (output / 'episodes' / (safe_id(eid) + '.json')).exists()]
            if limit is not None:
                tasks = tasks[:limit]
            # No service competition: C only runs after both A adapters and evaluations finish.
            with ThreadPoolExecutor(max_workers=4) as pool:
                for result in pool.map(one, tasks):
                    finished += 1
                    write_json(output / 'progress.json', {'finished_this_run': finished,
                                                         'elapsed_s': time.monotonic() - started})
        except Exception as exc:
            errors.append({'type': type(exc).__name__, 'message': str(exc)})
            raise
        finally:
            resources.close()
            saved = sum((output / 'episodes' / (safe_id(eid) + '.json')).exists() for eid in requested)
            write_json(output / 'status.json', {'requested': 130, 'saved': saved, 'all_attempted': saved == 130,
                'finished_this_run': finished, 'errors': errors, 'elapsed_s': time.monotonic() - started})
