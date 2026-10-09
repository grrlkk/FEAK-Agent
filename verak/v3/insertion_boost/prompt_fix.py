"""User-requested, versioned GLOBAL content-rule probe; old runs stay immutable."""
from collections import Counter
from pathlib import Path
import json
import random
import time

from feak_tc.runtime.openai import CallBudgetExceeded
from ..common import file_sha, read_json, sha_text, write_json
from ..train.teacher_bulk import BulkTeacher, atomic_new, collection_lock
from ..v2_ops.config import PHASE
from ..v2_ops.environment import V2RevisionEnv
from ..v2_ops.local import load_environment
from ..v2_ops.paid import V2API
from ..v2_ops.prompts import system_prompt as old_prompt
from ..v2_ops.runner import run_episode
from ..v2_ops.teacher import attempt_path, safe_id
from .teacher import Resources, prepare as prepare_original, recover_global
from .cpu_score import cached_gpu_score, queue_cpu
from .calibrate import layout_text


RULE = "글에 없는 사실은 만들지 않는다. 단, 앞에 소개되지 않은 대상을 가리키는 문장(예: '그 식당', '이러한 문제', '첫 번째 이유')이 있으면, 글의 다른 부분에 있는 내용으로 그 대상을 소개하는 문장을 INSERT 한다."
VERSION = 'content_rule_v1'
SEED = 89
SIZE = 20
MIN_INSERTIONS = 6


def system_prompt(role):
    text = old_prompt(role)
    if role != 'global':
        return text
    replacements = [
        ('글쓴이의 주장과 내용을 지킨다. 원문에 없는 구체적 사실·경험·수치·출처·새 주장을 만들지 않는다.',
         '글쓴이의 주장과 내용을 지킨다.'),
        ('글 어딘가에 이미 있는 내용은 옮기고, 연결하고, 다시 표현할 수 있다. 글 어디에도 없는 이유·사실·사례·통계·수치·출처·이름·경험은 일반 상식이어도 새로 넣지 않는다.', RULE),
        ('내용이 부족하면 본문에 만들어 넣지 말고 STOP summary에 글쓴이가 보충해야 할 내용을 적는다. ', ''),
        ('문단 ID는 해당 문단의 처음/끝이다. 글에 없는 내용은 만들지 않는다.',
         '문단 ID는 해당 문단의 처음/끝이다.'),
    ]
    for before, after in replacements:
        if text.count(before) != 1:
            raise ValueError('Frozen v2 content-rule anchor changed')
        text = text.replace(before, after, 1)
    return text


def directory(config):
    return config['paths'][PHASE + '_output'] / 'prompt_fix'


def save_once(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Immutable prompt-fix artifact changed: {path}')
    else:
        atomic_new(path, value)


def slot_path(config, phase, attempt, eid):
    return attempt_path(directory(config) / phase, attempt, eid)


def prepare(config):
    root = config['paths'][PHASE + '_output']
    previous, corpus = prepare_original(config)
    pause_path = directory(config) / 'pause/snapshot.json'
    pause = read_json(pause_path)
    if pause['status'] != 'paused' or not pause['old_controller_must_not_resume']:
        raise ValueError('Old teacher must be paused before the independent prompt test')
    for row in pause['files']:
        if file_sha(row['path']) != row['sha256']:
            raise ValueError('An archived old-prompt artifact changed after pause')
    saved = {(a, eid) for a in (1, 2) for eid in previous['orders'][str(a)]
             if attempt_path(root, a, eid).exists()}
    if len(saved) != pause['saved_attempts']:
        raise ValueError('Old-prompt collection advanced after the pause snapshot')
    remaining = [(a, previous['orders'][str(a)][i]) for i in range(len(corpus)) for a in (1, 2)
                 if (a, previous['orders'][str(a)][i]) not in saved]
    # Exactly one fresh request trajectory per source, occupying a previously unsaved slot.
    choices = {}
    for a, eid in remaining:
        choices.setdefault(corpus[eid]['source_id'], (a, eid))
    source_ids = sorted(choices)
    random.Random(SEED).shuffle(source_ids)
    if len(source_ids) < SIZE:
        raise ValueError('Fewer than 20 distinct QC-passing uncompleted sources')
    test = [choices[s] for s in source_ids[:SIZE]]
    test_keys = set(test)
    phases = {'test_v1': test, 'resume_v1': [x for x in remaining if x not in test_keys]}
    prompts = {r: system_prompt(r) for r in ('global', 'korean')}
    for phase, tasks in phases.items():
        value = {'version': VERSION, 'phase': phase, 'model': previous['model'],
            'reasoning': 'low', 'context': 8192, 'output_limit': 1024, 'sampling_seed': None,
            'ordering_seed': SEED, 'original_design_sha256': file_sha(root / 'teacher_design.json'),
            'pause_snapshot_sha256': file_sha(pause_path), 'runtime_sha256': previous['runtime_sha256'],
            'prompt': prompts, 'prompt_sha256': {r: sha_text(p) for r, p in prompts.items()},
            'replacement_rule': RULE, 'KOREAN_byte_identical': prompts['korean'] == previous['prompt']['korean'],
            'gate': {'denominator': SIZE, 'required_explicit_accepted_INSERT_essays': MIN_INSERTIONS,
                     'errors_count_in_denominator': True, 'legacy_EDIT_does_not_count': True},
            'tasks': [{'attempt': a, 'episode_id': eid, 'source_id': corpus[eid]['source_id'],
                       'candidate': previous['corpus'][eid]} for a, eid in tasks],
            'shared_ledger': str(root / 'api/ledger.sqlite'), 'shared_cap_usd': 6,
            'old_slots_saved': len(saved), 'original_total_slots': 2*len(corpus),
            'gpu_calls': 0, 'training': False}
        save_once(directory(config) / phase / 'design.json', value)
    return read_json(directory(config) / 'test_v1/design.json'), corpus


def accepted_insertions(raw, *, explicit=True):
    """Count executed one-sentence insertions, never thoughts, summaries or invalid actions."""
    found = []
    for action in raw.get('actions_by_role', {}).get('global', []):
        if not action.get('valid') or not action.get('created_sids') or action.get('before_hash') == action.get('after_hash'):
            continue
        if action['action'] == 'INSERT':
            found.append({'text': action['args']['text'], 'action': 'INSERT',
                          'created_sids': action['created_sids'], 't': action['t']})
        elif not explicit and action['action'] == 'EDIT' and action['args'].get('target', '').startswith(('before:', 'after:')):
            found.append({'text': action['args']['new_text'], 'action': 'EDIT',
                          'created_sids': action['created_sids'], 't': action['t']})
    return found


def run_phase(config, phase, *, max_api_calls, paid_approved):
    if not paid_approved:
        raise PermissionError('Explicit paid authorization is required')
    prepare(config)
    base = directory(config) / phase
    design_path = base / 'design.json'
    design = read_json(design_path)
    if phase == 'resume_v1':
        gate = read_json(directory(config) / 'test_v1/result.json')
        if gate['decision'] != 'resume' or gate['explicit_INSERT_essays'] < MIN_INSERTIONS:
            raise ValueError('Remaining generation is forbidden after a failed 20-essay gate')
    errors = []
    root = config['paths'][PHASE + '_output']
    with collection_lock(base):
        load_environment(config)
        resources = Resources(config)
        api = V2API(config, max_api_calls, kind='luna', paid_approved=True)
        before = api.accounting()
        if (base / 'budget_before.json').exists():
            before = read_json(base / 'budget_before.json')
        else:
            atomic_new(base / 'budget_before.json', before)
        try:
            api.client()
            api.settle_interrupted()
            for item in design['tasks']:
                a, eid = item['attempt'], item['episode_id']
                path = slot_path(config, phase, a, eid)
                row = read_json(item['candidate']['path'])
                if file_sha(item['candidate']['path']) != item['candidate']['sha256']:
                    raise ValueError('Frozen passing candidate changed')
                if path.exists():
                    raw = read_json(path)
                    if raw['v2']['design_sha256'] != file_sha(design_path):
                        raise ValueError('Saved new-prompt trajectory has a different design')
                else:
                    state = resources.worker()
                    episode = {k: row[k] for k in ('episode_id', 'source_id', 'question', 'genre',
                        'level', 'records', 'preexisting_spell_spans')}
                    episode.update(source=resources.source(row), document=resources.restore(row['corrupted_layout']))
                    env = V2RevisionEnv(config, analysis=state.analysis, tokenizer=state.tokenizer)
                    backend = BulkTeacher(api, a, state.tokenizer)
                    backend.condition = f'insertion_{VERSION}_{phase}_a{a}_{design["prompt_sha256"]["global"][:12]}'
                    raw = run_episode(env, episode, backend, prompt_factory=system_prompt,
                        prompt_variant=VERSION, event_path=base / f'attempt_{a}/events' / (safe_id(eid)+'.jsonl'))
                    raw['v2'] = {'attempt': a, 'design_sha256': file_sha(design_path),
                        'design_path': str(design_path), 'prompt_sha256': design['prompt_sha256'],
                        'operator': 'G_DEL_LINK', 'sampling_seed': None, 'ordering_seed': SEED,
                        'experiment': 'insertion_boost_prompt_fix', 'phase': phase,
                        'original_slot': {'attempt': a, 'episode_id': eid}}
                    atomic_new(path, raw)
                if raw.get('stage1_layout') is not None:
                    for text in (row['corrupted_text'], layout_text(raw['stage1_layout'])):
                        if cached_gpu_score(config, row['question'], text) is None:
                            queue_cpu(config, row['question'], text, requester=f'insertion:{phase}:{a}:{eid}')
                    recovery_path = base / f'attempt_{a}/recovery' / path.name
                    if not recovery_path.exists():
                        try:
                            stages = recover_global(config, row, raw, resources, api)
                            atomic_new(recovery_path, {'episode_sha256': file_sha(path), 'stages': stages})
                        except Exception as exc:
                            error = {'episode_id': eid, 'stage': 'recovery', 'type': type(exc).__name__, 'message': str(exc)}
                            errors.append(error)
                            write_json(base / f'attempt_{a}/recovery_errors' / path.name, error)
                            if isinstance(exc, CallBudgetExceeded):
                                raise
                if raw.get('runtime_error'):
                    errors.append({'episode_id': eid, 'stage': 'episode', **raw['runtime_error']})
                    if raw['runtime_error']['type'] == 'CallBudgetExceeded':
                        raise CallBudgetExceeded(raw['runtime_error']['message'])
                write_json(base / 'progress.json', {'last': item, 'budget': api.accounting(), 'at': time.time()})
                print(json.dumps({'phase': phase, 'attempt': a, 'episode_id': eid,
                    'accepted_INSERT': len(accepted_insertions(raw)), 'completed': raw['completed'],
                    'confirmed_usd': api.accounting()['confirmed_usd']}), flush=True)
        except Exception as exc:
            errors.append({'stage': 'collection', 'type': type(exc).__name__, 'message': str(exc)})
        finally:
            api.close()
            status = {'planned': len(design['tasks']),
                'saved': sum(slot_path(config, phase, x['attempt'], x['episode_id']).exists() for x in design['tasks']),
                'errors': errors, 'budget_before': before, 'budget': api.accounting(), 'gpu_used': False, 'training': False}
            write_json(base / 'status.json', status)
    return status


def summarize_test(config):
    base = directory(config) / 'test_v1'
    design = read_json(base / 'design.json')
    status = read_json(base / 'status.json')
    if status['saved'] != SIZE and not any(e['type'] == 'CallBudgetExceeded' for e in status['errors']):
        raise RuntimeError('20-essay test stopped unexpectedly; do not infer a gate result')
    rows, actions, valid = [], Counter(), Counter()
    for item in design['tasks']:
        path = slot_path(config, 'test_v1', item['attempt'], item['episode_id'])
        candidate = read_json(item['candidate']['path'])
        row = {**{k: item[k] for k in ('attempt', 'episode_id', 'source_id')}, 'path': str(path),
               'deleted_original': ' / '.join(candidate['records'][0]['original_text'].values()),
               'inserted': [], 'explicit_insertions': [], 'GLOBAL_main_recovery': None,
               'R_over': None, 'saved': path.exists()}
        if path.exists():
            raw = read_json(path)
            row.update(sha256=file_sha(path), completed=raw['completed'], runtime_error=raw.get('runtime_error'),
                termination=raw['termination'], explicit_insertions=accepted_insertions(raw),
                inserted=accepted_insertions(raw, explicit=False), steps=raw['steps'])
            for a in raw['actions_by_role'].get('global', []):
                actions[a['action']] += 1
                valid[a['action']] += bool(a['valid'])
            rp = base / f'attempt_{item["attempt"]}/recovery' / path.name
            if rp.exists():
                rec = read_json(rp)
                if rec['episode_sha256'] != file_sha(path):
                    raise ValueError('Recovery file does not match the fresh trajectory')
                stage = rec['stages'].get('global')
                if stage:
                    row.update(GLOBAL_main_recovery=stage['per_record'][0]['main'], R_over=stage['R_over'])
        rows.append(row)
    n = sum(bool(r['explicit_insertions']) for r in rows)
    judged = [r['GLOBAL_main_recovery'] for r in rows if r['GLOBAL_main_recovery'] is not None]
    value = {'status': 'complete' if status['saved'] == SIZE else 'budget_stop',
        'design_path': str(base / 'design.json'), 'design_sha256': file_sha(base / 'design.json'),
        'denominator': SIZE, 'saved': status['saved'], 'distinct_sources': len({r['source_id'] for r in rows}),
        'explicit_INSERT_essays': n, 'explicit_INSERT_rate': n/SIZE,
        'any_accepted_insertion_essays': sum(bool(r['inserted']) for r in rows),
        'decision': 'resume' if n >= MIN_INSERTIONS and status['saved'] == SIZE else 'stop_remaining_generation',
        'required_INSERT_essays': MIN_INSERTIONS, 'actions': dict(actions), 'valid_actions': dict(valid),
        'recovery': {'judged': len(judged), 'full': judged.count(1), 'partial': judged.count(.5),
                     'none': judged.count(0), 'mean_judged': sum(judged)/len(judged) if judged else None,
                     'sum_over_all20': sum(judged)/SIZE, 'unknown': SIZE-len(judged)},
        'rows': rows, 'budget': status['budget'],
        'test_confirmed_usd': status['budget']['confirmed_usd']-status['budget_before']['confirmed_usd'],
        'test_errors': status['errors'], 'role_R': 'pending mandatory GPU reference; not used for this action gate',
        'gpu_calls': 0, 'training': False}
    examples = sorted(rows, key=lambda r: (not bool(r['inserted']), -(r['GLOBAL_main_recovery'] or 0), r['source_id']))[:5]
    value['examples'] = examples
    save_once(base / 'result.json', value)
    lines = ['## GLOBAL content-rule test (fresh 20 essays)', '', RULE, '',
        f"Seed {SEED}; 20 distinct QC-passing training sources; one independent unseeded Luna-low attempt each. "
        'KOREAN and v1 are byte-identical. The old prompt and trajectories remain frozen.', '',
        f"Accepted explicit INSERT: **{n}/20 ({n/SIZE:.1%})**; required 6/20. Decision: **{value['decision']}**. "
        'Thoughts, STOP suggestions, invalid INSERTs and legacy EDIT insertion do not count toward the gate. '
        'An accepted insertion later undone still counts as an executed action; recovery measures the final stage.', '',
        f"Recovery: full {judged.count(1)}, partial {judged.count(.5)}, none {judged.count(0)}, "
        f"unknown {SIZE-len(judged)}. Mean among judged: {value['recovery']['mean_judged']}. "
        'Quality reward and GLOBAL selection remain pending mandatory GPU reference scoring.', '',
        f"New test confirmed cost ${value['test_confirmed_usd']:.6f}; shared confirmed "
        f"${status['budget']['confirmed_usd']:.6f}, reserved ${status['budget']['reserved_usd']:.6f}, cap $6.", '',
        '| GLOBAL action | Calls | Accepted |', '|---|---:|---:|']
    lines += [f'| {a} | {actions[a]} | {valid[a]} |' for a in sorted(actions)]
    lines += ['', 'Five distinct original/actual-insertion pairs (missing insertions are shown honestly):', '']
    for row in examples:
        lines += [f"- `{row['source_id']}`; main recovery={row['GLOBAL_main_recovery']}; R_over={row['R_over']}",
            '  - Deleted original: '+row['deleted_original'],
            '  - Actual inserted: '+(' / '.join(x['text'] for x in row['inserted']) or '(none)'), '']
    (base / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return value


def run_test(config, *, max_api_calls, paid_approved):
    run_phase(config, 'test_v1', max_api_calls=max_api_calls, paid_approved=paid_approved)
    return summarize_test(config)


def finish_test_report(config):
    """File-only detail appendix, safe even when the collecting worker predates this helper."""
    base = directory(config) / 'test_v1'
    result = read_json(base / 'result.json')
    design = read_json(base / 'design.json')
    pause = read_json(directory(config) / 'pause/snapshot.json')
    diagnosis = read_json(directory(config) / 'diagnosis82.json')
    diagnosed = len(diagnosis['rows'])
    retained_reservation = sum(r['reserved'] for r in pause['pending'])
    rows = []
    for row, task in zip(result['rows'], design['tasks']):
        item = dict(row)
        candidate = read_json(task['candidate']['path'])
        target = candidate['records'][0]['recovery_target']
        item['source_target'] = target
        item['recovery_position_tolerance'] = 1
        rp = base / f'attempt_{task["attempt"]}/recovery' / Path(row['path']).name
        recovery = read_json(rp)['stages'].get('global') if rp.exists() else None
        item['recovery_details'] = recovery['per_record'][0]['recovery_details'] if recovery else None
        if item['GLOBAL_main_recovery'] is None:
            item['reason'] = 'Recovery is unmeasured; not a zero.'
        elif not item['inserted']:
            item['reason'] = 'No accepted sentence insertion.'
        elif item['GLOBAL_main_recovery'] == 1:
            item['reason'] = 'At least one inserted sentence within source position ±1 received Luna yes.'
        elif item['GLOBAL_main_recovery'] == .5:
            item['reason'] = 'Best inserted sentence within source position ±1 received Luna partial.'
        elif not item['recovery_details']:
            item['reason'] = 'No surviving insertion at the source paragraph and source position ±1.'
        else:
            item['reason'] = 'Inserted sentence within source position ±1 received Luna no.'
        rows.append(item)
    positive = sorted([r for r in rows if r['inserted']], key=lambda r: (-(r['GLOBAL_main_recovery'] or 0), r['source_id']))
    failure = [r for r in rows if r['GLOBAL_main_recovery'] != 1]
    examples = positive[:5]
    if failure and len(examples) == 5 and all(r['GLOBAL_main_recovery'] == 1 for r in examples):
        examples[-1] = failure[0]
    for row in rows:
        if len(examples) >= 5:
            break
        if row['source_id'] not in {r['source_id'] for r in examples}:
            examples.append(row)
    mismatch = result['explicit_INSERT_essays'] < MIN_INSERTIONS <= result['any_accepted_insertion_essays']
    over = [r['R_over'] for r in rows if r['R_over'] is not None]
    value = {'result_path': str(base / 'result.json'), 'result_sha256': file_sha(base / 'result.json'),
        'legacy_changes_gate': mismatch, 'completed': sum(r.get('completed', False) for r in rows),
        'explicit_INSERT_essays': result['explicit_INSERT_essays'],
        'any_accepted_insertion_essays': result['any_accepted_insertion_essays'],
        'legacy_only_insertion_essays': sum(bool(r['inserted']) and not r['explicit_insertions'] for r in rows),
        'observed_recovery_lower_bound_over_all20': result['recovery']['sum_over_all20'],
        'unknown_recovery_count': result['recovery']['unknown'],
        'mean_R_over': sum(over)/len(over) if over else None, 'R_over_available': len(over),
        'examples': examples, 'rows': rows, 'old_saved_at_pause': pause['saved_attempts'],
        'diagnostic_saved_cohort': diagnosed,
        'extra_old_attempts_preserved_outside_diagnosis': pause['saved_attempts']-diagnosed,
        'pause_unknown_api_reserved_usd': retained_reservation, 'new_paid_calls': 0}
    save_once(base / 'detail.json', value)
    paragraphs = ['## GLOBAL content-rule test (fresh 20 essays)', '', RULE, '',
        f"Seed {SEED}; 20 distinct QC-passing training sources; one fresh independent Luna-low attempt each. "
        f"The sample occupies existing missing slots, preserving the {design['original_total_slots']}-slot total. "
        'KOREAN and v1 remain byte-identical. '
        f"The old-prompt supervisor was paused at {pause['saved_attempts']} saved attempts: the requested first"
        f"{diagnosed} diagnostic plus{pause['saved_attempts']-diagnosed} later saved attempts are all preserved. "
        f"Interrupted API outcomes retain their ${retained_reservation:.8f} reservation.", '',
        f"Completed: **{value['completed']}/20**. Accepted explicit INSERT: **{result['explicit_INSERT_essays']}/20 "
        f"({result['explicit_INSERT_rate']:.1%})**. Any accepted whole-sentence insertion including EDIT: "
        f"**{result['any_accepted_insertion_essays']}/20**; legacy-only essays {value['legacy_only_insertion_essays']}. "
        f"Required:6/20. Decision:**{result['decision']}**. "
        'The gate counts accepted actions that created a sentence and changed the text. Thoughts, STOP suggestions, '
        'invalid actions and SPLIT do not qualify. Later UNDO is reported by final-stage recovery.', '',
        f"Recovery: full {result['recovery']['full']}, partial {result['recovery']['partial']}, "
        f"none {result['recovery']['none']}, unknown {result['recovery']['unknown']}. "
        f"Mean among judged={result['recovery']['mean_judged']}; observed lower bound over all20="
        f"{result['recovery']['sum_over_all20']}. Unknown cases are not assigned zero. "
        f"Mean R_over={value['mean_R_over']} over {len(over)} measured trajectories. "
        'Role R and selections await mandatory GPU reference scores; this gate uses actions only.', '',
        f"Test confirmed cost **${result['test_confirmed_usd']:.6f}**; total shared confirmed "
        f"${result['budget']['confirmed_usd']:.6f}, reserved ${result['budget']['reserved_usd']:.6f}, $6 cap.", '',
        '| GLOBAL action | Calls | Accepted |', '|---|---:|---:|']
    paragraphs += [f"| {a} | {n} | {result['valid_actions'][a]} |" for a, n in sorted(result['actions'].items())]
    paragraphs += ['', 'Five distinct original/actual-insertion pairs, including a failure when available:', '']
    for row in examples:
        paragraphs += [f"- `{row['source_id']}`; recovery={row['GLOBAL_main_recovery']}; R_over={row['R_over']}",
            '  - Deleted original: '+row['deleted_original'],
            '  - Actual inserted: '+(' / '.join(x['text'] for x in row['inserted']) or '(none)'),
            '  - Recovery finding: '+row['reason'], '']
    if mismatch:
        paragraphs += ['Legacy EDIT insertions change the 30% decision; automatic continuation is held for interpretation.', '']
    (base / 'report.md').write_text('\n'.join(paragraphs)+'\n', encoding='utf-8')
    return value


def continue_run(config, *, max_api_calls):
    """Wait for the independently running test, then enforce its predeclared gate."""
    import os
    root = config['paths'][PHASE + '_output']
    base = directory(config)
    def update(stage, **more):
        write_json(base / 'continuation_status.json', {'pid': os.getpid(), 'stage': stage,
            'at': time.time(), 'training': False, 'gpu_calls': 0, **more})
    with collection_lock(base / 'controller'):
        try:
            update('waiting_for_20_essay_test')
            result_path = base / 'test_v1/result.json'
            while not result_path.exists():
                time.sleep(5)
            test = read_json(result_path)
            detail = finish_test_report(config)
            if detail['legacy_changes_gate']:
                raise RuntimeError('Legacy whole-sentence EDIT insertions change the 30% gate; notify parent')
            if test['decision'] == 'resume':
                update('resuming_remaining_slots', test_sha256=file_sha(result_path))
                resume = run_phase(config, 'resume_v1', max_api_calls=max_api_calls, paid_approved=True)
                if resume['saved'] != resume['planned'] and not any(e['type'] == 'CallBudgetExceeded' for e in resume['errors']):
                    raise RuntimeError('Remaining teacher dispatch stopped unexpectedly')
            else:
                resume = {'planned': len(read_json(base / 'resume_v1/design.json')['tasks']),
                    'saved': 0, 'errors': [], 'status': 'not_dispatched_by_user_gate'}
                save_once(base / 'resume_v1/status.json', resume)
            from .collection import entries
            from ..v2_ops.report import accounting
            saved = list(entries(config))
            errors = test['test_errors'] + resume['errors']
            stopped_at_cap = any(e['type'] == 'CallBudgetExceeded' for e in errors)
            reason = ('budget_cap' if stopped_at_cap else
                      'user_prompt_gate_failed' if test['decision'] != 'resume' else 'all_requested_slots_complete')
            total_slots = read_json(base / 'test_v1/design.json')['original_total_slots']
            status = {'planned_attempts': total_slots, 'saved_attempts': len(saved), 'errors': errors,
                'stop_reason': reason, 'prompt_test_denominator': 20,
                'prompt_test_insertions': test['explicit_INSERT_essays'],
                'prompt_test_sha256': file_sha(result_path), 'budget': accounting(root),
                'teacher_collection_finished': True, 'training': False, 'gpu_used': False}
            write_json(root / 'teacher_status.json', status)
            update('global_scoring', teacher=status)
            from .evaluate import run as evaluate
            evaluate(config)
            from .report import report
            result = report(config, final=True)
            update('complete', report=result['report'], selected_counts=result['selected_counts'])
            return result
        except Exception as exc:
            update('error', error={'type': type(exc).__name__, 'message': str(exc)})
            raise
