"""Saved-data-only full v4 exports. No scorer, analyzer, teacher or GPU calls."""
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path

from .common import REPO, atomic_new, file_sha, read_json, sha_text, safe_id, write_json
from .content_env import dumps
from .dryrun_common import BASE, REVISION, tokenizer
from .dryrun_data import audit_sample
from .render_v43 import assert_rendered
from . import orchestrator_v44 as orch
from verak.v3.train.formatting import encode_labels, lengths
from verak.v3.rft1.oneshot import encode_final_essay, messages_for

ROOT = REPO / 'verak/v4/outputs/train44'
SCALE = REPO / 'verak/v4/outputs/scale3'
MAPS = REPO / 'verak/v4/outputs/prep3/maps/scale2000'
DRY = REPO / 'verak/v4/outputs/dryrun'
ROLES = ('korean', 'revision', 'orchestrator', 'oneshot')


def ref(path):
    return {'path': str(path), 'sha256': file_sha(Path(path))}


def verified(reference):
    if file_sha(Path(reference['path'])) != reference['sha256']:
        raise ValueError('Saved artifact changed: ' + reference['path'])
    return read_json(Path(reference['path']))


def normalized(text):
    return sha_text(''.join(text.split()))


def shared_split(role_sources, *, seed=71):
    """Exact ceil(5%) per role with one shared source partition.

    Solve only up to fifteen membership strata, never individual random labels.
    Absolute stratum deviation favors representative membership proportions.
    SHA ordering then deterministically chooses the essays in each stratum.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    roles = list(role_sources)
    union = set().union(*role_sources.values())
    strata = defaultdict(list)
    for source in sorted(union):
        mask = tuple(int(source in role_sources[r]) for r in roles)
        strata[mask].append(source)
    masks = sorted(strata); n = len(masks)
    target = np.array([math.ceil(.05 * len(role_sources[r])) for r in roles], dtype=float)
    A = np.zeros((len(roles)+n, 3*n))
    A[:len(roles), :n] = np.array(masks).T
    expected = np.array([.05*len(strata[m]) for m in masks])
    A[len(roles):, :n] = np.eye(n)
    A[len(roles):, n:2*n] = -np.eye(n)
    A[len(roles):, 2*n:] = np.eye(n)
    rhs = np.r_[target, expected]
    result = milp(np.r_[np.arange(n)*1e-6, np.ones(2*n)],
        integrality=np.r_[np.ones(n), np.zeros(2*n)],
        bounds=Bounds(np.zeros(3*n), np.r_[[len(strata[m]) for m in masks], np.full(2*n, np.inf)]),
        constraints=LinearConstraint(A, rhs, rhs), options={'time_limit': 30})
    if not result.success:
        raise ValueError('Cannot form exact shared 5% source split: '+str(result.message))
    held = set()
    for mask, size in zip(masks, result.x[:n]):
        ordered = sorted(strata[mask], key=lambda s: (sha_text(f'{seed}:{s}'), s))
        held.update(ordered[:round(size)])
    assert all(len(held & role_sources[r]) == int(t) for r, t in zip(roles, target))
    return held


def metadata(row, *, source_text=None):
    text = source_text if source_text is not None else row['text']
    return {'source_id': row['source_id'], 'question_hash': sha_text(row['question']),
            'normalized_text_hash': normalized(text),
            'normalized_space_hash': sha_text(' '.join(text.split()))}


def paragraphs_for(attempt, role, delegation, fallback):
    call = next((c for c in attempt.get('calls', []) if c['role'] == role and c['delegation'] == delegation), None)
    if call:
        return assert_rendered(call['public_messages'])['paragraphs']
    return deepcopy(fallback)


def observed_report(stage):
    if stage is None:
        return {'status': 'unavailable'}
    # Status is factual. Do not leak privileged task IDs or free-text teacher
    # reports quoting rubric feedback into another agent's observation.
    return {k: deepcopy(stage.get(k)) for k in ('role', 'delegation', 'ending', 'stop_status', 'steps')}


def plan_samples(row, plan, attempt, provenance, notes=None):
    """Partially supervised decisions with real saved worker states/reports.

    No SCORE results or MAPs are fabricated to fill gaps in the saved teacher
    data. Independent MAP examples below teach the MAP action. The observation
    renderer and action schema are exactly those used by integrated inference.
    """
    current, history, reports = deepcopy(row['paragraphs']), [], []
    def sample(action, phase, paragraphs):
        checked, _ = orch.validate(dumps(action), paragraphs)
        assert checked == action
        return {'source_id': row['source_id'], 'role': 'orchestrator', 'kind': provenance['kind'],
            'provenance': provenance, 'messages': orch.messages(row['question'], paragraphs,
                phase=phase, history=history, reports=reports)+[{'role': 'assistant', 'content': dumps(action)}]}
    yield sample({'action': 'SCORE'}, 'score', current)
    revisions = [i for i in plan['items'] if i['owner'] == 'revision']
    korean = [i for i in plan['items'] if i['owner'] == 'korean']
    for number, start in enumerate(range(0, len(revisions), 2), 1):
        current = paragraphs_for(attempt, 'revision', number, current)
        action = {'action': 'DELEGATE', 'to': 'revision', 'items': orch.public_items(revisions[start:start+2])}
        yield sample(action, 'plan', current)
        history.append(action)
        stage = next((s for s in attempt.get('delegations', []) if s['delegation'] == number), None)
        reports.append(observed_report(stage))
    current = paragraphs_for(attempt, 'korean', 1, current)
    # A saved Revision attempt can delete a later task's sentence. Do not teach
    # an impossible delegation. Keep the remaining tasks and mandatory whole-
    # essay pass, while retaining the actual blocked worker report below.
    ids = {p['id'] for p in current} | {s['id'] for p in current for s in p['sentences']}
    obsolete = [i for i in korean if any(x not in ids for x in i['location'])]
    if obsolete:
        korean = [i for i in korean if i not in obsolete]
        if notes is not None:
            notes.append({'kind': 'orchestrator_obsolete_locations', 'source_id': row['source_id'],
                'episode_id': provenance.get('episode_id'),
                'removed_task_locations': [i['location'] for i in obsolete],
                'reason': 'Prior saved Revision deleted the target; no recovery is inferred.'})
    action = {'action': 'DELEGATE', 'to': 'korean', 'items': orch.public_items(korean)}
    yield sample(action, 'plan', current)
    history.append(action); reports.append(observed_report(attempt.get('korean_stage')))
    current = attempt.get('final_paragraphs') or current
    unfinished = any(r.get('ending') != 'STOP' or r.get('stop_status') == 'blocked' for r in reports)
    action = {'action': 'FINISH', 'summary': '일부 위임이 완료되지 않았다.' if unfinished else '배정한 단계가 STOP으로 종료되었다.',
              'writer_notes': deepcopy(plan.get('writer_notes', []))}
    yield sample(action, 'plan', current)


def cited_final(attempt):
    citations = {c['sentence_id']: c['source'] for c in attempt.get('source_citations', [])}
    if not citations:
        return attempt['final_text']
    return '\n\n'.join(' '.join(s['text'] + (' [출처: '+citations[s['id']]['title']+' | '+
        citations[s['id']]['passage_id']+']' if s['id'] in citations else '')
        for s in p['sentences']) for p in attempt['final_paragraphs'])


def build(*, root=ROOT):
    root = Path(root)
    if (root/'data/manifest.json').exists():
        return verify(root=root)
    tok = tokenizer(); orch.freeze(tok)
    inputs, rows, candidates, notes = {}, {}, {r: [] for r in ROLES}, []
    def load(path):
        path = Path(path); inputs[str(path)] = file_sha(path)
        return read_json(path)
    def keep_row(row, source_text=None):
        meta = metadata(row, source_text=source_text)
        if row['source_id'] in rows and rows[row['source_id']] != meta:
            raise ValueError('Conflicting original source identity: '+row['source_id'])
        rows[row['source_id']] = meta
    evaluation = load(DRY/'evaluation/manifest.json')
    excluded_ids, excluded_hashes = set(), set()
    for cohort in evaluation['cohorts'].values():
        for case in cohort['cases']:
            excluded_ids.add(case['source_id'])
            evidence = case.get('prepared') or case.get('row')
            saved = verified(evidence)
            raw = saved.get('policy_row', saved)
            excluded_hashes.add(normalized(saved.get('corpus', {}).get('source_text', raw['text'])))
    design = load(REPO/'verak/v4/outputs/prep/design.json')
    test_questions = set(design['selected_test_questions'])
    excluded_ids.update(design['test100'])
    for sid in design['test100']:
        excluded_hashes.add(design['source_metadata'][sid]['normalized_source_hash'])
    # Keep original v4.3 messages, IDs, labels and canonical action targets.
    for component in ('B2', 'B3'):
        base = SCALE/component/('content/export' if component == 'B2' else 'export')
        contract = load(base/'contract.json')
        for role in ('revision', 'korean'):
            path = Path(contract['roles'][role]['path'])
            if file_sha(path) != contract['roles'][role]['sha256']:
                raise ValueError('Accepted worker export changed')
            inputs[str(path)] = file_sha(path)
            with path.open() as stream:
                for line in stream:
                    sample = json.loads(line)
                    stats = audit_sample(sample, tok)
                    candidates[role].append({**sample, **stats, 'kind': component,
                        'provenance': {'path': str(path), 'sha256': inputs[str(path)]}})
            print(json.dumps({'export_read': component+'/'+role, 'targets': len(candidates[role])}), flush=True)
    # B2: all 1,000 saved plans; the quality-kept final attempts for one-shot.
    selections = load(SCALE/'B2/selections.json')
    kept = {(s['source_id'], s['attempt']) for s in selections if s['export_keep']}
    for target_path in sorted((SCALE/'B2/orchestrator_targets').glob('*.json')):
        target = load(target_path); sid = target['source_id']
        row = load(SCALE/'B2/essays'/(safe_id(sid)+'.json')); keep_row(row)
        judgment = load(SCALE/'B2/content/judgments'/(safe_id(sid)+'.json'))
        attempts = [load(SCALE/'B2/content/attempts'/f'{safe_id(sid)}_a{i}.json') for i in (1, 2)]
        preferred = next((a for a in attempts if str(a['attempt']) == str(judgment.get('preferred'))), attempts[0])
        plan = target['target']
        provenance = {**ref(target_path), 'kind': 'B2_plan', 'saved_attempt_for_observable_state': preferred['attempt']}
        try:
            candidates['orchestrator'].extend(list(plan_samples(row, plan, preferred, provenance, notes)))
        except (ValueError, KeyError) as exc:
            # A malformed target is visible; never silently repair its content.
            notes.append({'kind': 'B2_plan_export_failure', 'source_id': sid, 'error': str(exc)})
        for attempt in attempts:
            if (sid, attempt['attempt']) in kept:
                candidates['oneshot'].append({'source_id': sid, 'role': 'oneshot', 'kind': 'B2',
                    'attempt': attempt['attempt'], 'messages': messages_for(row['question'], row['text'])+
                    [{'role': 'assistant', 'content': cited_final(attempt)}],
                    'provenance': ref(SCALE/'B2/content/attempts'/f'{safe_id(sid)}_a{attempt["attempt"]}.json')})
    # B3: all active practices for planning; accepted role selections stay fixed.
    from .scale3_corruption import role_eligible
    selected = load(SCALE/'B3/gpu_selection.json')
    eligible_inputs = set().union(*(set(s) for s in selected['selected'].values()))
    for path in sorted((SCALE/'B3/essays').glob('*.json')):
        prepared = load(path); corpus, row = prepared['corpus'], prepared['policy_row']
        keep_row(row, corpus['source_text'])
        eid = corpus['episode_id']; stem = safe_id(eid)
        attempts = [load(SCALE/'B3/content/attempts'/f'{stem}_a{i}.json') for i in (1, 2)]
        measured = []
        for i in (1, 2):
            p = SCALE/'B3/gpu_measured'/f'{stem}_a{i}.json'
            if p.exists():
                a = load(p)
                if a.get('completed') and (a.get('reward') or {}).get('combined'):
                    measured.append(a)
        best = max(measured, key=lambda a: (a['reward']['combined']['R'], -a['attempt']), default=None)
        planning_attempt = next((a for a in attempts if best and a['attempt'] == best['attempt']), attempts[0])
        try:
            candidates['orchestrator'].extend(list(plan_samples(row, prepared['plan'], planning_attempt,
                {**ref(path), 'kind': 'B3_plan', 'episode_id': eid, 'saved_attempt_for_observable_state': planning_attempt['attempt']}, notes)))
        except (ValueError, KeyError) as exc:
            notes.append({'kind': 'B3_plan_export_failure', 'episode_id': eid, 'error': str(exc)})
        reason = 'not_in_agent_selection' if eid not in eligible_inputs else 'no_completed_GPU_reward' if best is None else None
        if reason is None:
            for role in ('global', 'korean'):
                if role == 'global' and not any(r['level'] == 'GLOBAL' for r in corpus['records']):
                    continue
                accepted, why = role_eligible(best, role, corpus)
                if not accepted:
                    reason = 'higher_combined_attempt_fails_'+role+':'+why
                    break
        if reason is not None:
            notes.append({'kind': 'B3_oneshot_excluded', 'episode_id': eid, 'reason': reason})
        else:
            raw = next(a for a in attempts if a['attempt'] == best['attempt'])
            candidates['oneshot'].append({'source_id': row['source_id'], 'corpus_episode_id': eid,
                'role': 'oneshot', 'kind': 'B3', 'attempt': best['attempt'],
                'messages': messages_for(row['question'], row['text'])+[{'role': 'assistant', 'content': cited_final(raw)}],
                'provenance': ref(SCALE/'B3/gpu_measured'/f'{stem}_a{best["attempt"]}.json')})
    source_files = load(MAPS.parent/'source_files.json')
    map_design = load(MAPS/'design.json')
    for sid in map_design['source_ids']:
        reference = source_files[sid]
        row = verified(reference); inputs[reference['path']] = reference['sha256']; keep_row(row)
        path = MAPS/'consensus'/(safe_id(sid)+'.json'); consensus = load(path)
        if consensus['status'] != 'valid':
            notes.append({'kind': 'map_unavailable', 'source_id': sid, 'status': consensus['status']})
            continue
        action = {'action': 'MAP', 'map': consensus['map']}
        checked, diagnostics = orch.validate(dumps(action), row['paragraphs'])
        if checked != action or diagnostics['dropped_edges']:
            raise ValueError('Saved consensus changed under the shared map validator')
        candidates['orchestrator'].append({'source_id': sid, 'role': 'orchestrator', 'kind': 'Luna_MAP',
            'provenance': ref(path), 'messages': orch.messages(row['question'], row['paragraphs'], phase='map')+
            [{'role': 'assistant', 'content': dumps(action)}]})
    # Normalize aliases of an identical source to one split group.
    aliases = defaultdict(list)
    for sid, meta in rows.items():
        aliases[meta['normalized_text_hash']].append(sid)
    group = {sid: min(values) for values in aliases.values() for sid in values}
    rejected, encoded = [], {r: [] for r in ROLES}
    for role in ROLES:
        for sample in candidates[role]:
            sid = sample['source_id']; meta = rows[sid]
            if (sid in excluded_ids or meta['normalized_text_hash'] in excluded_hashes
                    or meta['normalized_space_hash'] in excluded_hashes or meta['question_hash'] in test_questions):
                rejected.append({'role': role, 'source_id': sid, 'kind': sample['kind'], 'reason': 'diagnostic_or_test_exclusion'})
                continue
            if role not in ('korean', 'revision'):
                encoding = encode_final_essay(tok, sample['messages']) if role == 'oneshot' else encode_labels(tok, sample['messages'], last_assistant_only=True)
                sample.update(encoding)
                prefix = len(tok.apply_chat_template(sample['messages'][:-1], tokenize=True, add_generation_prompt=True))
                sample.update(prefix_tokens=prefix, total_tokens=len(sample['input_ids']),
                              target_tokens=sum(x != -100 for x in sample['labels']))
            if sample['total_tokens'] > 8192:
                rejected.append({'role': role, 'source_id': sid, 'kind': sample['kind'], 'reason': 'training_context_limit',
                                 'tokens': sample['total_tokens']})
                continue
            sample['source_group'] = group[sid]
            encoded[role].append(sample)
    source_sets = {r: {s['source_group'] for s in samples} for r, samples in encoded.items()}
    held = shared_split(source_sets)
    result = {'schema_version': 1, 'base_revision': REVISION, 'input_artifacts': inputs,
        'worker_training_prompt_version': 'v4.3', 'inference_environment_version': 'v4.4',
        'orchestrator_contract': ref(orch.FROZEN), 'roles': {}, 'notes': notes, 'excluded_targets': rejected,
        'source_metadata': rows, 'source_groups': group, 'shared_validation_groups': sorted(held),
        'split': {'seed': 71, 'fraction': .05, 'rounding': 'ceil per role; shared normalized-source groups',
                  'no_cross_role_or_oneshot_leakage': True},
        'test_exclusion': {'source_ids': sorted(excluded_ids), 'test_question_hashes': sorted(test_questions),
                           'test_cohort_evaluated': False},
        'worker_tokens_unchanged': True, 'packing': False, 'truncation': False,
        'new_API_calls': 0, 'GPU_calls': 0,
        'one_shot_filter': 'B2 all kept final attempts; B3 higher-combined-R completed attempt first, then every relevant role SFT gate; no lower-attempt fallback',
        'orchestrator_supervision': 'partial decisions; real saved pre-delegation states and status-only reports; no invented SCORE/MAP outputs'}
    (root/'data').mkdir(parents=True, exist_ok=True)
    for role in ROLES:
        result['roles'][role] = {}
        for part in ('train', 'validation'):
            samples = [s for s in encoded[role] if (s['source_group'] in held) == (part == 'validation')]
            path = root/'data'/f'{role}.{part}.jsonl.gz'
            with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
                for s in samples:
                    stream.write((dumps({**s, 'partition': part})+'\n').encode())
            result['roles'][role][part] = {**ref(path), 'turns': len(samples),
                'sources': len({s['source_id'] for s in samples}), 'source_groups': len({s['source_group'] for s in samples}),
                'source_ids': sorted({s['source_id'] for s in samples}), 'kinds': dict(Counter(s['kind'] for s in samples)),
                'tokens': sum(s['total_tokens'] for s in samples), 'target_tokens': sum(s['target_tokens'] for s in samples),
                'lengths': lengths([s['total_tokens'] for s in samples]),
                'actions': dict(Counter(json.loads(s['messages'][-1]['content'])['action'] for s in samples)) if role != 'oneshot' else {},
                'trajectory_count': len({(s['source_id'], s.get('corpus_episode_id'), s.get('attempt')) for s in samples})}
    atomic_new(root/'data/manifest.json', result)
    return result


def verify(*, root=ROOT):
    manifest = read_json(Path(root)/'data/manifest.json')
    if manifest['base_revision'] != REVISION:
        raise ValueError('Pinned base changed')
    orch.verify()
    for role in ROLES:
        for part in ('train', 'validation'):
            r = manifest['roles'][role][part]
            if file_sha(Path(r['path'])) != r['sha256']:
                raise ValueError('Frozen training data changed')
    return manifest


if __name__ == '__main__':
    value = build()
    print(json.dumps({'complete': True, 'roles': {r: {p: x['turns'] for p, x in v.items()}
          for r, v in value['roles'].items()}, 'notes': len(value['notes'])}), flush=True)
