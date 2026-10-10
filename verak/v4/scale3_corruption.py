"""B3: v4.3 delegation on the unchanged 1,430 active corruption practices.

Collection is API/CPU only. The unchanged v1 arithmetic is applied to exact
projections with saved GPU-reference Q endpoints, only on explicit finalization.
This module never schedules a scorer or training job.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import fcntl
from pathlib import Path
from types import SimpleNamespace
import json
import math
import os
import random
import tempfile
import time

from verak.v3.common import pair_key
from .common import (REPO, atomic_new, collection_lock, file_sha, load_config,
                     read_json, safe_id, sha_text, write_json)
from .scale3_common import (ROOT, ATTEMPTS, api_for, bounded_episodes, component_accounting,
                           component_contract, dispatch_limit, freeze, reward_contract, trim_metrics)
from .scale2_projection import (ProjectionUnknown, initial_recovery, project_document,
    restore_document, reward_actions, save_document, public_document, private_document,
    structural_split_attempt)

ROLE_NAMES = {'revision': 'global', 'korean': 'korean'}


def assert_public_teacher(messages, prepared):
    """Check hidden supervision without changing a single inference-visible byte."""
    from .content_env import dumps
    public = dumps(messages)
    forbidden = ('recovery_target', 'original_text', 'corrupted_text', 'record_id',
                 'grader_1_scores', 'grader_2_scores', 'LLM_rubric_feedback',
                 'stable_id_aliases', 'public_to_private_sentences', 'structural_split_attempt')
    if any('"'+key+'"' in public or '\\"'+key+'\\"' in public for key in forbidden):
        raise ValueError('Private B3 supervision leaked into public observations')
    identifiers = [r['record_id'] for r in prepared['corpus']['records']]
    identifiers += [i['item_id'] for i in prepared['plan']['items']]
    if any(identity and identity in public for identity in identifiers):
        raise ValueError('Private B3 record/task identity leaked into public observations')


@contextmanager
def finalize_lock(root):
    """Concurrent file-only finalizers wait, then reuse the immutable selection."""
    root.mkdir(parents=True, exist_ok=True)
    with (root/'finalize.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def freeze_sample(root=ROOT):
    """File-only active-corpus identity; does not pass or bypass the paid gate."""
    from verak.v3.phase2 import read_jsonl
    root = Path(root)
    config = load_config()
    path = config['paths']['active_corrupt']/'agent_train.jsonl'
    rows = read_jsonl(path)
    if len(rows) != 1430 or any(r['split'] != 'agent_train' for r in rows):
        raise ValueError('B3 requires all 1,430 active agent_train practices unchanged')
    fingerprints = {r['scorer_fingerprint'] for r in rows}
    if len(fingerprints) != 1:
        raise ValueError('Active corpus has more than one reference scorer fingerprint')
    order = [r['episode_id'] for r in rows]
    if len(set(order)) != len(order):
        raise ValueError('Duplicate active corruption episode identity')
    random.Random(541).shuffle(order)
    manifest = {'schema_version': 1, 'component': 'B3', 'collection_version': 'v4.3',
        'corpus_path': str(path), 'corpus_sha256': file_sha(path), 'episode_ids': order,
        'source_essays': len({r['source_id'] for r in rows}), 'practices': len(rows),
        'attempts_per_practice': 2, 'planned_attempts': 2860, 'ordering_seed': 541,
        'provider_sampling_seed': None, 'additional_source_exclusions': False,
        'operators': dict(Counter(x['op'] for r in rows for x in r['records'])),
        'no_GLOBAL_practices': sum(not any(x['level'] == 'GLOBAL' for x in r['records']) for r in rows),
        'no_GLOBAL_exception': 'valid terminal STOP within 2 steps, no structural attempt, R_over=0; <=1 rejection',
        'role_R_threshold': .8, 'terminal_STOP': True, 'max_rejected_actions': 1,
        'RFT_STOP_cap_or_operator_duplication': False, 'reference_gpu_fingerprint': next(iter(fingerprints)),
        'quality_endpoints': 'actual normalized initial, Revision final, Korean final; never raw corpus q_corrupted',
        'unrepresentable_projection': 'reward unknown and unselected; not zero or a replacement essay'}
    freeze(root/'B3/sample.json', manifest)
    return manifest, {r['episode_id']: r for r in rows}


def corruption_plan(row, prepared, aliases=None):
    """Reveal a task/site, never a recovery target, source quote or destination."""
    paragraphs = [p['id'] for p in prepared['paragraphs']]
    mapping = prepared['normalization']['sid_mapping']
    private_to_public = {private: public for public, private in aliases['public_to_private_sentences'].items()} if aliases else {}
    items = []
    instructions = {
        'G_PARA_SWAP': '문단의 현재 순서가 글의 주제 전개에 맞는지 검토하고 필요한 문단 배치를 고친다.',
        'G_SENT_MOVE': '지정 문장의 현재 위치와 앞뒤 연결을 검토하고 글의 흐름에 맞게 배치를 고친다.',
        'G_OFFTOPIC': '지정 문장이 문항과 주변 글의 흐름에 필요한지 검토하고, 관련 없는 문장만 삭제한다.',
        'L_CONN': '지정 위치의 연결어미가 앞뒤 절의 관계에 맞는지 검토하고 필요한 부분만 고친다.',
        'L_CONJ': '지정 위치의 문두 접속사가 앞 문장과의 관계에 맞는지 검토하고 필요한 부분만 고친다.',
        'L_REGISTER': '지정 위치의 종결체가 글 전체의 문체와 일관되는지 검토하고 필요한 부분만 고친다.',
        'L_SUBJ_INSERT': '지정 위치의 주어 표현과 생략이 앞뒤 문맥에서 자연스러운지 검토하고 필요한 부분만 고친다.',
        'L_SPACING': '지정 위치의 띄어쓰기를 검토하고 필요한 부분만 고친다.',
        'L_POLARITY': '지정 위치의 부정·양태 표현이 문법과 문맥에 맞는지 검토하고 필요한 부분만 고친다.',
    }
    for number, record in enumerate(row['records'], 1):
        op = record['op']
        if op not in instructions:
            raise ValueError('Unexpected active B3 operator: '+op)
        role = 'revision' if record['level'] == 'GLOBAL' else 'korean'
        locations = list(dict.fromkeys(s for sid in record['sids'] for s in mapping.get(private_to_public.get(sid, sid), [])))
        if role == 'revision' and op in {'G_PARA_SWAP', 'G_SENT_MOVE'}:
            locations = list(dict.fromkeys(locations+paragraphs))
        if not locations:
            raise ValueError('A corruption task has no surviving normalized source location')
        items.append({'item_id': row['episode_id']+f':B3I{number}', 'owner': role,
            'location': locations, 'action': 'MOVE' if op in {'G_PARA_SWAP', 'G_SENT_MOVE'} else 'DELETE' if op == 'G_OFFTOPIC' else 'EDIT',
            'instruction': instructions[op], 'record_id': record['record_id'], 'op': op, 'needs_search': 'no'})
    if not any(i['owner'] == 'revision' for i in items):
        items.insert(0, {'item_id': row['episode_id']+':B3_REVIEW', 'owner': 'revision',
            'location': paragraphs, 'action': 'EDIT', 'record_id': None, 'op': None,
            'instruction': '글 전체의 조직과 문장 배치를 한 번 검토한다. 필요한 구조 수정이 없으면 바로 STOP으로 보고한다.',
            'needs_search': 'no'})
    return {'source_id': row['source_id'], 'corpus_episode_id': row['episode_id'], 'status': 'completed',
        'items': items, 'writer_notes': [], 'provenance': 'record-derived locations/types only; answer targets private',
        'no_GLOBAL_review_delegation': not any(r['level'] == 'GLOBAL' for r in row['records'])}


def prepare_corruption(row, *, root=ROOT, config=None, examples=None):
    """Freeze all Bareun evidence during collection; later reward code is file-only."""
    from verak.v3.corrupt.document import Document, source_document
    from verak.v3.view_data import load_episode_examples
    from .environment_v43 import BoundaryParagraphs, normalize_row, normalize_document, document_profile
    root = Path(root)
    path = root/'B3/essays'/(safe_id(row['episode_id'])+'.json')
    if path.exists():
        saved = read_json(path)
        if saved['corpus'] != row:
            raise ValueError('B3 prepared corruption differs from its frozen corpus row')
        return saved, path
    config = config or load_config()
    if row.get('preexisting_spell_spans'):
        raise ValueError('This frozen B3 corpus has no spelling spans; new offset mapping needs review')
    examples = examples or {e.id: e for e in load_episode_examples(config, 'agent_train')}
    analysis = BoundaryParagraphs(config, cache_dir=root/'B3/bareun_sources')
    source = source_document(config, examples[row['source_id']], SimpleNamespace(seed=lambda _: None))
    raw_corrupted = Document.restore(row['corrupted_layout'], SimpleNamespace(tokens=lambda _: []))
    if source.text != row['source_text'] or raw_corrupted.text != row['corrupted_text']:
        raise ValueError('B3 stable source/corrupted text differs from the active corpus')
    # ParagraphAnalyzer.refresh (not BoundaryParagraphs.refresh) is required for
    # this raw audit copy: normalizing its whitespace would erase the evidence.
    from verak.v3.env.analysis import ParagraphAnalyzer
    ParagraphAnalyzer.refresh(analysis, raw_corrupted, {p.pid for p in raw_corrupted.paragraphs})
    public_corrupted, aliases = public_document(raw_corrupted)
    base = {'source_id': row['source_id'], 'corpus_episode_id': row['episode_id'], 'genre': row['genre'],
        'question': row['question'], 'text': raw_corrupted.text, 'profile': document_profile(raw_corrupted).to_dict()}
    prepared = normalize_row(base, analysis, document=public_corrupted)
    normalized_source = source.clone()
    source_mapping = normalize_document(normalized_source, analysis)
    from .environment_v43 import prepared_document
    normalized_corrupted = prepared_document(prepared)
    source_projection = project_document(normalized_source, source_mapping)
    corrupted_projection = private_document(project_document(normalized_corrupted, prepared['normalization']), aliases)
    source_profile = config['paths']['phase2_output']/'bareun_profiles'/f"{examples[row['source_id']].source_line}.json"
    value = {'schema_version': 1, 'corpus': deepcopy(row), 'policy_row': prepared,
        'source_profile': {'path': str(source_profile), 'sha256': file_sha(source_profile)},
        'raw_source': save_document(source), 'raw_corrupted': save_document(raw_corrupted),
        'stable_id_aliases': aliases, 'private_source_positions_hidden_from_policy': True,
        'normalized_source': save_document(normalized_source), 'source_normalization': source_mapping,
        'reward_source': save_document(source_projection), 'reward_initial': save_document(corrupted_projection),
        'initial_recovery': initial_recovery(source, raw_corrupted, source_projection, corrupted_projection, row['records']),
        'source_tag': 'v43_corruption_teacher', 'reward_formula': 'unchanged verak.v3.reward.total.rewards',
        'initial_normalization_is_not_an_agent_action': True}
    value['plan'] = corruption_plan(row, prepared, aliases)
    atomic_new(path, value)
    return value, path


def teach_corruption(prepared, prepared_path, attempt, api, tokenizer, *, root=ROOT):
    """Reuse the exact Dv3 turn loop; bind only storage identity and snapshots."""
    from . import policy_prompts_v43 as prompts
    from .environment_v43 import V43Environment, BoundaryParagraphs
    from .runtime_v43 import bind, teach_impl as procedure
    root = Path(root)
    corpus, row = prepared['corpus'], prepared['policy_row']
    episode, source = corpus['episode_id'], corpus['source_id']
    target = root/'B3/content/attempts'/f'{safe_id(episode)}_a{attempt}.json'
    digest = file_sha(prepared_path)
    if target.exists():
        result = read_json(target)
        if (result['corpus_episode_id'] != episode or result['source_id'] != source or
                result['prepared_sha256'] != digest or result['prompt_version'] != prompts.VERSION):
            raise ValueError('Saved B3 teacher belongs to a different immutable contract')
        for call in result.get('calls', []):
            prompts.assert_messages(call['public_messages'], call['role'])
            assert_public_teacher(call['public_messages'], prepared)
        return result
    holder = {}
    class TrackingEnvironment(V43Environment):
        def __init__(self, observed, analysis):
            super().__init__(observed, analysis)
            holder['env'] = self
            self.saved_initial = save_document(self.document)
            self.saved_stage1 = None
        def start_korean(self, items=()):
            self.saved_stage1 = {'document': save_document(self.document), 'lineage': deepcopy(self.lineage)}
            return super().start_korean(items)
        def step(self, raw, role):
            split = role == 'revision' and structural_split_attempt(raw, self.document)
            record = super().step(raw, role)
            # Audit metadata only. The frozen environment already copied its
            # public last_result; this field never changes an inference prefix.
            if split:
                record['structural_split_attempt'] = True
            return record
    class EpisodeAPI:
        def request(self, messages, **kwargs):
            assert_public_teacher(messages, prepared)
            item = kwargs['item_id']
            if not item.startswith(source+':a'):
                raise ValueError('Unexpected B3 API attempt identity')
            return api.request(messages, **{**kwargs, 'item_id': episode+item[len(source):]})
    def storage_id(value):
        return safe_id(episode) if value == source else safe_id(value)
    def persist(path, result):
        if Path(path) != target:
            raise ValueError('B3 teacher attempted to write another episode')
        env = holder['env']
        final = {'document': save_document(env.document), 'lineage': deepcopy(env.lineage)}
        stages = {'initial': {'document': env.saved_initial, 'lineage': deepcopy(row['normalization']['lineage'])},
                  'stage1': env.saved_stage1, 'final': final}
        actions = {'global': reward_actions(result['actions']['revision'], row['normalization'],
                                            (env.saved_stage1 or final)['lineage'], prepared.get('stable_id_aliases')),
                   'korean': reward_actions(result['actions']['korean'], row['normalization'], final['lineage'], prepared.get('stable_id_aliases'))}
        result.update(corpus_episode_id=episode, source_id=source, prepared_path=str(prepared_path), prepared_sha256=digest,
            source_tag='v43_corruption_teacher', states=stages, normalization=deepcopy(row['normalization']),
            actions_by_role=actions, steps={r: len(a) for r, a in actions.items()},
            v43_termination=deepcopy(result['termination']),
            termination={ROLE_NAMES[k]: v for k, v in result['termination'].items()},
            completed=result['status'] == 'completed' and env.saved_stage1 is not None,
            reward=None, global_only_reward=None, score_source=None, scorer_calls=0, gpu_used=False)
        projections, errors = {}, {}
        for name, state in stages.items():
            if state is None:
                continue
            try:
                projections[name] = save_document(private_document(project_document(restore_document(state['document']),
                    row['normalization'], state['lineage']), prepared.get('stable_id_aliases')))
            except ProjectionUnknown as exc:
                errors[name] = str(exc)
        result['reward_projection'] = {'states': projections, 'unknown': errors,
                                      'definition': 'exact original-SID projection; no approximate reward'}
        result['mechanical_terms'] = mechanical_terms(prepared, result, load_config()['reward'])
        atomic_new(path, result)
    result = bind(procedure, ROOT=root/'B3', V4Environment=TrackingEnvironment,
        BoostParagraphs=BoundaryParagraphs, safe_id=storage_id, atomic_new=persist,
        assert_messages=prompts.assert_messages, verify_frozen=prompts.verify_frozen)(row, prepared['plan'], attempt, EpisodeAPI(), tokenizer)
    return result


def mechanical_terms(prepared, raw, config):
    """Recovery/over-edit evidence only: no invented Q, total R, or selection."""
    from statistics import mean
    from verak.v3.reward.recovery import recover_records
    from verak.v3.reward.overedit import overedit
    from verak.v3.reward.total import LOCAL_LEVELS
    docs = raw['reward_projection']['states']
    source = restore_document(prepared['reward_source']) if 'reward_source' in prepared else None
    result = {'global': None, 'korean': None, 'combined': None, 'quality': None,
              'total_R': None, 'selection': None, 'GPU_reference_pending': True}
    if source is None or 'initial' not in docs:
        return result
    initial = restore_document(docs['initial'])
    records = prepared['corpus']['records']
    def recovered(document):
        return recover_records(source, document, records, corrupted=initial,
                               coupled_weight=config.get('dependents_weight', .3))
    def terms(rows, values, before, after, actions):
        return {'R_rec': mean(values) if values else 0., 'per_record': rows,
            'R_over': overedit(source, before, after, records, actions=actions)['value'],
            'R_step': len(actions) if actions is not None else sum(raw['steps'].values()),
            'R_q': None, 'R': None}
    middle = restore_document(docs['stage1']) if 'stage1' in docs else None
    if middle is not None:
        rows = [r for r in recovered(middle) if r['level'] == 'GLOBAL']
        result['global'] = terms(rows, [r['main'] for r in rows], initial, middle, raw['actions_by_role']['global'])
    if raw['completed'] and 'final' in docs:
        final = restore_document(docs['final'])
        all_rows = recovered(final)
        result['combined'] = terms(all_rows, [r['recovery'] for r in all_rows], source, final, None)
        if middle is not None:
            rows = [r for r in all_rows if r['level'] in LOCAL_LEVELS or r['coupled'] is not None]
            values = [r['main'] for r in all_rows if r['level'] in LOCAL_LEVELS]+[
                r['coupled'] for r in all_rows if r['level'] == 'GLOBAL' and r['coupled'] is not None]
            result['korean'] = terms(rows, values, middle, final, raw['actions_by_role']['korean'])
    return result


def role_eligible(row, role, corpus):
    """Existing SFT exception plus the requested valid STOP/rejection conditions."""
    from verak.v3.train.formatting import structural_action
    reward = (row.get('reward') or {}).get(role) or (row.get('global_only_reward') if role == 'global' else None)
    actions = row.get('actions_by_role', {}).get(role, [])
    no_global = role == 'global' and not any(r['level'] == 'GLOBAL' for r in corpus['records'])
    if reward is None:
        return False, 'reward_unknown'
    if not no_global and reward['R'] < .8:
        return False, 'role_R_below_0.80'
    if row.get('termination', {}).get(role) != 'STOP' or not actions or actions[-1]['action'] != 'STOP' or not actions[-1].get('valid'):
        return False, 'no_valid_terminal_STOP'
    if sum(not a.get('valid', False) for a in actions) > 1:
        return False, 'more_than_one_rejected_action'
    if role == 'korean' and not row.get('completed'):
        return False, 'incomplete_korean'
    if no_global and (row['steps']['global'] > 2 or reward['R_over'] != 0 or
                      any(structural_action(a) or a.get('structural_split_attempt', False) for a in actions)):
        return False, 'no_GLOBAL_STOP_rule'
    return True, 'no_GLOBAL_STOP_rule' if no_global else 'role_R_ge_0.80'


def run(root=ROOT, *, limit=None):
    from verak.v3.view_data import load_episode_examples
    from verak.v3.v2_ops.local import load_environment
    from transformers import AutoTokenizer
    root = Path(root)
    with collection_lock(root/'B3'):
        component_contract('B3', root)
        sample, corpus = freeze_sample(root)
        config = load_config()
        load_environment(config)
        examples = {e.id: e for e in load_episode_examples(config, 'agent_train')}
        tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
        api = api_for('B3', root=root)
        jobs = sample['episode_ids'] if limit is None else sample['episode_ids'][:limit]
        observed = sum(bool(read_json(p).get('calls')) for p in (root/'B3/content/attempts').glob('*.json'))
        try:
            api.settle_interrupted()
            def one(episode):
                account = component_accounting('B3', root)
                if account['confirmed_usd']+account['reserved_usd'] >= account['component_cap_usd']:
                    return {'episode_id': episode, 'budget_stop': True}
                failure_path = root/'B3/preparation_errors'/(safe_id(episode)+'.json')
                if failure_path.exists():
                    return {'episode_id': episode, 'preparation_error': True}
                try:
                    prepared, path = prepare_corruption(corpus[episode], root=root, config=config, examples=examples)
                except Exception as exc:
                    atomic_new(failure_path, {'episode_id': episode, 'source_id': corpus[episode]['source_id'],
                        'error': type(exc).__name__+': '+str(exc), 'attempts_unstarted': 2, 'scorer_calls': 0})
                    return {'episode_id': episode, 'preparation_error': True}
                existing = {a: (root/'B3/content/attempts'/f'{safe_id(episode)}_a{a}.json').exists() for a in ATTEMPTS}
                attempts = [teach_corruption(prepared, path, a, api, tokenizer, root=root) for a in ATTEMPTS]
                return {'episode_id': episode, 'new_started_attempts': sum(bool(a['calls']) and not existing[a['attempt']] for a in attempts),
                        'budget_stop': any(a['status'] == 'budget_stop' for a in attempts)}
            def progress(number, value, started):
                write_json(root/'B3/status.json', {'processed_this_run': number, **value,
                    'observed_started_attempts': started, 'current_worker_limit': dispatch_limit('B3', root, started),
                    'api': component_accounting('B3', root), 'at': time.time()})
            bounded_episodes('B3', jobs, one, root=root, observed_attempts=observed, progress=progress)
        finally:
            api.close()
        # A deliberately bounded development invocation does not close the full
        # corpus manifest. Resume can collect the remaining authorized practices.
        return report(root) if limit is not None and limit < 1430 else prepare_gpu_manifest(root)


def prepare_gpu_manifest(root=ROOT):
    """Freeze ready files only. They do not dispatch any scorer/GPU job."""
    root = Path(root)
    sample = read_json(root/'B3/sample.json')
    account = component_accounting('B3', root)
    if account['pending']:
        raise RuntimeError('Cannot freeze B3 while a paid call is live')
    episodes, requests = [], {}
    for episode in sample['episode_ids']:
        for attempt in ATTEMPTS:
            path = root/'B3/content/attempts'/f'{safe_id(episode)}_a{attempt}.json'
            if not path.exists():
                continue
            raw = read_json(path)
            if file_sha(raw['prepared_path']) != raw['prepared_sha256']:
                raise ValueError('B3 prepared evidence changed before GPU handoff')
            prepared = read_json(raw['prepared_path'])
            keys = {}
            # There is no role reward without a completed Revision endpoint.
            # Missing/unknown projections do not trigger unnecessary GPU calls.
            projected = raw['reward_projection']['states']
            needed = ['initial', 'stage1'] if 'stage1' in projected else []
            if needed and raw['completed'] and 'final' in projected:
                needed.append('final')
            for name in needed:
                text = restore_document(raw['states'][name]['document']).text
                question = prepared['corpus']['question']
                key = pair_key(question, text)
                keys[name] = key
                requests[key] = {'key': key, 'question': question, 'text': text}
            episodes.append({'episode_id': episode, 'source_id': raw['source_id'], 'attempt': attempt,
                'raw_path': str(path), 'raw_sha256': file_sha(path),
                'prepared_path': raw['prepared_path'], 'prepared_sha256': raw['prepared_sha256'],
                'quality_keys': keys, 'projection_unknown': raw['reward_projection']['unknown'],
                'provisional_path': None, 'provisional_sha256': None, 'provisional_R': None})
    manifest = {'schema_version': 1, 'component': 'B3', 'version': 'v4.3',
        'sample_path': str(root/'B3/sample.json'), 'sample_sha256': file_sha(root/'B3/sample.json'),
        'contract_path': str(root/'B3/contract.json'), 'contract_sha256': file_sha(root/'B3/contract.json'),
        'reward_contract': read_json(root/'B3/contract.json')['reward_contract'],
        'reference_gpu_fingerprint': sample['reference_gpu_fingerprint'], 'episodes': episodes,
        'requests': [requests[k] for k in sorted(requests)], 'planned_attempts': 2860,
        'uncollected_attempts': 2860-len(episodes), 'CPU_scoring_performed': False,
        'GPU_dispatch': 'root only, explicitly after one-shot evaluation',
        'teacher_collection_finished': True, 'no_live_paid_calls': True, 'api': account}
    freeze(root/'B3/gpu_manifest.json', manifest)
    freeze(root/'B3/gpu_ready.json', {'status': 'ready_for_parent_scheduling',
        'manifest_path': str(root/'B3/gpu_manifest.json'), 'manifest_sha256': file_sha(root/'B3/gpu_manifest.json'),
        'teacher_collection_finished': True, 'no_live_paid_calls': True,
        'automatic_GPU_dispatch': False, 'final_selection_pending_GPU_reference': True})
    return report(root)


def rebuild_gpu_reward(prepared, raw, scores, config):
    """Pure file arithmetic, actual v4.3 Q endpoints and unchanged v1 function."""
    from verak.v3.reward.total import rewards
    value = deepcopy(raw)
    docs = raw['reward_projection']['states']
    value.update(reward=None, global_only_reward=None, score_source='gpu_reference', quality_scores=scores)
    if 'stage1' not in docs or any(k not in scores for k in ('initial', 'stage1')):
        value['reward_status'] = 'unknown_stage1_projection_or_GPU_score'
        return value
    source = restore_document(prepared['reward_source'])
    initial, stage1 = (restore_document(docs[k]) for k in ('initial', 'stage1'))
    full = raw['completed'] and 'final' in docs and 'final' in scores
    final = restore_document(docs['final']) if full else stage1
    candidate = prepared['corpus']
    result = rewards(source, initial, final, candidate['records'], genre=candidate['genre'],
        q_corrupted=scores['initial']['mean'], q_stage1=scores['stage1']['mean'],
        q_final=scores['final']['mean'] if full else scores['stage1']['mean'],
        config=config['reward'], mode='two_stage', stage1=stage1,
        stage1_actions=raw['actions_by_role']['global'],
        stage2_actions=raw['actions_by_role']['korean'] if full else [],
        preexisting_spell_spans=candidate.get('preexisting_spell_spans', ()))
    if full:
        value.update(reward=result, reward_status='full_gpu_reference')
    else:
        value.update(global_only_reward=result['global'], reward_status='global_only_gpu_reference',
                     unmeasured_roles=['korean', 'combined'])
    return value


def gpu_finalize(root=ROOT, *, tokenizer=None):
    """Explicit file-only consumer; root owns creation of the GPU-complete marker."""
    root = Path(root)
    with finalize_lock(root/'B3/gpu_finalize'):
        manifest_path = root/'B3/gpu_manifest.json'
        manifest = read_json(manifest_path)
        if (file_sha(manifest['sample_path']) != manifest['sample_sha256'] or
            file_sha(manifest['contract_path']) != manifest['contract_sha256'] or
            manifest['reward_contract'] != reward_contract()):
            raise ValueError('B3 frozen corpus/reward formula or weights changed')
        complete_path = root/'B3/gpu_rescore/complete.json'
        complete = read_json(complete_path)
        identity = {'manifest_sha256': file_sha(manifest_path), 'gpu_complete_sha256': file_sha(complete_path)}
        if (complete.get('manifest_sha256') != identity['manifest_sha256'] or
            complete.get('fingerprint') != manifest['reference_gpu_fingerprint'] or
            complete.get('slot') != 'post_oneshot_evaluation' or complete.get('scheduled_by') != 'root'):
            raise ValueError('B3 GPU pass lacks the authorized post-one-shot reference contract')
        selection_path = root/'B3/gpu_selection.json'
        if selection_path.exists():
            saved = read_json(selection_path)
            if saved['identity'] != identity:
                raise ValueError('Published B3 selection belongs to different GPU inputs')
            for role in saved['selected'].values():
                for item in role.values():
                    if file_sha(item['path']) != item['sha256']:
                        raise ValueError('Published B3 selected reward changed')
            if tokenizer is not None:
                export_selected(saved, tokenizer, root=root)
            report(root)
            return saved
        responses, proofs, failures = {}, [], []
        for request in manifest['requests']:
            key = request['key']
            path = Path(complete['responses_root'])/(key+'.json')
            try:
                score = read_json(path)
                result = score['result']
                if (pair_key(request['question'], request['text']) != key or score.get('error') or
                    score.get('execution_device') != 'gpu_reference' or score['fingerprint'] != complete['fingerprint'] or
                    result['cache_key'] != key or not math.isfinite(result['mean'])):
                    raise ValueError('GPU reference response fingerprint/input mismatch')
                responses[key] = result
                proofs.append({'path': str(path), 'sha256': file_sha(path), 'key': key})
            except (OSError, KeyError, ValueError) as exc:
                failures.append({'key': key, 'error': type(exc).__name__+': '+str(exc)})
        selected, reasons, measured = {'global': {}, 'korean': {}}, Counter(), []
        config = load_config()
        for item in manifest['episodes']:
            for kind in ('raw', 'prepared'):
                if file_sha(item[kind+'_path']) != item[kind+'_sha256']:
                    raise ValueError('B3 frozen raw/evidence changed: '+item[kind+'_path'])
            raw, prepared = read_json(item['raw_path']), read_json(item['prepared_path'])
            scores = {name: responses[key] for name, key in item['quality_keys'].items() if key in responses}
            row = rebuild_gpu_reward(prepared, raw, scores, config)
            row['scorer_fingerprint'] = complete['fingerprint']
            target = root/'B3/gpu_measured'/Path(item['raw_path']).name
            freeze(target, row)
            measured.append({'episode_id': item['episode_id'], 'attempt': item['attempt'],
                'path': str(target), 'sha256': file_sha(target), 'reward_status': row['reward_status']})
            for role in selected:
                keep, reason = role_eligible(row, role, prepared['corpus'])
                reasons[role+':'+reason] += 1
                if not keep:
                    continue
                reward = (row.get('reward') or {}).get(role) or row['global_only_reward']
                entry = {'path': str(target), 'sha256': file_sha(target), 'R': reward['R'],
                    'attempt': item['attempt'], 'source_id': item['source_id'], 'prepared_path': item['prepared_path'],
                    'prepared_sha256': item['prepared_sha256'], 'source_tag': 'v43_corruption_teacher', 'weight': 1}
                old = selected[role].get(item['episode_id'])
                if old is None or (entry['R'], -entry['attempt']) > (old['R'], -old['attempt']):
                    selected[role][item['episode_id']] = entry
        result = {'schema_version': 1, 'version': 'v4.3', 'component': 'B3', 'score_source': 'gpu_reference',
            'fingerprint': complete['fingerprint'], 'identity': identity, 'selected': selected,
            'selection_reasons': dict(reasons), 'measured': measured, 'score_proofs': proofs,
            'GPU_failures': failures, 'CPU_comparison': {'observed': 0, 'unknown': len(manifest['episodes'])},
            'STOP_only_cap': None, 'operator_duplication': False, 'source_cap': None, 'training': False}
        atomic_new(selection_path, result)
        if tokenizer is not None:
            export_selected(result, tokenizer, root=root)
        report(root)
        return result


def export_selected(selection, tokenizer, *, root=ROOT):
    from verak.v3.train.formatting import encode_labels
    from . import policy_prompts_v43 as prompts
    from .content_env import dumps, parse_json
    root = Path(root)
    output = root/'B3/export'
    output.mkdir(parents=True, exist_ok=True)
    identity = {'selection_sha256': sha_text(json.dumps(selection, ensure_ascii=False, sort_keys=True)),
                'prompt_manifest_sha256': file_sha(prompts.FROZEN), 'context_limit': 8192}
    contract_path = output/'contract.json'
    if contract_path.exists():
        saved = read_json(contract_path)
        if saved['identity'] != identity:
            raise ValueError('Published B3 export belongs to different selection/prompt inputs')
        for row in saved['roles'].values():
            if file_sha(row['path']) != row['sha256']:
                raise ValueError('Published B3 action export changed')
        return saved['roles']
    counts = {}
    for role, policy_role in (('global', 'revision'), ('korean', 'korean')):
        path, count, maximum = output/(policy_role+'.jsonl'), 0, 0
        # A crash before contract publication can leave only an unpublished
        # temporary file. Resume rebuilds it from the same immutable selections.
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output, prefix='.'+policy_role,
                                         suffix='.partial', delete=False) as stream:
            temporary = Path(stream.name)
            for episode, proof in selection['selected'][role].items():
                if file_sha(proof['path']) != proof['sha256']:
                    raise ValueError('Selected B3 episode changed before export')
                row = read_json(proof['path'])
                prepared = read_json(row['prepared_path']) if row.get('prepared_path') else None
                if prepared is not None and file_sha(row['prepared_path']) != row['prepared_sha256']:
                    raise ValueError('Selected B3 preparation changed before export')
                for call in row['calls']:
                    if call['role'] != policy_role or not call['action']['valid']:
                        continue
                    prompts.assert_messages(call['public_messages'], policy_role)
                    if prepared is not None:
                        assert_public_teacher(call['public_messages'], prepared)
                    target = call.get('canonical_action')
                    if (not isinstance(target, str) or not isinstance(parse_json(target), dict) or
                            parse_json(target) != call['action']['value']):
                        raise ValueError('Valid v4.3 action lacks its canonical single-JSON target')
                    messages = deepcopy(call['public_messages'])+[{'role': 'assistant', 'content': target}]
                    encoded = encode_labels(tokenizer, messages, last_assistant_only=True)
                    prefix = tokenizer.apply_chat_template(call['public_messages'], tokenize=True, add_generation_prompt=True)
                    if (len(encoded['input_ids']) > 8192 or encoded['input_ids'][:len(prefix)] != prefix or
                        any(v != -100 for v in encoded['labels'][:len(prefix)]) or
                        encoded['labels'][len(prefix):] != encoded['input_ids'][len(prefix):]):
                        raise ValueError('B3 action-only inference prefix/mask mismatch')
                    public = dumps(call['public_messages'])
                    if any(k in public for k in ('recovery_target', 'original_text', 'corrupted_text', 'record_id',
                                                  'grader_1_scores', 'grader_2_scores', 'LLM_rubric_feedback')):
                        raise ValueError('Private B3 supervision leaked into public observations')
                    stream.write(dumps({'corpus_episode_id': episode, 'source_id': row['source_id'],
                        'source_tag': 'v43_corruption_teacher', 'attempt': row['attempt'], 'role': policy_role,
                        'delegation': call['delegation'], 'turn': call['turn'], 'prompt_version': prompts.VERSION,
                        'raw_response_sha256': sha_text(call['raw']), 'teacher_trim': call.get('teacher_trim'),
                        'messages': messages, **encoded})+'\n')
                    count += 1
                    maximum = max(maximum, len(encoded['input_ids']))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        counts[policy_role] = {'trajectories': len(selection['selected'][role]), 'action_targets': count,
            'path': str(path), 'sha256': file_sha(path), 'max_tokens': maximum}
    atomic_new(contract_path, {'identity': identity, 'roles': counts, 'score_source': 'gpu_reference',
        'action_targets_only': True, 'observations_tool_outputs_notices_masked': True,
        'raw_trace_and_rejections_preserved_for_selection': True, 'invalid_targets_exported': False,
        'canonical_action_targets_only': True, 'teacher_only_trimming': True,
        'RFT_STOP_cap_or_duplication': False, 'training': False})
    return counts


def operator_metrics(sample, measured, prepared, selected):
    """Own-role main recovery only, with unknown/uncollected denominators intact."""
    evidence = {p['corpus']['episode_id']: p for p in prepared}
    counters = {op: Counter() for op in sample['operators']}
    selected_sources = {op: set() for op in counters}
    for episode in measured:
        eid = episode['corpus_episode_id']
        source = evidence[eid]
        initial = {r['record_id']: r for r in source['initial_recovery']}
        chosen_ops = set()
        for record in source['corpus']['records']:
            op = record['op']
            role = 'global' if record['level'] == 'GLOBAL' else 'korean'
            count = counters[op]
            count['saved_attempt_records'] += 1
            reward = (episode.get('reward') or {}).get(role) or (episode.get('global_only_reward') if role == 'global' else None)
            values = [r for r in (reward or {}).get('per_record', []) if r['record_id'] == record['record_id']]
            if len(values) > 1:
                raise ValueError('Duplicate per-record GPU reward evidence')
            if not values:
                count['unknown_saved_attempt_records'] += 1
                continue
            value = values[0]['main']
            count['observed_attempt_records'] += 1
            count['main_sum'] += value
            count['fully_recovered_records'] += value == 1.
            baseline = initial[record['record_id']]['normalized_initial_main']
            count['observed_initially_incomplete_records'] += baseline < 1.
            count['fully_recovered_from_initial_damage'] += baseline < 1. and value == 1.
            proof = selected[role].get(eid)
            if proof and proof['attempt'] == episode['attempt']:
                count['selected_fully_recovered_records'] += value == 1.
                chosen_ops.add(op)
        for op in chosen_ops:
            counters[op]['selected_trajectories'] += 1
            selected_sources[op].add(episode['source_id'])
    result = {}
    for op, count in counters.items():
        planned = sample['operators'][op]*2
        observed = count['observed_attempt_records']
        if count['saved_attempt_records'] > planned:
            raise ValueError('Saved B3 operator count exceeds its two-attempt denominator')
        result[op] = {key: count[key] for key in (
            'saved_attempt_records', 'observed_attempt_records', 'unknown_saved_attempt_records',
            'fully_recovered_records', 'observed_initially_incomplete_records',
            'fully_recovered_from_initial_damage', 'selected_trajectories', 'selected_fully_recovered_records')}
        result[op].update(own_role='global' if op.startswith('G_') else 'korean',
            planned_attempt_records=planned, uncollected_attempt_records=planned-count['saved_attempt_records'],
            mean_main_recovery=count['main_sum']/observed if observed else None,
            full_recovery_rate_observed=count['fully_recovered_records']/observed if observed else None,
            selected_source_essays=len(selected_sources[op]), score_source='gpu_reference')
    return result


def report(root=ROOT):
    root = Path(root)
    sample = read_json(root/'B3/sample.json')
    raw = [read_json(p) for p in (root/'B3/content/attempts').glob('*.json')]
    prepared = [read_json(p) for p in (root/'B3/essays').glob('*.json')]
    initialized = [r for p in prepared for r in p['initial_recovery']]
    calls = [c for a in raw for c in a['calls']]
    selection_path = root/'B3/gpu_selection.json'
    selection = read_json(selection_path) if selection_path.exists() else None
    stats = {'component': 'B3', 'planned_practices': 1430, 'planned_attempts': 2860,
        'source_essays': sample['source_essays'], 'prepared_practices': len(prepared), 'saved_attempts': len(raw),
        'attempts_started': sum(bool(a['calls']) for a in raw), 'remaining_uncollected_attempts': 2860-len(raw),
        'execution_status': dict(Counter(a['status'] for a in raw)), 'valid_actions': sum(c['action']['valid'] for c in calls),
        'returned_actions': len(calls),
        'teacher_trim_diagnostics': trim_metrics(calls),
        'role_endings': {role: dict(Counter(a['termination'].get(role, 'not_started') for a in raw)) for role in ('global', 'korean')},
        'projection_unknown': dict(Counter(k+':'+reason.split(':')[0] for a in raw
            for k, reason in a['reward_projection']['unknown'].items())),
        'environment_initial_recovery': {op: {'records': sum(r['op'] == op for r in initialized),
            'fully_recovered_by_normalization': sum(r['op'] == op and r['environment_recovered_full'] for r in initialized),
            'normalized_initial_full': sum(r['op'] == op and r['normalized_initial_main'] == 1 for r in initialized)}
            for op in sorted({r['op'] for r in initialized})},
        'no_GLOBAL_exception': sample['no_GLOBAL_exception'], 'RFT_STOP_cap_or_operator_duplication': False,
        'final_reward_status': 'gpu_reference' if selection else 'pending_GPU_reference',
        'selected': {r: len(x) for r, x in selection['selected'].items()} if selection else None,
        'CPU_rewards_used_for_selection_or_reporting': False, 'CPU_scorer_calls': 0,
        'api': component_accounting('B3', root), 'GPU_calls_by_collector': 0, 'training': False}
    for row in stats['environment_initial_recovery'].values():
        row['fully_recovered_by_normalization_rate'] = row['fully_recovered_by_normalization']/row['records']
        row['normalized_initial_full_rate'] = row['normalized_initial_full']/row['records']
    stats['preparation_errors'] = [read_json(p) for p in (root/'B3/preparation_errors').glob('*.json')]
    if selection:
        measured = [read_json(item['path']) for item in selection['measured']]
        roles = {}
        for role in ('global', 'korean', 'combined'):
            rows = [(a.get('reward') or {}).get(role) or (a.get('global_only_reward') if role == 'global' else None) for a in measured]
            known = [r for r in rows if r is not None]
            roles[role] = {'observed': len(known), 'unknown': len(rows)-len(known),
                'mean_R': sum(r['R'] for r in known)/len(known) if known else None,
                'mean_R_over': sum(r['R_over'] for r in known)/len(known) if known else None}
        stats['GPU_only_role_rewards'] = roles
        stats['selection_reasons'] = selection['selection_reasons']
        stats['GPU_failures'] = selection['GPU_failures']
        stats['GPU_only_operator_recovery'] = operator_metrics(sample, measured, prepared, selection['selected'])
    write_json(root/'B3/metrics.json', stats)
    lines = ['# B3 v4.3 corruption teachers', '',
        f"Saved {len(raw)}/2860 attempts on {len(prepared)}/1430 practices; {sample['source_essays']} original source essays.",
        f"Valid executed first actions {stats['valid_actions']}/{stats['returned_actions']}; "
        f"teacher-only trimmed responses {stats['teacher_trim_diagnostics']['trimmed']}, "
        f"of which {stats['teacher_trim_diagnostics']['trimmed_first_action_valid']} executed validly.",
        'The active corpus and v1 reward code are unchanged. Revision receives structure-record tasks; '
        'Korean always reviews the whole essay. No-GLOBAL essays receive an explicit structure-review delegation.',
        'No-GLOBAL selection retains the existing STOP-within-2, no structural attempt, R_over=0 exception. '
        'All selected roles require valid terminal STOP and at most one rejected action. '
        'A Revision EDIT split is a structural attempt even if rejected or undone. '
        'No RFT 35% STOP cap or 2x duplication is applied.',
        'Only canonical single-action JSON is exported. Full teacher output and trim diagnostics remain in '
        'the raw trace; student parsing remains strict.',
        'Initial segmentation/whitespace normalization is baseline preparation, not an agent action. '
        'Its independently recovered records are reported below. Split states that cannot project exactly '
        'to the original stable units have unknown reward and are excluded, without replacing their denominator.',
        f"Final reward/selection status: {stats['final_reward_status']}. Q endpoints are the actual normalized initial, "
        'Revision and Korean texts; raw corpus q_corrupted is never silently reused. Only saved GPU-reference scores '
        'can enter final reward tables and selections. The collector made no CPU/GPU scorer calls.', '',
        '|Operator|Prepared records|Full after normalization|Newly fully recovered by environment|Environment recovery rate|',
        '|---|---:|---:|---:|---:|']
    for op, row in stats['environment_initial_recovery'].items():
        lines.append(f"|{op}|{row['records']}|{row['normalized_initial_full']}|{row['fully_recovered_by_normalization']}|{row['fully_recovered_by_normalization_rate']:.2%}|")
    if selection:
        lines += ['', '|Role|GPU reward observed|Unknown|Mean R|Mean R_over|Selected|', '|---|---:|---:|---:|---:|---:|']
        for role, values in stats['GPU_only_role_rewards'].items():
            lines.append(f"|{role}|{values['observed']}|{values['unknown']}|{values['mean_R']}|{values['mean_R_over']}|{stats['selected'].get(role, '—')}|")
        lines += ['', 'Operator recovery below is the main term at the operator owner’s endpoint: '
            'Revision for GLOBAL, Korean for local records. These are GPU-measured episodes only. '
            'Unknown saved records and uncollected attempts stay separate; neither is filled with zero. '
            'Full recovery includes any record already restored by normalization; the final column isolates '
            'model recovery from the remaining initial damage.', '',
            '|Operator|Owner|Planned attempt-records|Observed|Unknown saved|Uncollected|Full / observed|Mean main|Selected trajectories|Selected sources|Recovered from remaining damage|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for op, row in stats['GPU_only_operator_recovery'].items():
            lines.append(f"|{op}|{row['own_role']}|{row['planned_attempt_records']}|{row['observed_attempt_records']}|"
                f"{row['unknown_saved_attempt_records']}|{row['uncollected_attempt_records']}|"
                f"{row['fully_recovered_records']}/{row['observed_attempt_records']}|{row['mean_main_recovery']}|"
                f"{row['selected_trajectories']}|{row['selected_source_essays']}|"
                f"{row['fully_recovered_from_initial_damage']}/{row['observed_initially_incomplete_records']}|")
    account = stats['api']
    lines += ['', f"Cost ${account['confirmed_usd']:.8f}; reserved ${account['reserved_usd']:.8f}; cap $30. "
        f"Uncollected attempts {stats['remaining_uncollected_attempts']}; preparation errors {len(stats['preparation_errors'])}. "
        'No training was started.', '']
    (root/'B3/report.md').write_text('\n'.join(lines))
    return stats
