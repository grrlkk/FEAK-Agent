"""v4.3 scaled collection: exact rewards, gates, canonical targets and source proof."""
from copy import deepcopy
import json
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from verak.src.schemas import Token
from verak.v3.corrupt.document import Document, Paragraph, Unit
from verak.v3.reward.total import rewards
from verak.v4 import scale3_common as common
from verak.v4 import scale3_content as content
from verak.v4 import scale3_corruption as corruption
from verak.v4.scale2_projection import (ProjectionUnknown, project_document,
    restore_document, reward_actions, save_document, public_document, private_document, structural_split_attempt)


def unit(sid, text, leading=''):
    return Unit(sid, text, [Token(text, 'NNG', 0, len(text))], leading)


@pytest.fixture
def split_pair():
    raw = Document([Paragraph('P2', [unit('S8', '하나다.둘이다.'), unit('S3', '끝이다.', ' ')]),
                    Paragraph('P1', [unit('S5', '다른 문단이다.')])], ['', '\n'])
    normalized = Document([Paragraph('P2', [unit('S8', '하나다.'), unit('S8a', '둘이다.', ' '), unit('S3', '끝이다.', ' ')]),
                           Paragraph('P1', [unit('S5', '다른 문단이다.')])], ['', '\n'])
    lineage = {'S8': {'parent_sid': 'S8', 'source_start': 0, 'source_end': 4, 'initial': True},
               'S8a': {'parent_sid': 'S8', 'source_start': 4, 'source_end': 8, 'initial': True},
               'S3': {'parent_sid': 'S3', 'source_start': 0, 'source_end': 4, 'initial': True},
               'S5': {'parent_sid': 'S5', 'source_start': 0, 'source_end': 8, 'initial': True}}
    proof = {'original_layout': raw.snapshot(), 'normalized_layout': normalized.snapshot(),
             'sid_mapping': {'S8': ['S8', 'S8a'], 'S3': ['S3'], 'S5': ['S5']}, 'lineage': lineage}
    return raw, normalized, proof


def test_saved_tokens_roundtrip_without_analyzer(split_pair):
    _, doc, _ = split_pair
    saved = save_document(doc)
    assert restore_document(saved) == doc
    saved['tokens']['S8'][0]['end'] = 999
    with pytest.raises(ValueError, match='offset'):
        restore_document(saved)


def test_exact_projection_preserves_surface_tokens_and_original_nonsequential_ids(split_pair):
    _, doc, proof = split_pair
    projected = project_document(doc, proof)
    assert [u.sid for u in projected.units] == ['S8', 'S3', 'S5']
    assert [p.pid for p in projected.paragraphs] == ['P2', 'P1']
    assert projected.text == doc.text
    assert projected.units[0].text == '하나다. 둘이다.'
    assert projected.units[0].tokens[1].start == len('하나다. ')
    assert len(doc.units) == 4  # no mutation of the policy state


def test_corrupted_reading_order_aliases_hide_source_positions_and_reverse_exactly(split_pair):
    raw, _, _ = split_pair
    public, aliases = public_document(raw)
    assert [p.pid for p in public.paragraphs] == ['P1', 'P2']
    assert [u.sid for u in public.units] == ['S1', 'S2', 'S3']
    assert public.text == raw.text and private_document(public, aliases) == raw
    assert aliases['public_to_private_sentences'] == {'S1': 'S8', 'S2': 'S3', 'S3': 'S5'}
    public.paragraphs.reverse()
    public.gaps.reverse()
    restored = private_document(public, aliases)
    assert [p.pid for p in restored.paragraphs] == ['P1', 'P2']
    assert restored.paragraphs[0].units[0].sid == 'S5'


@pytest.mark.parametrize('change,reason', [('partial', 'partially_deleted'), ('interleave', 'not_contiguous'),
    ('reverse', 'order_changed'), ('paragraph', 'cross_paragraphs')])
def test_unrepresentable_fragments_are_unknown_not_failed_recovery(split_pair, change, reason):
    _, doc, proof = split_pair
    if change == 'partial':
        doc.paragraphs[0].units.pop(1)
    elif change == 'interleave':
        doc.paragraphs[0].units[1:3] = list(reversed(doc.paragraphs[0].units[1:3]))
    elif change == 'reverse':
        doc.paragraphs[0].units[:2] = list(reversed(doc.paragraphs[0].units[:2]))
    else:
        doc.paragraphs[1].units.append(doc.paragraphs[0].units.pop(1))
    with pytest.raises(ProjectionUnknown, match=reason):
        project_document(doc, proof)


def test_whole_unit_delete_move_and_split_then_undo_remain_exact(split_pair):
    _, doc, proof = split_pair
    moved = doc.clone()
    moved.paragraphs[1].units.extend(moved.paragraphs[0].units[:2])
    moved.paragraphs[0].units = moved.paragraphs[0].units[2:]
    assert project_document(moved, proof).locate('S8')[0] == 1
    deleted = doc.clone()
    deleted.paragraphs[0].units = deleted.paragraphs[0].units[2:]
    assert 'S8' not in {u.sid for u in project_document(deleted, proof).units}
    lineage = {**proof['lineage'], 'S8b': {'parent_sid': 'S8', 'initial': False, 'action_index': 1}}
    # UNDO restored the exact original state; an unused lineage event is harmless.
    assert project_document(doc, proof, lineage) == project_document(doc, proof)


def test_reward_action_aliases_delete_insert_and_rejected_structural_attempts(split_pair):
    from verak.v3.train.formatting import structural_action
    _, _, proof = split_pair
    raw = [{'action': 'DELETE', 'value': {'sentence': 'S8a'}, 'valid': False, 'changed_sids': []},
           {'action': 'INSERT', 'value': {'after': 'S3', 'text': '새 설명.'}, 'valid': True, 'changed_sids': ['N1']},
           {'action': 'MOVE', 'value': {'sentence': 'S8a', 'to': {'after': 'S5'}}, 'valid': True, 'changed_sids': ['S8a']},
           {'action': 'EDIT', 'value': {'sentence': 'S8a', 'old': '둘', 'new': '두 개'}, 'valid': True, 'changed_sids': ['S8a']}]
    normalized = reward_actions(raw, proof, proof['lineage'])
    assert [structural_action(a) for a in normalized] == [True, True, True, False]
    assert normalized[0]['args'] == {'target': 'S8', 'new_text': ''}
    assert normalized[2]['changed_sids'] == ['S8']
    assert raw[0]['action'] == 'DELETE'


def reward_config():
    return {'w_rec': 1., 'w_q': .3, 'w_over': .5, 'w_step': .01,
            'quality': {'noise_floor': .2, 'noise_floor_by_genre': {'논증': .2}}}


def stop():
    return {'action': 'STOP', 'valid': True, 'args': {}}


def test_normalized_initial_baseline_does_not_charge_environment_changes(split_pair):
    _, normalized, proof = split_pair
    initial = project_document(normalized, proof)
    expected = rewards(initial, initial, initial, [], genre='논증', q_corrupted=4., q_stage1=4., q_final=4.,
        config=reward_config(), mode='two_stage', stage1=initial, stage1_actions=[stop()], stage2_actions=[stop()])
    assert all(expected[role]['R_over'] == 0 and expected[role]['R_q'] == 0 for role in ('global', 'korean', 'combined'))
    raw = {'completed': True, 'reward_projection': {'states': {k: save_document(initial) for k in ('initial', 'stage1', 'final')}},
           'actions_by_role': {'global': [stop()], 'korean': [stop()]}}
    prepared = {'reward_source': save_document(initial), 'corpus': {'records': [], 'genre': '논증',
                 'q_corrupted': 1000., 'corrupted_score': {'mean': 1000.}, 'preexisting_spell_spans': []}}
    scores = {k: {'mean': 4.} for k in ('initial', 'stage1', 'final')}
    actual = corruption.rebuild_gpu_reward(prepared, raw, scores, {'reward': reward_config()})
    assert actual['reward'] == expected  # includes every v1 component and detail
    assert actual['reward']['global']['quality']['before'] == 4.  # never raw q_corrupted
    assert actual['score_source'] == 'gpu_reference'


def test_unknown_final_keeps_only_true_global_reward(split_pair):
    _, normalized, proof = split_pair
    doc = project_document(normalized, proof)
    prepared = {'reward_source': save_document(doc), 'corpus': {'records': [], 'genre': '논증'}}
    raw = {'completed': True, 'reward_projection': {'states': {k: save_document(doc) for k in ('initial', 'stage1')}},
           'actions_by_role': {'global': [stop()], 'korean': [stop()]}}
    measured = corruption.rebuild_gpu_reward(prepared, raw, {k: {'mean': 4.} for k in ('initial', 'stage1')}, {'reward': reward_config()})
    assert measured['reward'] is None and measured['global_only_reward']['R'] == -.01
    assert measured['unmeasured_roles'] == ['korean', 'combined']


def test_b3_existing_no_global_exception_and_stop_rejection_gates():
    corpus = {'records': []}
    row = {'global_only_reward': {'R': -.01, 'R_over': 0}, 'termination': {'global': 'STOP'},
           'steps': {'global': 1}, 'actions_by_role': {'global': [stop()]}}
    assert corruption.role_eligible(row, 'global', corpus) == (True, 'no_GLOBAL_STOP_rule')
    bad = deepcopy(row)
    bad['actions_by_role']['global'].insert(0, {'action': 'MOVE', 'valid': False, 'args': {}})
    bad['steps']['global'] = 2
    assert not corruption.role_eligible(bad, 'global', corpus)[0]
    bad = deepcopy(row)
    bad['actions_by_role']['global'][-1]['valid'] = False
    assert not corruption.role_eligible(bad, 'global', corpus)[0]
    assert corruption.role_eligible({'reward': None}, 'global', corpus) == (False, 'reward_unknown')
    bad = deepcopy(row)
    bad['actions_by_role']['global'].insert(0, {'action': 'EDIT', 'valid': False, 'args': {'target': 'S1:x'},
                                             'structural_split_attempt': True})
    bad['steps']['global'] = 2
    assert not corruption.role_eligible(bad, 'global', corpus)[0]


def test_split_attempt_audit_covers_valid_rejected_and_undone_edits(split_pair):
    _, doc, _ = split_pair
    raw = json.dumps({'action': 'EDIT', 'sentence': 'S8', 'old': '하나다.', 'new': '하나다. 추가다.'})
    assert structural_split_attempt(raw, doc)
    rejected = json.dumps({'action': 'EDIT', 'sentence': 'S8', 'old': '하나다.', 'new': '하나다. 추가다. 또다.'})
    assert structural_split_attempt(rejected, doc)
    assert not structural_split_attempt('{"action":"UNDO"}', doc)
    assert not structural_split_attempt('{broken', doc)


def test_b3_tasks_do_not_reveal_answer_targets(split_pair):
    _, doc, proof = split_pair
    row = {'source_id': 'valid:1', 'episode_id': 'practice:one', 'records': [
        {'record_id': 'private', 'level': 'GLOBAL', 'op': 'G_SENT_MOVE', 'sids': ['S8'],
         'original_text': {'S8': 'SECRET ORIGINAL'}, 'recovery_target': {'paragraph': 2, 'answer': 'SECRET'}}]}
    prepared = {'normalization': proof, 'paragraphs': [{'id': p.pid} for p in doc.paragraphs]}
    plan = corruption.corruption_plan(row, prepared)
    public = [{k: i[k] for k in ('location', 'action', 'instruction')} for i in plan['items']]
    assert 'SECRET' not in json.dumps(public) and 'recovery_target' not in json.dumps(public)
    assert {'S8', 'S8a', 'P1', 'P2'} <= set(plan['items'][0]['location'])
    row['records'] = []
    review = corruption.corruption_plan(row, prepared)
    assert review['no_GLOBAL_review_delegation'] and review['items'][0]['owner'] == 'revision'


def ledger(path, cost=.4, pending=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE calls(id INTEGER,stage TEXT,item_id TEXT,status TEXT,reserved REAL,confirmed REAL,path TEXT)')
        db.execute('INSERT INTO calls VALUES(1,?,?,?,?,?,?)', ('teacher', 'one', 'pending' if pending else 'completed',
                   .1 if pending else 0., cost, None))


def test_smoke_gate_and_cumulative_cap_frozen_before_any_api(tmp_path, monkeypatch):
    root = tmp_path/'scale3'
    monkeypatch.setattr(common, 'REPO', tmp_path)
    prompt_path = tmp_path/'prompts.json'
    prompt_path.write_text('{}')
    fake = SimpleNamespace(VERSION='v43_fixture', FROZEN=prompt_path, verify_frozen=lambda: {})
    monkeypatch.setitem(sys.modules, 'verak.v4.policy_prompts_v43', fake)
    import verak.v4
    monkeypatch.setattr(verak.v4, 'policy_prompts_v43', fake, raising=False)
    common.freeze(root/'B1/gate.json', {'passed': False, 'stop_all_B': True})
    common.freeze(root/'B1/complete.json', {'status': 'failed'})
    common.freeze(root/'B1/contract.json', {'prompt_version': fake.VERSION})
    with pytest.raises(RuntimeError, match='has not passed'):
        common.component_contract('B2', root)
    (root/'B1/gate.json').write_text(json.dumps({'passed': True, 'stop_all_B': False,
        'all20_started': True, 'all20_saved': True, 'minimum_valid_action_rate': .9,
        'returned_actions': 100, 'valid_actions': 90}))
    common.freeze(root/'B1/metrics.json', {'saved_outcomes': 20})
    (root/'B1/complete.json').write_text(json.dumps({'status': 'passed', 'no_live_paid_calls': True,
        'gate': common.read_json(root/'B1/gate.json'), 'gate_sha256': common.file_sha(root/'B1/gate.json'),
        'metrics_path': str(root/'B1/metrics.json'), 'metrics_sha256': common.file_sha(root/'B1/metrics.json')}))
    ledger(tmp_path/'verak/v4/outputs/scale/B1/api/ledger.sqlite', .441109825)
    ledger(tmp_path/'verak/v4/outputs/scale2/B1/api/ledger.sqlite', .11151045)
    ledger(root/'B1/api/ledger.sqlite', .6)
    common.freeze(root/'B2/sample.json', {'source_ids': ['new:1']})
    common.freeze(root/'B3/sample.json', {'episode_ids': ['practice:1']})
    common.freeze(root/'B1/sample.json', {'source_ids': ['old:'+str(i) for i in range(20)]})
    common.freeze(tmp_path/'verak/v4/outputs/scale2/B1/sample.json', {'source_ids': ['old:'+str(i) for i in range(20)]})
    value = common.component_contract('B2', root)
    assert value['cap_usd'] == pytest.approx(50-.441109825-.11151045-.6)
    assert common.component_contract('B3', root)['cap_usd'] == 30
    with sqlite3.connect(root/'B1/api/ledger.sqlite') as db:
        db.execute('UPDATE calls SET reserved=.1,status="pending"')
    with pytest.raises(RuntimeError, match='reservations'):
        common.component_contract('B2', root)


def test_ramp_and_pause_only_control_new_episode_dispatch(tmp_path):
    assert common.dispatch_limit('B2', tmp_path, 19) == 2
    assert common.dispatch_limit('B2', tmp_path, 20) == 6
    assert common.dispatch_limit('B3', tmp_path, 20) == 4
    common.freeze(tmp_path/'dispatch_policy.json', {'pause_new_episodes': True})
    assert common.dispatch_limit('B2', tmp_path, 200) == 0
    (tmp_path/'dispatch_policy.json').write_text(json.dumps({'max_workers': {'B2': 1, 'B3': 2}}))
    assert common.dispatch_limit('B2', tmp_path, 200) == 1
    assert common.dispatch_limit('B3', tmp_path, 200) == 2


def test_bounded_workers_drain_inflight_jobs_without_dispatching_after_budget_stop(tmp_path):
    from threading import Event, Lock
    entered, exited, mutex, all_running = [], [], Lock(), Event()
    def worker(number):
        with mutex:
            entered.append(number)
            if len(entered) == 6:
                all_running.set()
        assert all_running.wait(2)
        with mutex:
            exited.append(number)
        return {'budget_stop': True, 'new_started_attempts': 2, 'number': number}
    results = common.bounded_episodes('B2', range(100), worker, root=tmp_path, observed_attempts=20)
    assert sorted(entered) == sorted(exited) == list(range(6))
    assert len(results) == 6


def test_b2_materialization_is_lazy_and_preserves_frozen_source_order(tmp_path, monkeypatch):
    """Preparing row two must not hold up API work on the already yielded row one."""
    from verak.v4 import data, environment_v43
    from verak.v3 import common as v3common
    from verak.v3.insertion_boost import resources
    from verak.v3.v2_ops import local
    from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis
    from verak.src.schemas import Profile, Sentence
    calls = []
    rows = [(i, {'question': '문항', 'text': f'{i}번 글이다.', 'grader_1_scores': [3]*8,
                 'grader_2_scores': [3]*8, 'assistant': 'feedback'}) for i in (1, 2)]
    class FakeAnalysis(FakeBoundaryAnalysis):
        def __init__(self, *a, **k): super().__init__()
        def profile(self, text, neighbors=()):
            calls.append(text)
            return Profile([Sentence('S1', 1, 0, len(text), text, [])], [], 'bareun', 'fake')
    sample = {'input_path': 'unused', 'source_ids': ['train:2', 'train:1'],
              'source_metadata': {f'train:{i}': {'source_id': f'train:{i}', 'genre': 'argumentative',
                  'essay_hash': common.sha_text(r['text'])} for i, r in rows}}
    monkeypatch.setattr(content, 'freeze_sample', lambda root: sample)
    monkeypatch.setattr(content, 'component_contract', lambda *a: {})
    monkeypatch.setattr(content, 'load_config', lambda: {})
    monkeypatch.setattr(local, 'load_environment', lambda config: None)
    monkeypatch.setattr(data, 'read_rows', lambda path: iter(rows))
    monkeypatch.setattr(data, 'feedback_parts', lambda raw: ['feedback']*8)
    monkeypatch.setattr(v3common, 'extract_question_essay', lambda row: (row['question'], row['text']))
    monkeypatch.setattr(resources, 'BoostParagraphs', FakeAnalysis)
    iterator = content.materialize(tmp_path)
    assert not calls
    first = next(iterator)
    assert first['source_id'] == 'train:2' and all(t.startswith('2') for t in calls)
    assert first['environment_version'] == environment_v43.ENVIRONMENT_VERSION
    assert next(iterator)['source_id'] == 'train:1'
    with pytest.raises(StopIteration):
        next(iterator)
    assert set(common.read_json(tmp_path/'B2/source_files.json')) == set(sample['source_ids'])


def test_new_source_selection_is_stratified_human_low_mid_and_excludes_duplicates(monkeypatch):
    from verak.v3 import common as v3common
    from verak.v4 import data
    monkeypatch.setattr(v3common, 'extract_question_essay', lambda row: (row['question'], row['text']))
    monkeypatch.setattr(data, 'genre_of', lambda row: row['genre'])
    rows = []
    for g, genre in enumerate(content.GENRES):
        for i in range(9):
            rows.append((g*10+i+1, {'question': 'question'+str(i), 'text': f'{genre} text {i}', 'genre': genre,
                'grader_1_scores': [1+i%5]*8, 'grader_2_scores': [1+i%5]*8}))
    prior = {'train:1': {'normalized_source_hash': data.normalized_source(rows[0][1]['text'])}}
    picked, eligibility, exclusions = content.select_sources(rows, prior, {common.sha_text('question1')},
        quotas={g: 2 for g in content.GENRES})
    assert len(picked) == 6 and 'train:1' not in {r['source_id'] for r in picked}
    assert exclusions['prior_source_or_normalized_text'] == 1 and exclusions['test_question'] == 3
    assert all(r['human_mean'] <= eligibility[r['genre']]['low_middle_cutoff'] for r in picked)
    assert all(v['chosen'] == 2 for v in eligibility.values())


def planner_row():
    return {'source_id': 'train:private', 'question': '문항',
        'paragraphs': [{'id': 'P1', 'sentences': [{'id': 'S1', 'text': '공개 사례의 근거가 필요하다.'}]}]}


def plan_item(**changes):
    return {'rubric': 'specificity', 'feedback_index': 1, 'problem': '근거 부족', 'location': ['S1'],
        'evidence': '공개 사례의 근거', 'owner': 'revision', 'action': 'INSERT',
        'instruction': '관련 공개 사례를 검색하고 근거 문단을 인용해 한 문장으로 연결한다.',
        'needs_search': 'yes', **changes}


def test_search_planner_keeps_grounding_caps_and_feedback_free_orchestrator_target():
    row = planner_row()
    raw = {'items': [plan_item()], 'dropped': [], 'writer_notes': ['경험은 글쓴이에게 질문한다.']}
    value = content.validate_plan(raw, row)
    assert value['items'][0]['needs_search'] == 'yes'
    assert 'needs_search' in content.plan_schema(row)['properties']['items']['items']['required']
    target = content.orchestrator_target(row, value)
    public = json.dumps(target['input'], ensure_ascii=False)+json.dumps(target['target'], ensure_ascii=False)
    for forbidden in ('feedback_index', 'LLM_feedback', 'grader_1_scores', 'train:private', ':D3I'):
        assert forbidden not in public
    assert target['target']['items'][0]['item_id'] == 'T1'
    assert not target['Orchestrator_tokens_exported']
    invalid = {'items': [plan_item(owner='korean', action='EDIT')], 'dropped': [], 'writer_notes': []}
    with pytest.raises(ValueError, match='Revision INSERT'):
        content.validate_plan(invalid, row)
    two = {'items': [plan_item(), plan_item(instruction='another task')], 'dropped': [], 'writer_notes': []}
    checked = content.validate_plan(two, row)
    assert len(checked['items']) == 1
    assert checked['mechanically_dropped'][0]['reason'] == 'one_INSERT_task_per_essay'
    bad = {'items': [plan_item(evidence='ABSENT REFERENCE')], 'dropped': [], 'writer_notes': []}
    assert content.validate_plan(bad, row)['items'] == []


def test_support_is_one_extra_gate_without_relaxing_existing_quality_or_fake_none():
    items = [{'item_id': 'private', 'owner': 'revision'}]
    attempt = {'attempt': 1, 'source_citations': [], 'search_history': [], 'status': 'completed', 'complete': True}
    verdict = {'attempt': 1, 'items': [{'item_id': 'private', 'status': 'addressed'}],
        'invented_specifics': 'no', 'invented_experiences': 'no', 'meaning_preserved': 'yes',
        'better_than_original': 'yes', 'repetition_introduced': 'no', 'edits_introduced_awkwardness': 'no',
        'source_support_applicable': 'none', 'source_support': []}
    # The existing base validator names are used below, rather than rewriting
    # quality rules in the scale implementation.
    from verak.v4.content import selection as base
    assert base(items, attempt, verdict)['export_keep']
    assert content.selection(items, attempt, verdict)['export_keep']
    bad = {**verdict, 'source_support_applicable': 'yes'}
    assert not content.selection(items, attempt, bad)['export_keep']
    bad = {**verdict, 'edits_introduced_awkwardness': 'yes'}
    assert not content.selection(items, attempt, bad)['export_keep']
    from verak.v4.tests.test_search_v43 import final_attempt, verdict as support_verdict
    sourced = {**final_attempt(), 'complete': True}
    supported = {**verdict, **support_verdict()}
    assert content.selection(items, sourced, supported)['export_keep']
    copied = {**verdict, **support_verdict(paraphrased='no')}
    assert not content.selection(items, sourced, copied)['export_keep']
    missing = {k: v for k, v in supported.items() if k != 'source_support'}
    assert not content.selection(items, sourced, missing)['export_keep']


def test_pair_judge_embeds_actual_citations_in_one_call_and_resumes_without_payment(tmp_path):
    from verak.v4.tests.test_search_v43 import final_attempt, verdict as support_verdict
    sourced = {**final_attempt(), 'phase_text': {'revision': '수정'}, 'final_text': '수정',
        'status': 'completed', 'delegations': [], 'complete': True}
    none = {**sourced, 'attempt': 2, 'source_citations': [], 'search_history': [], 'final_paragraphs': []}
    items = [{'item_id': 'private', 'owner': 'revision'}]
    common_fields = {'items': [{'item_id': 'private', 'status': 'addressed'}],
        'invented_specifics': 'no', 'invented_experiences': 'no', 'meaning_preserved': 'yes',
        'better_than_original': 'yes', 'repetition_introduced': 'no', 'edits_introduced_awkwardness': 'no',
        'reason': '판정'}
    response = {'attempts': [{**common_fields, **support_verdict()},
        {**common_fields, 'attempt': 2, 'source_support_applicable': 'none', 'source_support': []}],
        'preferred': 'tie', 'reason': '독립 비교'}
    class API:
        count = 0
        def request(self, messages, **kwargs):
            self.count += 1
            payload = json.loads(messages[1]['content'])
            citation = payload['attempts'][0]['source_citations'][0]
            assert citation['text'] == sourced['source_citations'][0]['text']
            assert citation['passage'] == sourced['source_citations'][0]['passage']
            assert payload['attempts'][1]['source_support_applicable'] == 'none'
            assert 'source_support' in kwargs['schema']['properties']['attempts']['items']['required']
            return {'raw': json.dumps(response, ensure_ascii=False), 'phase_call': 1}
    api = API()
    row = {**planner_row(), 'feedback': ['LLM feedback']*8}
    plan = {'items': items, 'writer_notes': []}
    first = content.judge(row, plan, [sourced, none], api, output_root=tmp_path)
    assert first['status'] == 'completed', first
    assert content.judge(row, plan, [sourced, none], api, output_root=tmp_path) == first
    assert api.count == 1


class PrefixTokenizer:
    """Prefix-stable tiny test tokenizer; no model/library/runtime initialization."""
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        value = ''.join('<'+m['role']+'>'+m['content']+'<end>' for m in messages)
        if add_generation_prompt:
            value += '<assistant>'
        return [ord(c) for c in value] if tokenize else value


def test_real_dv3_loop_b3_storage_identity_saved_snapshots_resume_and_export(tmp_path):
    from verak.v4.environment_v43 import normalize_row
    from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis, row as source_row
    from verak.v4.common import file_sha
    observed = normalize_row({**source_row('첫 문장이다.둘째 문장이다.'),
        'source_id': 'valid:7', 'genre': '논증'}, FakeBoundaryAnalysis())
    class FakeAPI:
        def __init__(self): self.calls = []
        def request(self, messages, **kwargs):
            self.calls.append(kwargs['item_id'])
            # Teacher-only trimming retains the first action. Raw two-action
            # wrappers must never enter the action-only training target.
            action = {'action': 'STOP', 'status': 'done', 'summary': '검토 완료', 'issues': []}
            return {'raw': json.dumps({'actions': [action, {'action': 'UNDO'}]}, ensure_ascii=False),
                    'phase_call': len(self.calls)}
    api = FakeAPI()
    prepared_rows = []
    for variant in (1, 2):
        corpus = {'source_id': 'valid:7', 'episode_id': f'practice:{variant}', 'records': [], 'genre': '논증'}
        prepared = {'corpus': corpus, 'policy_row': observed,
                    'plan': corruption.corruption_plan(corpus, observed)}
        path = tmp_path/f'prepared{variant}.json'
        common.freeze(path, prepared)
        result = corruption.teach_corruption(prepared, path, 1, api, PrefixTokenizer(), root=tmp_path)
        assert result['source_id'] == 'valid:7' and result['corpus_episode_id'] == f'practice:{variant}'
        assert result['completed'] and result['termination'] == {'global': 'STOP', 'korean': 'STOP'}
        assert result['reward'] is None and result['score_source'] is None
        assert all(c['teacher_trim']['trimmed'] and json.loads(c['canonical_action'])['action'] == 'STOP'
                   for c in result['calls'])
        assert result['states']['initial']['document']['text_sha256'] == common.sha_text(observed['text'])
        assert not result['reward_projection']['unknown']
        for name in ('initial', 'stage1', 'final'):
            assert restore_document(result['states'][name]['document']).text == observed['text']
            assert restore_document(result['reward_projection']['states'][name]).units[0].sid == 'S1'
        raw_path = tmp_path/'B3/content/attempts'/f'practice_{variant}_a1.json'
        prepared_rows.append((result, raw_path))
        again = corruption.teach_corruption(prepared, path, 1, api, PrefixTokenizer(), root=tmp_path)
        assert again == result
    assert len(api.calls) == 4 and len(set(api.calls)) == 4
    assert all('practice:' in key and 'valid:7' not in key for key in api.calls)
    selected = {'selected': {'global': {r['corpus_episode_id']: {'path': str(p), 'sha256': file_sha(p)} for r, p in prepared_rows}, 'korean': {}}}
    exported = corruption.export_selected(selected, PrefixTokenizer(), root=tmp_path)
    assert exported['revision']['trajectories'] == 2 and exported['revision']['action_targets'] == 2
    values = [json.loads(line) for line in (tmp_path/'B3/export/revision.jsonl').read_text().splitlines()]
    assert len({v['corpus_episode_id'] for v in values}) == 2
    assert {v['source_id'] for v in values} == {'valid:7'}
    for value in values:
        assert json.loads(value['messages'][-1]['content'])['action'] == 'STOP'
        assert 'actions' not in json.loads(value['messages'][-1]['content'])
        prefix = PrefixTokenizer().apply_chat_template(value['messages'][:-1], add_generation_prompt=True)
        assert value['input_ids'][:len(prefix)] == prefix
        assert value['labels'][:len(prefix)] == [-100]*len(prefix)
        assert value['labels'][len(prefix):] == value['input_ids'][len(prefix):]
    exported_sha = file_sha(tmp_path/'B3/export/contract.json')
    assert corruption.export_selected(selected, PrefixTokenizer(), root=tmp_path) == exported
    assert file_sha(tmp_path/'B3/export/contract.json') == exported_sha


def test_complete_teacher_public_trace_hides_private_ids_answers_and_audit_metadata(tmp_path, monkeypatch):
    """Actual frozen Dv3 loop, EDIT split/UNDO/notices/handoff; only API/Bareun are fake."""
    from verak.v4 import environment_v43 as environment
    from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis, row as source_row
    raw = Document([Paragraph('P37', [unit('S908', '첫 내용이다.두 번째 내용이다.'),
                                      unit('S907', '마지막 내용이다.', ' ')])], [''])
    public, aliases = public_document(raw)
    observed = environment.normalize_row({**source_row(raw.text), 'source_id': 'valid:private', 'genre': '논증'},
        FakeBoundaryAnalysis(), document=public)
    corpus = {'source_id': 'valid:private', 'episode_id': 'private_practice', 'genre': '논증', 'records': [
        {'record_id': 'HIDDEN_RECORD_75', 'level': 'GLOBAL', 'op': 'G_SENT_MOVE', 'sids': ['S908'],
         'original_text': {'S908': 'HIDDEN_ANSWER_CONTENT'}, 'recovery_target': {'destination': 'P999'}}]}
    prepared = {'corpus': corpus, 'policy_row': observed, 'stable_id_aliases': aliases,
                'plan': corruption.corruption_plan(corpus, observed, aliases)}
    pp = tmp_path/'prepared.json'
    common.freeze(pp, prepared)
    monkeypatch.setattr(environment, 'BoundaryParagraphs', lambda *a, **k: FakeBoundaryAnalysis())
    actions = iter([
        {'action': 'EDIT', 'sentence': 'S1', 'old': '첫 내용이다.', 'new': '첫 내용이다. 관계를 설명한다.'},
        {'action': 'UNDO'},
        {'action': 'STOP', 'status': 'done', 'summary': '검토 완료', 'issues': []},
        {'action': 'STOP', 'status': 'done', 'summary': '검토 완료', 'issues': []}])
    class FakeAPI:
        def request(self, messages, **kwargs):
            serialized = json.dumps(messages, ensure_ascii=False)
            for forbidden in ('HIDDEN_RECORD_75', 'HIDDEN_ANSWER_CONTENT', 'P999', 'S908', 'S907', 'P37',
                              'record_id', 'recovery_target', 'stable_id_aliases', 'structural_split_attempt'):
                assert forbidden not in serialized
            return {'raw': json.dumps(next(actions), ensure_ascii=False), 'phase_call': 1}
    result = corruption.teach_corruption(prepared, pp, 1, FakeAPI(), PrefixTokenizer(), root=tmp_path)
    assert result['completed'] and len(result['calls']) == 4
    assert all(c['action']['valid'] for c in result['calls'])
    assert result['actions_by_role']['global'][0]['structural_split_attempt']
    assert result['actions_by_role']['global'][0]['changed_sids'] == ['S908']
    assert result['calls'][0]['action']['created_sids'] == ['S1b']
    assert result['final_text'] == observed['text']
    assert restore_document(result['reward_projection']['states']['final']).units[0].sid == 'S908'
    from verak.v4.render_v43 import unpack_payload
    handoff = unpack_payload(json.loads(result['calls'][-1]['public_messages'][1]['content']))['revision_handoff']
    assert any(n.get('kind') == 'sentence_split' for n in handoff['marker_change_notices'])


def test_operator_reporting_uses_owner_endpoint_preserves_unknown_and_initial_baseline():
    corpus = {'episode_id': 'one', 'source_id': 'source', 'records': [
        {'record_id': 'g', 'op': 'G_SENT_MOVE', 'level': 'GLOBAL'},
        {'record_id': 'l', 'op': 'L_SPACING', 'level': 'WORD'}]}
    prepared = [{'corpus': corpus, 'initial_recovery': [
        {'record_id': 'g', 'normalized_initial_main': 0.}, {'record_id': 'l', 'normalized_initial_main': 1.}]}]
    one = {'corpus_episode_id': 'one', 'source_id': 'source', 'attempt': 1, 'reward': {
        'global': {'per_record': [{'record_id': 'g', 'main': 1.}]},
        'korean': {'per_record': [{'record_id': 'g', 'main': 0.}, {'record_id': 'l', 'main': 1.}]}}}
    two = {'corpus_episode_id': 'one', 'source_id': 'source', 'attempt': 2, 'reward': None}
    selected = {'global': {'one': {'attempt': 1}}, 'korean': {'one': {'attempt': 1}}}
    # Another practice was never collected: its two record-attempts remain unknown coverage.
    values = corruption.operator_metrics({'operators': {'G_SENT_MOVE': 2, 'L_SPACING': 2}}, [one, two], prepared, selected)
    for op in values:
        assert values[op]['observed_attempt_records'] == 1
        assert values[op]['unknown_saved_attempt_records'] == 1
        assert values[op]['uncollected_attempt_records'] == 2
        assert values[op]['mean_main_recovery'] == 1.
        assert values[op]['selected_trajectories'] == 1
    assert values['G_SENT_MOVE']['fully_recovered_from_initial_damage'] == 1
    assert values['L_SPACING']['fully_recovered_from_initial_damage'] == 0


@pytest.mark.parametrize('operator', ['G_PARA_SWAP', 'G_SENT_MOVE', 'G_OFFTOPIC', 'L_CONJ',
    'L_CONN', 'L_REGISTER', 'L_SUBJ_INSERT', 'L_SPACING', 'L_POLARITY'])
def test_all_active_record_rewards_equal_v1_after_exact_stable_id_projection(operator, tmp_path):
    """Use stored real Bareun fixture tokens, never a live analyzer/scorer."""
    from pathlib import Path
    from verak.v3.common import read_json, load_config
    from verak.v3.phase2 import restore_profile
    from verak.v3.corrupt.document import BareunBank
    from verak.v3.corrupt.operators import Proposal, apply
    from verak.v4.environment_v43 import normalize_document
    fixture = read_json(Path(__file__).parents[2]/'v3/tests/fixtures/corrupt_bareun.json')
    config = load_config()
    bank = BareunBank(config, cache_dir=tmp_path)
    bank.tokens = lambda text: [Token(**t) for t in fixture['tokens'][text]]
    source = Document.from_profile(fixture['text'], restore_profile(fixture['profile']))
    initial, record = apply(source, Proposal(**fixture['proposals'][operator]), bank)
    class SavedOnly:
        def refresh(self, document, pids):
            for p in document.paragraphs:
                if p.pid in pids:
                    for u in p.units:
                        u.tokens = bank.tokens(u.text)
    def normalized(doc):
        value = doc.clone()
        mapping = normalize_document(value, SavedOnly())
        return project_document(value, mapping)
    logs1 = [{'action': 'MOVE' if record['level'] == 'GLOBAL' else 'STOP',
              'valid': True, 'changed_sids': record['sids']}, stop()]
    logs2 = [{'action': 'EDIT', 'valid': True, 'changed_sids': record['sids']}, stop()]
    middle = source if record['level'] == 'GLOBAL' else initial
    kwargs = dict(genre='논증', q_corrupted=4., q_stage1=5., q_final=5.5, config=config['reward'],
                  mode='two_stage', stage1_actions=logs1, stage2_actions=logs2)
    old = rewards(source, initial, source, [record], stage1=middle, **kwargs)
    projected = rewards(normalized(source), normalized(initial), normalized(source), [record],
                        stage1=normalized(middle), **kwargs)
    assert projected == old


@pytest.mark.parametrize('valid_reference', [True, False])
def test_gpu_handoff_uses_actual_normalized_inputs_and_immutable_gpu_only_selection(tmp_path, monkeypatch, split_pair, valid_reference):
    from verak.v3.common import pair_key
    _, normalized, proof = split_pair
    doc = project_document(normalized, proof)
    candidate = {'episode_id': 'fixture:one', 'source_id': 'valid:1', 'genre': '논증',
                 'question': '질문', 'records': [], 'q_corrupted': 999.}
    prepared = {'corpus': candidate, 'reward_source': save_document(doc)}
    pp = tmp_path/'B3/essays/fixture_one.json'
    common.freeze(pp, prepared)
    raw = {'source_id': 'valid:1', 'corpus_episode_id': 'fixture:one', 'attempt': 1,
        'completed': True, 'status': 'completed', 'calls': [],
        'prepared_path': str(pp), 'prepared_sha256': common.file_sha(pp),
        'states': {k: {'document': save_document(normalized), 'lineage': proof['lineage']} for k in ('initial', 'stage1', 'final')},
        'reward_projection': {'states': {k: save_document(doc) for k in ('initial', 'stage1', 'final')}, 'unknown': {}},
        'steps': {'global': 1, 'korean': 1}, 'termination': {'global': 'STOP', 'korean': 'STOP'},
        'actions_by_role': {'global': [stop()], 'korean': [stop()]}, 'reward': None}
    ap = tmp_path/'B3/content/attempts/fixture_one_a1.json'
    common.freeze(ap, raw)
    common.freeze(tmp_path/'B3/sample.json', {'episode_ids': ['fixture:one'], 'reference_gpu_fingerprint': 'GPU_REFERENCE'})
    common.freeze(tmp_path/'B3/contract.json', {'cap_usd': 30., 'reward_contract': common.reward_contract()})
    monkeypatch.setattr(corruption, 'report', lambda root: {})
    monkeypatch.setattr(corruption, 'load_config', lambda: {'reward': reward_config()})
    corruption.prepare_gpu_manifest(tmp_path)
    manifest_path = tmp_path/'B3/gpu_manifest.json'
    manifest = common.read_json(manifest_path)
    assert len(manifest['requests']) == 1  # identity states share a reference input
    request = manifest['requests'][0]
    assert request['text'] == normalized.text and request['key'] == pair_key('질문', normalized.text)
    response_root = tmp_path/'responses'
    complete = {'manifest_sha256': common.file_sha(manifest_path), 'fingerprint': 'GPU_REFERENCE',
                'slot': 'post_oneshot_evaluation', 'scheduled_by': 'root', 'responses_root': str(response_root)}
    complete_path = tmp_path/'B3/gpu_rescore/complete.json'
    common.freeze(complete_path, {**complete, 'slot': 'during_A_training'})
    with pytest.raises(ValueError, match='post-one-shot'):
        corruption.gpu_finalize(tmp_path)
    complete_path.write_text(json.dumps(complete))
    common.freeze(response_root/(request['key']+'.json'), {'execution_device': 'gpu_reference' if valid_reference else 'cpu',
        'fingerprint': 'GPU_REFERENCE', 'result': {'cache_key': request['key'], 'mean': 4.}})
    result = corruption.gpu_finalize(tmp_path)
    assert len(result['selected']['global']) == int(valid_reference) and not result['selected']['korean']
    assert result['STOP_only_cap'] is None and not result['operator_duplication']
    item = result['measured'][0]
    measured = common.read_json(item['path'])
    if valid_reference:
        assert measured['reward']['global']['quality']['before'] == 4.
    else:
        assert measured['reward'] is None and measured.get('global_only_reward') is None
        assert result['selection_reasons']['global:reward_unknown'] == 1
        assert result['GPU_failures']
    assert measured['score_source'] == 'gpu_reference'
    digest = common.file_sha(tmp_path/'B3/gpu_selection.json')
    # The first call may finalize rewards before export; a later tokenizer pass
    # produces the exact same immutable selection and only fills the export.
    assert corruption.gpu_finalize(tmp_path, tokenizer=PrefixTokenizer()) == result
    assert common.file_sha(tmp_path/'B3/gpu_selection.json') == digest
    export_sha = common.file_sha(tmp_path/'B3/export/contract.json')
    assert corruption.gpu_finalize(tmp_path, tokenizer=PrefixTokenizer()) == result
    assert common.file_sha(tmp_path/'B3/export/contract.json') == export_sha
