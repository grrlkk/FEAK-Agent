import json
import sqlite3

import pytest

from verak.v5 import type1


def case():
    return {'case_id': 'essay:1', 'cohort': 'smoke20', 'question': '질문',
            'original': '나는 간다.\n\n그는 남는다.',
            'variants': {name: '나는 간다. 이것은 남는다.' for name in type1.SYSTEMS}}


def verdict(spec):
    return {'variants': [{'id': alias, 'links': [{'type': 'reference',
        'original_context': '그는 남는다.', 'revised_context': '이것은 남는다.',
        'explanation': '앞선 사람을 이것으로 받아 참조 대상이 어긋났다.'}]} for alias in spec['aliases']]}


def test_blind_full_text_dedup_and_unchanged():
    source = case()
    source['variants']['base_8b'] = '  나는 간다. 그는 남는다.  '
    spec = type1.judge_spec(source)
    payload = json.loads(spec['messages'][1]['content'])
    assert payload['original'] == '나는 간다. 그는 남는다.'
    assert len(payload['revisions']) == 1
    assert spec['unchanged'] == ['base_8b']
    assert all(name not in spec['messages'][1]['content'] for name in type1.SYSTEMS)
    assert type1.judge_spec(source) == spec


@pytest.mark.parametrize('corruption', ('missing_id', 'duplicate_id', 'unknown_type', 'invented_quote', 'duplicate_link'))
def test_invalid_judgments_rejected(corruption):
    spec = type1.judge_spec(case())
    value = verdict(spec)
    entry = value['variants'][0]
    if corruption == 'missing_id':
        value['variants'] = []
    elif corruption == 'duplicate_id':
        value['variants'].append(entry)
    elif corruption == 'unknown_type':
        entry['links'][0]['type'] = 'spelling'
    elif corruption == 'invented_quote':
        entry['links'][0]['original_context'] = '원문에 없는 글'
    elif corruption == 'duplicate_link':
        entry['links'].append(dict(entry['links'][0]))
    with pytest.raises(ValueError):
        type1.validate_judgment(json.dumps(value, ensure_ascii=False), spec)


def test_counts_separate_instances_from_affected_essays_and_missing():
    first = case()
    first['variants']['base_8b'] = first['original']
    second = case()
    second['case_id'] = 'essay:2'
    specs = [type1.judge_spec(c) for c in (first, second)]
    good = verdict(specs[0])
    good['variants'][0]['links'].append({'type': 'style', 'original_context': '나는 간다.',
        'revised_context': '나는 간다.', 'explanation': 'Test aggregation fixture only.'})
    data = type1.aggregate([first, second], specs, {
        first['case_id']: {'verdict': good}, second['case_id']: {'verdict': None}})
    assert data['summary']['base_8b']['unchanged'] == 1
    assert data['summary']['base_8b']['judged'] == 1
    assert data['summary']['base_8b']['links'] == 0
    assert data['summary']['luna_alone']['links'] == 2
    assert data['summary']['luna_alone']['affected_essays'] == 1
    assert data['summary']['luna_alone']['judged'] == 1
    assert [r for r in data['rows'] if r['case_id'] == second['case_id']][0]['count'] is None


def test_base_api_rejects_prompt_drift_before_dispatch():
    api = object.__new__(type1.BaseAPI)
    api.expected = [{'role': 'user', 'content': 'saved'}]
    with pytest.raises(ValueError, match='byte for byte'):
        api.request([{'role': 'user', 'content': 'changed'}], item_id='test', max_output=4096)


def test_explicit_decisions_exclude_nonerrors_and_uncertainty():
    spec = type1.judge_spec(case())
    alias = next(iter(spec['aliases']))
    payload = json.loads(spec['messages'][1]['content'])
    payload['previous_candidates'] = [{'id': alias, 'candidates': [{'candidate_id': 'C1'}, {'candidate_id': 'C2'}]}]
    spec['messages'][1]['content'] = json.dumps(payload)
    link = verdict(spec)['variants'][0]['links'][0]
    value = {'variants': [{'id': alias, 'candidates': [
        {**link, 'candidate_id': 'C1', 'status': 'not_new_error'},
        {**link, 'candidate_id': 'C2', 'status': 'uncertain'}], 'additional_links': []}]}
    final, decisions = type1.validate_decisions(json.dumps(value), spec)
    assert final['variants'][0]['links'] == []
    assert len(decisions['variants'][0]['candidates']) == 2
    value['variants'][0]['candidates'][0].update(status='new_error', explanation='원문과 같으므로 오류가 아니다.')
    with pytest.raises(ValueError, match='contradicts'):
        type1.validate_decisions(json.dumps(value), spec)
    value['variants'][0]['candidates'] = value['variants'][0]['candidates'][:1]
    with pytest.raises(ValueError, match='candidate IDs'):
        type1.validate_decisions(json.dumps(value), spec)


def test_partitioned_variant_repair_requires_complete_disjoint_candidate_ids():
    spec = type1.judge_spec(case())
    alias = next(iter(spec['aliases']))
    payload = json.loads(spec['messages'][1]['content'])
    payload['previous_candidates'] = [{'id': alias, 'candidates': [{'candidate_id': 'C1'}, {'candidate_id': 'C2'}]}]
    spec['messages'][1]['content'] = json.dumps(payload)
    link = verdict(spec)['variants'][0]['links'][0]
    value = {'variants': [{'id': alias, 'candidates': [{**link, 'candidate_id': cid, 'status': 'not_new_error'}],
                           'additional_links': []} for cid in ('C1', 'C2')]}
    final, decisions = type1.merge_partitioned_decisions(value, spec)
    assert len(decisions['variants']) == 1 and final['variants'][0]['links'] == []
    value['variants'][1]['candidates'][0]['candidate_id'] = 'C1'
    with pytest.raises(ValueError, match='candidate IDs'):
        type1.merge_partitioned_decisions(value, spec)


def test_existing_list_executor_accepts_isolated_artifact_root(tmp_path, monkeypatch):
    from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis, row
    from verak.v4 import policy_prompts_v43
    from verak.v5 import reviser
    monkeypatch.setattr(policy_prompts_v43, 'verify_frozen', lambda **kw: {})
    monkeypatch.setattr(reviser, 'ROOT', tmp_path / 'old_experiment')
    sent = []

    class Local:
        def request(self, messages, **kwargs):
            sent.append(kwargs)
            return {'raw': json.dumps({'actions': [
                {'action': 'EDIT', 'sentence': 'S1', 'old': '첫', 'new': '첫째'}], 'summary': '수정'})}

    value = reviser.edit_list(row('첫 문장이다. 둘째 문장이다.'), Local(), FakeBoundaryAnalysis(),
        tmp_path / 'new_experiment', request_item_id='TYPE1:test')
    assert value['no_protocol_error']
    assert sent == [{'item_id': 'TYPE1:test', 'max_output': 4096}]


@pytest.mark.parametrize('error_type', ['ConnectError', 'ReadTimeout', 'WriteError'])
def test_connection_recovery_preserves_evidence_and_only_releases_unsent(tmp_path, monkeypatch, error_type):
    monkeypatch.setattr(type1, 'ROOT', tmp_path)
    calls = tmp_path / 'api' / 'calls'
    calls.mkdir(parents=True)
    path = calls / 'failed.json'
    original = json.dumps({'status': 'error', 'error_type': error_type, 'reserved_usd': .2})
    path.write_text(original)
    with sqlite3.connect(tmp_path / 'api' / 'ledger.sqlite') as db:
        db.execute('CREATE TABLE calls (fingerprint TEXT, status TEXT, reserved REAL, confirmed REAL, path TEXT)')
        db.execute('INSERT INTO calls VALUES (?,?,?,?,?)', ('failed', 'error', .2, 0, str(path)))
    judgments = tmp_path / 'judgments'
    judgments.mkdir()
    judgment = judgments / 'essay.json'
    judgment.write_text(json.dumps({'status': 'unjudged'}))
    dispatched = []
    monkeypatch.setattr(type1, 'judge', lambda: dispatched.append(True))
    if error_type == 'ConnectError':
        type1.recover_connections()
        assert dispatched == [True]
        assert (tmp_path / 'connection_failure' / 'essay.json').exists()
        assert not judgment.exists()
        assert (tmp_path / 'connection_recovery.json').exists()
        expected = ('not_sent_connect_error', 0, 0)
    else:
        with pytest.raises(ValueError, match='pre-send ConnectError'):
            type1.recover_connections()
        assert dispatched == [] and judgment.exists()
        expected = ('error', .2, 0)
    assert path.read_text() == original
    with sqlite3.connect(tmp_path / 'api' / 'ledger.sqlite') as db:
        assert db.execute('SELECT status,reserved,confirmed FROM calls').fetchone() == expected
