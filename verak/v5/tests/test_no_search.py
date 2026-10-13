"""SEARCH deferral must preserve real work, provenance and failed endings."""
from copy import deepcopy
import json

import pytest

from verak.v4.content_env import dumps
from verak.v4.fix1_infer import run_case
from verak.v4.render_v43 import assert_rendered
from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis, row
from verak.v5.cooperate_env import CooperationEnvironment
from verak.v5.tests.test_cooperate import Tokens


SEARCH = {'location': ['S1'], 'action': 'INSERT',
          'instruction': '이 설명을 뒷받침하는 외부 사례를 확인한다.', 'needs_search': 'yes'}
EDIT = {'location': ['S1'], 'action': 'EDIT',
        'instruction': '문장이라는 표현을 글로 바꾼다.', 'needs_search': 'no'}
FINISH = {'action': 'FINISH', 'summary': '검토했다.', 'writer_notes': ['기존 작성자 메모']}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from verak.v4 import policy_prompts_v43, search_v43
    monkeypatch.setattr(policy_prompts_v43, 'verify_frozen', lambda **kw: {})
    def unexpected(query, **kwargs):
        pytest.fail('No index or retrieval is permitted in this test')
    monkeypatch.setattr(search_v43, 'local_search', unexpected)


class Backend:
    def __init__(self, items, *, repeats=1, fail=False):
        self.observations = []
        self.revision_calls = 0
        self.actions = [
            {'action': 'SCORE'},
            {'action': 'MAP', 'map': {
                'genre': 'argumentative', 'paragraph_roles': [{'paragraph': 'P1', 'role': 'body', 'key_sentence': 'S1'}],
                'sentence_relations': [], 'paragraph_relations': [], 'off_topic': []}},
            *[{'action': 'DELEGATE', 'to': 'revision', 'items': deepcopy(items)} for _ in range(repeats)],
            *([{'action': 'INVALID'}]*3 if fail else []),
            *[deepcopy(FINISH) for _ in range(4)],
        ]

    def generate(self, messages, role):
        self.observations.append((role, deepcopy(messages)))
        if role == 'orchestrator':
            action = self.actions.pop(0)
        elif role == 'revision' and self.revision_calls == 0:
            action = {'action': 'EDIT', 'sentence': 'S1', 'old': '문장', 'new': '글'}
            self.revision_calls += 1
        else:
            action = {'action': 'STOP', 'status': 'done', 'summary': '검토했다.', 'issues': []}
        return {'raw': dumps(action), 'usage': {'prompt_tokens': 1}, 'finish_reason': 'stop'}


def run(path, backend, *, enabled=False):
    source = row('문장이다.'); source['genre'] = 'argumentative'
    return run_case(source, case_id='fixture', output=path, backend=backend,
        tok=Tokens(), analysis=FakeBoundaryAnalysis(),
        score=lambda *args: {'expected': [5]*8, 'integers': [5]*8},
        cooperation=True, capture_states=True, search_enabled=enabled)


def test_search_only_delegation_becomes_notes_without_a_fabricated_worker_stage(tmp_path):
    backend = Backend([SEARCH])
    result = run(tmp_path, backend)
    assert result['completion']['no_protocol_error']
    assert result['deferred_task_count'] == 1
    assert [s['role'] for s in result['stages']] == ['korean']
    assert not any(role == 'revision' for role, _ in backend.observations)
    assert result['final']['text'] == result['initial']['text']
    assert result['finish_report'] == FINISH
    assert result['writer_notes'][0] == '기존 작성자 메모'
    assert SEARCH['instruction'] in result['writer_notes'][1]
    record = next(a for a in result['orchestrator_actions'] if a['action'] == 'DELEGATE')
    assert record['value']['items'] == [SEARCH]
    assert not record['worker_dispatched'] and not record['executed_items']
    assert record['deferred_to_writer_notes'][0]['origin'] == 'controller'
    last = json.loads([m for r, m in backend.observations if r == 'orchestrator'][-1][1]['content'])
    assert last['integration']['search_available'] is False
    assert last['integration']['deferred_search_tasks'][0]['status'] == 'deferred_to_writer'
    assert result['source_citations']['search_history'] == []


def test_mixed_delegation_still_executes_the_non_search_edit(tmp_path):
    backend = Backend([SEARCH, EDIT])
    result = run(tmp_path, backend)
    assert result['final']['text'] == '글이다.'
    assert [s['role'] for s in result['stages']] == ['revision', 'korean']
    assert result['stages'][0]['assigned_tasks'] == [EDIT]
    assert result['deferred_task_count'] == 1
    payload = next(assert_rendered(m) for r, m in backend.observations if r == 'revision')
    assert all(t['needs_search'] == 'no' for t in payload['assigned_tasks'])
    assert payload['search_history'] == []
    assert payload['cooperation']['search_available'] is False
    assert result['completion']['all_stages_STOP_done']
    # A done stage does not mean its controller-deferred task was performed.
    assert result['deferred_search_tasks'][0]['status'] == 'deferred_to_writer'


def test_repeated_search_requests_keep_one_note_and_never_dispatch_a_worker(tmp_path):
    result = run(tmp_path, Backend([SEARCH], repeats=3))
    assert result['deferred_task_count'] == 1
    assert len(result['writer_notes']) == 2
    assert len([a for a in result['orchestrator_actions'] if a['action'] == 'DELEGATE']) == 3
    assert [s['role'] for s in result['stages']] == ['korean']


def test_notes_survive_an_orchestrator_failure_without_fabricating_finish(tmp_path):
    result = run(tmp_path, Backend([SEARCH], fail=True))
    assert not result['completion']['no_protocol_error']
    assert 'finish_report' not in result
    assert result['deferred_task_count'] == 1
    assert SEARCH['instruction'] in result['writer_notes'][0]


def test_bypassing_the_router_cannot_enable_search():
    value = CooperationEnvironment(row('문장이다.'), FakeBoundaryAnalysis(), search_enabled=False)
    with pytest.raises(ValueError, match='Defer SEARCH'):
        value.delegate([SEARCH], 1)
    value.delegate([EDIT], 1)
    result = value.step(dumps({'action': 'SEARCH', 'item_id': 'D1T1', 'query': '사례'}), 'revision')
    assert not result['valid']
    assert value.search.history == []
    with pytest.raises(ValueError, match='SEARCH is disabled'):
        value.search.backend('사례')


def test_search_setting_cannot_reuse_an_output_from_the_other_condition(tmp_path):
    run(tmp_path, Backend([EDIT]), enabled=True)
    with pytest.raises(ValueError, match='Saved SEARCH setting differs'):
        run(tmp_path, Backend([EDIT]), enabled=False)


def test_same_condition_resume_does_not_call_the_model(tmp_path):
    expected = run(tmp_path, Backend([SEARCH]))
    backend = Backend([EDIT])
    assert run(tmp_path, backend) == expected
    assert backend.observations == []


def test_default_search_enabled_routing_keeps_the_frozen_contract():
    value = CooperationEnvironment(row('문장이다.'), FakeBoundaryAnalysis())
    items, deferred = value.route_revision_items([SEARCH, EDIT])
    assert items == [SEARCH, EDIT] and deferred == []
    value.delegate(items, 1)
    payload = assert_rendered(value.public_messages('revision'))
    assert [i['needs_search'] for i in payload['assigned_tasks']] == ['yes', 'no']
    assert value.deferred_search_tasks == []
