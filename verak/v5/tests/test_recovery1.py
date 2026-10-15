"""Resumed observations must preserve the state used to justify teacher labels."""
import pytest

from verak.v4.content_env import dumps
from verak.v4.render_v43 import assert_rendered
from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis, row
from verak.v5.cooperate_env import CooperationEnvironment
from verak.v5.recovery1 import restore_seed


@pytest.fixture(autouse=True)
def no_contract_io(monkeypatch):
    from verak.v4 import policy_prompts_v43
    monkeypatch.setattr(policy_prompts_v43, 'verify_frozen', lambda **kw: {})


def seed(env):
    return {'source_id': 'train:example', 'messages': env.public_messages('korean')+
            [{'role':'assistant','content':dumps({'action':'STOP','status':'done','summary':'검토 완료','issues':[]})}]}


def test_restore_preserves_text_ids_relations_and_rejected_history():
    analysis = FakeBoundaryAnalysis()
    env = CooperationEnvironment(row('학생은 책을 읽었다. 공부했다.'), analysis)
    env.start_korean([])
    env.relations = {'S1': {'type':'main','target':'Q'}}
    action = {'action':'EDIT','sentence':'S1','old':'학생','new':'학생'}
    assert not env.step(dumps(action),'korean')['valid']
    restored = restore_seed(seed(env), analysis)
    before = assert_rendered(env.public_messages('korean'))
    after = assert_rendered(restored.public_messages('korean'))
    for field in ('paragraphs','rendering','declared_relations','work_journal','steps_left','last_result','allowed_scope'):
        assert before[field] == after[field]
    assert restored.rejected_in_a_row == 1
    assert restored.document.text == env.document.text
    assert not restored.step(dumps(action),'korean')['valid']
    assert restored.document.text == env.document.text
    assert restored.step(seed(env)['messages'][-1]['content'],'korean')['valid']


def test_do_not_resume_controller_stop_boundary():
    env = CooperationEnvironment(row('학생은 책을 읽었다.'), FakeBoundaryAnalysis())
    env.start_korean([])
    action = {'action':'EDIT','sentence':'S1','old':'없는말','new':'말'}
    for _ in range(2): assert not env.step(dumps(action),'korean')['valid']
    with pytest.raises(ValueError, match='controller stop boundary'):
        restore_seed(seed(env), FakeBoundaryAnalysis())


def test_restore_keeps_retrieved_evidence_and_citation_source():
    env = CooperationEnvironment(row('학생은 책을 읽었다.'), FakeBoundaryAnalysis())
    env.start_korean([])
    env.search.history = [{'query':'책','passages':[]}]
    env.search.successful_sourced_inserts = 1
    # Use the actual citation source contract rather than a fake URL-only label.
    env.search.citations = {'S1': {'inserted_text':env.document.text,
        'source':{'title':'책','passage_id':'wiki:1'},
        'passage':{'text':'학생이 책을 읽었다.'}}}
    sample = seed(env)
    restored = restore_seed(sample, FakeBoundaryAnalysis())
    p = assert_rendered(sample['messages'][:-1])
    result = assert_rendered(restored.public_messages('korean'))
    for field in ('source_citations','search_history','successful_sourced_inserts'):
        assert p[field] == result[field]
