from copy import deepcopy
from types import SimpleNamespace
import json
import pytest

from verak.v3.corrupt.document import Document, Paragraph, Unit
from verak.v4.content_env import dumps
from verak.v4.environment_v46 import V46Environment
from verak.v4.render_v43 import assert_rendered
from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis, row
from verak.v5.alignment import align
from verak.v5 import detector
from verak.v5.pipeline import ScopedEnvironment, stage, request


@pytest.fixture(autouse=True)
def frozen(monkeypatch):
    from verak.v4 import policy_prompts_v43
    monkeypatch.setattr(policy_prompts_v43,'verify_frozen',lambda **kw:{})


def task(scope):
    return {'location':scope,'action':'EDIT','instruction':'지정한 항목만 고친다.','needs_search':'no'}


def test_oracle_destination_exception_does_not_change_a1_or_movable_targets():
    source=row('첫 문장이다. 둘째 문장이다. 셋째 문장이다.')
    old=V46Environment(source,FakeBoundaryAnalysis());new=ScopedEnvironment(source,FakeBoundaryAnalysis())
    for env in (old,new):env.delegate([task(['S1'])],1)
    new.oracle_move_targets={'S1'}
    action=dumps({'action':'MOVE','sentence':'S1','to':{'after':'S1b'}})
    assert not old.step(action,'revision')['valid']
    result=new.step(action,'revision')
    assert result['valid'],result
    assert new.document.units[-1].sid=='S1'
    assert not new.step(dumps({'action':'MOVE','sentence':'S1a','to':{'before':'S1'}}),'revision')['valid']


def test_scoped_korean_rejections_consume_budget_and_force_failure_at_three():
    env=ScopedEnvironment(row('첫 문장이다. 둘째 문장이다.'),FakeBoundaryAnalysis())
    env.start_korean([task(['S1'])]);env.scope={'S1'};env.narrow_korean=True
    payload=assert_rendered(env.public_messages('korean'))
    assert payload['allowed_scope']==['S1'] and not payload['whole_essay_typo_spacing_pass']
    bad=dumps({'action':'EDIT','sentence':'S1a','old':'둘째','new':'두 번째'})
    for i in range(3):
        result=env.step(bad,'korean');assert not result['valid'];assert env.steps_used==i+1
    assert env.closed and env.controller_stop['failure']
    assert env.document.units[1].text=='둘째 문장이다.'


def test_whole_korean_pass_still_uses_full_v46_contract():
    env=ScopedEnvironment(row('첫 문장이다. 둘째 문장이다.'),FakeBoundaryAnalysis())
    env.start_korean([])
    assert assert_rendered(env.public_messages('korean'))['whole_essay_typo_spacing_pass']
    assert env.step(dumps({'action':'EDIT','sentence':'S1a','old':'둘째','new':'두 번째'}),'korean')['valid']


class NoTokens:
    def apply_chat_template(self,*a,**kw):return [1]


def test_empty_notice_set_requires_no_model_call(tmp_path):
    from verak.v5.pipeline import repair
    from verak.v4.scale2_projection import save_document
    env=ScopedEnvironment(row('학생이다.'),FakeBoundaryAnalysis())
    result=repair(env.row,save_document(env.document),[],None,None,NoTokens(),FakeBoundaryAnalysis(),tmp_path)
    assert result['no_protocol_error'] and result['stages']==[]


def test_request_keeps_unfinished_attempt_and_never_regenerates(tmp_path):
    from verak.v5.common import atomic_new
    path=tmp_path/'call';atomic_new(path.with_suffix('.prefix.json'),{'old':'state'})
    with pytest.raises(RuntimeError,match='Unfinished'):
        request(None,NoTokens(),[{'role':'user','content':'x'}],'korean',path)


def make_doc(paras):
    n=0;paragraphs=[]
    for pi,texts in enumerate(paras,1):
        units=[]
        for text in texts:
            n+=1;units.append(Unit(f'S{n}',text,[],'' if not units else ' '))
        paragraphs.append(Paragraph(f'P{pi}',units))
    return Document(paragraphs,['']+['\n\n']*(len(paragraphs)-1))


class NullAnalysis:
    def refresh(self,*a):pass


def test_alignment_tracks_sentence_and_paragraph_moves_without_gold():
    original=make_doc([['고양이는 야옹 한다.','달은 밝게 빛난다.'],['바다는 파랗다.','겨울에는 눈이 온다.']])
    final,audit=align(original,'바다는 파랗다. 겨울에는 눈이 온다.\n\n달은 밝게 빛난다. 고양이는 야옹 한다.',NullAnalysis())
    assert [p.pid for p in final.paragraphs]==['P2','P1']
    assert [u.sid for u in final.units]==['S3','S4','S2','S1']
    assert not audit['failures'] and any(x['order_inversion'] for x in audit['matches'])


def test_alignment_duplicate_sentences_are_explicitly_ambiguous():
    original=make_doc([['같은 문장이다.','같은 문장이다.']])
    final,audit=align(original,original.text,NullAnalysis())
    assert final.text==original.text and audit['alignment_failed']
    assert any(x['type']=='ambiguous_sentence_match' for x in audit['failures'])


def base_phi():
    return {'S1':{'ec_categories':['CAUSE'],'conjunction_category':None,
        'antecedent':{'sid':None,'text_sha256':None},'subject_omitted':False,'omission_uncertain':False,
        'style':'한다','majority_style':'한다','off_style':False},
        'S2':{'ec_categories':[],'conjunction_category':'RESULT',
        'antecedent':{'sid':'S1','text_sha256':'old'},'subject_omitted':True,'omission_uncertain':False,
        'style':'한다','majority_style':'한다','off_style':False}}


def test_notices_compare_category_and_antecedent_text_not_marker_surface(monkeypatch):
    old=base_phi();new=deepcopy(old);new['S2']['antecedent']['text_sha256']='new'
    before=SimpleNamespace(units=[SimpleNamespace(sid=x) for x in old]);after=SimpleNamespace(units=before.units)
    monkeypatch.setattr(detector,'phi',lambda d:old if d is before else new)
    notices=detector.changes(before,after,{'sentence_relations':[{'source':'S2','target':'S1','type':'support'}]})
    assert {x['type'] for x in notices}=={'conjunction_antecedent','omitted_subject_antecedent'}
    assert all(x['scope']==['S2','S1'] and not x['error_verdict'] for x in notices)
    assert all(x['original_graph_relations'] for x in notices)


def test_inserted_style_and_unknown_resolution_are_not_silently_successful(monkeypatch):
    old=base_phi();new=deepcopy(old);new['S3']=deepcopy(old['S1']);new['S3']['style']='합니다';new['S3']['off_style']=True
    before=SimpleNamespace(units=[SimpleNamespace(sid=x) for x in old]);after=SimpleNamespace(units=[SimpleNamespace(sid=x) for x in new])
    monkeypatch.setattr(detector,'phi',lambda d:old if d is before else new)
    notices=detector.changes(before,after)
    style=next(n for n in notices if n['type']=='style_vs_majority')
    assert style['inserted'] and style['sid']=='S3'
    resolved=detector.resolved(notices,after)
    assert next(x for x in resolved if x['type']=='style_vs_majority')['resolved_by_rule'] is False
    assert any(x['resolved_by_rule'] is None for x in resolved)


def test_each_changed_connective_occurrence_gets_one_notice(monkeypatch):
    old=base_phi();new=deepcopy(old)
    old['S1']['ec_categories']=['CAUSE','CONDITION','CAUSE']
    new['S1']['ec_categories']=['CAUSE','ADVERSATIVE','PURPOSE']
    before=SimpleNamespace(units=[SimpleNamespace(sid=x) for x in old]);after=SimpleNamespace(units=before.units)
    monkeypatch.setattr(detector,'phi',lambda d:old if d is before else new)
    notices=detector.changes(before,after)
    ec=[n for n in notices if n['type']=='ec_category']
    assert len(ec)==2 and [n['ec_index_after'] for n in ec]==[1,2]
    assert all(r['resolved_by_rule'] is None for r in detector.resolved(ec,after))
