"""v4.2 boundary regressions; no model/GPU or live Bareun needed."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
from verak.src.schemas import Profile,Sentence
from verak.v3.corrupt.document import Document
from verak.v4 import policy_prompts_v42 as prompts
from verak.v4.environment_v42 import (V42Environment,BoundaryParagraphs,sentence_spans,
    normalize_row,prepared_document)
from verak.v4.smoke42 import remap_plan,spacing_regression


class FakeBoundaryAnalysis(BoundaryParagraphs):
    def __init__(self): self.refreshed=[]
    def profile(self,text,neighbors=()):
        # Deliberately disagree with terminal segmentation: character-sized
        # analyzer sentences must not affect the new boundary guard.
        return Profile([Sentence(str(i),1,i,i+1,c,[]) for i,c in enumerate(text) if not c.isspace()],[], 'bareun','fake')
    def refresh(self,document,paragraph_ids):
        self.refreshed.append(sorted(paragraph_ids))
        return super().refresh(document,paragraph_ids)


def row(text):
    return {'source_id':'fixture','question':'설명하시오.','text':text,
        'profile':Profile([Sentence('S1',1,0,len(text),text,[])],[],'bareun','fake').to_dict()}


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(prompts,'verify_frozen',lambda **kwargs:{'version':prompts.VERSION})
    return V42Environment(row('대중문화가 있음.과거와는 다르다.'),FakeBoundaryAnalysis())


def action(env,role,name,**kwargs):
    return env.step(json.dumps({'action':name,**kwargs},ensure_ascii=False),role)


def test_no_space_boundaries_decimal_url_markers_and_runs():
    text='값은 3.14다.다음은 #@...#이다?!이 주소 https://example.com/a.b?q=1.2 를 본다...끝!'
    assert [text[a:b] for a,b in sentence_spans(text)]==[
        '값은 3.14다.','다음은 #@...#이다?!','이 주소 https://example.com/a.b?q=1.2 를 본다...','끝!']
    text='주소는 https://example.com/a.b. 다음이다.'
    assert [text[a:b] for a,b in sentence_spans(text)]==['주소는 https://example.com/a.b.','다음이다.']
    text='“맞다!”다음이다.'
    assert [text[a:b] for a,b in sentence_spans(text)]==['“맞다!”','다음이다.']


def test_initial_spacing_normalization_and_source_lineage(env):
    assert env.document.text=='대중문화가 있음. 과거와는 다르다.'
    assert [u.sid for u in env.document.units]==['S1','S1a']
    assert env.normalization['sid_mapping']=={'S1':['S1','S1a']}
    assert env.lineage['S1a']['source_start']==len('대중문화가 있음.')
    assert not env.actions['revision'] and not env.actions['korean']
    assert env.normalization['source_characters_preserved_except_whitespace']


def test_ending_edit_passes_despite_analyzer_count_and_korean_cannot_split(env):
    env.start_korean([])
    assert action(env,'korean','EDIT',sentence='S1',old='있음',new='있다')['valid']
    assert env.document.text=='대중문화가 있다. 과거와는 다르다.'
    before=env.document.text
    bad=action(env,'korean','EDIT',sentence='S1',old='있다.',new='있다. 중요하다.')
    assert not bad['valid'] and '경계' in bad['error'] and env.document.text==before
    assert not action(env,'korean','EDIT',sentence='S1',old='있다.',new='있다!')['valid']


def test_revision_split_scope_new_id_notice_handoff_and_undo(env):
    env.delegate([{'location':['S1'],'action':'EDIT','instruction':'설명한다.'}],1)
    before=env.document.text
    result=action(env,'revision','EDIT',sentence='S1',old='대중문화가 있음.',new='대중문화가 있다. 영향을 준다.')
    assert result['valid'] and result['created_sids']==['S1b']
    assert result['split_notice']['result_sids']==['S1','S1b']
    assert 'S1b' in env.scope and [u.sid for u in env.document.units]==['S1','S1b','S1a']
    payload=json.loads(env.public_messages('revision')[1]['content'])
    assert payload['sentence_lineage']['S1b']['parent_sid']=='S1'
    assert action(env,'revision','UNDO')['valid'] and env.document.text==before
    assert not action(env,'revision','EDIT',sentence='S1b',old='영향',new='관계')['valid']
    result=action(env,'revision','EDIT',sentence='S1',old='대중문화가 있음.',new='대중문화가 있다. 영향을 준다.')
    assert result['valid'] and result['created_sids']==['S1c']
    env.start_korean([])
    assert any(n.get('kind')=='sentence_split' for n in env.handoff['marker_change_notices'])
    assert action(env,'korean','EDIT',sentence='S1c',old='영향',new='도움')['valid']
    assert action(env,'korean','UNDO')['valid']


def test_three_sentence_split_scope_and_multiple_json_stay_invalid(env):
    env.delegate([{'location':['S1'],'action':'EDIT','instruction':'고친다.'}],1)
    before=env.document.text
    bad=action(env,'revision','EDIT',sentence='S1',old='대중문화가 있음.',new='하나다. 둘이다. 셋이다.')
    assert not bad['valid'] and env.document.text==before
    assert not action(env,'revision','EDIT',sentence='S1a',old='다르다',new='같다')['valid']
    assert not env.step('{"action":"UNDO"}\n{"action":"UNDO"}','revision')['valid']
    assert env.remaining==3


def test_prepared_roundtrip_and_supplied_document_is_not_mutated(monkeypatch):
    monkeypatch.setattr(prompts,'verify_frozen',lambda **kwargs:{'version':prompts.VERSION})
    original=row('대중문화가 있음.과거와는 다르다.');analysis=FakeBoundaryAnalysis()
    from verak.v3.phase2 import restore_profile
    document=Document.from_profile(original['text'],restore_profile(original['profile']))
    snapshot=document.snapshot()
    prepared=normalize_row(original,analysis,document=document)
    assert document.snapshot()==snapshot and original['text']=='대중문화가 있음.과거와는 다르다.'
    assert prepared_document(prepared).snapshot()==prepared['layout']
    env=V42Environment(prepared,analysis)
    assert env.document.snapshot()==prepared['layout']
    assert normalize_row(prepared,analysis)==prepared


def test_derived_ids_move_order_marker_guard_and_undo(env):
    env.delegate([{'location':['S1','S1a'],'action':'MOVE','instruction':'순서를 바꾼다.'}],1)
    before=env.document.snapshot()
    assert action(env,'revision','MOVE',sentence='S1a',to={'before':'S1'})['valid']
    assert [u.sid for u in env.document.units]==['S1a','S1']
    assert env.document.text=='과거와는 다르다. 대중문화가 있음.'
    assert action(env,'revision','UNDO')['valid'] and env.document.snapshot()==before
    # A mask-internal period is not a boundary and protected edits still reject.
    assert action(env,'revision','EDIT',sentence='S1',old='대중문화',new='#@가림...#')['valid'] is False


def test_plan_id_mapping_preserves_task_evidence_and_all_children():
    original=row('대중문화가 있음.과거와는 다르다.')
    prepared=normalize_row(original,FakeBoundaryAnalysis())
    plan={'source_id':'fixture','status':'completed','phase_call':71,'items':[
        {'location':['S1'],'instruction':'S1의 첫 문장 종결을 고친다.','evidence':'대중문화가 있음.',
         'item_id':'fixture:D3I1','owner':'revision','action':'EDIT','problem':'unchanged'}],
        'writer_notes':['unchanged'],'dropped':[]}
    saved=deepcopy(plan);mapped=remap_plan(plan,prepared)
    assert plan==saved and mapped['writer_notes']==saved['writer_notes']
    item=mapped['items'][0]
    assert item['location']==['S1','S1a'] and item['instruction'].startswith(saved['items'][0]['instruction'])
    assert item['evidence']==saved['items'][0]['evidence']
    assert mapped['v42_reuse']['item_count_unchanged'] and mapped['v42_reuse']['new_planner_calls']==0
    from verak.v4.prep2_content import public_tasks
    assert set(public_tasks(mapped['items'])[0])=={'location','action','instruction'}


def test_saved_twenty_plans_and_thirteen_spacing_regressions(monkeypatch):
    # Local archived evidence; the generic unit tests above also run in CI.
    from verak.v4.smoke42 import PRIOR
    if not (PRIOR/'sample.json').exists(): pytest.skip('Local frozen smoke artifacts absent')
    sample=json.loads((PRIOR/'sample.json').read_text());rows=[];n=0
    for source in sample['source_ids']:
        name=source.replace(':','_')
        old=json.loads((PRIOR/'essays'/f'{name}.json').read_text())
        prepared=normalize_row(old,FakeBoundaryAnalysis());rows.append(prepared)
        plan=json.loads((PRIOR/'content/items'/f'{name}.json').read_text())
        result=remap_plan(plan,prepared)
        assert result['v42_reuse']['item_count_unchanged']
        assert ''.join(old['text'].split())==''.join(prepared['text'].split())
        n+=1
    assert n==20 and spacing_regression(rows)['archived_rejections']==13
    target=next(r for r in rows if r['source_id']=='train:58244')
    monkeypatch.setattr(prompts,'verify_frozen',lambda **kwargs:{'version':prompts.VERSION})
    env=V42Environment(target,FakeBoundaryAnalysis())
    assert target['normalization']['sid_mapping']['S2']==['S2','S2a']
    env.delegate([{'location':['S2','S2a'],'action':'EDIT','instruction':'종결을 고친다.'}],1)
    assert action(env,'revision','EDIT',sentence='S2',old='있음',new='있다')['valid']


def test_runtime_demands_prepared_rows_and_prompts_are_exact_addition():
    from verak.v4.runtime_v42 import require_prepared
    from verak.v4 import policy_prompts_v41 as prior
    with pytest.raises(ValueError): require_prepared(row('한 문장이다.'))
    assert prompts.KOREAN==prior.KOREAN
    assert prompts.REVISION.replace('\n'+prompts.REVISION_SPLIT_RULE,'')==prior.REVISION
    assert 'JSON 객체 하나만 출력한다' in prompts.REVISION
    assert 'JSON 객체 하나만 출력한다' in prompts.KOREAN


def test_report_categories_keep_multi_json_invalid_and_no_adjusted_gate(env):
    from verak.v4.smoke42_report import rejection_category
    env.start_korean([])
    record=env.step('{"action":"UNDO"}{"action":"UNDO"}','korean')
    assert rejection_category({'action':record})=='malformed_JSON'
    from verak.v4.smoke41 import action_metrics
    metrics=action_metrics([{'calls':[{'action':record,'role':'korean'}]}],
        {'by_stage':{'teacher':{'logical_requests':1}}})
    assert metrics['returned_action_calls']==1 and metrics['invalid_actions']==1
    assert metrics['primary_gate_pass'] is False
