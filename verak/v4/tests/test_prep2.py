from copy import deepcopy
import json
import pytest

from verak.v4.maps import extraction_request, validate, consensus
from verak.v4.prep2_maps import gate


def map_fixture(n):
    row = {'genre':'explanatory','question':'설명하시오.', 'paragraphs':[
        {'id':f'P{i}','sentences':[{'id':f'S{i}a','text':'중심 설명이다.'},{'id':f'S{i}b','text':'관련 설명이다.'}]}
        for i in range(1,n+1)]}
    value = {'genre':'explanatory','paragraph_roles':[
        {'paragraph':f'P{i}','role':'explanation','key_sentence':f'S{i}a'} for i in range(1,n+1)],
        'sentence_relations':[{'source':'S1b','target':'S2b','type':'sequence'}],
        'paragraph_relations':[], 'off_topic':[]}
    return row,value


def test_cross_paragraph_rule_only_above_six_without_changing_old_default():
    for n in (2,6):
        row,value = map_fixture(n)
        assert validate(value,row,long_only=True)==value
        with pytest.raises(ValueError,match='main/key'):
            validate(value,row)
        result,_ = consensus(value,value,row,long_only=True)
        assert result['sentence_relations']==value['sentence_relations']
    row,value = map_fixture(7)
    with pytest.raises(ValueError,match='main/key'):
        validate(value,row,long_only=True)
    value['sentence_relations'][0].update(source='S1a',target='S2a')
    validate(value,row,long_only=True)


def test_relaxed_length_rule_never_relaxes_ids_or_paragraph_membership():
    row,value = map_fixture(2)
    value['paragraph_roles'][0]['key_sentence']='S2a'
    with pytest.raises(ValueError,match='belong'):
        validate(value,row,long_only=True)
    row,value = map_fixture(2)
    value['sentence_relations'][0]['target']='S99'
    with pytest.raises(ValueError,match='Unknown ID'):
        validate(value,row,long_only=True)
    messages,_ = extraction_request(row,long_only=True)
    payload=json.loads(messages[1]['content'])
    assert payload['valid_ids']['sentence_paragraph']['S1b']=='P1'
    assert payload['cross_paragraph_main_key_required'] is False
    assert 'advisory' not in json.dumps(payload)  # no additional private field is sent
    assert '참고 정보' in messages[0]['content']


def test_gate_requires_both_runs_for_39_of_43_fixed_sources():
    assert not gate([{'status':'valid'}]*38+[{'status':'unavailable'}]*5)['passed']
    assert gate([{'status':'valid'}]*39+[{'status':'unavailable'}]*4)['passed']
    with pytest.raises(ValueError):
        gate([{'status':'valid'}]*42)


@pytest.fixture
def delegated():
    from verak.src.schemas import Profile,Sentence
    from verak.v3.tests.test_environment import StubParagraphAnalysis
    from verak.v4.prep2_content import DelegatedEnv
    text='첫 주장을 쓴다. 이를 설명한다.'
    profile=Profile([Sentence('S1',1,0,9,'첫 주장을 쓴다.',[]),
        Sentence('S2',1,10,len(text),'이를 설명한다.',[])],[],'bareun','fixture')
    env=DelegatedEnv({'source_id':'train:fixture','question':'설명하시오.','text':text,
        'profile':profile.to_dict()},StubParagraphAnalysis())
    env.delegate([{'location':['S1'],'action':'EDIT','instruction':'첫 주장을 명료화한다.'}],1)
    return env


def test_delegation_scope_is_atomic_and_cannot_undo_previous_work(delegated):
    env=delegated
    before=env.document.text
    assert not env.step(json.dumps({'action':'DELETE','sentence':'S2'}),'revision')['valid']
    assert not env.step(json.dumps({'action':'MOVE','sentence':'S1','to':{'after':'S2'}}),'revision')['valid']
    assert env.document.text==before
    assert env.step(json.dumps({'action':'EDIT','sentence':'S1','old':'첫','new':'중심'}),'revision')['valid']
    env.delegate([{'location':['S2'],'action':'EDIT','instruction':'설명을 명료화한다.'}],2)
    assert not env.step('{"action":"UNDO"}','revision')['valid']
    assert '중심' in env.document.text


def test_insert_cap_persists_across_undo_and_delegations(delegated):
    env=delegated
    insert={'action':'INSERT','after':'S1','text':'구체적인 설명이다.',
        'relation':{'type':'support','target':'S1'},'source':None}
    assert env.step(json.dumps(insert),'revision')['valid']
    assert env.step('{"action":"UNDO"}','revision')['valid']
    assert env.step(json.dumps(insert),'revision')['valid']
    env.delegate([{'location':['S1'],'action':'INSERT','instruction':'연결한다.'}],2)
    before=env.document.text
    assert not env.step(json.dumps(insert),'revision')['valid']
    assert env.document.text==before and env.insert_count==2
    payload=json.loads(env.public_messages('revision',2)[1]['content'])
    assert payload['inserts_remaining']==0 and 'notice' in payload


def test_public_assignment_excludes_raw_feedback_and_private_ids(delegated):
    env=delegated
    env.tasks[0].update(item_id='PRIVATE_ID',problem='PRIVATE_PROBLEM',evidence='PRIVATE_EVIDENCE',feedback_index=1)
    text=json.dumps(env.public_messages('revision',6),ensure_ascii=False)
    assert 'PRIVATE_' not in text and '첫 주장을 명료화한다.' in text
    env.start_korean([])
    payload=json.loads(env.public_messages('korean',14)[1]['content'])
    assert payload['whole_essay_typo_spacing_pass'] and payload['scope']=='whole_essay'
    assert not env.step('{"action":"DELETE","sentence":"S1"}','korean')['valid']


def test_quality_gate_adds_redundancy_and_naturalness_without_hiding_step_limits():
    from verak.v4.prep2_content import selection
    items=[{'item_id':'x','owner':'revision'}]
    verdict={'items':[{'item_id':'x','status':'addressed'}],'meaning_preserved':'yes',
        'invented_specifics':'no','invented_experiences':'no','better_than_original':'yes',
        'repetition_introduced':'no','reads_naturally':'yes'}
    assert selection(items,{'complete':False},verdict)['quality_keep']
    for key,bad in [('repetition_introduced','yes'),('reads_naturally','unknown')]:
        changed={**verdict,key:bad}
        assert not selection(items,{'complete':True},changed)['quality_keep']


def test_items_require_at_most_four_and_exact_located_evidence():
    from verak.v4.prep2_content import validate_plan
    row,_=map_fixture(2)
    row['source_id']='train:fixture'
    item={'rubric':'clarity','feedback_index':1,'problem':'불명료한 설명','location':['S1a'],
        'evidence':'중심 설명','owner':'revision','action':'EDIT','instruction':'해당 설명의 지시 대상을 명료화한다.'}
    value={'items':[item],'dropped':[],'writer_notes':[]}
    assert len(validate_plan(value,row)['items'])==1
    bad=deepcopy(value); bad['items'][0]['evidence']='없는 내용'
    result=validate_plan(bad,row)
    assert not result['items'] and len(result['mechanically_dropped'])==1
    with pytest.raises(ValueError,match='four'):
        validate_plan({**value,'items':[item]*5},row)


def test_score_audit_compares_targets_without_treating_feedback_as_gold():
    from verak.v4.prep2_scorer_audit import compare_row
    row={'assistant':'6 6 6 6 6 6 6 6\n\n### Feedback\nLLM feedback',
        'grader_1_scores':[3.]*8,'grader_2_scores':[4.]*8}
    scores,r1,r2,mapped=compare_row(row)
    assert scores==mapped==[6]*8
    assert all((s+1)/2==(a+b)/2 for s,a,b in zip(scores,r1,r2))
    row['grader_1_scores'][0]=6
    with pytest.raises(ValueError,match='1–5'):
        compare_row(row)


def test_saved_prep1_default_maps_replay_identically_when_artifacts_available():
    from verak.v4.common import ROOT,read_json,safe_id
    if not (ROOT/'C/final_complete.json').exists():
        pytest.skip('Local private PREP1 artifacts are not committed')
    design=read_json(ROOT/'design.json')
    checked=0
    for source in design['maps150']:
        row=read_json(ROOT/'essays'/(safe_id(source)+'.json'))
        attempts=[read_json(ROOT/'C'/f'attempt_{a}'/(safe_id(source)+'.json')) for a in (1,2)]
        for attempt in attempts:
            if attempt['status']=='valid':
                assert validate(attempt['value'],row)==attempt['value']
            elif attempt['status']=='invalid':
                with pytest.raises(ValueError):
                    validate(attempt['parsed'],row)
        saved=read_json(ROOT/'C/consensus'/(safe_id(source)+'.json'))
        if saved['status']=='valid':
            value,diag=consensus(*(a['value'] for a in attempts),row)
            assert value==saved['map'] and diag==saved['diagnostics']
        checked+=1
    assert checked==150
