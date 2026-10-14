import json
from copy import deepcopy

import pytest

from verak.src.schemas import Profile,Sentence
from verak.v3.tests.test_environment import StubParagraphAnalysis
from verak.v4.content_env import ContentEnv,dumps
from verak.v4.content import teacher_messages,selection,validate_judgment,export_cases


@pytest.fixture
def env():
    text='첫 주장을 쓴다. 이를 설명한다.'
    profile=Profile([Sentence('S1',1,0,9,'첫 주장을 쓴다.',[]),
        Sentence('S2',1,10,len(text),'이를 설명한다.',[])],[],'bareun','fixture')
    return ContentEnv({'source_id':'train:fixture','question':'설명하시오.','text':text,
                       'profile':profile.to_dict()},StubParagraphAnalysis())


def act(env,action,role='revision',**kwargs):
    return env.step(dumps({'action':action,**kwargs}),role)


def test_insert_undo_stable_ids_and_korean_boundary(env):
    original=env.document.text
    assert act(env,'INSERT',after='S1',text='이것은 일반적 설명이다.',
        relation={'type':'support','target':'S1'},source=None)['valid']
    assert env.relations['N1']['target']=='S1'
    assert act(env,'UNDO')['valid'] and env.document.text==original
    assert not env.relations
    assert act(env,'INSERT',after='S1',text='또 다른 설명이다.',
        relation={'type':'support','target':'S1'},source=None)['created_sids']==['N2']
    env.start_korean()
    current=env.document.text
    assert not act(env,'MOVE',role='korean',sentence='N2',to={'before':'S1'})['valid']
    assert not act(env,'UNDO',role='korean')['valid']
    assert env.document.text==current
    assert act(env,'EDIT',role='korean',sentence='N2',old='다른',new='구체적')['valid']


def test_invalid_split_and_relation_are_atomic(env):
    original=env.document.text
    for text,target in [('하나다. 둘이다.','S1'),('하나다.','S99')]:
        assert not act(env,'INSERT',after='S1',text=text,
            relation={'type':'support','target':target},source=None)['valid']
        assert env.document.text==original and not env.relations
    assert not act(env,'EDIT',sentence='S1',old='쓴다',new='쓴다. 추가한다')['valid']
    assert env.document.text==original
    assert not env.step('{"action":"STOP","action":"UNDO"}','revision')['valid']


def test_privileged_items_never_enter_public_state_or_handoff(env):
    items=[{'item_id':'PRIVATE_ITEM_1','problem':'SECRET_TEACHER_TEXT','owner':'revision'}]
    public=env.public_messages('revision',14)
    private=teacher_messages(public,items,'revision')
    assert 'SECRET_TEACHER_TEXT' in dumps(private)
    assert 'SECRET_TEACHER_TEXT' not in dumps(public)
    act(env,'STOP',status='blocked',summary='SECRET_TEACHER_TEXT',issues=['PRIVATE_ITEM_1'])
    env.start_korean()
    exported=env.public_messages('korean',14)
    assert 'SECRET_TEACHER_TEXT' not in dumps(exported)
    assert 'PRIVATE_ITEM_1' not in dumps(exported)


def test_quality_gate_counts_full_items_not_partials_or_deferrals():
    items=[{'item_id':str(i),'owner':'revision' if i<3 else 'writer'} for i in range(4)]
    verdict={'items':[{'item_id':str(i),'status':'addressed' if i in (0,3) else 'partly'} for i in range(4)],
        'meaning_preserved':'yes','invented_specifics':'no','invented_experiences':'no','better_than_original':'yes'}
    attempt={'complete':True}
    assert not selection(items,attempt,verdict)['quality_keep']
    verdict['items'][1]['status']='addressed'
    assert selection(items,attempt,verdict)['quality_keep']
    verdict['invented_experiences']='yes'
    assert not selection(items,attempt,verdict)['quality_keep']
    verdict['invented_experiences']='no'
    result=selection(items,{'complete':False},verdict)
    assert result['quality_keep'] and result['export_keep']


def test_judge_rejects_missing_or_duplicate_item_verdicts():
    items=[{'item_id':'a'},{'item_id':'b'}]
    attempt={'attempt':1,'items':[{'item_id':'a','status':'addressed'},{'item_id':'b','status':'not'}],
        'meaning_preserved':'yes','invented_specifics':'no','invented_experiences':'no',
        'better_than_original':'yes','reason':'판정'}
    two=deepcopy(attempt)
    two['attempt']=2
    value={'attempts':[attempt,two],'preferred':'1','reason':'비교'}
    validate_judgment(value,items)
    two['items'][1]['item_id']='a'
    with pytest.raises(ValueError,match='missing or duplicated'):
        validate_judgment(value,items)


def test_actual_policy_export_masks_all_observations(tmp_path,monkeypatch,env):
    from transformers import AutoTokenizer
    from verak.v4 import content
    from verak.v4.common import load_config
    tokenizer=AutoTokenizer.from_pretrained(str(load_config()['paths']['policy_base']),local_files_only=True)
    monkeypatch.setattr(content,'ROOT',tmp_path)
    public=env.public_messages('revision',14)
    raw=dumps({'action':'STOP','status':'done','summary':'검토 완료','issues':[]})
    call={'role':'revision','turn':1,'public_messages':public,'raw':raw,'action':{'valid':True}}
    cases=[{'source_id':'train:fixture','selection':{'export_keep':True},'attempt':{'attempt':1,'calls':[call]}}]
    export_cases(cases,tokenizer)
    row=json.loads((tmp_path/'D/export/revision.jsonl').read_text())
    prefix=tokenizer.apply_chat_template(public,tokenize=True,add_generation_prompt=True)
    assert row['input_ids'][:len(prefix)]==prefix
    assert all(x==-100 for x in row['labels'][:len(prefix)])
    assert row['labels'][len(prefix):]==row['input_ids'][len(prefix):]
    assert row['messages']==public+[{'role':'assistant','content':raw}]
