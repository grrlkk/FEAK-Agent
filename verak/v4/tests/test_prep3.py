import json
import pytest


@pytest.fixture
def env():
    from verak.src.schemas import Profile, Sentence
    from verak.v3.tests.test_environment import StubParagraphAnalysis
    from verak.v4.policy_env import V4Environment
    text='첫 주장을 쓴다. 이를 설명한다.'
    profile=Profile([Sentence('S1',1,0,9,'첫 주장을 쓴다.',[]),
        Sentence('S2',1,10,len(text),'이를 설명한다.',[])],[],'bareun','fixture')
    value=V4Environment({'source_id':'train:fixture','question':'설명하시오.','text':text,
        'profile':profile.to_dict()},StubParagraphAnalysis())
    value.delegate([{'location':['S1'],'action':'EDIT','instruction':'첫 주장을 명료화한다.'}],1)
    return value


def test_hard_step_limit_scope_notice_and_terminal_stop(env):
    before=env.document.text
    for _ in range(4):
        assert not env.step('{"action":"DELETE","sentence":"S2"}','revision')['valid']
    assert env.remaining==2 and env.document.text==before
    payload=json.loads(env.public_messages('revision')[1]['content'])
    assert 'notice' in payload and payload['steps_left']==2
    for _ in range(2):
        assert not env.step('{"action":"UNDO"}','revision')['valid']
    with pytest.raises(RuntimeError,match='ended'):
        env.step('{"action":"DELETE","sentence":"S1"}','revision')
    assert env.document.text==before
    env.delegate([{'location':['S1'],'action':'EDIT','instruction':'명료화한다.'}],2)
    stop='{"action":"STOP","status":"blocked","summary":"범위 내 해결 불가","issues":["추가 근거 필요"]}'
    assert env.step(stop,'revision')['valid']
    with pytest.raises(RuntimeError,match='ended'):
        env.step('{"action":"DELETE","sentence":"S1"}','revision')


def test_insert_success_cap_survives_undo_and_korean_has_only_form_actions(env):
    insert=json.dumps({'action':'INSERT','after':'S1','text':'구체적인 설명이다.',
        'relation':{'type':'support','target':'S1'},'source':None})
    assert env.step(insert,'revision')['valid']
    assert env.step('{"action":"UNDO"}','revision')['valid']
    assert env.step(insert,'revision')['valid']
    assert not env.step(insert,'revision')['valid']
    assert env.insert_count==2
    env.start_korean([])
    before=env.document.text
    for raw in (insert,'{"action":"DELETE","sentence":"S1"}',
                '{"action":"MOVE","sentence":"S1","to":{"after":"S2"}}'):
        assert not env.step(raw,'korean')['valid']
    assert env.document.text==before
    for new in ('','첫 문장이다. 둘째 문장이다.'):
        raw=json.dumps({'action':'EDIT','sentence':'S1','old':'첫 주장을 쓴다.','new':new})
        assert not env.step(raw,'korean')['valid']
    assert env.document.text==before
    observation=json.loads(env.public_messages('korean')[1]['content'])
    assert observation['scope']=='whole_essay' and observation['whole_essay_typo_spacing_pass']


def test_exact_frozen_prompts_used_by_environment(env):
    from verak.v4.policy_prompts import verify_frozen, assert_messages, assert_editor_request
    manifest=verify_frozen()
    assert all(v['policy_tokens']<=400 and v['system_message_tokens']<=400 for v in manifest['roles'].values())
    messages=env.public_messages('revision')
    assert_messages(messages,'revision')
    assert assert_editor_request(messages)=='revision'
    messages[0]['content']+=' '
    with pytest.raises(ValueError,match='exact frozen'):
        assert_messages(messages,'revision')
    with pytest.raises(ValueError,match='exact frozen'):
        assert_editor_request(messages)


def test_edge_filter_keeps_map_and_independent_support_example_slots():
    from verak.v4.tests.test_prep2 import map_fixture
    from verak.v4.prep3_maps import filter_edges
    row,value=map_fixture(2)
    value['sentence_relations']=[
        {'source':'S1b','target':'Q','type':'support'},
        {'source':'S1b','target':'S1a','type':'support'},
        {'source':'S1b','target':'S2a','type':'support'},
        {'source':'S1b','target':'S2b','type':'example'},
        {'source':'S1b','target':'S2b','type':'example'},
    ]
    result=filter_edges(value,row)
    assert result['raw_edges']==5 and result['kept_edges']==2
    assert result['dropped_by_reason']=={'invalid_target_or_direction':1,'outgoing_support_limit':1,'duplicate_relation':1}
    assert {e['type'] for e in result['map']['sentence_relations']}=={'support','example'}


def test_long_edge_filter_does_not_consume_slot_for_invalid_cross_edge():
    from verak.v4.tests.test_prep2 import map_fixture
    from verak.v4.prep3_maps import filter_edges
    row,value=map_fixture(7)
    value['sentence_relations']=[
        {'source':'S1b','target':'S2b','type':'support'},
        {'source':'S1b','target':'S1a','type':'support'}]
    result=filter_edges(value,row)
    assert result['dropped_by_reason']=={'long_cross_paragraph_missing_main_or_key':1}
    assert result['map']['sentence_relations']==value['sentence_relations'][1:]
    row,value=map_fixture(6)
    assert filter_edges(value,row)['kept_edges']==1


def test_rejudge_replaces_only_naturalness_and_preserves_all_other_gates():
    from copy import deepcopy
    from verak.v4.prep3_rejudge import replace_gate
    from verak.v4.prep3_content import selection
    items=[{'item_id':'x','owner':'revision'}]
    verdict={'items':[{'item_id':'x','status':'addressed'}],'meaning_preserved':'yes',
        'invented_specifics':'no','invented_experiences':'no','better_than_original':'yes',
        'repetition_introduced':'no','reads_naturally':'no'}
    saved=deepcopy(verdict)
    old,new=replace_gate(items,{'complete':False},verdict,{'edits_introduced_awkwardness':'no'})
    assert not old['quality_keep'] and new['quality_keep'] and verdict==saved
    assert {k:v for k,v in old['criteria'].items() if k!='natural_yes'}=={
        k:v for k,v in new['criteria'].items() if k!='introduced_awkwardness_no'}
    for bad in ('yes','unknown'):
        assert not replace_gate(items,{'complete':False},verdict,{'edits_introduced_awkwardness':bad})[1]['quality_keep']
    revised={k:v for k,v in verdict.items() if k!='reads_naturally'}
    revised['edits_introduced_awkwardness']='no'
    assert selection(items,{'complete':False},revised)['quality_keep']
    revised['invented_specifics']='yes'
    assert not selection(items,{'complete':True},revised)['quality_keep']


def test_d3_planner_retains_one_grounded_insert_and_records_extra_drop():
    from verak.v4.prep3_content import validate_plan
    from verak.v4.tests.test_prep2 import map_fixture
    row,_=map_fixture(2); row['source_id']='train:fixture'
    item={'rubric':'clarity','feedback_index':1,'problem':'빠진 설명 관계','location':['S1a'],
        'evidence':'중심 설명','owner':'revision','action':'INSERT','instruction':'기존 내용의 관계를 연결한다.'}
    result=validate_plan({'items':[item.copy(),item.copy()],'dropped':[],'writer_notes':[]},row)
    assert len(result['items'])==1 and result['items'][0]['item_id'].endswith('D3I1')
    assert result['mechanically_dropped'][0]['reason']=='one_INSERT_task_per_essay'


def test_new_judge_schema_requires_introduced_awkwardness():
    from verak.v4.prep3_content import judge_schema,validate_judge
    fields=judge_schema([])['properties']['attempts']['items']
    assert 'edits_introduced_awkwardness' in fields['required']
    assert 'reads_naturally' not in fields['properties']
    sample={'attempts':[{'attempt':a,'items':[],'meaning_preserved':'yes','invented_specifics':'no',
        'invented_experiences':'no','better_than_original':'yes','repetition_introduced':'no',
        'edits_introduced_awkwardness':'no','reason':'changed text is fluent'} for a in (1,2)],
        'preferred':'tie','reason':'same'}
    assert validate_judge(sample,[])==sample
    sample['attempts'][0]['reads_naturally']='yes'
    with pytest.raises(ValueError):
        validate_judge(sample,[])


def test_action_export_matches_actual_policy_prefix_and_masks_observations(env,tmp_path,monkeypatch):
    from pathlib import Path
    from verak.v4.common import load_config
    base=Path(load_config()['paths']['policy_base'])
    if not base.exists(): pytest.skip('Pinned local policy tokenizer not available')
    from transformers import AutoTokenizer
    from verak.v4 import prep3_content_report as report
    tokenizer=AutoTokenizer.from_pretrained(str(base),local_files_only=True)
    monkeypatch.setattr(report,'ROOT',tmp_path)
    public=env.public_messages('revision')
    raw='{"action":"STOP","status":"done","summary":"검토 완료","issues":[]}'
    action=env.step(raw,'revision')
    cases=[{'source_id':'fixture','items':[{'item_id':'PRIVATE_FEEDBACK_ID'}],
        'attempt':{'attempt':1,'calls':[{'role':'revision','delegation':1,'turn':1,
            'public_messages':public,'raw':raw,'action':action}]}}]
    result=report.export(cases,tokenizer)
    assert result['revision']['action_targets']==1 and result['korean']['action_targets']==0
    row=json.loads((tmp_path/'content/export/revision.jsonl').read_text())
    prefix=tokenizer.apply_chat_template(public,tokenize=True,add_generation_prompt=True)
    assert row['input_ids'][:len(prefix)]==prefix
    assert all(t==-100 for t in row['labels'][:len(prefix)])
    assert row['labels'][len(prefix):]==row['input_ids'][len(prefix):]
