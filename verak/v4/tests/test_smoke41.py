"""v4.1 version isolation, unchanged Dv3 mechanics and honest smoke denominators."""
import json

import pytest

from verak.v4 import policy_prompts as original_prompts
from verak.v4 import policy_prompts_v41 as prompts
from verak.v4 import runtime_v41 as runtime
from verak.v4 import smoke41


@pytest.fixture
def pair():
    from verak.src.schemas import Profile,Sentence
    from verak.v3.tests.test_environment import StubParagraphAnalysis
    from verak.v4.policy_env import V4Environment
    text='첫 주장을 쓴다. 이를 설명한다.'
    profile=Profile([Sentence('S1',1,0,9,'첫 주장을 쓴다.',[]),
        Sentence('S2',1,10,len(text),'이를 설명한다.',[])],[],'bareun','fixture')
    row={'source_id':'train:fixture','question':'설명하시오.','text':text,'profile':profile.to_dict()}
    values=[cls(row,StubParagraphAnalysis()) for cls in (V4Environment,runtime.V41Environment)]
    for value in values:
        value.delegate([{'location':['S1'],'action':'EDIT','instruction':'첫 주장을 명료화한다.'}],1)
    return values


@pytest.fixture(scope='module')
def tokenizer():
    from verak.v4.common import load_config,constrain_cpu
    constrain_cpu()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(load_config()['paths']['policy_base']),local_files_only=True)


def test_new_frozen_lines_and_pinned_token_counts_without_changing_v4(tokenizer):
    old=original_prompts.FROZEN.read_bytes()
    manifest=prompts.verify_frozen()
    for role,text in prompts.PROMPTS.items():
        assert text.count(prompts.MASK_RULE)==1
        assert text.count('JSON 객체 하나만 출력한다')==1
        assert manifest['roles'][role]['policy_tokens']==len(tokenizer.encode(text,add_special_tokens=False))<=400
        assert manifest['roles'][role]['system_message_tokens']==len(tokenizer.apply_chat_template(
            [{'role':'system','content':text}],tokenize=True))<=400
    assert prompts.KOREAN.count(prompts.KOREAN_EDIT_RULE)==1
    assert prompts.KOREAN_EDIT_RULE not in original_prompts.KOREAN
    assert original_prompts.FROZEN.read_bytes()==old
    assert manifest['version']!=original_prompts.verify_frozen()['version']


def test_environment_body_limits_actions_and_scope_are_identical(pair):
    old,new=pair
    assert new.public_messages('revision')[1:]==old.public_messages('revision')[1:]
    assert new.step.__func__ is old.step.__func__
    for raw in ('not JSON','{"action":"DELETE","sentence":"S2"}',
                '{"action":"EDIT","sentence":"S1","old":"첫","new":"주요"}'):
        assert new.step(raw,'revision')==old.step(raw,'revision')
    assert new.document.text==old.document.text and new.remaining==old.remaining==3
    for value in pair: value.start_korean([])
    assert new.public_messages('korean')[1:]==old.public_messages('korean')[1:]
    assert new.remaining==old.remaining==14
    raw=json.dumps({'action':'EDIT','sentence':'S1','old':'주요 주장을 쓴다.','new':'한 문장이다. 또 하나다.'})
    assert new.step(raw,'korean')==old.step(raw,'korean')
    assert not new.actions['korean'][-1]['valid']


def test_teacher_export_and_inference_reject_cross_version_messages_before_dispatch(pair):
    old,new=pair
    old_messages=old.public_messages('revision');new_messages=new.public_messages('revision')
    assert prompts.assert_editor_request(new_messages)=='revision'
    for messages,checker in ((old_messages,prompts.assert_messages),(new_messages,original_prompts.assert_messages)):
        with pytest.raises(ValueError,match='exact frozen'):
            checker(messages,'revision')
    api=object.__new__(runtime.VersionedAPI)  # rejection must precede any DB/client access
    with pytest.raises(ValueError,match='exact frozen'):
        api.request(old_messages,stage='prep3_content_teacher',item_id='fixture')
    with pytest.raises(ValueError,match='explicit v4.1'):
        prompts.assert_messages(new_messages,'revision',version=original_prompts.VERSION)


def test_private_binding_keeps_original_code_and_module_globals(tmp_path):
    from verak.v4 import prep3_content as frozen
    before=frozen.ROOT
    bound=runtime.bind(frozen.make_plan,ROOT=tmp_path)
    assert bound.__code__ is frozen.make_plan.__code__
    assert bound.__globals__['ROOT']==tmp_path and frozen.ROOT==before
    with pytest.raises(ValueError,match='Unknown'):
        runtime.bind(frozen.make_plan,does_not_exist=True)


def test_cached_teacher_cannot_silently_use_old_version(tmp_path):
    from verak.v4.common import write_json
    write_json(tmp_path/'content/attempts/train_fixture_a1.json',
               {'prompt_version':original_prompts.VERSION,'calls':[]})
    with pytest.raises(ValueError,match='wrong prompt version'):
        runtime.teach({'source_id':'train:fixture'},None,1,None,None,output_root=tmp_path)


def test_rejected_and_malformed_actions_count_and_incomplete_requests_are_visible():
    def call(valid,name,role='revision',error=None):
        return {'role':role,'action':{'valid':valid,'action':name,'error':error}}
    calls=[call(True,'STOP') for _ in range(18)]+[
        call(False,'INVALID',error='JSONDecodeError'),call(False,'EDIT','korean','boundary changed')]
    account={'by_stage':{'v41_prep3_content_teacher':{'logical_requests':21}}}
    metrics=smoke41.action_metrics([{'calls':calls}],account)
    assert metrics['returned_action_calls']==20 and metrics['invalid_actions']==2
    assert metrics['primary_valid_action_rate']==.9 and metrics['primary_gate_pass']
    assert metrics['scheduled_without_returned_action']==1
    assert metrics['valid_per_all_scheduled_editor_calls']==18/21
    calls.append(call(False,'STOP',error='missing summary'))
    assert not smoke41.action_metrics([{'calls':calls}],account)['primary_gate_pass']
    empty=smoke41.action_metrics([],{"by_stage":{}})
    assert empty['primary_valid_action_rate'] is None and not empty['primary_gate_pass']


def test_export_preserves_new_inference_prefix_and_masks_nonaction_tokens(pair,tmp_path,tokenizer):
    env=pair[1];public=env.public_messages('revision')
    raw='{"action":"STOP","status":"done","summary":"검토 완료","issues":[]}'
    action=env.step(raw,'revision')
    cases=[{'source_id':'fixture','items':[{'item_id':'PRIVATE_FEEDBACK_ID'}],
        'attempt':{'attempt':1,'calls':[{'role':'revision','delegation':1,'turn':1,
            'public_messages':public,'raw':raw,'action':action}]}}]
    result=runtime.export(cases,tokenizer,output_root=tmp_path)
    assert result['revision']['action_targets']==1 and result['korean']['action_targets']==0
    row=json.loads((tmp_path/'content/export/revision.jsonl').read_text())
    prefix=tokenizer.apply_chat_template(public,tokenize=True,add_generation_prompt=True)
    assert row['prompt_version']==prompts.VERSION
    assert row['input_ids'][:len(prefix)]==prefix
    assert all(x==-100 for x in row['labels'][:len(prefix)])
    assert row['labels'][len(prefix):]==row['input_ids'][len(prefix):]
    assert row['messages']==public+[{'role':'assistant','content':raw}]
