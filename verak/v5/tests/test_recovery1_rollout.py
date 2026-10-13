from copy import deepcopy
import json

from verak.v4.content_env import dumps
from verak.v4.render_v43 import render
from verak.v5.recovery1_rollout import state_key, teacher_index


def sample(target):
    payload={'question':'생각을 쓰세요.','paragraphs':[{'id':'P1','sentences':[{'id':'S1','text':'글이다.'}]}],
        'assigned_tasks':[],'work_journal':[],'steps_left':14}
    return {'source_id':'train:1','messages':render(payload,'system')+[{'role':'assistant','content':dumps(target)}]}


def test_teacher_lookup_requires_same_whole_text_and_task():
    a=sample({'action':'STOP','status':'done','summary':'완료','issues':[]})
    b=deepcopy(a)
    packed=json.loads(b['messages'][1]['content'])
    packed['current_snapshot']['paragraphs'][0][1][0][1]='다른 글이다.'
    b['messages'][1]['content']=dumps(packed)
    assert state_key(a,a['messages'][:-1])!=state_key(b,b['messages'][:-1])
    assert state_key(b,b['messages'][:-1]) not in teacher_index([a])


def test_conflicting_teacher_next_actions_are_not_invented_as_a_consensus():
    a=sample({'action':'STOP','status':'done','summary':'완료','issues':[]})
    b=sample({'action':'EDIT','sentence':'S1','old':'글','new':'글쓰기'})
    assert not teacher_index([a,b])


def test_undo_requires_real_history_and_is_never_transferred_by_text_match():
    assert not teacher_index([sample({'action':'UNDO'})])


def test_equivalent_done_reports_use_history_independent_summary():
    a=sample({'action':'STOP','status':'done','summary':'고쳤습니다.','issues':[]})
    b=sample({'action':'STOP','status':'done','summary':'수정하지 않았습니다.','issues':[]})
    index=teacher_index([a,b])
    assert len(index)==1
    assert next(iter(index.values()))['summary']=='현재 글의 한국어 형식 검토를 마쳤습니다.'
