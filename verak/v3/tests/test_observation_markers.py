from verak.v3.observation.markers import update_state, at, contract
from verak.v3.observation.environment import ObservationEnv
from verak.v3.tests.test_environment import setup_env, action
from verak.v3.tests.test_observation_graph import discourse
import pytest


@pytest.mark.parametrize('setting',['current','text_only','graph'])
def test_partial_replay_moves_deletes_undo_and_handoff(setup_env,setting):
    base,ep,analysis=setup_env
    env=ObservationEnv(base.config,analysis=analysis,setting=setting,discourse=discourse())
    state=update_state(env.reset(ep))
    commands=[action('MOVE',target='S2',position='after:S3'),
        action('EDIT',target='S1',new_text=''),action('UNDO'),
        action('EDIT',target='after:S4',new_text='새 설명이다.'),
        action('STOP',summary='인계'),action('EDIT',target='S2:설명',new_text='해설'),action('STOP',summary='완료')]
    for cmd in commands:
        obs,_,_=env.step(cmd)
        state=update_state(obs,state)
        assert [(r['sid'],r['text']) for r in state]==[(env.sentence_ids[u.sid],u.text) for u in env.document.units]
    assert at(state,'S2')['previous']['sid']=='S3'


def test_judge_receives_only_final_pair():
    c={'fields':['conjunction','ending_style'],'final':{'previous':{'text':'마지막 앞 문장'},
        'sentence':{'text':'수정 뒤 대상'}},'before':'비공개','essay_id':'hidden','reward':1}
    messages,schema=contract(c)
    assert messages[1]['content']=='{"previous_sentence": "마지막 앞 문장", "affected_sentence": "수정 뒤 대상"}'
    assert set(schema['properties'])==set(c['fields'])
