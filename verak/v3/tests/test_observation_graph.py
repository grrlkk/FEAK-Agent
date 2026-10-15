from copy import deepcopy
from dataclasses import replace
import json

import pytest

from verak.src.schemas import Token
from verak.v3.observation.graph import validate, public_input, build, render, changes, clauses
from verak.v3.observation.environment import ObservationEnv, prompt
from verak.v3.agent.runner import split_handoff, system_prompt
from verak.v3.tests.test_environment import setup_env, action, StubBackend


def discourse():
    return {'sentence_edges':[{'source':'S2','target':'S1','label':'supports'},
        {'source':'S4','target':'S3','label':'concludes'}],
        'paragraph_edges':[{'source':'P2','target':'P1','label':'continues'}],
        'roles':[{'id':f'S{i}','role':'claim' if i%2 else 'support'} for i in range(1,5)]}


def test_graph_schema_exact_ids_and_outgoing_limit():
    d=discourse()
    assert validate(d,['S1','S2','S3','S4'],['P1','P2'])==d
    for extra in ({'source':'S2','target':'S3','label':'example_of'},
                  {'source':'S99','target':'S1','label':'supports'},
                  {'source':'S2','target':'S2','label':'elaborates'},d['sentence_edges'][0]):
        bad=deepcopy(d);bad['sentence_edges'].append(extra)
        with pytest.raises(ValueError):
            validate(bad,['S1','S2','S3','S4'],['P1','P2'])
    bad=deepcopy(d);bad['roles'].append(bad['roles'][0])
    with pytest.raises(ValueError):
        validate(bad,['S1','S2','S3','S4'],['P1','P2'])


def test_no_source_order_leak_in_extraction(setup_env):
    env,ep,_=setup_env
    rows,sm,pm=public_input(ep['document'])
    assert sm=={'S8':'S1','S4':'S2','S5':'S3','S1':'S4'}
    assert pm=={'P3':'P1','P1':'P2'}
    assert rows[0]['sentences'][0]['id']=='S1'


def graph_env(setup_env):
    env,ep,analysis=setup_env
    current=ObservationEnv(env.config,analysis=analysis,setting='graph',discourse=discourse())
    current.reset(ep)
    return current,ep


def test_move_frozen_edges_changed_membership_and_predecessor(setup_env):
    env,_=graph_env(setup_env)
    before=env.view_history[-1]['graph']
    obs,_,_=env.step(action('MOVE',target='S2',position='after:S3'))
    after=env.view_history[-1]['graph']
    assert before['sentence_edges']==after['sentence_edges']
    s2=next(m for m in after['sentences'] if m['id']=='S2')
    assert s2['paragraph']=='P2' and s2['predecessor']=='S3'
    assert '⚠ in P2, supports target S1 in P1' in obs
    assert 'graph-change notices' in obs and 'prev: S1 → S3' in obs
    assert 'marker-change notices' not in obs


def test_delete_dangling_insert_new_and_undo_restores_edges(setup_env):
    env,_=graph_env(setup_env)
    env.step(action('EDIT',target='S1',new_text=''))
    g=env.view_history[-1]['graph']
    assert {'source':'S2','target':'S1','label':'supports','kind':'sentence_edges'} in g['dangling']
    assert all(e['target']!='S1' for e in g['sentence_edges'])
    env.step(action('UNDO'))
    assert not env.view_history[-1]['graph']['dangling']
    assert env.view_history[-1]['graph']['sentence_edges']==discourse()['sentence_edges']
    env.step(action('EDIT',target='after:S1',new_text='새 설명이다.'))
    g=env.view_history[-1]['graph']
    assert next(s for s in g['sentences'] if s['id']=='N1')['new']
    assert all('N1' not in (e['source'],e['target']) for e in g['sentence_edges'])


def test_empty_paragraph_remains_a_paragraph_node(setup_env):
    env,_=graph_env(setup_env)
    env.step(action('EDIT',target='S1',new_text=''))
    env.step(action('EDIT',target='S2',new_text=''))
    g=env.view_history[-1]['graph']
    assert 'P1' in g['paragraphs']
    assert g['paragraph_edges']==discourse()['paragraph_edges']
    assert not any(e['kind']=='paragraph_edges' for e in g['dangling'])


def test_in_sentence_edit_rebuilds_clause_markers_but_keeps_discourse(setup_env, monkeypatch):
    env,_=graph_env(setup_env)
    analysis=setup_env[2]
    original_pieces=analysis.pieces
    def tagged_pieces(text):
        rows=original_pieces(text)
        for sentence,tokens,_ in rows:
            for form in ('면','니까'):
                if form in sentence:
                    start=sentence.index(form)
                    tokens.append(Token(form,'EC',start,start+len(form)))
            tokens.sort(key=lambda t:t.start)
        return rows
    monkeypatch.setattr(analysis,'pieces',tagged_pieces)
    env.step(action('STOP',summary='인계'))
    _,_,info=env.step(action('EDIT',target='S2',new_text='설명하면 좋다.'))
    assert info['action']['valid']
    before=env.view_history[-1]['graph']
    assert next(e for e in before['clause_edges'] if e['source']=='S2.C1')['class']=='CONDITION'
    obs,_,info=env.step(action('EDIT',target='S2:면',new_text='니까'))
    assert info['action']['valid']
    after=env.view_history[-1]['graph']
    assert next(e for e in after['clause_edges'] if e['source']=='S2.C1')['class']=='CAUSE'
    assert after['sentence_edges']==before['sentence_edges']==discourse()['sentence_edges']
    assert 'S2 clause edges:' in obs


def test_clauses_keep_absolute_offsets_and_embedded_boundaries(setup_env):
    env,ep,_=setup_env;env.reset(ep)
    ann=env.public_structure().annotations[1]
    ann.text='비가 오면 읽는 일을 한다고 말했다.'
    ann.tokens=[Token('면','EC',ann.start+4,ann.start+5),Token('는','ETM',ann.start+7,ann.start+8),
                Token('기','ETN',ann.start+10,ann.start+11),Token('고','JKQ',ann.start+14,ann.start+15)]
    ann.connectives=[{'span':[ann.start+4,ann.start+5],'form':'면','coarse_class':'CONDITION','eligible':True,'classification':'UNAMBIGUOUS'}]
    nodes,edges=clauses(ann)
    assert len(nodes)==5 and ''.join(n['text'] for n in nodes)==ann.text
    assert edges[0]['source']=='S2.C1' and edges[0]['target']=='S2.C2'
    assert edges[0]['class']=='CONDITION'
    assert [n['boundary'][0]['tag'] for n in nodes[:-1]]==['EC','ETM','ETN','JKQ']


@pytest.mark.parametrize('setting',['text_only','graph'])
def test_handoff_and_partial_view_without_profile(setup_env,setting):
    original,ep,analysis=setup_env
    env=ObservationEnv(original.config,analysis=analysis,setting=setting,discourse=discourse())
    first=env.reset(ep)
    assert '[전체 갱신]' in first and 'Korean document profile' not in first
    obs,_,_=env.step(action('MOVE',target='P2',position='before:P1'))
    assert '[부분 갱신:' in obs
    obs,_,_=env.step(action('STOP',summary='인계'))
    handoff,rest=split_handoff(obs)
    assert handoff and '[전체 갱신]' in rest
    assert 'Korean document profile' not in obs and 'marker-change notices' not in obs
    data=json.loads(handoff.split('\n',1)[1])
    if setting=='text_only':
        assert set(data)=={'actions'} and 'DEP' not in obs
    else:
        assert data['graph_change_notices']
    for _ in range(5):
        # Invalid actions still count, but avoid three consecutive errors.
        if env.done: break
        env.step(action('EDIT',target='S1:첫',new_text='첫'))


def test_current_condition_matches_existing_observation_and_prompts(setup_env):
    original,ep,analysis=setup_env
    new=ObservationEnv(original.config,analysis=analysis,setting='current')
    assert original.reset(ep)==new.reset(ep)
    cmd=action('MOVE',target='P2',position='before:P1')
    assert original.step(cmd)[0]==new.step(cmd)[0]
    assert prompt('global','current')==system_prompt('global')
    assert prompt('korean','current')==system_prompt('korean')
    for role in ('global','korean'):
        assert 'Korean document profile' not in prompt(role,'text_only')
        assert 'marker-change notices' not in prompt(role,'graph')


def test_optional_prompt_factory_uses_shared_runner(setup_env):
    from verak.v3.agent.runner import run_episode
    original,ep,analysis=setup_env
    env=ObservationEnv(original.config,analysis=analysis,setting='text_only')
    result=run_episode(env,ep,StubBackend([action('STOP',summary='인계'),action('STOP',summary='완료')]),
                       prompt_factory=lambda role:prompt(role,'text_only'))
    assert result['completed']
    assert result['messages_by_role']['korean'][0]['content']==prompt('korean','text_only')


def test_shared_budget_across_graph_teacher_and_judge(tmp_path):
    from verak.v3.observation.experiment import config_for
    from verak.v3.eval.api import Phase6API
    from feak_tc.runtime.openai import CallBudgetExceeded
    cfg=config_for();cfg['paths']['observation_test_output']=tmp_path
    luna=Phase6API(cfg,20,phase='observation_test')
    sol_cfg=deepcopy(cfg);sol_cfg['observation_test']['model']='gpt-6.1-sol'
    sol=Phase6API(sol_cfg,20,phase='observation_test')
    luna.reserve('graph_extract','one','one',7.9)
    with pytest.raises(CallBudgetExceeded):
        sol.reserve('marker_fit','two','two',.2)
    assert sol.accounting()['reserved_usd']==7.9


def test_degree_invalid_raw_graph_is_only_available_for_quality(tmp_path):
    from verak.v3.observation.experiment import config_for, graph_path, quality_graph
    from verak.v3.common import write_json
    import sqlite3
    cfg=config_for();cfg['paths']['observation_test_output']=tmp_path
    raw=discourse();raw['sentence_edges'].append({'source':'S2','target':'S3','label':'example_of'})
    original={'status':'error','error':{'message':'More than one outgoing support/example/conclusion'}}
    write_json(graph_path(cfg,'test'),original)
    api=tmp_path/'api';api.mkdir()
    write_json(api/'one.json',{'raw':json.dumps(raw),'phase_call':1})
    with sqlite3.connect(api/'ledger.sqlite') as db:
        db.execute('CREATE TABLE calls (id INTEGER,stage TEXT,item_id TEXT,status TEXT,path TEXT)')
        db.execute('INSERT INTO calls VALUES (1,?,?,?,?)',('graph_extract','test:run1','completed',str(api/'one.json')))
    measured=quality_graph(cfg,'test')
    assert measured['status']=='completed' and not measured['policy_graph_valid']
    assert measured['discourse']==raw
    assert json.loads(graph_path(cfg,'test').read_text())==original
