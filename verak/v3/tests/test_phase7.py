"""Main action space and teacher-pilot data contracts (no live services)."""
import pytest

from verak.v3.agent.runner import system_prompt
from verak.v3.tests.test_environment import setup_env, action
from verak.v3.train.pilot import stratified_order
from verak.v3.eval.api import Phase6API
from verak.v3.common import load_config
from feak_tc.runtime.openai import CallBudgetExceeded


@pytest.mark.parametrize('role', ['global', 'korean', 'single'])
def test_main_prompts_hide_check_and_name_observations(role):
    text = system_prompt(role)
    assert 'CHECK' not in text
    assert 'Korean document profile' in text and 'marker-change notices' in text
    assert 'CHECK' in system_prompt(role, allow_check=True)


def test_disabled_check_consumes_error_step_without_scoring_or_check_termination(setup_env):
    env, ep, _ = setup_env
    env.config['env']['global']['max_checks'] = 0
    initial = env.reset(ep)
    assert 'checks_left' not in initial and '[점수]' not in initial
    assert 'Korean document profile' in initial and 'marker-change notices' in initial
    _, done, info = env.step(action('CHECK'))
    assert not done and not info['handoff']
    assert info['action']['error_code'] == 'check_disabled'
    assert env.steps['global'] == 1 and env.checks['global'] == 0
    assert not env.scorer.calls
    observation, _, info = env.step(action('STOP', summary='변경 없음'))
    assert info['handoff'] and 'Korean document profile' in observation
    assert 'marker-change notices' in observation and 'checks_left' not in observation


def test_disabled_check_is_protocol_invalid_and_errors_never_advertise_it(setup_env):
    from verak.v3.agent.runner import validity
    assert validity(action('CHECK'),allow_check=False)==(True,False)
    assert validity(action('CHECK'),allow_check=True)==(True,True)
    env,ep,_=setup_env
    env.reset(ep)
    observation,_,info=env.step(action('SEARCH'))
    assert info['action']['error_code']=='unknown_action'
    assert 'CHECK' not in observation


def test_fixed_stratified_first_100_without_replacement():
    from collections import Counter
    rows=[{'episode_id':f'{level}:{i}', 'level':level} for level,n in
          [('L1',50),('L2',45),('L3',40),('L4',35)] for i in range(n)]
    a=stratified_order(rows)
    assert a==stratified_order(list(reversed(rows)))
    assert len(set(a))==len(rows)
    assert Counter(id.split(':')[0] for id in a[:100])=={k:25 for k in ('L1','L2','L3','L4')}


def test_pilot_cost_budget_is_separate_and_cannot_exceed_15(tmp_path):
    c=load_config()
    c['paths']['phase7_pilot_output']=tmp_path/'pilot'
    c['paths']['phase6_output']=tmp_path/'phase6'
    api=Phase6API(c,10,phase='phase7_pilot')
    api.reserve('teacher_pilot','a','a',1)
    with api.db() as db:
        db.execute("UPDATE calls SET status='completed', reserved=0, confirmed=14.9")
    with pytest.raises(CallBudgetExceeded):api.reserve('teacher_pilot','b','b',.11)
    assert not c['paths']['phase6_output'].exists()
    assert api.accounting()['confirmed_usd']==14.9


class TinyTemplate:
    def apply_chat_template(self,messages,*,tokenize=True,add_generation_prompt=False):
        s='BOS'+''.join('['+m['role']+']'+m['content']+'[END]' for m in messages)
        if add_generation_prompt:s+='[assistant]'
        return [ord(c) for c in s] if tokenize else s


def test_three_turn_transcript_masks_every_nonassistant_token():
    from verak.v3.train.formatting import encode_labels
    tok=TinyTemplate()
    messages=[{'role':'system','content':'rules'},{'role':'user','content':'essay'},
        {'role':'assistant','content':'A1 user system'},{'role':'user','content':'notice'},
        {'role':'assistant','content':'A2'},{'role':'user','content':'next'},
        {'role':'assistant','content':'STOP'}]
    result=encode_labels(tok,messages)
    supervised=''.join(chr(v) for v in result['labels'] if v!=-100)
    assert supervised=='A1 user system[END]A2[END]STOP[END]'
    assert all(label==-100 or label==token for token,label in zip(result['input_ids'],result['labels']))
    current=encode_labels(tok,messages,last_assistant_only=True)
    assert ''.join(chr(v) for v in current['labels'] if v!=-100)=='STOP[END]'


def test_inference_compaction_and_journal_are_shared_with_sft():
    from types import SimpleNamespace
    from verak.v3.agent.runner import fit_history
    from verak.v3.train.formatting import formatted_turns
    tok=TinyTemplate()
    history=[{'role':'system','content':system_prompt('global')},{'role':'user','content':'initial'}]
    logs=[]
    for i in range(12):
        history.extend([{'role':'assistant','content':'action'+str(i)}, {'role':'user','content':'notice '+('x'*200)}])
        logs.append({'action':'EDIT','args':{'target':f'S{i}'},'valid':True})
    budget=len(system_prompt('global'))+4400
    policy=SimpleNamespace(name='policy',context_limit=budget)
    expected,compacted=fit_history(history,policy,tok,logs)
    assert compacted
    row={'corpus_episode_id':'toy','actions_by_role':{'global':logs},'calls':[
        {'role':'global','turn':'13:0','messages':history,'raw':'STOP'}]}
    sample=next(formatted_turns(row,'global',tok,budget))
    assert sample['messages'][:-1]==expected
    assert sample['history_compacted'] and sample['work_journal_present']
    assert all(x==-100 for x in sample['labels'][:sample['prompt_tokens']])


def test_role_selection_uses_separate_populations_and_explicit_no_global_stop_rule():
    from verak.v3.train.formatting import select_roles
    episodes=[];corpus={}
    for i in range(10):
        id=str(i)
        episodes.append({'corpus_episode_id':id,'completed':True,'reward':{
            'global':{'R':i/10,'R_over':0 if i!=8 else .1},'korean':{'R':i/10}},
            'steps':{'global':1 if i!=9 else 3},'termination':{'global':'STOP'}})
        corpus[id]={'records':[{'level':'GLOBAL' if i<5 else 'WORD'}]}
    selected=select_roles(episodes,corpus)
    assert selected['thresholds']==pytest.approx({'global':.24,'korean':.54})
    assert selected['kept_ids']['global']==['3','4','5','6','7']
    assert selected['kept_ids']['korean']==['6','7','8','9']
