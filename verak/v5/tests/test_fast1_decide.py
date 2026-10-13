import math

import pytest

from verak.v4.content_env import dumps
from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis, row
from verak.v4.render_v43 import assert_rendered
from verak.v5.fast1_decide_env import DecideEnvironment, lenient_span, notice_sids


@pytest.fixture(autouse=True)
def no_contract_io(monkeypatch):
    from verak.v4 import policy_prompts_v43
    monkeypatch.setattr(policy_prompts_v43,'verify_frozen',lambda **kw:{})


def environment():
    return DecideEnvironment(row('첫 문장이다. 둘째 문장이다.'),FakeBoundaryAnalysis())


def edit(env,old,new,sid='S1',role='korean'):
    return env.step(dumps({'action':'EDIT','sentence':sid,'old':old,'new':new}),role)


def test_lenient_unique_and_ambiguous_matches():
    text='우리의 문화는 매우 중요하다.'
    span,diag=lenient_span(text,'우리의 문화는 매오 중요하다.')
    assert span==(0,len(text)) and diag['mode']=='lenient' and diag['similarity']>=.85
    span,diag=lenient_span('아주 좋은 문장과 아주 좋은 문장','아주 조은 문장')
    assert span is None and not diag['matched']


def test_noop_is_done_and_is_not_an_accepted_edit():
    env=environment();env.start_korean(scope=['S1'])
    result=edit(env,'첫','첫')
    assert result['valid'] and result['noop_EDIT'] and not result['text_changed']
    assert env.closed and env.steps_used==1 and env.controller_stop is None
    with pytest.raises(RuntimeError):edit(env,'첫','다른')


def test_noop_outside_scope_stays_rejected():
    env=environment();env.start_korean(scope=['S1'])
    result=edit(env,'둘째','둘째',sid='S2')
    assert not result['valid'] and not env.closed


def test_notice_scope_and_final_scope_are_explicit():
    env=environment();sid=env.document.units[-1].sid
    notices=[{'sid':sid,'type':'style_change'}]
    env.start_korean(scope=notice_sids(notices),notices=notices)
    observation=assert_rendered(env.public_messages('korean'))
    assert observation['allowed_scope']==[sid]
    assert observation['notices']==notices
    assert not observation['whole_essay_typo_spacing_pass']
    assert not edit(env,'첫','새')['valid']
    assert edit(env,'둘째','두 번째',sid=sid)['valid']


def test_safety_rejection_restores_text_notices_undo_and_lineage():
    env=environment();env.start_korean(scope=['S1'])
    before=env.document.text
    class Reject:
        def assess(self,*args):return {'reasons':['immediate_repeated_word'],'features':[]}
    env.safety=Reject()
    record=edit(env,'첫','첫 첫')
    assert not record['valid'] and record['rejection_reasons']==['immediate_repeated_word']
    assert env.document.text==before and not env.undo and not env.notices
    assert not record['text_changed'] and record['before_hash']==record['after_hash']
    assert '첫 첫' in record['candidate_text']


def test_unavailable_safety_is_a_rejection_not_a_leaked_candidate():
    env=environment();env.start_korean(scope=['S1']);before=env.document.text
    class Fails:
        def assess(self,*args):raise RuntimeError('offline')
    env.safety=Fails()
    record=edit(env,'첫','새')
    assert not record['valid'] and env.document.text==before
    assert record['rejection_reasons']==['safety_measurement_unavailable']


def test_revision_and_korean_each_cap_at_four_actions():
    env=environment();env.delegate([{'location':['S1'],'action':'EDIT','instruction':'검토','needs_search':'no'}],1)
    assert env.remaining==4
    env.start_korean(scope=['S1']);assert env.remaining==4
    old='첫'
    for new in ('다른','새로운','최종','검토'):
        assert edit(env,old,new)['valid'];old=new
    assert env.remaining==0
    with pytest.raises(RuntimeError):edit(env,old,'마지막')


def test_repeated_unsafe_edit_stops_after_three_rejections():
    env=environment();env.start_korean(scope=['S1'])
    class Reject:
        def assess(self,*args):return {'reasons':['new_proper_noun'],'features':[]}
    env.safety=Reject()
    first=edit(env,'첫','새')
    second=edit(env,'첫','새')
    third=edit(env,'첫','새')
    assert not first['valid'] and second['duplicate_rejection']
    assert third['controller_stop']['status']=='blocked' and env.closed
    assert env.document.text.startswith('첫')


def test_search_is_rejected_without_retrieval():
    env=environment();env.delegate([{'location':['S1'],'action':'EDIT','instruction':'검토','needs_search':'no'}],1)
    before=env.document.text
    record=env.step(dumps({'action':'SEARCH','item_id':'D1T1','query':'사실'}),'revision')
    assert not record['valid'] and record['error']=='search_disabled'
    assert env.document.text==before and not env.search.history


def test_number_gate_rejects_additions_but_not_retained_saved_numbers():
    from verak.v5.fast1_decide_env import SafetyGate
    env=environment()
    gate=SafetyGate(env.document,lambda text:{'mean_nll':0.})
    before=env.document.clone();after=before.clone()
    before.units[0].text='37년의 첫 문장이다.'
    after.units[0].text='37년의 다른 문장이다.'
    result=gate.assess(before,after,{'S1'})
    assert 'new_number' not in result['reasons']
    after.units[0].text='38년의 다른 문장이다.'
    result=gate.assess(before,after,{'S1'})
    assert 'new_number' in result['reasons']


class Tokens:
    def apply_chat_template(self,*args,**kwargs):return [1]
    def encode(self,*args,**kwargs):return [1]


def test_controller_no_edit_stops_and_empty_final_pass_is_skipped(tmp_path,monkeypatch):
    from verak.v5 import fast1_decide as runner
    calls=[]
    class Scripted:
        first=True
        def __init__(self,*args):self.number=0
        def generate(self,messages,role):
            calls.append(role)
            if role=='orchestrator':
                self.number+=1
                value=[{'action':'SCORE'}, {'action':'MAP','map':{
                    'genre':'argumentative','paragraph_roles':[{'paragraph':'P1','role':'body','key_sentence':'S1'}],
                    'sentence_relations':[{'source':'S1','target':'Q','type':'main'}],
                    'paragraph_relations':[],'off_topic':[]}},
                    {'action':'DELEGATE','to':'revision','items':[{
                        'location':['S1'],'action':'EDIT','instruction':'첫 문장 확인','needs_search':'no'}]}][self.number-1]
            else:value={'action':'EDIT','sentence':'S1','old':'첫','new':'첫'}
            return {'raw':dumps(value),'finish_reason':'stop','usage':{'prompt_tokens':1}}
        def close(self):pass
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'Hybrid',Scripted)
    monkeypatch.setattr(runner,'ppl',lambda text:{'mean_nll':0.})
    monkeypatch.setattr(runner,'score_cached',lambda *args:{'expected':[5]*8,'integers':[5]*8})
    source=row('첫 문장이다. 둘째 문장이다.');source['genre']='argumentative'
    runner.write_json(tmp_path/'source.json',source)
    result=runner.run_case({'case_id':'fixture','cohort':'smoke20','row':{'path':str(tmp_path/'source.json')}},
                          1,None,Tokens(),FakeBoundaryAnalysis(),{'max_log_ppl_rise':1.})
    assert result['status']=='completed',result.get('traceback')
    assert result['stop_reason']=='delegation_no_accepted_edit'
    assert result['protocol_complete'] and result['all_items_done']
    assert result['final_korean_skip']=='no_sentences_changed_in_run'
    assert calls==['orchestrator','orchestrator','orchestrator','revision']


def _link_candidate_data(candidate='후보 문장.'):
    row={'id':'K1','sentence_id':'S1','revision_attribution_available':True,
         'revision_start':'원문 문장.','before':'수정 전 문장.',
         'candidate':candidate}
    return {'rows':[{**row,'judge_id':'K1'}],'unique_candidates':[row]}


def test_link_evaluation_unknown_delivery_is_not_retried(tmp_path,monkeypatch):
    from verak.v5 import fast1_decide as runner
    class Unknown:
        def __init__(self,*args,**kwargs):self.calls=0
        def request(self,*args,**kwargs):
            self.calls+=1
            raise TimeoutError('response delivery unknown')
        def close(self):pass
    api=Unknown()
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'link_candidates',lambda:_link_candidate_data())
    monkeypatch.setattr(runner,'PaidAPI',lambda *args,**kwargs:api)
    result=runner.evaluate_links()
    saved=runner.read_json(tmp_path/'korean_judgments'/'000.json')
    assert api.calls==1
    assert saved['status']=='request_error_no_retry'
    assert saved['delivery_unknown']
    assert result['valid_judgments']==0


def test_link_evaluation_rejects_stale_cache_input(tmp_path,monkeypatch):
    from verak.v5 import fast1_decide as runner
    class NoCalls:
        def __init__(self,*args,**kwargs):self.calls=0
        def request(self,*args,**kwargs):
            self.calls+=1
            raise TimeoutError('response delivery unknown')
        def close(self):pass
    api=NoCalls()
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    current={'data':_link_candidate_data()}
    monkeypatch.setattr(runner,'link_candidates',lambda:current['data'])
    monkeypatch.setattr(runner,'PaidAPI',lambda *args,**kwargs:api)
    runner.evaluate_links()
    current['data']=_link_candidate_data('다른 후보 문장.')
    with pytest.raises(ValueError,match='Stale Korean-link cache'):
        runner.evaluate_links()
    assert api.calls==1
