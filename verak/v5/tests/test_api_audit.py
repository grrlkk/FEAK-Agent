from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from verak.v4.content_env import dumps
from verak.v4.scale2_projection import save_document
from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis, row
from verak.v5 import api, reviser
from verak.v5.pipeline import ScopedEnvironment
from verak.v5.judge import payload, validate
from verak.v5.measurement import problem_types, a1_planning, final_korean_changes, task_key
from verak.v5.report import resolution_delta
from verak.v5.common import write_json, ref


@pytest.fixture(autouse=True)
def frozen(monkeypatch):
    from verak.v4 import policy_prompts_v43
    monkeypatch.setattr(policy_prompts_v43,'verify_frozen',lambda **kw:{})


def setup_api(tmp_path,monkeypatch,*,cap=1.):
    monkeypatch.setattr(api,'ROOT',tmp_path)
    monkeypatch.setattr(api,'credentials',lambda:{'OPENAI_API_KEY':'fake-key-for-unit-test','ANTHROPIC_API_KEY':'fake-key-for-unit-test'})
    spec={'model':'unit-model','input_per_million':1.,'output_per_million':2.,'cap_usd':cap,'reasoning_effort':'none','temperature':0}
    write_json(tmp_path/'api_contract.json',{'user_approved':True,'total_cap_usd':cap,'luna':spec,'claude':spec})
    return api.PaidAPI('luna')


def test_paid_cache_replay_charges_once_and_records_actual_model(tmp_path,monkeypatch):
    client=setup_api(tmp_path,monkeypatch);sent=[]
    def serve(request):
        sent.append(request)
        return httpx.Response(200,json={'model':'unit-model','id':'one','status':'completed','usage':{'input_tokens':100,'output_tokens':20},
            'output':[{'content':[{'type':'output_text','text':'{}'}]}]})
    client.client.close();client.client=httpx.Client(transport=httpx.MockTransport(serve))
    try:
        one=client.request([{'role':'user','content':'x'}],item_id='one',max_output=10)
        two=client.request([{'role':'user','content':'x'}],item_id='one',max_output=10)
        assert one==two and len(sent)==1
        assert one['confirmed_usd']==pytest.approx(.00014) and one['response_model']=='unit-model'
    finally:client.close()


def test_budget_prevents_dispatch_and_unknown_outcome_cannot_retry(tmp_path,monkeypatch):
    client=setup_api(tmp_path,monkeypatch,cap=.00001)
    try:
        with pytest.raises(api.BudgetStop):client.request([{'role':'user','content':'x'}],item_id='cap')
        with client.db() as db:assert db.execute('SELECT COUNT(*) FROM calls').fetchone()[0]==0
    finally:client.close()
    client=setup_api(tmp_path,monkeypatch);sent=[]
    def lost(request):sent.append(1);raise httpx.ReadTimeout('dummy failure')
    client.client.close();client.client=httpx.Client(transport=httpx.MockTransport(lost))
    try:
        with pytest.raises(RuntimeError,match='terminal outcome'):
            client.request([{'role':'user','content':'x'}],item_id='lost')
        with pytest.raises(RuntimeError,match='Prior terminal'):
            client.request([{'role':'user','content':'x'}],item_id='lost')
        assert len(sent)==1
        with client.db() as db:
            reserved,confirmed=db.execute('SELECT reserved,confirmed FROM calls').fetchone()
        assert reserved>0 and confirmed==0
    finally:client.close()


@pytest.mark.parametrize('text_format', [None, {'type':'json_object'}])
def test_luna_json_mode_is_opt_in_and_preserves_input(tmp_path,monkeypatch,text_format):
    client=setup_api(tmp_path,monkeypatch);sent=[]
    if text_format is not None:client.spec['text_format']=text_format
    def serve(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200,json={'model':'unit-model','id':'one','status':'completed',
            'usage':{'input_tokens':100,'output_tokens':20},
            'output':[{'content':[{'type':'output_text','text':'{"action":"SCORE"}'}]}]})
    client.client.close();client.client=httpx.Client(transport=httpx.MockTransport(serve))
    messages=[{'role':'system','content':'One action JSON.'},{'role':'user','content':'unchanged'}]
    try:
        client.request(messages,item_id='format',max_output=4096)
        assert sent[0]['input']==messages
        assert sent[0]['max_output_tokens']==4096
        if text_format is None:assert 'text' not in sent[0]
        else:assert sent[0]['text']=={'format':text_format}
    finally:client.close()


@pytest.mark.parametrize('finish,choices', [('stop',1),('length',1),('stop',2)])
def test_single_chat_response_usage_and_terminal_failures(tmp_path,monkeypatch,finish,choices):
    client=setup_api(tmp_path,monkeypatch);sent=[]
    client.spec.update(endpoint='chat_completions',text_format={'type':'json_object'})
    def serve(request):
        sent.append(request)
        return httpx.Response(200,json={'model':'unit-model','id':'one',
            'usage':{'prompt_tokens':100,'completion_tokens':20,'prompt_tokens_details':{'cached_tokens':10}},
            'choices':[{'message':{'role':'assistant','content':'{"action":"SCORE"}'},
                        'finish_reason':finish}]*choices})
    client.client.close();client.client=httpx.Client(transport=httpx.MockTransport(serve))
    messages=[{'role':'system','content':'One action JSON.'},{'role':'user','content':'same'}]
    try:
        if finish=='stop' and choices==1:
            result=client.request(messages,item_id='chat')
            assert result['raw']=='{"action":"SCORE"}'
            assert result['usage']['input_tokens']==100 and result['usage']['output_tokens']==20
            assert result['provider_usage']['prompt_tokens']==100
            assert result['confirmed_usd']==pytest.approx(.00014)
        else:
            with pytest.raises(RuntimeError,match='terminal outcome'):client.request(messages,item_id='chat')
            with pytest.raises(RuntimeError,match='Prior terminal'):client.request(messages,item_id='chat')
        assert len(sent)==1 and sent[0].url.path=='/v1/chat/completions'
        body=json.loads(sent[0].content)
        assert body['messages']==messages and body['n']==1
        assert body['response_format']=={'type':'json_object'}
    finally:client.close()


class Reply:
    def __init__(self,value):self.value=value;self.messages=[]
    def request(self,messages,**kw):self.messages.append(messages);return {'raw':dumps(self.value)}


def test_edit_list_uses_editor_and_selfcheck_has_no_notice_or_graph(tmp_path,monkeypatch):
    monkeypatch.setattr(reviser,'ROOT',tmp_path)
    source=row('첫 문장이다. 둘째 문장이다.')
    call=Reply({'actions':[{'action':'EDIT','sentence':'S1','old':'첫','new':'첫째'}],'summary':'수정'})
    first=reviser.edit_list(source,call,FakeBoundaryAnalysis(),tmp_path/'revision',scores={'integers':[1]*8})
    assert first['no_protocol_error'] and first['actions'][0]['action']['valid']
    call2=Reply({'actions':[],'summary':'고칠 연결 없음'})
    second=reviser.edit_list(source,call2,FakeBoundaryAnalysis(),tmp_path/'check',initial_document=first['final'],self_check=True)
    observed=json.loads(call2.messages[0][1]['content'])
    assert set(observed)=={'question','paragraphs'}
    assert second['no_protocol_error'] and second['final']==first['final']


def test_judge_payload_is_blind_and_schema_rejects_missing_or_duplicate_ids():
    doc=save_document(ScopedEnvironment(row('학생이다.'),FakeBoundaryAnalysis()).document)
    values={name:{'initial':doc,'final':doc,'secret_score':9,'teacher_gold':'answer'} for name in ('onepass','onepass_feak')}
    notice={'id':'N1','sid':'S1','type':'style_vs_majority','before':'a','after':'b','scope':['S1']}
    view,key=payload({'question':'문항'},values,[notice],'onepass','case')
    raw=dumps(view)
    assert 'secret_score' not in raw and 'teacher_gold' not in raw and 'onepass' not in raw
    verdict={'notices':[{'id':'N1','broken_after_revision':'no','reason':'원래도 연결됨'}],
        'variants':[{'id':v,'better_than_original':'no','meaning_preserved':'yes','fabrication':'no','reason':'그대로',
            'notices':[{'id':'N1','fixed_after_repair':'not_applicable','undo_intended_change':'not_applicable','reason':'바뀌지 않음'}]} for v in key]}
    assert validate(dumps(verdict),['N1'],list(key))==verdict
    duplicate=deepcopy(verdict);duplicate['variants'][1]['id']=duplicate['variants'][0]['id']
    with pytest.raises(ValueError,match='IDs'):validate(dumps(duplicate),['N1'],list(key))
    invented=deepcopy(verdict);invented['variants'][0]['notices'][0]['fixed_after_repair']='yes'
    with pytest.raises(ValueError,match='Unbroken'):validate(dumps(invented),['N1'],list(key))


def test_rule_satisfaction_does_not_credit_already_aligned_style():
    initial=[{'notice_id':'N1','resolved_by_rule':True},{'notice_id':'N2','resolved_by_rule':None}]
    final=[{'notice_id':'N1','resolved_by_rule':True},{'notice_id':'N2','resolved_by_rule':True}]
    result=resolution_delta(initial,final)
    assert result['initially_rule_satisfied']==1 and result['newly_rule_satisfied']==1


def test_orchestrator_recall_uses_matching_problem_not_just_scope_or_tool(tmp_path):
    prepared={'corpus':{'records':[{'record_id':'r1','sids':['hidden1'],'op':'L_CONJ'}]},
        'stable_id_aliases':{'public_to_private_sentences':{'S1':'hidden1'}},
        'policy_row':{'normalization':{'sid_mapping':{'S1':['S1']}}}}
    snapshot={'paragraphs':[{'id':'P1','sentences':[{'id':'S1','text':'그러나 다음이다.'}]},
                            {'id':'P2','sentences':[{'id':'S2','text':'다른 문장이다.'}]}]}
    prefix=tmp_path/'prefix.json';write_json(prefix,{'messages':[{'role':'system','content':''},{'role':'user','content':dumps(snapshot)}]})
    def delegation(location,instruction):
        return {'role':'orchestrator','prefix':ref(prefix),'action':{'valid':True,'action':'DELEGATE',
            'value':{'to':'korean','items':[{'location':location,'instruction':instruction,'action':'EDIT'}]}}}
    result={'calls':[delegation(['P1'],'띄어쓰기를 고친다.'),delegation(['P2'],'접속사를 검토한다.')]}
    audit=a1_planning(prepared,result)
    assert audit['records'][0]['in_explicit_delegated_scope'] and not audit['records'][0]['matching_problem_type']
    assert audit['no_record_delegations']==1
    result['calls'].append(delegation(['P1'],'접속사가 앞 문장과 맞는지 고친다.'))
    assert a1_planning(prepared,result)['records'][0]['matching_problem_type']
    assert 'G_PARA_SWAP' in problem_types({'action':'EDIT','instruction':'문단 순서가 잘못됨'})
    # A reviewed ambiguous task must not silently become a definite match.
    review={}
    for call in result['calls']:
        item=call['action']['value']['items'][0]
        review[task_key(item,'korean')]={'operator_types':[], 'possible_types':['L_CONJ']}
    measured=a1_planning(prepared,result,review)
    assert not measured['records'][0]['matching_problem_type']
    assert measured['records'][0]['ambiguous_problem_type_only']
    with pytest.raises(ValueError,match='Unreviewed'):
        a1_planning(prepared,result,{})


def test_final_whole_korean_audit_ignores_earlier_stages(tmp_path):
    from verak.v4.scale2_projection import restore_document
    source=row('그러나 문장이다.');doc=ScopedEnvironment(source,FakeBoundaryAnalysis()).document
    prepared={'corpus':{'records':[{'record_id':'r','sids':['hidden1'],'op':'L_CONJ',
        'original_text':{'hidden1':'그래서 문장이다.'},'recovery_target':{'site':[0,3]}}]},
        'stable_id_aliases':{'public_to_private_sentences':{'S1':'hidden1'}},
        'policy_row':{'normalization':{'sid_mapping':{'S1':['S1']}}}}
    call={'before':save_document(doc),'action':{'valid':True,'action':'EDIT','value':
        {'sentence':'S1','old':'그러나','new':'그래서'}}}
    result={'stages':[{'calls':[call]},{'final_whole_essay_pass':True,'calls':[]}]}
    assert not final_korean_changes(prepared,result,'a2')[0]['sentence_edited']
    result['stages'][1]['calls']=[call]
    counted=final_korean_changes(prepared,result,'a2')[0]
    assert counted['position_edited'] and counted['edits'][0]['new']=='그래서'
    assert counted['position_text_changed']
    # A wide EDIT can cover the conjunction while changing only later words.
    call['action']['value'].update(old='그러나 문장이다.',new='그러나 좋은 문장이다.')
    counted=final_korean_changes(prepared,result,'a2')[0]
    assert counted['position_edited'] and counted['sentence_edited']
    assert not counted['position_text_changed']
