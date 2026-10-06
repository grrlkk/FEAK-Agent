"""No-secret, no-network calibration contracts and budget accounting."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from feak_tc.runtime.openai import CallBudgetExceeded
from verak.v3.cli.phase4_preflight import select_luna
from verak.v3.common import load_config,read_json,write_json
from verak.v3.reconstruction_data import reconstruction_payload,labeling_payload,round_robin_sites
from verak.v3.reconstruction_api import ReconstructionAPI,usage_cost
from verak.v3.reconstruction_local import parse_sentence
from verak.v3.reward.similarity import make_pairs,roc_metrics


def case():
    return {'question':'물 절약 방법을 설명하시오.','essay_style':'한다',
            'corrupted_paragraph_with_gap':'물을 아낀다. [MISSING_SENTENCE] 따라서 도움이 된다.',
            'previous_sentence':'물을 아낀다.','next_sentence':'따라서 도움이 된다.',
            'next_cross_paragraph':False,'deleted_sentence':'수도꼭지를 잠근다.',
            'private_answer':{'text':'NEVER SEND'},'item_id':'valid:1:S2'}


def test_hidden_deleted_sentence_and_private_fields_never_reach_generator():
    c=case()
    payload=reconstruction_payload(c)
    assert 'private_answer' not in payload and 'deleted_sentence' not in payload
    assert c['deleted_sentence'] not in json.dumps(payload,ensure_ascii=False)
    assert labeling_payload(c,'물을 담아 사용한다.')['deleted_sentence']==c['deleted_sentence']
    c['previous_sentence']=c['deleted_sentence']
    with pytest.raises(ValueError,match='leaked'):
        reconstruction_payload(c)
    c['deleted_sentence']='"인용문"이다.'
    c['previous_sentence']=c['deleted_sentence']
    with pytest.raises(ValueError,match='leaked'):
        reconstruction_payload(c)


def test_new_model_snapshot_preferred_without_substitution():
    assert select_luna(['gpt-6-luna','gpt-6-luna-2026-09-17','gpt-6-sol'])[0]=='gpt-6-luna-2026-09-17'
    assert select_luna(['gpt-6-luna'])[0]=='gpt-6-luna'
    assert select_luna(['gpt-5.6-luna','gpt-6-astra'])[0] is None
    c=load_config()
    assert c['role_teacher']=={'model':'gpt-6.1-sol','reasoning_effort':'low'}


@pytest.mark.parametrize('raw', ['복원한 문장이다.','{"sentence":"복원한 문장이다."}',
                                '{"sentence":"복원한 문장이다.","essay_style":"한다"}',
                                '"복원한 문장이다."',
                                '```json\n{"sentence":"복원한 문장이다."}\n```'])
def test_local_plain_or_json_without_rewriting(raw):
    assert parse_sentence(raw)=='복원한 문장이다.'


def test_no_truncation_or_repair_of_malformed_local_output():
    for value in ('','{"sentence":','[MISSING_SENTENCE]'):
        with pytest.raises(ValueError):
            parse_sentence(value)


def test_first_sentence_is_positional_not_a_quality_selection():
    from verak.v3.reconstruction_format import first_sentence
    assert first_sentence('첫째 줄.\n더 나은 둘째 줄.')=='첫째 줄.'
    assert first_sentence('정말 그런가? 네, 그렇다.')=='정말 그런가?'
    assert first_sentence('1.5배 늘었다. 다음 내용이다.')=='1.5배 늘었다.'
    assert parse_sentence('{"sentence":"첫 문장이다. 다음 문장이다."}')=='첫 문장이다.'


def test_same_case_sites_spread_across_essays_without_duplicate_padding():
    pool=[{'source_id':str(i),'item_id':f'{i}:{j}'} for i in range(5) for j in range(4)]
    first=round_robin_sites(pool,8,43)
    assert len({r['source_id'] for r in first[:5]})==5
    assert first==round_robin_sites(list(reversed(pool)),8,43)
    with pytest.raises(ValueError):
        round_robin_sites(pool,21,43)


def test_budget_cache_raw_persistence_and_model_contract(tmp_path):
    called=[]
    class Client:
        def __init__(self,cfg,on_record,gate):
            self.cfg,self.record=cfg,on_record
        def _load(self): pass
        def start_sample(self,sample): pass
        def close(self): pass
        def __call__(self,**kwargs):
            called.append(kwargs)
            result={'sentence':'물을 담아 사용한다.'}
            self.record({'status':'completed','raw':json.dumps(result,ensure_ascii=False),'usage':{
                'input_tokens':100,'output_tokens':20},'response_id':'synthetic'})
            return result
    api=ReconstructionAPI(tmp_path,1,cheap_model='gpt-6-luna-2026-09-17',client_factory=Client)
    api.gate=SimpleNamespace(acquire=lambda n:None,record=lambda r:None)
    first=api.request('reconstruction','luna:1',reconstruction_payload(case()))
    assert api.budget.used==1
    assert api.request('reconstruction','luna:1',reconstruction_payload(case()))==first
    restarted=ReconstructionAPI(tmp_path,1,cheap_model='gpt-6-luna-2026-09-17',client_factory=Client)
    assert restarted.request('reconstruction','luna:1',reconstruction_payload(case()))==first
    with pytest.raises(CallBudgetExceeded):
        api.request('reconstruction','luna:2',reconstruction_payload(case()))
    assert len(called)==1
    with pytest.raises(ValueError,match='contract'):
        api.request('reconstruction','luna:1',{**reconstruction_payload(case()),'question':'다른 질문'})
    with pytest.raises(ValueError):
        ReconstructionAPI(tmp_path,651,cheap_model='gpt-6-luna')


def test_usage_keeps_reasoning_as_output_subset_and_separate_cache_rates():
    result=usage_cost('gpt-6-luna-2026-09-17',{'input_tokens':1000,'output_tokens':100,
        'input_tokens_details':{'cached_tokens':100,'cache_write_tokens':500},
        'output_tokens_details':{'reasoning_tokens':70}})
    assert result['confirmed_usd']==pytest.approx((400*.1+100*.01+500*.125+100*.5)/1e6)
    assert result['reasoning']==70


def test_roc_finds_known_threshold_and_refuses_single_class():
    result=roc_metrics([False,False,True,True],[.1,.2,.8,.9])
    assert result['roc_auc']==1 and result['tau']==.8
    with pytest.raises(ValueError):
        roc_metrics([False,False],[.1,.3])


def test_union_uses_both_sources_and_unique_negatives():
    cases=[{'item_id':str(i),'deleted_sentence':'a','negative_sentence':'b'} for i in range(200)]
    rows=[{'item_id':str(i),'source':s,'reconstruction':'c','same_role':bool(i%2)}
          for s in ('kanana','luna') for i in range(200)]
    pairs=make_pairs(cases,rows)
    assert len(pairs)==600
    assert sum(r['source']=='negative' for r in pairs)==200
    with pytest.raises(ValueError):
        make_pairs(cases,rows[:-1])
