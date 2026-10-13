import json

import pytest

from verak.v5.recovery1_eval import validate, evidence_only


def verdict():
    scores={'task':3,'coherence':3,'language':4,'overall':3}
    return {'original_scores':scores,'variants':[
        {'id':label,'scores':dict(scores),'better_than_original':'yes','meaning_preserved':'yes',
         'new_specific_claims':'unknown','reason':'전체 글 연결 개선','evidence_reason':'추가 사실의 자료 부족'}
        for label in ('A','B')], 'preference':'tie','preference_reason':'비슷한 품질'}


def test_unknown_grounding_remains_unknown():
    value=validate(json.dumps(verdict()))
    assert value['variants'][0]['new_specific_claims']=='unknown'
    assert value['preference']=='tie'


@pytest.mark.parametrize('change', ['duplicate','bool_score','score_outside','missing_reason'])
def test_malformed_judgments_cannot_be_counted_as_success(change):
    value=verdict()
    if change=='duplicate': value['variants'][1]['id']='A'
    elif change=='bool_score': value['variants'][0]['scores']['overall']=True
    elif change=='score_outside': value['variants'][0]['scores']['overall']=6
    else: value['variants'][0]['reason']=''
    with pytest.raises(ValueError): validate(json.dumps(value))


def test_only_source_evidence_reaches_quality_judge():
    value={'successful_sourced_inserts':1,'search_history':[{'delegation':2,'item_id':'D2T1','query':'비밀 계획',
        'passages':[{'passage_id':'p1','title':'출처','text':'실제 근거','url':'https://example.com','bm25_score':99}]}],
        'source_citations':[{'sentence_id':'N1','text':'본문','source':{'title':'출처','passage_id':'p1'},
                             'inserted_text':'원래 삽입','internal_plan':'공개 금지'}]}
    clean=evidence_only(value)
    encoded=json.dumps(clean,ensure_ascii=False)
    assert '실제 근거' in encoded and '본문' in encoded
    for key in ('delegation','item_id','query','bm25_score','internal_plan','successful_sourced_inserts'):
        assert key not in encoded


def test_initial_variation_handles_worker_calls_saved_before_their_delegation(monkeypatch):
    from verak.v5 import recovery1_eval as module
    monkeypatch.setattr(module,'verified',lambda reference:reference)
    def episode(instruction):
        return {'calls':[
            {'role':'korean','action':{'action':'STOP'}},
            {'role':'orchestrator','action':{'action':'DELEGATE','value':{'instruction':instruction}},
             'prefix':{'messages':[{'role':'user','content':'same original observation'}]},
             'generation':{'temperature':0,'seed':47}}]}
    result=module.first_delegation_variation(episode('first plan'),episode('different plan'))
    assert result['same_input_settings_different_action'] is True
    changed=episode('different plan')
    changed['calls'][1]['prefix']['messages'][0]['content']='changed observation'
    assert module.first_delegation_variation(episode('first plan'),changed)['same_input_settings_different_action'] is False
