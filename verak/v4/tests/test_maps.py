from copy import deepcopy
import json
import pytest

from verak.v4.maps import (consensus, extraction_request, public_input, schema, sol_request,
                           validate, validate_sol, plan, extract_one, join_one, judge_one)


def source(genre='argumentative'):
    return {'source_id': 'train:1', 'genre': genre, 'question': '문항', 'text': '글',
        'feedback': 'PRIVILEGED FEEDBACK', 'scores': [1,2,3],
        'paragraphs': [{'id':'P1', 'sentences':[{'id':f'S{i}', 'text':f'문장 {i}.'} for i in range(1,6)]}]}


def graph(edges=(), flags=()):
    return {'genre':'argumentative', 'paragraph_roles':[{'paragraph':'P1','role':'body','key_sentence':'S1'}],
        'sentence_relations': [{'source':s,'target':t,'type':k} for s,t,k in edges],
        'paragraph_relations':[], 'off_topic':list(flags)}


def test_public_payload_excludes_feedback_scores_and_other_private_fields():
    text = json.dumps(extraction_request(source())[0], ensure_ascii=False)
    assert 'PRIVILEGED' not in text and 'scores' not in text and 'feedback' not in text
    assert public_input(source())['paragraphs'][0]['sentences'][0]['id'] == 'S1'


def test_support_example_share_one_outgoing_budget():
    with pytest.raises(ValueError, match='one outgoing'):
        validate(graph([('S2','S1','support'),('S2','S3','example')]), source())
    validate(graph([('S2','S1','support'),('S2','S3','contrast')]), source())


def test_intersection_drops_disagreements_but_union_protection_reaches_only_two_hops():
    left = graph([('S1','Q','main'),('S2','S1','support'),('S3','S2','example'),('S4','S3','support')],
                 ['S1','S2','S3','S4','S5'])
    right = graph([('S1','Q','main'),('S3','S2','example')], ['S1','S2','S3','S4','S5'])
    original = deepcopy((left,right))
    result, audit = consensus(left,right,source())
    assert len(result['sentence_relations']) == 2
    assert result['off_topic'] == ['S4','S5']
    assert audit['off_topic']['protected_override'] == ['S1','S2','S3']
    assert audit['protection']['depths'] == {'S1':0,'S2':1,'S3':2}
    assert not audit['protection']['is_proof_of_relevance']
    assert audit['relations']['sentence_relations:support']['intersection'] == 0
    assert (left,right) == original


def test_no_q_path_or_no_relations_never_implicitly_flags_off_topic():
    result, audit = consensus(graph(),graph(),source())
    assert result['off_topic'] == []
    assert audit['relations']['sentence_relations:main']['jaccard'] is None


def test_cause_effect_does_not_extend_relevance_protection():
    raw = graph([('S1','Q','main'),('S2','S1','cause_effect')], ['S2'])
    assert consensus(raw,raw,source())[0]['off_topic'] == ['S2']


def test_genre_roles_are_validated_and_disagreement_is_unknown():
    row=source('emotional'); left=graph(); left['genre']='emotional'
    with pytest.raises(ValueError,match='not allowed'):
        validate(left,row)
    left['paragraph_roles'][0]['role']='feeling'; right=deepcopy(left)
    right['paragraph_roles'][0]['role']='reflection'
    result,audit=consensus(left,right,row)
    assert result['paragraph_roles'][0]['role'] is None
    assert not audit['paragraph_roles'][0]['role_agrees']


def test_cross_paragraph_sentence_edges_need_hierarchy_anchors():
    row=source(); row['paragraphs']=[{'id':'P1','sentences':row['paragraphs'][0]['sentences'][:2]},
                                  {'id':'P2','sentences':row['paragraphs'][0]['sentences'][2:]}]
    raw=graph([('S1','Q','main'),('S4','S1','support')])
    raw['paragraph_roles'].append({'paragraph':'P2','role':'body','key_sentence':'S3'})
    with pytest.raises(ValueError,match='Cross-paragraph'):
        validate(raw,row)
    raw['sentence_relations'][1]['source']='S3'
    validate(raw,row)


def test_sol_checks_all_retained_edges_and_rejects_missing_or_duplicate_ids():
    row=source(); raw=graph([('S1','Q','main')]); messages, _, checks = sol_request(row,raw)
    assert 'PRIVILEGED' not in json.dumps(messages)
    good={'judgments':[{'id':c['id'],'verdict':'plausible','reason':'타당하다.'} for c in checks]}
    validate_sol(good,checks)
    with pytest.raises(ValueError,match='exactly once'):
        validate_sol({'judgments':good['judgments'][:-1]},checks)
    with pytest.raises(ValueError,match='exactly once'):
        validate_sol({'judgments':good['judgments']*2},checks)


@pytest.mark.parametrize('bad', [('S1','Q','support'),('S1','S2','main'),('S9','S2','support'),('S2','S2','example')])
def test_unknown_ids_wrong_q_direction_and_self_edges_are_invalid(bad):
    with pytest.raises(ValueError):
        validate(graph([bad]),source())


def test_fixed_plan_has_two_shortest_examples_and_twenty_sol_maps_per_genre():
    from collections import Counter
    rows=[]
    for genre in ('argumentative','explanatory','emotional'):
        for i in range(50):
            row=source(genre); row.update(source_id=f'train:{genre}:{i:02}',text='가'*(i+1)); rows.append(row)
    frozen=plan(rows,manifest_sha256='abc',model='pinned-luna')
    assert frozen==plan(rows,manifest_sha256='abc',model='pinned-luna')
    assert frozen['examples6']==[f'train:{g}:{i:02}' for g in ('argumentative','explanatory','emotional') for i in (0,1)]
    assert Counter(s.split(':')[1] for s in frozen['sol_maps60'])==Counter(dict.fromkeys(('argumentative','explanatory','emotional'),20))
    assert len(frozen['sol_maps60'])==60 and frozen['sampling_seed'] is None
    with pytest.raises(ValueError,match='150'):
        plan(rows[:-1],manifest_sha256='abc',model='pinned-luna')


class StubAPI:
    def __init__(self,raw):
        import sqlite3
        self.connection=sqlite3.connect(':memory:')
        self.connection.execute('CREATE TABLE calls(id INTEGER,stage TEXT,item_id TEXT,status TEXT,path TEXT)')
        self.raw=raw; self.requests=[]

    def request(self,messages,**kwargs):
        self.requests.append((messages,kwargs))
        return {'status':'completed','raw':json.dumps(self.raw),'phase_call':1,'response_id':'test_response'}

    def db(self):
        return self.connection


def minimal_frozen(row):
    return {'input_sha256':{row['source_id']:'input'},'request_sha256':{row['source_id']:'request'},
            'extraction_max_output':4096,'sol_max_output':8192}


def test_exact_resume_does_not_regenerate_invalid_maps_or_replace_fixed_sol_source(tmp_path):
    row=source(); invalid=graph([('S1','S2','main')]); api=StubAPI(invalid); frozen=minimal_frozen(row)
    first=extract_one(api,row,1,tmp_path,frozen)
    assert first['status']=='invalid' and first['parsed']==invalid
    assert extract_one(api,row,1,tmp_path,frozen)==first
    assert len(api.requests)==1
    extract_one(api,row,2,tmp_path,frozen)
    assert api.requests[0][1]['stage']!=api.requests[1][1]['stage']
    joined=join_one(row,tmp_path)
    assert joined['status']=='unavailable'
    judged=judge_one(api,row,tmp_path,frozen)
    assert judged['status']=='unmeasured_map_unavailable' and len(api.requests)==2
    broken=deepcopy(frozen); broken['input_sha256'][row['source_id']]='changed'
    with pytest.raises(ValueError,match='identity changed'):
        extract_one(api,row,1,tmp_path,broken)


def test_budget_stop_is_recorded_without_inventing_an_api_response(tmp_path):
    from feak_tc.runtime.openai import CallBudgetExceeded
    class NoFunds(StubAPI):
        def request(self,*args,**kwargs):
            raise CallBudgetExceeded('cap')
    row=source(); result=extract_one(NoFunds({}),row,1,tmp_path,minimal_frozen(row))
    assert result['status']=='budget_not_dispatched' and result['requests']==[]
    assert 'response_id' not in result and 'value' not in result


def test_report_keeps_unknown_wrong_and_unmeasured_separate_and_full_example_ids(tmp_path):
    from verak.v4.common import write_json
    from verak.v4.map_report import aggregate, example
    row=source(); frozen=minimal_frozen(row)
    frozen.update(source_manifest_sha256='manifest',sol_maps60=[row['source_id']],examples6=[row['source_id']])
    write_json(tmp_path/'design.json',frozen)
    for attempt in (1,2):
        extract_one(StubAPI(graph([('S1','Q','main')])),row,attempt,tmp_path,frozen)
    joined=join_one(row,tmp_path)
    checks=sol_request(row,joined['map'])[2]
    judged=judge_one(StubAPI({'judgments':[{'id':c['id'],'verdict':('wrong' if i==0 else 'unknown'),'reason':'근거'}
                                            for i,c in enumerate(checks)]}),row,tmp_path,frozen)
    metrics,_,_=aggregate([row],tmp_path,frozen,{'confirmed_usd':0})
    assert metrics['sol']['all']['by_check_type']['sentence_relation:main']=={'plausible':0,'wrong':1,'unknown':0,'unmeasured':0}
    assert metrics['sol']['all']['by_check_type']['paragraph_role:body']['unknown']==1
    rendered=example(row,joined,judged)
    assert 'train:1' in rendered and '```mermaid' in rendered and 'S1 -->|main| Q' in rendered
    assert all(f'S{i}' in rendered for i in range(1,6))
