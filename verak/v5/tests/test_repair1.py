from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from verak.v5 import repair1 as experiment
from verak.v5.repair1 import validate_decisions, repeat_key, controlled_stage, QueueEnvironment
from verak.v5.repair1_eval import transitions, judge_sites
from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis, row
from verak.v4.content_env import dumps
from verak.v4.scale2_projection import save_document


def test_gate_requires_exact_ids_and_abstention_stays_explicit():
    good={'decisions':[{'id':'C1','decision':'uncertain','reason':'관계 불확실'}]}
    assert validate_decisions(dumps(good),['C1'])[0]['decision']=='uncertain'
    for value in [dict(decisions=[]),dict(decisions=good['decisions']*2),
                  dict(decisions=[dict(good['decisions'][0],id='C2')]),
                  dict(decisions=[dict(good['decisions'][0],decision='done')])]:
        with pytest.raises(ValueError):validate_decisions(dumps(value),['C1'])


def test_frozen_source_and_binary_refs_are_verified_without_json_parsing(tmp_path):
    path=tmp_path/'weights.safetensors';path.write_bytes(b'\x00\xffnot-json')
    reference=experiment.ref(path)
    experiment.verify_frozen_file(reference)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):experiment.verify_frozen_file(reference)


def test_repeat_key_is_state_sensitive_and_ignores_error_wording():
    action={'value':{'action':'EDIT','sentence':'S1','old':'가','new':'가'},'error':'no-op'}
    assert repeat_key('same',action)==repeat_key('same',dict(action,error='different wording'))
    assert repeat_key('same',action)!=repeat_key('changed',action)


def setup_fake(monkeypatch, outputs):
    from verak.v4 import policy_prompts_v43
    monkeypatch.setattr(policy_prompts_v43,'verify_frozen',lambda **kw:{})
    iterator=iter(outputs)
    def generate(*args,**kwargs):
        return {'status':'completed','raw':dumps(next(iterator))}
    monkeypatch.setattr(experiment,'request',generate)
    return QueueEnvironment(row('첫 문장이다. 둘째 문장이다.'),FakeBoundaryAnalysis())


def test_two_identical_rejections_skip_only_item_and_never_claim_done(tmp_path,monkeypatch):
    noop={'action':'EDIT','sentence':'S1','old':'첫','new':'첫'}
    env=setup_fake(monkeypatch,[noop,noop])
    result=controlled_stage(env,[],None,None,tmp_path,limit=3,scope=['S1'])
    assert len(result['calls'])==2
    assert result['ending']=='controller_skip_unresolved'
    assert not result['no_protocol_error'] and not result['all_done']
    assert env.exhausted_sids=={'S1'}
    assert all(not c['action']['valid'] for c in result['calls'])


def test_whole_pass_continues_after_repeat_but_failure_remains(tmp_path,monkeypatch):
    noop={'action':'EDIT','sentence':'S1','old':'첫','new':'첫'}
    stop={'action':'STOP','status':'done','summary':'검토 종료','issues':[]}
    env=setup_fake(monkeypatch,[noop,noop,stop])
    result=controlled_stage(env,[],None,None,tmp_path,limit=4,whole=True)
    assert result['ending']=='STOP' and result['stop_status']=='done'
    assert len(result['forced_events'])==1 and not result['no_protocol_error']
    assert env.document.text=='첫 문장이다. 둘째 문장이다.'


def test_model_blocked_with_reason_is_protocol_report(tmp_path,monkeypatch):
    stop={'action':'STOP','status':'blocked','summary':'확신이 없다','issues':['연결 판단 불가']}
    env=setup_fake(monkeypatch,[stop])
    result=controlled_stage(env,[],None,None,tmp_path,limit=3,scope=['S1'])
    assert result['no_protocol_error'] and not result['all_done']


def test_failed_sentence_is_not_editable_in_later_whole_pass(tmp_path,monkeypatch):
    action={'action':'EDIT','sentence':'S1','old':'첫','new':'새'}
    stop={'action':'STOP','status':'done','summary':'종료','issues':[]}
    env=setup_fake(monkeypatch,[action,stop]);env.exhausted_sids={'S1'}
    result=controlled_stage(env,[],None,None,tmp_path,limit=4,whole=True)
    assert result['calls'][0]['action']['scope_rejection']
    assert env.document.text.startswith('첫')


def test_link_transitions_keep_unknown_separate_and_do_not_equate_edit_with_error():
    before={'S1':'yes','S2':'no','S3':'no','S4':'uncertain','S5':'yes'}
    after={'S1':'no','S2':'yes','S3':'no','S4':'yes','S5':'unjudged'}
    result=transitions(before,after,{'S2','S3'})
    assert result['fixed']==1 and result['new_broken']==1
    assert result['edited_previously_fine']==2
    assert result['uncertain_or_unjudged_pairs']==2


def test_judge_site_union_includes_unflagged_edit_and_successor():
    env=QueueEnvironment(row('첫 문장이다. 둘째 문장이다. 셋째 문장이다.'),FakeBoundaryAnalysis())
    original=save_document(env.document)
    env.start_korean([])
    sid=env.document.units[1].sid
    next_sid=env.document.units[2].sid
    assert env.step(dumps({'action':'EDIT','sentence':sid,'old':'둘째','new':'두 번째'}),'korean')['valid']
    changed=save_document(env.document)
    episodes={c:{'initial':original,'final':changed if c=='queue' else original} for c in experiment.CONDITIONS}
    assert judge_sites({'case_id':'one'},episodes,[])==[sid,next_sid]


def test_queue_continues_next_item_and_keeps_total_cap_and_failure(tmp_path,monkeypatch):
    noop={'action':'EDIT','sentence':'S1','old':'첫','new':'첫'}
    stop={'action':'STOP','status':'done','summary':'검토 종료','issues':[]}
    env=setup_fake(monkeypatch,[noop,noop,stop,stop])
    monkeypatch.setattr(experiment,'restore_env',lambda *args:env)
    suggestions=[{'id':f'C{i+1}','sid':u.sid,'scope':[u.sid],
        'item':{'location':[u.sid],'action':'EDIT','instruction':'검토','needs_search':'no'}}
        for i,u in enumerate(env.document.units)]
    case={'case_id':'test','cohort':'smoke20','revision_protocol':True}
    result=experiment.run_arm(case,'queue',suggestions,None,None,None,tmp_path)
    assert result['worker_requests']==4 and result['gate_requests']==0
    assert result['tasks'][0]['status']=='unresolved'
    assert result['tasks'][1]['status']=='model_done'
    assert result['stages'][-1]['whole_essay_pass']
    assert not result['korean_no_protocol_error'] and not result['all_done']


def test_queue_model_blocked_is_valid_report_without_task_success(tmp_path,monkeypatch):
    blocked={'action':'STOP','status':'blocked','summary':'연결 불확실','issues':['연결 판단 불가']}
    stop={'action':'STOP','status':'done','summary':'점검 종료','issues':[]}
    env=setup_fake(monkeypatch,[blocked,stop])
    monkeypatch.setattr(experiment,'restore_env',lambda *args:env)
    suggestions=[{'id':'C1','sid':'S1','scope':['S1'],'item':
        {'location':['S1'],'action':'EDIT','instruction':'검토','needs_search':'no'}}]
    case={'case_id':'test','cohort':'smoke20','revision_protocol':True}
    result=experiment.run_arm(case,'queue',suggestions,None,None,None,tmp_path)
    assert result['korean_no_protocol_error'] and not result['all_done']
    assert result['unresolved_tasks'][0]['status']=='model_blocked'


def test_gate_uncertainty_never_becomes_solved_task(tmp_path,monkeypatch):
    stop={'action':'STOP','status':'done','summary':'점검 종료','issues':[]}
    env=setup_fake(monkeypatch,[stop])
    monkeypatch.setattr(experiment,'restore_env',lambda *args:env)
    monkeypatch.setattr(experiment,'assess',lambda *args:{'valid':True,'calls':[{}],
        'decisions':[{'id':'C1','decision':'uncertain','reason':'알 수 없음'}]})
    suggestions=[{'id':'C1','sid':'S1','scope':['S1'],'item':
        {'location':['S1'],'action':'EDIT','instruction':'검토','needs_search':'no'}}]
    case={'case_id':'test','cohort':'smoke20','revision_protocol':True}
    result=experiment.run_arm(case,'queue_gate',suggestions,None,None,None,tmp_path)
    assert result['korean_no_protocol_error'] and not result['all_done']
    assert result['gate_requests']==1 and result['worker_requests']==1
    assert result['unresolved_tasks'][0]['status']=='model_uncertain'
