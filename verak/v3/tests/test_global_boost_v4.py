from copy import deepcopy
from types import SimpleNamespace

import pytest

from verak.v3.common import read_json, write_json
from verak.v3.global_boost.config import config_for, PHASE
from verak.v3.global_boost.v4_data import position_signature
from verak.v3.global_boost.v4_selection import eligible, source_cap


def test_source_cap_keeps_best_gpu_practices_including_old_concentrated_source():
    entries={f'old_{i}':{'source_id':'valid:1','operator':'G_SENT_MOVE','attempt':1,
        'R':.80+i/100,'score_source':'gpu_reference'} for i in range(8)}
    entries['new']={'source_id':'valid:2','operator':'G_SENT_MOVE','attempt':2,
        'R':.81,'score_source':'gpu_reference'}
    kept,dropped=source_cap(entries)
    assert set(kept)=={'old_4','old_5','old_6','old_7','new'}
    assert len(dropped)==4 and set(entries)==set(kept)|{e['episode_id'] for e in dropped}
    entries['old_0']['score_source']='cpu_provisional'
    with pytest.raises(ValueError,match='GPU-reference'):
        source_cap(entries)


def test_v4_selection_requires_valid_stop_and_no_more_than_one_rejection():
    row={'global_only_reward':{'R':.80},'termination':{'global':'STOP'},
         'actions_by_role':{'global':[{'action':'MOVE','valid':False},{'action':'STOP','valid':True}]}}
    assert eligible(row)
    row['actions_by_role']['global'].insert(0,{'action':'MOVE','valid':False})
    assert not eligible(row)
    row['actions_by_role']['global']=[{'action':'MOVE','valid':True}]
    assert not eligible(row)
    row['actions_by_role']['global']=[{'action':'STOP','valid':False}]
    assert not eligible(row)


def test_position_proof_ignores_sampling_seed_but_rejects_same_structural_position():
    record={'op':'G_PARA_SWAP','sids':['S1','S2'],'params':{'paragraph_indices':[1,3],'seed':71,'span':None}}
    original=position_signature('valid:1','hash',record)
    record['params'].update(paragraph_indices=[3,1],seed=72)
    assert position_signature('valid:1','hash',record)==original
    record['params']['paragraph_indices']=[0,3]
    assert position_signature('valid:1','hash',record)!=original
    move={'op':'G_SENT_MOVE','sids':['S2'],'params':{'from_paragraph':0,'from_position':1,'to_paragraph':2,'to_position':3}}
    key=position_signature('valid:1','hash',move)
    move['params']['to_position']=2
    assert position_signature('valid:1','hash',move)!=key


def test_rescue_gate_requires_two_completed_luna_failures_of_triggered_operator(tmp_path,monkeypatch):
    from verak.v3.global_boost import v4_teacher as module
    from verak.v3.global_boost.teacher import attempt_path
    config=config_for();config['paths'][PHASE+'_output']=tmp_path
    write_json(tmp_path/'v4/sol_luna_same92.json',{'triggers':{
        'G_SENT_MOVE':{'sol_rescue_authorized':True},'G_PARA_SWAP':{'sol_rescue_authorized':False}}})
    rows={eid:{'source_id':eid,'operator':op} for eid,op in
          [('fail','G_SENT_MOVE'),('partial','G_SENT_MOVE'),('success','G_SENT_MOVE'),('incomplete','G_SENT_MOVE'),('untriggered','G_PARA_SWAP')]}
    for eid in rows:
        for attempt in (1,2):
            write_json(attempt_path(tmp_path,attempt,eid),{'completed':eid!='incomplete','main':1 if eid=='success' else .5 if eid=='partial' else 0})
    monkeypatch.setattr(module,'batch_configs',lambda _:[config])
    monkeypatch.setattr(module,'corpus',lambda _:rows)
    monkeypatch.setattr(module,'structural_main',lambda _,r:r['main'])
    plan=module.rescue_plan(config)
    assert {t['episode_id'] for t in plan['tasks']}=={'fail','partial'}
    assert plan['teacher_output_global']==4096 and plan['teacher_output_korean']==1024
    assert plan['policy_output_limit']==1024 and plan['policy_context_limit']==8192


def test_sol_rescue_returns_only_json_under_policy_target_limit(monkeypatch):
    from verak.v3.global_boost.v4_teacher import RescueTeacher
    from verak.v3.train.teacher_bulk import BulkTeacher
    seen=[]
    def generate(self,messages,**kwargs):
        seen.append((self.model,self.max_output,kwargs['role']))
        return {'raw':'  {"action":"STOP","args":{"summary":"정리"}}  '}
    monkeypatch.setattr(BulkTeacher,'generate',generate)
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):return list(range(len(messages)*5))
    backend=RescueTeacher(SimpleNamespace(model='Sol'),SimpleNamespace(model='Luna'),Tokenizer())
    global_result=backend.generate([{'role':'system','content':'v1'}],episode_id='x',role='global',turn='1:0')
    backend.generate([{'role':'system','content':'v1'}],episode_id='x',role='korean',turn='1:0')
    assert seen==[('Sol',4096,'global'),('Luna',1024,'korean')]
    assert global_result['raw'].startswith('{') and '```' not in global_result['raw']
    assert global_result['teacher_raw'].startswith('  ')


@pytest.mark.parametrize('partial_cpu_snapshot',[False,True])
def test_gpu_handoff_applies_v4_cap_and_stop_gate_before_immutable_publication(tmp_path,monkeypatch,partial_cpu_snapshot):
    from verak.v3.common import file_sha
    from verak.v3.global_boost import aggregate, measure, v4_data
    from verak.v3.global_boost.measure import measured_path
    from verak.v3.global_boost.teacher import attempt_path
    config=config_for();root=tmp_path/'global';config['paths'][PHASE+'_output']=root
    write_json(root/'v4/plan.json',{'cap':4})
    config[PHASE].update(measurement_source='gpu_reference',score_fingerprint='reference')
    items=[]
    for i in range(8):
        eid=f'case{i}'
        raw_path=attempt_path(root,1,eid);write_json(raw_path,{'episode_id':eid})
        candidate_path=root/f'candidate{i}.json'
        write_json(candidate_path,{'source_id':'valid:1','operator':'G_SENT_MOVE'})
        provisional=root/f'provisional{i}.json';write_json(provisional,{'R':.8+i/100})
        reward={'R':.8+i/100,'R_rec':1.,'R_over':0.}
        write_json(measured_path(config,1,raw_path),{'global_only_reward':reward,'score_source':'gpu_reference',
            'quality_scores':{'before':{'execution_device':'gpu_reference','scorer_fingerprint':'reference'}},
            'termination':{'global':'STOP'},'actions_by_role':{'global':[{'action':'STOP','valid':i!=7}]}})
        items.append({'episode_id':eid,'attempt':1,'batch':str(root),'raw_path':str(raw_path),'raw_sha256':file_sha(raw_path),
            'candidate_path':str(candidate_path),'candidate_sha256':file_sha(candidate_path),
            'provisional_path':str(provisional),'provisional_sha256':file_sha(provisional),
            'provisional_global_eligible':True,'provisional_R':reward['R'],'quality_inputs':[]})
        if partial_cpu_snapshot and i in (3,7):
            items[-1].update(provisional_path=None,provisional_sha256=None,provisional_global_eligible=None,provisional_R=None)
    if partial_cpu_snapshot:
        # One observed attempt is insufficient for comparing a practice's best
        # sample when another saved attempt has no CPU snapshot.
        second=deepcopy(items[0]);second_raw=attempt_path(root,2,'case0')
        write_json(second_raw,{'episode_id':'case0','attempt':2})
        second.update(attempt=2,raw_path=str(second_raw),raw_sha256=file_sha(second_raw),
            provisional_path=None,provisional_sha256=None,provisional_global_eligible=None,provisional_R=None)
        measured=read_json(measured_path(config,1,attempt_path(root,1,'case0')))
        measured['global_only_reward']['R']=.805
        write_json(measured_path(config,2,second_raw),measured)
        items.append(second)
    manifest_path=root/'gpu_rescore_manifest.json'
    manifest={'episodes':items,'requests':[],'reference_gpu_fingerprint':'reference','provisional_selected':{}}
    if partial_cpu_snapshot:
        manifest.update(schema_version=2,handoff_contract='teacher_complete_gpu_reference_v2')
    write_json(manifest_path,manifest)
    write_json(root/'cpu_ready.json',{'manifest_path':str(manifest_path),'manifest_sha256':file_sha(manifest_path)})
    write_json(tmp_path/'gpu_rescore/global_complete.json',{'component':'global','manifest_sha256':file_sha(manifest_path),
        'request_count':0,'fingerprint':'reference','slot':'pre_rft_training','errors':[]})
    monkeypatch.setattr(aggregate,'batch_configs',lambda _:[config])
    monkeypatch.setattr(measure,'run',lambda _:None)
    monkeypatch.setattr(v4_data,'holdouts',lambda _:(set(),tmp_path/'manifest.json'))
    result=aggregate.gpu_finalize(config)
    assert set(result['selected'])=={'case3','case4','case5','case6'}
    assert result['gpu_eligible_before_source_cap']==7 and len(result['source_cap_exclusions'])==3
    assert result['selected_global']==4 and result['selected_korean']==0
    assert result['selection_changes']['attempts_unknown']==(3 if partial_cpu_snapshot else 0)
    assert result['selection_changes']['attempt_eligibility_flip_count']==(0 if partial_cpu_snapshot else 1)
    assert result['selection_changes']['practices_compared']==(5 if partial_cpu_snapshot else 8)
    if partial_cpu_snapshot:
        assert 'case0' not in {c['episode_id'] for c in result['selection_changes']['best_attempt_or_membership_changes']}
    old_sha=file_sha(root/'gpu_selection.json')
    assert aggregate.gpu_finalize(config)==result and file_sha(root/'gpu_selection.json')==old_sha


def test_rescue_export_masks_non_json_target_without_rewriting_runtime_context_or_reward():
    from verak.v3.global_boost.v4_selection import rescue_action_targets
    malformed={'role':'global','turn':'1:0','raw':'not JSON','messages':[{'role':'user','content':'question'}]}
    valid={'role':'global','turn':'1:1','raw':'{"action":"STOP","args":{"summary":"done"}}',
        'messages':[{'role':'user','content':'question'},{'role':'assistant','content':'not JSON'},
                    {'role':'user','content':'JSON retry'}]}
    korean={'role':'korean','turn':'1:0','raw':'unchanged KOREAN'}
    row={'calls':[malformed,valid,korean],'global_only_reward':{'R':.95},
        'actions_by_role':{'global':[{'action':'STOP','valid':True}]},'termination':{'global':'STOP'}}
    before=deepcopy(row);exported=rescue_action_targets(row)
    assert row==before and exported['calls']==[valid,korean]
    assert exported['calls'][0]['messages'][1]['content']=='not JSON'
    assert exported['global_only_reward']==row['global_only_reward']
    assert exported['actions_by_role']==row['actions_by_role']
    assert len(exported['action_only_export']['masked_non_action_targets'])==1


def test_pre_gpu_report_exposes_raw_cap_inventory_without_role_rewards(tmp_path,monkeypatch):
    from verak.v3.global_boost import v4_pre_gpu as module
    from verak.v3.global_boost.teacher import attempt_path
    from verak.v3.global_boost.prepare import safe_id
    config=config_for();config['paths'][PHASE+'_output']=tmp_path
    write_json(tmp_path/'expansion_status.json',{'stage':'generation_finished','stop_reason':'quota'})
    write_json(tmp_path/'v4/plan.json',{'old_source_capped_counts':{'G_SENT_MOVE':4,'G_PARA_SWAP':0}})
    write_json(tmp_path/'v4/sol_luna_same92.json',{'trigger':'G_SENT_MOVE'})
    rows={f'old{i}':{'source_id':'source1','operator':'G_SENT_MOVE'} for i in range(5)}
    rows['new']={'source_id':'source2','operator':'G_SENT_MOVE','source_provenance':{'active_corpus_source':True}}
    for eid in rows:
        write_json(tmp_path/'qc'/(safe_id(eid)+'.json'),{'passed':True})
        raw={'corpus_episode_id':eid,'generation_completed':True,'termination':{'global':'STOP','korean':'STOP'},
             'actions_by_role':{'global':[{'action':'STOP','valid':True}]},'runtime_error':None}
        write_json(attempt_path(tmp_path,1,eid),raw)
    failed={**raw,'corpus_episode_id':'new','generation_completed':False,
            'runtime_error':{'type':'IncompleteResponse','message':'kept as an error'}}
    write_json(attempt_path(tmp_path,2,'new'),failed)
    write_json(attempt_path(tmp_path,3,'old0'),{**raw,'corpus_episode_id':'old0'})
    monkeypatch.setattr(module,'batch_configs',lambda _:[config])
    monkeypatch.setattr(module,'corpus',lambda _:rows)
    account={'pending':0,'confirmed_usd':11.5,'reserved_usd':0}
    monkeypatch.setattr(module,'BoostAPI',lambda *_:SimpleNamespace(accounting=lambda:account,close=lambda:None))
    result=module.report(config)
    op=result['operators']['G_SENT_MOVE']
    assert op['Luna_saved_attempts']==7 and op['Luna_completed_attempts']==6
    assert op['Sol_rescue_saved_attempts']==1 and len(op['teacher_errors'])==1
    assert op['legacy_Luna_unattempted_slots']==5 and op['new_Luna_unattempted_slots']==0
    assert op['source_cap_inventory']['potential_saved_excess_before_GPU_eligibility']==1
    assert op['source_cap_inventory']['actual_final_cap_exclusions'] is None
    assert result['final_role_R'] is None and result['final_selected_GLOBAL'] is None
    assert result['distinct_sources']==2 and result['distinct_new_sources']==1
    assert read_json(tmp_path/'component_pre_gpu.json')==read_json(tmp_path/'v4/component_pre_gpu.json')
    assert not (tmp_path/'complete.json').exists() and not (tmp_path/'cpu_ready.json').exists()
    write_json(tmp_path/'expansion_status.json',{'stage':'v4_collecting'})
    with pytest.raises(RuntimeError,match='Finish all authorized'):
        module.report(config)


def test_teacher_complete_handoff_freezes_all_inputs_without_waiting_for_cpu(tmp_path,monkeypatch):
    from verak.v3.common import file_sha
    from verak.v3.global_boost import v4_handoff as module
    from verak.v3.global_boost.measure import measured_path
    from verak.v3.global_boost.teacher import attempt_path
    config=config_for();root=tmp_path/'global';config['paths'][PHASE+'_output']=root
    config[PHASE].update(measurement_source='cpu_provisional',score_fingerprint='cpu-frozen')
    write_json(root/'expansion_status.json',{'stage':'generation_finished','stop_reason':'quota'})
    write_json(root/'v4/plan.json',{'cap':4})
    write_json(tmp_path/'cpu_scorer/status.json',{'fingerprint':'cpu-frozen'})
    rows,files={},[]
    layout={'paragraphs':[{'units':[{'text':'revised','leading':''}]}],'gaps':[''],'tail':''}
    for eid in ('full','partial','none'):
        candidate={'source_id':eid,'operator':'G_SENT_MOVE','question':'Q','corrupted_text':eid}
        path=root/(eid+'.json');write_json(path,candidate);rows[eid]=candidate
        files.append({'episode_id':eid,'path':str(path),'sha256':file_sha(path)})
        for attempt in (1,2):
            raw_path=attempt_path(root,attempt,eid)
            write_json(raw_path,{'corpus_episode_id':eid,'stage1_layout':layout})
            if eid=='full' or (eid=='partial' and attempt==1):
                write_json(measured_path(config,attempt,raw_path),{'raw_generation_sha256':file_sha(raw_path),
                    'score_source':'cpu_provisional','global_only_reward':{'R':.85},'termination':{'global':'STOP'},
                    'actions_by_role':{'global':[{'action':'STOP','valid':True}]},
                    'quality_scores':{'corrupted':{'execution_device':'cpu','scorer_fingerprint':'cpu-frozen'}}})
    write_json(root/'source_plan.json',{'candidates':{'G_SENT_MOVE':files}})
    monkeypatch.setattr(module,'batch_configs',lambda _:[config])
    monkeypatch.setattr(module,'corpus',lambda _:rows)
    monkeypatch.setattr(module,'original_teacher_running',lambda _:False)
    monkeypatch.setattr(module,'reference_gpu_fingerprint',lambda _:'reference')
    account={'pending':0,'confirmed_usd':12.5,'reserved_usd':0}
    monkeypatch.setattr(module,'BoostAPI',lambda *_:SimpleNamespace(accounting=lambda:account,close=lambda:None))
    ready=module.freeze_ready(config)
    manifest=read_json(ready['manifest_path'])
    assert ready['handoff_contract']=='teacher_complete_gpu_reference_v2' and not ready['cpu_measurements_finished']
    assert manifest['schema_version']==2 and len(manifest['episodes'])==6 and len(manifest['requests'])==4
    assert manifest['provisional_observation']['attempts_observed']==3
    assert manifest['provisional_observation']['practices_fully_observed']==1
    assert set(manifest['provisional_selected'])=={'full'}
    absent=[e for e in manifest['episodes'] if e['provisional_path'] is None]
    assert len(absent)==3 and all(e['provisional_global_eligible'] is None and e['provisional_R'] is None for e in absent)
    # CPU files can arrive after the immutable handoff, including after a crash
    # between manifest publication and the readiness marker. Never change it.
    late=attempt_path(root,2,'partial')
    write_json(measured_path(config,2,late),{'late':'not part of the frozen snapshot'})
    (root/'cpu_ready.json').unlink()
    resumed=module.freeze_ready(config)
    assert resumed['manifest_sha256']==ready['manifest_sha256']
    assert resumed['provisional_observation']==ready['provisional_observation']
    account['pending']=1
    with pytest.raises(RuntimeError,match='paid calls are live'):
        module.freeze_ready(config)


def _completed_audit_fixture():
    audit={'status':'complete','source_essays':200,'unique_source_essays':200,'unknown_essays':0,
        'state_comparisons':600,'unique_score_inputs_completed':534,'mean_abs_Q_delta':.123456,
        'max_abs_Q_delta':.654321,'unique_input_mean_abs_Q_delta':.111111,
        'generated_digit_agreement_by_rubric':[.91,.92,.93,.94,.95,.96,.97,.98],
        'argmax_digit_agreement_by_rubric':[.98]*8,'score_line_changes':23,
        'selection_changes_by_role':{'global':3,'korean':5},'global_record_selection_changes':2}
    changes={'attempts_compared':12,'attempts_unknown':4,'attempts_missing_CPU_snapshot':3,
        'attempt_eligibility_flip_count':2,'practices_compared':5,'practices_unknown':3,
        'best_attempt_or_membership_change_count':1,'cpu_provisional_selected':3,
        'gpu_selected_before_cap_in_comparable_practices':4}
    return audit,changes


def test_final_report_prints_numeric_audit_and_observed_selection_denominators():
    from verak.v3.global_boost.v4_report import scorer_comparison_lines
    audit,changes=_completed_audit_fixture()
    text='\n'.join(scorer_comparison_lines(audit,changes))
    for number in ('0.123456','0.654321','0.111111','534','23 / 600','3 / 200','5 / 200','2 / 12','1 / 5'):
        assert number in text
    for i in range(91,99):
        assert f'{i}.00%' in text
    assert '| Attempts with unknown comparison | 4 |' in text
    assert '| Practices with unknown best-attempt comparison | 3 |' in text
    assert 'saved GPU cache hits' in text and 'not population estimates' in text
    for value in ({**audit,'status':'incomplete'},{**audit,'unique_source_essays':199},{**audit,'unknown_essays':1}):
        with pytest.raises(ValueError,match='at least 200 distinct'):
            scorer_comparison_lines(value,changes)


def test_final_v4_metrics_copies_keep_identical_completion_metadata(tmp_path,monkeypatch):
    from verak.v3.common import file_sha
    from verak.v3.global_boost import v4_report as module
    config=config_for();root=tmp_path/'global';config['paths'][PHASE+'_output']=root
    audit,changes=_completed_audit_fixture()
    audit_path=tmp_path/'audit_200/calibration.json';write_json(audit_path,audit)
    approval={'fingerprint':'gpu-reference','calibration_path':str(audit_path),'calibration_sha256':file_sha(audit_path)}
    plan={'old_raw_counts':{'G_PARA_SWAP':0,'G_SENT_MOVE':0},
        'old_source_capped_counts':{'G_PARA_SWAP':0,'G_SENT_MOVE':0},'source_inventory':{'counts':{}}}
    write_json(root/'v4/plan.json',plan)
    terminal=[{'episode_id':'failed','attempt':1,'failed_input_keys':['failed-key']}]
    write_json(root/'gpu_selection.json',{'score_source':'gpu_reference','fingerprint':'gpu-reference','selected':{},
        'selection_changes':changes,'source_cap_exclusions':[],'gpu_eligible_before_source_cap':0,'failed_gpu_episodes':terminal})
    write_json(root/'gpu_rescore_manifest.json',{'provisional_score_sources':[]})
    write_json(root/'expansion_status.json',{'stage':'generation_finished','stop_reason':'budget_cap'})
    write_json(root/'v4/sol_luna_same92.json',{})
    monkeypatch.setattr(module,'batch_configs',lambda _:[])
    account={'pending':0,'confirmed_usd':40.,'reserved_usd':0}
    monkeypatch.setattr(module,'BoostAPI',lambda *_:SimpleNamespace(accounting=lambda:account,close=lambda:None))
    result=module.report(config,approval)
    assert result['measurements_finished'] is True and result['terminal_measurement_errors']==terminal
    assert file_sha(root/'component_metrics.json')==file_sha(root/'v4/component_metrics.json')
    assert file_sha(root/'component_report.md')==file_sha(root/'v4/component_report.md')
    assert '0.123456' in (root/'component_report.md').read_text()
    # The existing finalizer adds these same fields again before its marker.
    result.update(terminal_measurement_errors=terminal,measurements_finished=True)
    write_json(root/'component_metrics.json',result)
    assert file_sha(root/'component_metrics.json')==file_sha(root/'v4/component_metrics.json')
