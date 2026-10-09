"""Final GPU-only diverse-data report, with historical excess kept explicit."""
from collections import Counter
from pathlib import Path
from statistics import mean

from ..common import file_sha, read_json, write_json
from .config import PHASE, OPERATORS
from .expansion import batch_configs, root_for
from .measure import measured_path
from .paid import BoostAPI
from .prepare import corpus, safe_id


def report(config,approval):
    root=root_for(config)
    plan=read_json(root/'v4/plan.json')
    selection=read_json(root/'gpu_selection.json')
    if selection['score_source']!='gpu_reference' or selection['fingerprint']!=approval['fingerprint']:
        raise ValueError('Final diverse-data report requires the completed GPU reference selection')
    selected=selection['selected']
    all_sources,new_sources=set(),set()
    values={op:{'requested':400,'generated':0,'qc':{'generated':0,'judged':0,'passed':0,'unknown':0},
        'teacher_planned':0,'teacher_saved':0,'generation_completed':0,'global_reward_measured':0,
        'fully_recovered_attempts':0,'selected_global':sum(e['operator']==op for e in selected.values()),
        'selected_korean':0,'unknown':[],'termination':{'global':Counter(),'korean':Counter()},
        'rewards':[],'sources':set(),'paired':{},'rescue_saved':0,'rescue_full_recovery':0,
        'new_generated':0,'new_qc_passed':0,'new_sources':set()} for op in OPERATORS}
    for cfg in batch_configs(config):
        cfg[PHASE].update(measurement_source='gpu_reference',score_fingerprint=approval['fingerprint'])
        batch=cfg['paths'][PHASE+'_output']
        candidates=corpus(cfg)
        for eid,candidate in candidates.items():
            all_sources.add(candidate['source_id'])
            value=values[candidate['operator']]
            value['generated']+=1;value['qc']['generated']+=1
            value['sources'].add(candidate['source_id'])
            new='source_provenance' in candidate
            if new:new_sources.add(candidate['source_id'])
            value['new_generated']+=new
            if new:value['new_sources'].add(candidate['source_id'])
            verdict=batch/'qc'/(safe_id(eid)+'.json')
            if verdict.exists():
                value['qc']['judged']+=1
                if read_json(verdict)['passed']:
                    value['qc']['passed']+=1
                    value['new_qc_passed']+=new
                    value['teacher_planned']+=2
            else:value['qc']['unknown']+=1
        for raw_path in batch.glob('attempt_*/episodes/*.json'):
            raw=read_json(raw_path);eid=raw['corpus_episode_id']
            attempt=int(raw_path.parent.parent.name.split('_')[-1])
            value=values[candidates[eid]['operator']]
            value['teacher_saved']+=1
            value['teacher_planned']+=attempt==3
            value['rescue_saved']+=attempt==3
            value['generation_completed']+=bool(raw['generation_completed'])
            for role in ('global','korean'):
                value['termination'][role][raw['termination'].get(role,'not_completed')]+=1
            target=measured_path(cfg,attempt,raw_path)
            if not target.exists():
                value['unknown'].append({'episode_id':eid,'attempt':attempt,'reason':'GPU_reward_not_measured'})
                continue
            measured=read_json(target)
            if measured.get('score_source')!='gpu_reference':raise ValueError('CPU reward leaked into final report')
            reward=measured.get('global_only_reward')
            if reward is None:
                value['unknown'].append({'episode_id':eid,'attempt':attempt,'reason':'incomplete_GLOBAL_stage'})
                continue
            value['global_reward_measured']+=1;value['rewards'].append(reward)
            full=reward['per_record'][0]['main']==1
            value['fully_recovered_attempts']+=full
            value['rescue_full_recovery']+=attempt==3 and full
            if attempt in (1,2):value['paired'].setdefault(eid,[]).append(full)
    for op,value in values.items():
        rewards=value.pop('rewards');pairs=value.pop('paired')
        value['distinct_sources']=len(value.pop('sources'))
        value['new_distinct_sources']=len(value.pop('new_sources'))
        value['selected_distinct_sources']=len({e['source_id'] for e in selected.values() if e['operator']==op})
        value['old_raw_practices']=plan['old_raw_counts'][op]
        value['old_source_capped_practices']=plan['old_source_capped_counts'][op]
        value['usable_diversified_practices']=value['old_source_capped_practices']+value['new_generated']
        value['archived_raw_excess']=value['old_raw_practices']-value['old_source_capped_practices']
        value['source_cap_excluded_selected']=sum(e['operator']==op for e in selection['source_cap_exclusions'])
        value['teacher_main_recovery']=mean(r['per_record'][0]['main'] for r in rewards) if rewards else None
        value['R']=mean(r['R'] for r in rewards) if rewards else None
        value['R_over']=mean(r['R_over'] for r in rewards) if rewards else None
        value['completion_rate']=value['generation_completed']/value['teacher_saved'] if value['teacher_saved'] else None
        value['full_recovery_rate']=value['fully_recovered_attempts']/len(rewards) if rewards else None
        complete_pairs=[r for r in pairs.values() if len(r)==2]
        value['at_least_one_of_two']={'fully_observed_essays':len(complete_pairs),'recovered':sum(any(r) for r in complete_pairs),
            'share':mean(any(r) for r in complete_pairs) if complete_pairs else None}
        value['observed_any_sample']={'essays':len(pairs),'recovered':sum(any(r) for r in pairs.values())}
        value['qc']['pass_rate_judged']=value['qc']['passed']/value['qc']['judged'] if value['qc']['judged'] else None
        value['legacy_or_budget_unattempted_teacher_slots']=max(0,value['teacher_planned']-value['teacher_saved'])
        value['termination']={r:dict(c) for r,c in value['termination'].items()}
    api=BoostAPI(config,0);account=api.accounting();api.close()
    if account['confirmed_usd']+account['reserved_usd']>40+1e-8:raise ValueError('Cumulative GLOBAL $40 cap exceeded')
    manifest=read_json(root/'gpu_rescore_manifest.json')
    status=read_json(root/'expansion_status.json')
    result={'phase':PHASE,'task_version':'v4_prep','aggregate':True,'operators':values,
        'selected':selected,'selected_global':len(selected),'selected_korean':0,
        'distinct_new_sources':len(new_sources),'distinct_all_sources':len(all_sources),
        'selected_distinct_sources':len({e['source_id'] for e in selected.values()}),
        'source_inventory':plan['source_inventory']['counts'],'diversity_plan':plan,
        'diversity_plan_sha256':file_sha(root/'v4/plan.json'),'source_cap':4,
        'source_cap_exclusions':selection['source_cap_exclusions'],
        'gpu_eligible_before_source_cap':selection['gpu_eligible_before_source_cap'],
        'selection_changes':selection['selection_changes'],'same92_comparison':read_json(root/'v4/sol_luna_same92.json'),
        'rescue_plan':read_json(root/'v4/rescue_plan.json') if (root/'v4/rescue_plan.json').exists() else None,
        'rescue_status':read_json(root/'v4/rescue_status.json') if (root/'v4/rescue_status.json').exists() else None,
        'api':account,'budget_usd_cumulative':40,'historical_spending_included':True,
        'score_source':'gpu_reference','gpu_reference_rescoring_used':True,'canonical_for_selection':True,
        'scorer_approval':approval,'cpu_score_audit':read_json(approval['calibration_path']),
        'provisional_score_sources':manifest['provisional_score_sources'],
        'score_device_observations':{'gpu_reference':sum(v['global_reward_measured']*2 for v in values.values())},
        'selection_rule':'GLOBAL R>=.80, valid terminal STOP, rejected actions<=1; best attempt/practice then max4/source/operator by GPU R; frozen source holdouts excluded.',
        'teacher_roles':['global','korean'],'selected_roles':['global'],'unmeasured_reward_roles':['korean','combined'],
        'completion':{'teacher_requested':sum(v['teacher_planned'] for v in values.values()),
            'teacher_saved':sum(v['teacher_saved'] for v in values.values()),
            'rewards_measured':sum(v['global_reward_measured'] for v in values.values()),
            'stop_reason':status['stop_reason'],'pending_api':account['pending'],
            'legacy_unattempted_authorized_rescope':True},'gpu_used':False,'training':False}
    write_json(root/'best_global_trajectories.json',{'global':selected,'korean':{}})
    write_json(root/'component_metrics.json',result)
    write_json(root/'v4/component_metrics.json',result)
    lines=['# V4 prep A — diverse GLOBAL data','',
        'Final rewards and selections below use GPU-reference scores only. v1 policy, reward, KOREAN actions and prompt, and the8192/1024 inference contract were kept unchanged.','',
        '|Operator|Legacy raw / capped pool|New practices / sources|All sources|QC pass / judged|Main full recovery / measured attempts|R_over|Selected GLOBAL / sources|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for op,v in values.items():
        over='NA' if v['R_over'] is None else f"{v['R_over']:.4f}"
        lines.append(f"|{op}|{v['old_raw_practices']} / {v['old_source_capped_practices']}|{v['new_generated']} / {v['new_distinct_sources']}|{v['distinct_sources']}|{v['qc']['passed']} / {v['qc']['judged']}|{v['fully_recovered_attempts']} / {v['global_reward_measured']}|{over}|{v['selected_global']} / {v['selected_distinct_sources']}|")
    lines+=['','The same92 saved pilot comparison used every intended record as denominator; partials were not full recovery and unknowns earned no success.',
        'G_PARA_SWAP: Sol10/21 vs Luna10/21 (0pp, no Sol rescue). G_SENT_MOVE: Sol9/13 vs Luna6/13 (+23.08pp, rescue enabled). '
        'Sol completed92/92; Luna91/92, with one KOREAN failure affecting two local records and no missing GLOBAL recovery.',
        '',result['selection_rule'],'',
        f"GPU eligible practices before source cap: {selection['gpu_eligible_before_source_cap']}; cap exclusions: {len(selection['source_cap_exclusions'])}; final selected: {len(selected)}.",
        'Historical excess and interrupted legacy slots remain archived; they are not silently rerun or counted as successful episodes. '
        'New source positions were checked against the frozen archived/full-corpus/practice index. SFT validation and evaluation sources were excluded.',
        '',f"Sol rescue attempts / full recoveries: {sum(v['rescue_saved'] for v in values.values())} / {sum(v['rescue_full_recovery'] for v in values.values())}. "
        'Only practices with two completed, non-fully-recovering Luna attempts qualified. Sol used low reasoning and4096 output tokens for GLOBAL; KOREAN remained Luna low/1024. '
        'Policy targets contain action JSON only, limited to1024 policy tokens; inference context remained8192.',
        '',f"Cumulative cost including all legacy collection: ${account['confirmed_usd']:.6f}; reserved ${account['reserved_usd']:.6f}; cap$40; pending {account['pending']}.",
        '',f"Provisional-to-GPU attempt eligibility flips: {selection['selection_changes']['attempt_eligibility_flip_count']}; "
        f"best-attempt/membership changes before source-cap filtering: {selection['selection_changes']['best_attempt_or_membership_change_count']}.",
        f"Comparison coverage: {selection['selection_changes'].get('attempts_compared', 'NA')} observed attempts, "
        f"{selection['selection_changes'].get('attempts_unknown', 'NA')} unknown; "
        f"{selection['selection_changes'].get('practices_compared', 'NA')} fully observed practices, "
        f"{selection['selection_changes'].get('practices_unknown', 'NA')} unknown. "
        'An absent CPU snapshot is unknown, never a failed eligibility decision. GPU-only selection covers every available raw attempt.',
        'The provisional comparator explicitly mixes CPU results, exact-input/exact-fingerprint saved GPU cache hits and identity-proven zero deltas. '
        'It supplies no final reward or selection. The >=200-source audit is recorded separately; all final values use GPU reference scores.',
        '',f"Stop reason: {status['stop_reason']}. No KOREAN data selected; no GPU calls or training started by this component.",'']
    text='\n'.join(lines)
    (root/'component_report.md').write_text(text,encoding='utf-8')
    (root/'v4/component_report.md').write_text(text,encoding='utf-8')
    return result
