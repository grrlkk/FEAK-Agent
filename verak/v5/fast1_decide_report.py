"""Reproducible FAST1 C aggregation; saved outcomes stay individually auditable."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

from .common import read_json,write_json,sha_text,safe_id
from .fast1_common import normalize_whitespace,score_cached,sentence_change_counts
from .fast1_decide import ROOT,REPORT,episodes,source_cases


def read_evaluations():
    root=ROOT/'evaluation';manifest=read_json(root/'judge_manifest.json')
    values={}
    for request in manifest['requests']:
        judgment_path=root/'judgments'/(request['id']+'.json')
        value=read_json(judgment_path) if judgment_path.exists() else {'status':'missing'}
        entry=values.setdefault(request['case_id'],{'absolute':{},'primary':{},'reverse':{},'status':{}})
        entry['status'][request['order']]=value['status']
        aliases=request['aliases'];reverse={v:k for k,v in aliases.items()}
        for name in request['identical_conditions']:
            entry['absolute'][name]={'improved':'no','meaning_preserved':'yes','fabrication':'absent','no_change':True,'derived':True}
        if value['status'] not in {'valid','derived','code_only'}:
            continue
        verdict=value.get('verdict',{})
        for a in verdict.get('absolute',[]):
            entry['absolute'][reverse[a['id']]]={**a,'no_change':False,'derived':False}
        expected=set(request['expected_pair_ids'])
        labels={p['id']:p['preference'] for p in verdict.get('pairs',[])}
        for index,baseline in enumerate(manifest['baseline_conditions'],1):
            pair=f'P{index}'
            label=labels.get(pair,'tie' if pair not in expected else 'unjudged')
            if label in {'left','right'}:
                anchor_left=request['order']=='primary'
                label='decide2' if (label=='left')==anchor_left else baseline
            entry[request['order']][baseline]=label
    return manifest,values




def saved_request_artifacts(result):
    """Load request prefixes/responses, including raw responses not folded into episode.calls."""
    request_dir=ROOT/f"run{result['run']}"/'requests'/safe_id(result['case_id'])
    prefixes=sorted(request_dir.glob('*.prefix.json')) if request_dir.exists() else []
    recorded={Path(c.get('prefix',{}).get('path','')).name:c
              for c in result.get('calls',[]) if c.get('prefix')}
    raw_responses=[];gaps=[]
    for prefix_path in prefixes:
        raw_path=prefix_path.with_name(prefix_path.name.replace('.prefix.json','.raw.json'))
        if not raw_path.exists():
            try: prefix=read_json(prefix_path);role=prefix.get('role','unknown')
            except Exception: role='unknown'
            context=any(x.get('role')==role for x in result.get('context_limits',[]))
            gaps.append({'prefix':prefix_path.name,'role':role,
                'status':'blocked_by_context_preflight' if context else 'no_saved_response'})
            continue
        response=read_json(raw_path)
        response['_prefix_name']=prefix_path.name
        response['_has_episode_call_record']=prefix_path.name in recorded
        raw_responses.append(response)
        if prefix_path.name not in recorded:
            detail={'prefix':prefix_path.name,'role':response.get('role','unknown'),
                'status':'raw_response_not_in_episode_calls','fingerprint':response.get('fingerprint')}
            if response.get('role')=='orchestrator':
                try:
                    raw=json.loads(response.get('raw',''))
                    action=raw.get('action') if isinstance(raw,dict) else None
                except Exception: action=None
                detail['unprocessed_action_intent']=action or 'unparseable'
            gaps.append(detail)
    return prefixes,raw_responses,gaps

def aggregate():
    outputs=episodes();manifest,judged=read_evaluations()
    aliases=read_json(ROOT/'evaluation_aliases.json')['rows']
    alias={(r['case_id'],r['run']):r['evaluation_case_id'] for r in aliases}
    baselines=manifest['baseline_conditions']
    source={c['case_id']:read_json(c['row']['path']) for c in source_cases()}
    # The exact raw essay sent to one-pass is the frozen source for scoring.
    from .fast1_eval import load_cases
    raw_cases={c['case_id']:c for c in load_cases()}
    korean=read_json(ROOT/'korean_evaluation.json')
    label_by_call={(r['case_id'],r['run'],r['call_index']):r for r in korean['rows']}
    rows=[];korean_rows=[]
    cache_audit={};seen_fingerprints=set();request_artifacts={}
    for episode in sorted(outputs,key=lambda x:(x['run'],x['case_id'])):
        prefixes,raw_responses,gaps=saved_request_artifacts(episode)
        request_artifacts[(episode['case_id'],episode['run'])]=(prefixes,raw_responses,gaps)
        logical=unique_dispatches=replays=first_replays=0
        for call in raw_responses:
            if call.get('role')!='orchestrator':continue
            logical+=1
            fingerprint=call.get('fingerprint')
            if not fingerprint:
                raise ValueError(f"Missing raw orchestrator fingerprint: {episode['case_id']} run{episode['run']}")
            if fingerprint in seen_fingerprints:
                replays+=1
                item_id=(call.get('request') or {}).get('item_id','')
                if 'first_round' in item_id:first_replays+=1
            else:
                seen_fingerprints.add(fingerprint);unique_dispatches+=1
        cache_audit[(episode['case_id'],episode['run'])]={
            'logical_orchestrator_calls':logical,
            'unique_orchestrator_fingerprints_dispatched':unique_dispatches,
            'replayed_orchestrator_calls':replays,
            'first_round_cache_replays':first_replays}
    for result in outputs:
        raw_original=raw_cases[result['case_id']]['original']
        question=raw_cases[result['case_id']]['question']
        if normalize_whitespace(result['initial']['text'])!=normalize_whitespace(raw_original):
            raise ValueError(f"C initial text differs from raw FAST1 original after whitespace normalization: {result['case_id']}")
        score_original=score_cached(ROOT,question,raw_original)
        score_final=score_cached(ROOT,question,result['final']['text'])
        if score_original.get('fingerprint')!=score_final.get('fingerprint'):
            raise ValueError('C scorer fingerprint changed within one essay')
        same=normalize_whitespace(raw_original)==normalize_whitespace(result['final']['text'])
        delta=score_final['mean']-score_original['mean']
        evidence=judged.get(alias[(result['case_id'],result['run'])],{})
        stages=result['stages'];worker=[c for c in result['calls'] if c['role']!='orchestrator']
        reasons=Counter()
        for call in worker:
            if not call['action']['valid']:
                reasons.update(call['action'].get('rejection_reasons') or [call['action'].get('error','unknown')])
        notice_received=notice_attempted=notice_accepted=0
        for stage in stages:
            if stage['role']!='korean':continue
            calls=[c for c in worker if c['stage']==stage['number']]
            attempted={c['action'].get('value',{}).get('sentence') for c in calls if c['action']['action']=='EDIT'}
            accepted={c['action'].get('value',{}).get('sentence') for c in calls if c['action']['action']=='EDIT' and c['action']['valid'] and c['action'].get('text_changed')}
            notices=stage['notices_received'];notice_received+=len(notices)
            notice_attempted+=sum(n.get('sid') in attempted or bool(set(n.get('result_sids',[]))&attempted) for n in notices)
            notice_accepted+=sum(n.get('sid') in accepted or bool(set(n.get('result_sids',[]))&accepted) for n in notices)
        for index,call in enumerate(result['calls']):
            if call['role']!='korean' or call['action'].get('action')!='EDIT':continue
            row=label_by_call[(result['case_id'],result['run'],index)]
            korean_rows.append(row)
        delegations=[r for r in result['orchestrator_actions'] if r.get('valid') and r.get('action')=='DELEGATE']
        request_prefixes,raw_responses,request_gaps=request_artifacts[(result['case_id'],result['run'])]
        raw_model_calls=dict(Counter(c.get('role','unknown') for c in raw_responses))
        recorded_model_calls=dict(Counter(c.get('role','unknown') for c in result['calls']))
        pending_delegates=sum(g.get('unprocessed_action_intent')=='DELEGATE' for g in request_gaps)
        missing_orchestrator_dispatches=sum(g.get('role')=='orchestrator' for g in request_gaps)
        rows.append({'case_id':result['case_id'],'run':result['run'],'cohort':result['cohort'],
            'status':result['status'],'error':result.get('error'),'protocol_complete':result['protocol_complete'],
            'all_items_done':result['all_items_done'],'stop_reason':result['stop_reason'],
            'context_limits':result.get('context_limits',[]),
            'incomplete_stages':sum(s.get('ending') is None or s.get('stop_status') is None for s in stages),
            'request_prefix_count':len(request_prefixes),'unrecorded_response_prefixes':sum(g.get('status')!='raw_response_not_in_episode_calls' for g in request_gaps),
            'unrecorded_orchestrator_calls':missing_orchestrator_dispatches,
            'unprocessed_orchestrator_delegate_intents':pending_delegates,'request_gaps':request_gaps,
            'delegations':result['model_delegations'],
            'delegations_with_accepted_edit':sum(r.get('accepted_edits',0)>0 for r in delegations),
            'model_calls':raw_model_calls,'recorded_action_calls':recorded_model_calls,
            **cache_audit[(result['case_id'],result['run'])],
            'worker_rejections':dict(reasons),'stage_endings':dict(Counter(s['ending'] for s in stages)),
            'notices_received':notice_received,'notices_acted_on':notice_attempted,'notices_accepted_edit':notice_accepted,
            'korean_no_op_edits':sum(c['role']=='korean' and c['action'].get('noop_EDIT',False) for c in worker),
            'korean_kept_edits':sum(c['role']=='korean' and c['action']['valid'] and c['action'].get('text_changed',False) for c in worker),
            'korean_rejected_edits':sum(c['role']=='korean' and c['action']['action']=='EDIT' and not c['action']['valid'] for c in worker),
            'lenient_matches':sum(c['action'].get('span_match',{}).get('mode')=='lenient' and c['action']['span_match'].get('matched',False) for c in worker),
            'delta_q':delta,'no_change':same,'sanity_violation':same and delta!=0.,
            'raw_original_sha256':sha_text(raw_original),
            'initial_matches_raw_original_after_whitespace_normalization':True,
            'changes':sentence_change_counts(raw_original,result['final']['text']),
            'evaluation':evidence,'wall_seconds':result['wall_seconds']})
    def group(items):
        n=len(items)
        return {'n':n,'protocol_complete':sum(r['protocol_complete'] for r in items),
            'all_items_done':sum(r['all_items_done'] for r in items),'no_change':sum(r['no_change'] for r in items),
            'execution_statuses':dict(Counter(r['status'] for r in items)),
            'execution_errors':dict(Counter(r['error'] for r in items if r.get('error'))),
            'context_overflow_episodes':sum(bool(r['context_limits']) for r in items),
            'incomplete_stages':sum(r['incomplete_stages'] for r in items),
            'unrecorded_response_prefixes':sum(r['unrecorded_response_prefixes'] for r in items),
            'unrecorded_orchestrator_calls':sum(r['unrecorded_orchestrator_calls'] for r in items),
            'unprocessed_orchestrator_delegate_intents':sum(r['unprocessed_orchestrator_delegate_intents'] for r in items),
            'mean_delta_q':statistics.mean(r['delta_q'] for r in items),
            'stop_reasons':dict(sum((Counter(r['stop_reason'] for r in items),),Counter())),
            'model_calls':dict(sum((Counter(r['model_calls']) for r in items),Counter())),
            'worker_rejections':dict(sum((Counter(r['worker_rejections']) for r in items),Counter())),
            'absolute':{k:dict(Counter(r['evaluation'].get('absolute',{}).get('decide2',{}).get(k,'unjudged') for r in items))
                for k in ('improved','meaning_preserved','fabrication')},
            'pairs':{order:{b:dict(Counter(r['evaluation'].get(order,{}).get(b,'unjudged') for r in items)) for b in baselines} for order in ('primary','reverse')},
            **{k:sum(r[k] for r in items) for k in ('delegations','delegations_with_accepted_edit','notices_received','notices_acted_on','notices_accepted_edit','korean_no_op_edits','korean_kept_edits','korean_rejected_edits','lenient_matches','first_round_cache_replays','logical_orchestrator_calls','unique_orchestrator_fingerprints_dispatched','replayed_orchestrator_calls')}}
    groups={'all':group(rows),**{f'run{i}':group([r for r in rows if r['run']==i]) for i in range(1,4)}}
    link_audit_path=ROOT/'link_audit'/'result.json'
    if not link_audit_path.exists():
        raise FileNotFoundError(f'Required standalone Revision-stage link audit is missing: {link_audit_path}; run fast1_link_audit.evaluate first')
    link_audit=read_json(link_audit_path)
    if not isinstance(link_audit.get('summary'),dict) or not isinstance(link_audit.get('rows'),list):
        raise ValueError(f'Invalid standalone link-audit schema: {link_audit_path}')
    if link_audit['summary'].get('revision_stage_denominator') != len(link_audit['rows']):
        raise ValueError('Link-audit denominator does not match its all-stage rows')
    design=read_json(ROOT/'design.json')
    started_epoch=float(design['started_at']);ended_epoch=time.time()
    def utc(epoch):
        return datetime.fromtimestamp(epoch,timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
    a_complete=ROOT.parent/'a_eval'/'judge_complete.json'
    a_complete_epoch=a_complete.stat().st_mtime if a_complete.exists() else None
    measurement_window={'start_epoch':started_epoch,'start_utc':utc(started_epoch),
        'end_epoch':ended_epoch,'end_utc':utc(ended_epoch),'wall_seconds':ended_epoch-started_epoch,
        'start_source':str(ROOT/'design.json'),
        'end_event':'C aggregation completed after final evaluation scoring and report-data generation',
        'includes_job_a_wait':bool(a_complete_epoch is not None and started_epoch<a_complete_epoch),
        'job_a_complete_utc':utc(a_complete_epoch) if a_complete_epoch is not None else None}
    result={'rows':rows,'groups':groups,'korean_rows':korean_rows,'protocol_prompt_sha256':manifest['prompt_sha256'],
        'calibration':read_json(ROOT/'calibration/frozen.json'),
        'accounting':read_json(ROOT/'api/accounting.json'),
        'wall_seconds':measurement_window['wall_seconds'],'measurement_window':measurement_window,
        'inference_wall_seconds':read_json(ROOT/'inference_complete.json')['wall_seconds'],
        'link_audit':link_audit,
        'sanity_violations':[r for r in rows if r['sanity_violation']]}
    write_json(ROOT/'metrics.json',result)
    return result


def report():
    m=aggregate();a=m['groups']['all'];cost=m['accounting']['by_provider'];c=m['calibration']
    def pct(num,den):return '—' if not den else f'{100*num/den:.1f}%'
    comparisons=[]
    for baseline,label in (('orchestrator1_luna','저장 ORCHESTRATOR1'),('luna_alone','Luna 단독')):
        pairs=[a['pairs'][order][baseline] for order in ('primary','reverse')]
        comparisons.append(label+' 대비 DECIDE2 우세/상대 우세는 '+', '.join(
            f"{p.get('decide2',0)}/{p.get(baseline,0)}" for p in pairs)+'건(1차, 역순)')
    lines=['# FAST1 C — DECIDE2','',
        f"계획된 C30 반복 실험에서 저장된 episode {a['n']}건을 집계했다. 프로토콜 종료 {a['protocol_complete']}/{a['n']}, 모든 작업 done {a['all_items_done']}/{a['n']}, 정규화 후 원문과 동일 {a['no_change']}/{a['n']}이다. 실행 성공은 품질 향상을 뜻하지 않는다.",
        '주요 비교: '+ '; '.join(comparisons)+'.',
        '동일한 30편의 반복이며 90개의 독립 표본이 아니다. 학습 실행은 없었고 frozen test cohort는 사용하지 않았다. 입력은 smoke20 20편(train split)과 agent_dev10 10편이다. Revision epoch1 / Korean epoch2와 작업자 system prompt를 고정했다. Luna 총괄은 reasoning none이다.','',
        '| 작업 | 확인 비용 | 미확정 예약 | 상한 |','|---|---:|---:|---:|',
        f"| GPT / Luna 총괄 | ${cost['luna']['confirmed_usd']:.6f} | ${cost['luna']['reserved_usd']:.6f} | $1 |",
        f"| Claude / 평가 | ${cost['claude']['confirmed_usd']:.6f} | ${cost['claude']['reserved_usd']:.6f} | $5 |",'',
        f"실험 실행 window: {m['measurement_window']['start_utc']}–{m['measurement_window']['end_utc']} ({m['wall_seconds']:.1f}초; 설계 시작부터 최종 평가 점수와 집계 데이터 생성까지). Job A 완료는 {m['measurement_window']['job_a_complete_utc']}이며 이 window가 그 전 시작해 대기 시간을 포함한다: {m['measurement_window']['includes_job_a_wait']}. C90 추론은 {m['inference_wall_seconds']:.1f}초다. GPU 서버는 A/B/C가 공유했으므로 실행 시간을 독립 장비 속도로 해석하지 않는다.",
        '', '## 고정한 실행 및 안전 규칙','',
        'SEARCH를 차단하고 필요한 새 사실은 writer_notes에만 남겼다. 총괄 DELEGATE 최대4회, 항목별 작업자 행동 최대4회다. 자동 Korean 단계 수는 총괄 위임 수와 별도로 기록한다. 원문 구절은 지목 문장 안에서만 similarity≥0.85인 유일한 최상 일치를 허용하고 모든 일치 진단을 저장했다. 동일한 EDIT는 실제 변경으로 세지 않고 항목 done으로 끝냈다.',
        '각 Revision 항목 직후 그 단계의 변화 알림이 지목한 문장만 Korean이 검토했다. 알림이 없으면 건너뛰고, 실행 끝에는 이번 실행에서 바뀐 현존 문장만 최종 점검했다. FINISH/4회 위임/채택 수정 없는 위임/연속2개 blocked 보고가 종료 조건이다. 종료 조건 준수와 작업 done, 글 품질은 별도 집계다.',
        '안전 검사는 바로 반복된 단어, 원문에 없는 숫자·Bareun NNP 고유명사, 알려진 문장 종결체의 실제 과반 문체 불일치, frozen base 모델 문장 perplexity의 급상승을 차단한다. NNP·종결체가 불확실하면 추정으로 확정하지 않는다. 새 문장에는 분할 부모 문장, 부모가 없는 INSERT에는 편집 직전 글의 평균 NLL을 비교 기준으로 쓴다. 거절 시 문서·알림·undo·인용·계보를 되돌리고 후보와 사유를 보존한다.',
        f"저장된 수정 {sum(c['all_labels'].values())}개를 Claude가 평가했으며 fine {c['fine_evaluated']}개 중 {c['fine_blocked']}개({100*c['fine_blocked_rate']:.2f}%)를 안전 규칙이 차단했다. 고정한 perplexity 배수 상한은 {c['perplexity_ratio']:.4f} (Δmean NLL>{c['max_log_ppl_rise']:.6f})다. 이는 저장 사례 안에서의 보정 결과이며 새 사례의 오차율 보장은 아니다.",
        '', '## 품질 — Job A와 같은 평가','',
        f"Job A 완료 후 동일 prompt `{m['protocol_prompt_sha256']}`를 사용했다. 원시 전체 글 양쪽에 같은 공백 정규화를 적용하고, 일치하면 no change 및 ΔQ=0으로 처리했다. 동일 텍스트의 비영 ΔQ 위반은 {len(m['sanity_violations'])}건이다. 동일한 네 출력 조합은 두 순서 판정을 공유하고 원래90개 실행에 매핑했다.",
        '| 반복 | 프로토콜 종료 | 모든 항목 done | 평균 ΔQ | 개선 yes/no/유보/미판정 | 의미 yes/no/유보/미판정 | fabrication 없음/있음/유보/미판정 | no change |',
        '|---|---:|---:|---:|---|---|---|---:|']
    def counts(d,keys):return '/'.join(str(d.get(k,0)) for k in keys)
    for name,g in m['groups'].items():
        v=g['absolute'];lines.append(f"|{name}|{g['protocol_complete']}/{g['n']}|{g['all_items_done']}/{g['n']}|{g['mean_delta_q']:+.5f}|{counts(v['improved'],('yes','no','uncertain','unjudged'))}|{counts(v['meaning_preserved'],('yes','no','uncertain','unjudged'))}|{counts(v['fabrication'],('absent','present','uncertain','unjudged'))}|{g['no_change']}|")
    lines+=['','## 실행 완결성','',
        '실행 오류와 미완료 episode를 완료 실행처럼 합치지 않고 분리했다. context overflow는 요청 prefix 검사에서 발생하므로 해당 요청은 모델에 dispatch되지 않는다. raw 응답이 저장됐지만 episode action log에 반영되지 않은 API 호출도 따로 센다.',
        '| 반복 | 저장 episode | 상태별 실행 수 | 실행 오류 | context overflow | 미완료 stage | prefix만 있는 요청 | raw 응답이 action log에서 빠진 총괄 요청 | 누락된 DELEGATE 의도 |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for name in ('run1','run2','run3'):
        g=m['groups'].get(name)
        if not g:continue
        errors=', '.join(f"{k}:{v}" for k,v in g['execution_errors'].items()) or '—'
        lines.append(f"|{name}|{g['n']}|{g['execution_statuses']}|{errors}|{g['context_overflow_episodes']}|{g['incomplete_stages']}|{g['unrecorded_response_prefixes']}|{g['unrecorded_orchestrator_calls']}|{g['unprocessed_orchestrator_delegate_intents']}|")
    lines+=['',f"전체 실행 상태: {a['execution_statuses']}; 오류 사유: {a['execution_errors'] or '없음'}. context overflow {a['context_overflow_episodes']}건, 미완료 stage {a['incomplete_stages']}건, raw 응답은 있으나 action log에서 빠진 총괄 호출 {a['unrecorded_orchestrator_calls']}건, 누락된 DELEGATE 의도 {a['unprocessed_orchestrator_delegate_intents']}건, prefix만 있고 응답이 없는 요청 {a['unrecorded_response_prefixes']}건이다."]
    lines+=['','| 순서 | 상대 | DECIDE2 우세 | 상대 우세 | 동률 | 유보 | 미판정 |','|---|---|---:|---:|---:|---:|---:|']
    for order,groups in a['pairs'].items():
        for b,v in groups.items():lines.append(f"|{order}|{b}|{v.get('decide2',0)}|{v.get(b,0)}|{v.get('tie',0)}|{v.get('uncertain',0)}|{v.get('unjudged',0)}|")
    lines+=['','비교 대상 ORCHESTRATOR1은 요청에 지정된 저장 `orchestrator1_chat`의 검색 사용 조건이다. DECIDE2는 검색 OFF와 여러 실행 규칙을 동시에 바꿨으므로 총괄만의 인과 효과를 추정하지 않는다. 두 순서는 같은 Claude의 순서 민감도 검사이며 독립 평가자가 아니다.','',
        '## 실행·협력','', '| 항목 | 수치 | 비율 |','|---|---:|---:|',
        f"| 채택 수정 있는 총괄 위임 | {a['delegations_with_accepted_edit']}/{a['delegations']} | {pct(a['delegations_with_accepted_edit'],a['delegations'])} |",
        f"| Korean에 전달된 알림 중 EDIT 시도 | {a['notices_acted_on']}/{a['notices_received']} | {pct(a['notices_acted_on'],a['notices_received'])} |",
        f"| 알림 문장에서 채택된 EDIT | {a['notices_accepted_edit']}/{a['notices_received']} | {pct(a['notices_accepted_edit'],a['notices_received'])} |",
        f"| Korean 채택 / 거절 / no-op EDIT | {a['korean_kept_edits']} / {a['korean_rejected_edits']} / {a['korean_no_op_edits']} |",
        f"| lenient 일치 기록 | {a['lenient_matches']} | — |",
        f"| 총괄 논리 호출 | {a['logical_orchestrator_calls']} | — |",
        f"| 고유 fingerprint 실제 dispatch | {a['unique_orchestrator_fingerprints_dispatched']} | — |",
        f"| 동일 fingerprint 캐시 재생 | {a['replayed_orchestrator_calls']} | — |",
        f"| first-round 캐시 재생 | {a['first_round_cache_replays']} (매회 SCORE/MAP/첫 위임 1회씩이면 기준 180) | — |",'',
        '알림 수는 단계별 전달 횟수다. 최종 점검에 다시 전달된 알림은 다시 센다. ‘acted on’은 같은 문장에서 EDIT를 시도했다는 뜻이며 수정 성공이나 오류 해결의 증거가 아니다.','',
        '| 역할 | 호출 합계 | 전체 역할 호출 중 비율 | 편당 평균 |','|---|---:|---:|---:|']
    total_role_calls=sum(a['model_calls'].values())
    for role,n in a['model_calls'].items():lines.append(f"|{role}|{n}|{pct(n,total_role_calls)}|{n/a['n']:.3f}|")
    lines+=['','| 종료 사유 | 편수 |','|---|---:|']
    for reason,n in a['stop_reasons'].items():lines.append(f'|{reason}|{n}|')
    lines+=['','| 작업자 거절 사유 | 횟수 |','|---|---:|']
    for reason,n in sorted(a['worker_rejections'].items(),key=lambda x:-x[1]):lines.append(f'|{reason.replace(chr(10)," ").replace("|","/")}|{n}|')
    labels=[r for r in m['korean_rows'] if r.get('judgment')]
    effects=Counter(r['judgment']['effect'] for r in labels if r['accepted'])
    inconsistent=[r for r in labels if r['judgment']['effect']=='fixed_link' and r['judgment']['broken_before']=='no']
    incomplete_fixes=[r for r in labels if r['accepted'] and r['judgment']['effect']=='fixed_link' and r['judgment']['broken_after']!='no']
    audit=m['link_audit'];summary=audit['summary']
    yes_rows=[r for r in audit['rows'] if r.get('revision_created_broken_link')=='yes']
    yes_effects=Counter(r.get('korean_effect','not_run') for r in yes_rows)
    lines+=['','## Revision 단계와 Korean 연결 판정','',
        f"독립 stage-audit의 전체 Revision 단계 분모는 {summary['revision_stage_denominator']}다. 생성된 연결 오류 라벨 전체는 {summary['revision_created_broken_link']}이며, 단계 상태는 {summary['stage_status']}다.",
        f"broken link를 새로 만들었다고 판정된 단계 {len(yes_rows)}개의 Korean 효과: {dict(yes_effects)} (분류: fixed_link / left_link / introduced_error / uncertain / not_run). Revision 직후 Korean 단계가 없는 경우 not_run이다. 최종 통합 Korean pass는 이 감사 입력에 포함하지 않아 개별 Revision에 귀속하지 않는다.",
        f"채택 Korean 수정의 개별 판정: {dict(effects)}. 아래는 모든 Korean EDIT 시도다. no-op은 텍스트가 같아 별도 판정을 만들지 않았으며, 같은 전후 문맥은 판정을 공유한다.", '',
        f"fixed_link로 판정된 채택 수정 중 {len(incomplete_fixes)}건은 broken_after가 no가 아니어서 완전한 오류 해결로 해석하지 않는다. 비채택 후보를 포함한 고유 판정 {len({r['judge_id'] for r in inconsistent})}개는 fixed_link와 broken_before=no가 함께 나오는 라벨 불일치가 있다. 원시 판정을 보존하고 아래에 표시했다. 후보 문장을 재구성할 수 없었던 거절 행동 {sum(r['status']=='unconstructable_rejected_action' for r in m['korean_rows'])}건은 미판정이다.", '',
        '| 글 | 반복 | 호출 | 문장 | 채택 | 판정 | Revision 생성 연결 오류 | 근거 |',
        '|---|---:|---:|---|---|---|---|---|']
    for row in m['korean_rows']:
        j=row.get('judgment') or {};effect=j.get('effect','no change (코드)' if row['no_op'] else '미판정: '+row['status'])
        if j.get('effect')=='fixed_link':
            if j.get('broken_before')=='no':effect+=' (라벨 불일치: 수정 전 오류 없음)'
            elif j.get('broken_after')!='no':effect+=' (잔여 오류: '+j.get('broken_after','unknown')+')'
        reason=j.get('reason','').replace('|','/').replace('\n',' ')
        lines.append(f"|{row['case_id']}|{row['run']}|{row['call_index']}|{row['sentence_id']}|{'yes' if row['accepted'] else 'no'}|{effect}|{j.get('revision_created_broken_link','—')}|{reason}|")
    lines+=['','| 글 | 반복 | 총괄 raw 호출 | Revision raw 호출 | Korean raw 호출 | 반영된 위임 | 채택 수정 위임 | 알림 수신 | 알림 시도 | 알림 채택 | 논리 총괄 호출 | 고유 dispatch | 캐시 재생 | 1차 캐시 재생 | 미처리 DELEGATE | 오류 | context 제한 | 미완료 stage | 누락 요청 (역할/상태) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|']
    for row in m['rows']:
        calls=row['model_calls']
        gaps=', '.join(f"{g['role']}:{g['status']}" for g in row['request_gaps']) or '—'
        contexts=', '.join(f"{x.get('role')}:{x.get('prefix_tokens')} tokens" for x in row['context_limits']) or '—'
        error=(row.get('error') or '—').replace('|','/')
        lines.append(f"|{row['case_id']}|{row['run']}|{calls.get('orchestrator',0)}|{calls.get('revision',0)}|{calls.get('korean',0)}|{row['delegations']}|{row['delegations_with_accepted_edit']}|{row['notices_received']}|{row['notices_acted_on']}|{row['notices_accepted_edit']}|{row['logical_orchestrator_calls']}|{row['unique_orchestrator_fingerprints_dispatched']}|{row['replayed_orchestrator_calls']}|{row['first_round_cache_replays']}|{row['unprocessed_orchestrator_delegate_intents']}|{error}|{contexts}|{row['incomplete_stages']}|{gaps}|")
    lines+=['',
        '캐시 재생은 저장된 호출 fingerprint의 정확한 중복으로 계산했다. SCORE/MAP/첫 위임을 각 1회씩 부르면 2·3번째 반복에서 편당 6회를 재생한다. 도구 재호출과 조기 종료에 따라 실제 수는 달라질 수 있다.',
        '전체 단계별 분모와 행별 판정은 `verak/v5/outputs/fast1/C/link_audit/result.json`에 저장한다.','']
    lines+=['','기록: `verak/v5/outputs/fast1/C/metrics.json`, 각 run의 attempts/requests, calibration, evaluation, korean_evaluation.json, api/ledger.sqlite. 원문·출력·판정은 로컬에 보존한다.','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
    return m


if __name__=='__main__':report()
