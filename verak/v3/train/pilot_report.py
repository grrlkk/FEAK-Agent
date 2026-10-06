"""Pilot metrics, role selection, inference-format export and local Markdown report."""
from collections import Counter
from statistics import mean
import gzip
import json

from ..common import read_json,write_json
from ..phase2 import write_jsonl
from ..agent.runner import system_prompt
from ..eval.summary import condition_stats
from .pilot import prepare,safe_id,LEVELS
from .formatting import select_roles,export


def action_type(action):
    name,args=action['action'],action['args']
    if name=='EDIT':
        if args['target'].startswith(('before:','after:')):return 'sentence_insert'
        if ':' not in args['target'] and not args['new_text'].strip():return 'sentence_delete'
        return 'in_sentence_EDIT'
    return name


def group_summary(rows):
    return {'n':len(rows),
        'rewards':{role:{metric:mean(r['reward'][role][metric] for r in rows) if rows else None
                      for metric in ('R','R_rec','R_q','R_over','R_step')}
                   for role in ('global','korean','combined')},
        'steps':{role:mean(r['steps'][role] for r in rows) if rows else None for role in ('global','korean')},
        'cost_per_episode':mean(r['cost_usd'] for r in rows) if rows else None}


def validate(config,design,rows,episodes,formatting,api):
    checks=[]
    def require(ok,name):
        checks.append({'name':name,'passed':bool(ok)})
    require(len(episodes)==100,'Exactly 100 completed episodes')
    require({r['corpus_episode_id'] for r in episodes}==set(design['pilot_ids']),'Only frozen pilot IDs')
    require(len(list((config['paths']['phase7_pilot_output']/'episodes').glob('*.json')))<=100,'No remaining teacher episodes')
    for r in episodes:
        id=r['corpus_episode_id']
        require(rows[id]['split']=='agent_train',id+': agent_train')
        require(r['model']=='gpt-6.1-sol' and r['mode']=='two_stage',id+': teacher/mode')
        require(all(v==0 for v in r['checks'].values()),id+': zero CHECK')
        require(all(s['purpose'].startswith('terminal_reward') for s in r['score_calls']),id+': terminal scorer only')
        for role in ('global','korean'):
            calls=[c for c in r['calls'] if c['role']==role]
            require(r['messages_by_role'][role][0]['content']==system_prompt(role),id+': '+role+' prompt')
            require(all(c['reasoning_effort']=='low' for c in calls),id+': '+role+' low')
            require(all(a['messages']==b['messages'][:len(a['messages'])] for a,b in zip(calls,calls[1:])),id+': '+role+' append-only prefix')
            obs=r['messages_by_role'][role][1]['content']
            require('Korean document profile' in obs and 'marker-change notices' in obs,id+': '+role+' observations')
            require(all('[점수]' not in m['content'] and 'recovery_target' not in m['content']
                        for c in calls for m in c['messages']),id+': '+role+' no hidden scores/targets')
    for role,fmt in formatting['roles'].items():
        require(not fmt['format_errors'],role+': all selected trajectories format')
        n=0
        with gzip.open(fmt['turns_path'],'rt',encoding='utf-8') as handle:
            for line in handle:
                sample=json.loads(line);n+=1
                require(all(x==-100 for x in sample['labels'][:sample['prompt_tokens']]),role+': context masked '+str(n))
                require(all(y==x for x,y in zip(sample['input_ids'][sample['prompt_tokens']:],sample['labels'][sample['prompt_tokens']:])),role+': target unmasked '+str(n))
                require(sample['messages'][0]['content']==system_prompt(role),role+': inference prompt '+str(n))
        require(n==fmt['inference_turns']['n'],role+': sample count')
    account=api.accounting()
    require(account['confirmed_usd']<=15 and account['pending']==0,'Budget <= $15; no pending calls')
    result={'passed':all(c['passed'] for c in checks),'count':len(checks),
            'failed':[c for c in checks if not c['passed']], 'checks':checks}
    write_json(config['paths']['phase7_pilot_output']/'validation.json',result)
    return result


def summarize(config,api):
    from transformers import AutoTokenizer
    design,corpus,_=prepare(config)
    root=config['paths']['phase7_pilot_output']
    all_rows=[read_json(root/'episodes'/(safe_id(id)+'.json')) for id in design['pilot_ids']
              if (root/'episodes'/(safe_id(id)+'.json')).exists()]
    rows=[r for r in all_rows if r.get('completed') and r.get('reward')]
    stats=condition_stats(rows)
    by_level={level:group_summary([r for r in rows if r['level']==level]) for level in LEVELS}
    ops=sorted({rec['op'] for r in corpus.values() for rec in r['records']})
    by_op={op:group_summary([r for r in rows if any(rec['op']==op for rec in corpus[r['corpus_episode_id']]['records'])]) for op in ops}
    action_counts={role:{kind:dict(Counter(action_type(a) for r in rows for a in r['actions_by_role'][role]
                        if kind=='attempted' or a['valid'])) for kind in ('attempted','accepted')}
                   for role in ('global','korean')}
    account=api.accounting()
    tokens=Counter();errors=Counter()
    for path in (root/'api/requests').glob('*.json'):
        call=read_json(path)
        tokens.update({k:call.get('cost',{}).get(k,0) for k in ('input','output','reasoning','cache_read','cache_write')})
        if call['status']!='completed':errors[str(call.get('error_type',call['status']))]+=1
    counts=Counter(r['level'] for r in corpus.values())
    projection=sum(counts[level]*by_level[level]['cost_per_episode'] for level in LEVELS) if all(by_level[l]['n'] for l in LEVELS) else None
    selection=select_roles(rows,corpus)
    no_global_ids={id for id,reason in selection['global_reasons'].items() if reason.startswith('no_GLOBAL')}
    selection['no_GLOBAL_kept_action_counts']=dict(Counter(action_type(a) for r in rows
        if r['corpus_episode_id'] in no_global_ids for a in r['actions_by_role']['global'] if a['valid']))
    write_json(root/'selection.json',selection)
    tokenizer=AutoTokenizer.from_pretrained(str(config['paths']['policy_base']),local_files_only=True)
    formatting=export(config,rows,selection,tokenizer)
    validation=validate(config,design,corpus,rows,formatting,api)
    result={'episodes':len(rows),'requested':100,'stats':stats,'by_level':by_level,'by_operator':by_op,
        'actions':action_counts,'api':account,'api_tokens':dict(tokens),'api_errors':dict(errors),
        'cost_per_episode_confirmed':account['confirmed_usd']/len(rows) if rows else None,
        'corpus_counts_by_level':dict(counts),'projected_all_1573_by_level_usd':projection,
        'projected_remaining_1473_by_level_usd':projection-sum(r['cost_usd'] for r in rows) if projection is not None else None,
        'projected_all_1573_simple_usd':1573*account['confirmed_usd']/len(rows) if rows else None,
        'selection':selection,'formatting':formatting,
        'validation':{'passed':validation['passed'],'count':validation['count'],'failed':validation['failed']}}
    write_json(root/'metrics.json',result)
    write_jsonl(root/'episodes.jsonl',all_rows)
    path=render(config,result)
    return path


def render(config,result):
    root=config['paths']['phase7_pilot_output'];s=result['stats'];a=result['api'];sel=result['selection'];fmt=result['formatting']
    def number(v):return '해당 없음' if v is None else f'{v:.6f}'
    lines=['# V3 Phase 7 teacher pilot', '',
        '## 범위와 결정', '',
        'Part A → B → C 순서로 완료. Addendum Section 10에 two_stage 주 방법, CHECK 기본 비활성화, '
        'recoverability-filtered corpus 1,573/397, Korean document profile 및 marker-change notices 용어를 기록했다. '
        '`env.enable_check=true`에서만 기존 CHECK 경로가 활성화된다. 본 파일럿의 채점은 종료 후 보상 계산에만 사용했다.', '',
        '`imple/reports/V3_PHASE_6_EXAMPLES.md`는 Phase 6 저장된 dev/real 기록으로 작성했다. '
        '이 단계는 GPT/Bareun 호출, 생성, 재채점 0회다. Judge의 의미 변경 span은 원래 스키마에 없어 '
        '판정문과 코드로 발췌한 전후 문장을 구분했다. Figure 1의 DEP 사례는 위치 변화에 대한 대응이며 의미 오류의 입증은 아니다.', '',
        '## 표본 및 실행', '',
        f"- 입력: `verak/v3/data/corrupt_recoverable/agent_train.jsonl` 1,573편. seed 71로 level별 셔플한 뒤 level 순서를 매 라운드 섞어 교차 배치했다. 처음 100편은 L1–L4 각 25편이며 중복 episode가 없다.",
        '- 전체 순서는 `design.json`에 고정했다. 같은 원본의 서로 다른 corruption variant는 서로 다른 corpus episode다.',
        f"- teacher: `gpt-6.1-sol`, reasoning `low`; two_stage; GLOBAL/KOREAN 각 최대 14 step; CHECK 없음. 완료 {result['episodes']}/100.",
        '- 동시 요청 최대 4, transport timeout 600초, retry 최대 3회(10/40/120초). 별도 공유 ledger, `--max-api-calls 6000`, confirmed cost $15 상한. 요청별 상한을 먼저 예약해 동시 호출도 상한을 넘지 않게 했다.',
        '- scorer와 Bareun은 고정된 Phase 1/2c 구현이다. scorer train/test split을 열지 않았다. 정책 생성·LoRA 학습·SFT·RFT·나머지 1,473편 실행 없음.', '',
        '## 비용과 사용량', '',
        f"실제 API 요청 {a['calls']}회, confirmed usage **${a['confirmed_usd']:.6f}**, 미정산 예약 ${a['reserved_usd']:.6f}. "
        f"편당 **${result['cost_per_episode_confirmed']:.6f}**. API 오류: `{json.dumps(result['api_errors'])}`.", '',
        '| input | output | reasoning (output에 포함) | cache read | cache write |', '|---:|---:|---:|---:|---:|',
        '|'+ '|'.join(str(result['api_tokens'].get(k,0)) for k in ('input','output','reasoning','cache_read','cache_write'))+'|', '',
        '비용은 저장 usage에 input $2/M, cache read $0.10/M, cache write $2.50/M, output $10/M을 적용한 추정이며 청구서 금액은 아니다. reasoning은 output과 중복 과금하지 않는다.', '',
        f"전체 1,573편 예상: **${result['projected_all_1573_by_level_usd']:.2f}** "
        '(level별 실측 비용 × 실제 corpus level 수). '
        f"나머지 1,473편만 추가 실행하는 예상: **${result['projected_remaining_1473_by_level_usd']:.2f}**. "
        f"단순 편당 평균 × 1,573은 ${result['projected_all_1573_simple_usd']:.2f}. 재시도와 향후 캐시 적중률 변화는 확정할 수 없다.", '',
        f"cached input share: {s['cached_share']:.2%}. corpus level 수: `{json.dumps(result['corpus_counts_by_level'])}`.", '',
        '## 보상과 행동', '',
        '| level | n | GLOBAL R | KOREAN R | combined R | combined R_rec | R_over | G steps | K steps | $/episode |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    groups={'전체':group_summary_from_stats(s),**result['by_level']}
    for level,g in groups.items():
        rr=g['rewards']
        lines.append('|'+level+'|'+str(g['n'])+'|'+'|'.join(number(x) for x in
            [rr['global']['R'],rr['korean']['R'],rr['combined']['R'],rr['combined']['R_rec'],rr['combined']['R_over'],g['steps']['global'],g['steps']['korean'],g['cost_per_episode']])+'|')
    lines+=['','operator별 역할 보상은 해당 operator가 하나 이상 있는 episode들의 평균이다. '
            '복합 손상은 여러 행에 포함되므로 operator의 인과 효과로 해석할 수 없다. record recovery는 그 operator record만의 평균이다.', '',
            '| operator | essays | GLOBAL R | KOREAN R | combined R | R_over | record recovery |', '|---|---:|---:|---:|---:|---:|---:|']
    for op,g in result['by_operator'].items():
        rr=g['rewards']
        lines.append('|'+op+'|'+str(g['n'])+'|'+'|'.join(number(x) for x in
            [rr['global']['R'],rr['korean']['R'],rr['combined']['R'],rr['combined']['R_over'],s['by_operator'].get(op,{}).get('mean')])+'|')
    lines+=['','| role | MOVE | sentence insert | sentence delete | in-sentence EDIT | UNDO | STOP |', '|---|---:|---:|---:|---:|---:|---:|']
    for role,c in result['actions'].items():
        for kind in ('accepted','attempted'):
            lines.append('|'+role+' '+kind+'|'+'|'.join(str(c[kind].get(k,0)) for k in
                         ('MOVE','sentence_insert','sentence_delete','in_sentence_EDIT','UNDO','STOP'))+'|')
    lines+=['',f"invalid-action rate: {s['invalid_actions']}/{s['action_count']} = {s['invalid_action_rate']:.2%}; role violations {s['role_violations']}.", '',
        '종료 사유: `'+json.dumps({r:v['terminations'] for r,v in s['roles'].items()},ensure_ascii=False)+'`.', '',
        '| role | mean R | mean R_rec | mean R_q | mean R_over | mean R_step |', '|---|---:|---:|---:|---:|---:|']
    for role,metrics in s['reward'].items():
        lines.append('|'+role+'|'+'|'.join(number(metrics[k]['mean']) for k in ('R','R_rec','R_q','R_over','R_step'))+'|')
    lines+=['',
        '## Pilot SFT 선별', '',
        '최신 사용자 규칙만 적용했다. 과거 spec의 별도 R≥0.4 조건이나 공통 STOP 조건을 추가하지 않았다. '
        '분위수는 선형 보간이며 경계와 같은 값은 포함한다. GLOBAL 기록 없는 경우만 STOP≤2 step 및 역할 R_over=0을 검사한다.', '',
        '| role | p60 계산 모집단 | threshold | 선별 궤적 |', '|---|---:|---:|---:|']
    for role in ('global','korean'):
        lines.append('|'+role+'|'+str(sel['threshold_population'][role])+'|'+number(sel['thresholds'][role])+'|'+str(len(sel['kept_ids'][role]))+'|')
    lines+=['','GLOBAL 선별 이유: `'+json.dumps(sel['global_reason_counts'],ensure_ascii=False)+'`.', '',
        'GLOBAL 기록 없이 선택된 궤적의 실제 행동 수: `'+json.dumps(sel['no_GLOBAL_kept_action_counts'],ensure_ascii=False)+'`. '
        '`R_over=0`은 무수정과 같은 조건이 아니다. MOVE 또는 EDIT 1회 뒤 STOP한 경우도 사용자 규칙대로 포함했다.', '',
        '## Policy 입력 형식과 토큰 길이', '',
        'pinned Kanana tokenizer의 실제 chat template를 사용했다. system prompt, 각 관찰의 profile/notices, '
        'KOREAN 첫 관찰의 hand-off를 그대로 보존했다. runtime의 `fit_history`를 그대로 호출하므로 '
        '32,768 context − 2,048 generation reserve를 넘으면 최초 system/관찰, 마지막 8 conversation message와 '
        '행동 작업 일지를 사용한다. 새 축약 방식은 추가하지 않았다.', '',
        '실제 SFT 입력은 행동별 `*.turns.jsonl.gz`다. 추론 시점의 문맥 뒤에 당시 teacher 응답을 붙이고 '
        '현재 assistant 응답과 end-of-turn에만 label을 부여한다. 이전 assistant는 문맥으로 마스킹하고 '
        '각 응답을 별도 target으로 한 번씩 포함하여 중복 loss를 피한다. 모든 system/user/profile/notices/hand-off/journal 및 역할 header는 -100이다. '
        '`*.trajectories.jsonl.gz`는 전체 대화와 모든 assistant loss를 저장한 감사용 원본이다.', '',
        '| role / 단위 | n | median | p90 | max | >8,192 |', '|---|---:|---:|---:|---:|---:|']
    for role,value in fmt['roles'].items():
        for key,label in [('full_transcript','축약 전 전체 궤적'),('inference_turns','실제 추론 형식 turn'),('max_inference_turn_per_trajectory','궤적별 최장 turn')]:
            v=value[key]
            lines.append('|'+role+' / '+label+'|'+'|'.join(f'{v[k]:.1f}' if isinstance(v[k],float) else str(v[k]) for k in ('n','median','p90','max','above_8192'))+'|')
    for role,value in fmt['roles'].items():
        lines+=['',f"{role}: 작업 일지로 축약한 turn {value['compacted_turns']}, loss 대상 토큰 {value['assistant_loss_tokens']}, 형식 오류 {len(value['format_errors'])}.",
            f"선별 궤적에 포함된 JSON parse 오류 turn {value['selected_json_parse_error_turns']}, protocol 오류 turn {value['selected_protocol_invalid_turns']}, 실행 거절 행동 {value['selected_invalid_actions']}. 사용자 선별 규칙 외에 별도 제거하지 않았으며 당시 오류 관찰과 후속 수정도 보존했다.",'']
    lines+=['8,192 초과 항목도 자르거나 버리지 않았다. 현재 추론 context는 32,768이고 기존 SFT recipe 상한은 8,192이므로, '
        '이 파일럿의 길이 분포를 보고 실제 학습 전에 길이 처리 방침을 확정해야 한다. 이 단계에서는 학습을 실행하지 않았다.', '',
        'KOREAN 선별·형식화 결과도 이번 teacher 파일럿의 진단용이다. Addendum Section 6에 따라 실제 KOREAN 학습은 '
        '나중에 학습된 GLOBAL이 만든 hand-off를 기반으로 진행하는 순서를 유지한다.', '',
        '## 관찰된 약점', '',
        f"- G_DELETE_SUPPORT: {s['by_operator']['G_DELETE_SUPPORT']['n']} records의 평균 recovery {s['by_operator']['G_DELETE_SUPPORT']['mean']:.3f}. "
        f"L_CONJ: {s['by_operator']['L_CONJ']['n']} records의 평균 recovery {s['by_operator']['L_CONJ']['mean']:.3f}. 전체 평균이 이 두 항목의 낮은 복구를 가리지 않도록 별도로 확인해야 한다.",
        '- GLOBAL 기록이 없는 50편에서는 그 역할의 기본 R_rec가 0이므로 전체 GLOBAL R의 평균만으로 역할 간 성능을 비교할 수 없다. 역할 보상과 combined 보상은 단순 합산 관계도 아니다.',
        '- 표본은 agent_train의 teacher 파일럿이다. 학습된 policy의 성능이나 held-out 일반화 성능을 측정한 결과가 아니다.', '',
        '## 검증·차이·다음 단계', '',
        '- prompt/config 변경 직후 전체 테스트: **796 passed, 25 skipped**, 2개의 기존 SWIG deprecation warnings. '
        '최초 sandbox 실행의 로컬 HTTP 소켓 권한 오류 4개는 허용된 환경의 동일 테스트에서 해소했다.',
        f"- 파일·표본·프롬프트·CHECK 비사용·비밀 정답 비노출·mask·예산 검증: **{result['validation']['count']} checks**, passed={result['validation']['passed']}.",
        '- 최종 전체 테스트: **'+(root/'tests.txt').read_text().strip().splitlines()[-1]+'**. 원시 로그: `verak/v3/outputs/phase7_pilot/tests.txt`.',
        '- 현재 corpus·분석기·채점기·기존 Phase 6 생성물은 변경하지 않았다. Phase 6 사례 파일은 새로 작성했다.',
        '- 편당 과거 대화를 계속 덧붙이는 teacher 형식은 그대로이며, SFT export에서 추론 시의 policy context 규칙을 재현했다. '
        '단일 거대 transcript를 8,192로 무조건 자르는 학습 파일을 만들지 않았다.',
        '- 범위 종료: 100편 파일럿과 선별/형식화만 수행했다. bulk teacher, SFT, RFT는 별도 승인 전 실행하지 않는다.', '',
        '## 로컬 산출물', '',
        '- `verak/v3/outputs/phase7_pilot/design.json`: 전체 고정 순서 및 실제 100편 ID.',
        '- `episodes/`, `events/`, `episodes.jsonl`: 원시 대화·행동·알림·인계·역할별 보상.',
        '- `metrics.json`, `selection.json`, `validation.json`: 수치·역할별 선별 ID·검증.',
        '- `sft_pilot/global.turns.jsonl.gz`, `sft_pilot/korean.turns.jsonl.gz`: 추론과 같은 문맥 및 input_ids/labels.',
        '- `sft_pilot/*.trajectories.jsonl.gz`, `sft_pilot/formatting.json`: 전체 궤적과 길이 통계.',
        '- `api/ledger.sqlite`, `api/requests/`, `api/accounting.json`: 원시 usage와 별도 예산 기록.',
        '- `imple/reports/V3_PHASE_6_EXAMPLES.md`: 요청된 Phase 6 사례 전체.', '']
    path=config['paths']['repo']/'imple/reports/V3_PHASE_7_PILOT.md'
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return path


def group_summary_from_stats(s):
    return {'n':s['n'],'rewards':{role:{k:v['mean'] for k,v in fields.items()} for role,fields in s['reward'].items()},
        'steps':{role:s['roles'][role]['steps']['mean'] for role in ('global','korean')},
        'cost_per_episode':s['cost']['mean']}
