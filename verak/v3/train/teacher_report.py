"""Absolute role yields and level-weighted teacher cost projections."""
from collections import Counter, defaultdict
from statistics import mean
from types import SimpleNamespace
import json

from ..common import read_json, write_json, file_sha
from ..phase2 import read_jsonl, write_jsonl
from ..agent.runner import fit_history, system_prompt
from ..eval.api import Phase6API
from ..eval.summary import paired
from .teacher_comparison import PHASE, prepare, absolute_selection
from .pilot import safe_id
from .formatting import lengths

ROLES = ('global', 'korean', 'combined')


def role_reward(row, role):
    if row.get('completed') and row.get('reward'):
        return row['reward'][role]
    return row.get('global_only_reward') if role == 'global' else None


def summary(rows, corpus):
    result = {'attempted': len(rows), 'completed': sum(r['completed'] for r in rows),
        'both_stopped': sum(all(r['termination'].get(k) == 'STOP' for k in ROLES[:2]) for r in rows),
        'cost_total': sum(r.get('confirmed_episode_cost', r['cost_usd']) for r in rows)}
    result['completion_rate'] = result['completed']/len(rows) if rows else None
    result['cost_per_episode'] = result['cost_total']/len(rows) if rows else None
    result['rewards'], result['roles'] = {}, {}
    for role in ROLES:
        values = [v for r in rows if (v := role_reward(r, role)) is not None]
        result['rewards'][role] = {'n': len(values), **{k: mean(v[k] for v in values) if values else None
            for k in ('R', 'R_rec', 'R_q', 'R_over', 'R_step')}, **{
            k: mean(v['overedit'][k] for v in values) if values else None for k in ('morpheme', 'order')}}
        if role == 'combined':
            continue
        actions = [a for r in rows for a in r['actions_by_role'].get(role, [])]
        n = len(actions)
        invalid = sum(not a['valid'] for a in actions)
        result['roles'][role] = {'actions': n, 'invalid': invalid, 'invalid_rate': invalid/n if n else None,
            'steps_per_attempted_episode': mean(r['steps'].get(role, 0) for r in rows) if rows else None,
            'steps_per_started_role': mean(r['steps'][role] for r in rows if r['steps'].get(role, 0))
                if any(r['steps'].get(role, 0) for r in rows) else None,
            'terminations': dict(Counter(r['termination'].get(role, 'not_completed') for r in rows)),
            'role_forbidden': sum(a.get('error_code') == 'role_forbidden' for a in actions)}
    selections = {r['corpus_episode_id']: absolute_selection(r, corpus[r['corpus_episode_id']]) for r in rows}
    result['kept'] = {role: sum(s[role] for s in selections.values()) for role in ROLES[:2]}
    result['global_kept_by_rule'] = dict(Counter(s['global_rule'] for s in selections.values() if s['global']))
    records = defaultdict(list)
    for row in rows:
        if not row['completed'] or not row.get('reward'):
            continue
        for rec in row['reward']['combined']['per_record']:
            records[rec['op']].append(rec)
    result['record_recovery'] = {op: {'n': len(v), 'main': mean(x['main'] for x in v),
        'recovery': mean(x['recovery'] for x in v)} for op, v in records.items()}
    result['selection'] = selections
    return result


def projection(levels, population):
    if any(not levels[l]['attempted'] for l in population):
        return None
    return {'cost_usd': sum(n*levels[l]['cost_per_episode'] for l, n in population.items()),
        'global': sum(n*levels[l]['kept']['global']/levels[l]['attempted'] for l, n in population.items()),
        'korean': sum(n*levels[l]['kept']['korean']/levels[l]['attempted'] for l, n in population.items()),
        'global_record_threshold': sum(n*levels[l]['global_kept_by_rule'].get('GLOBAL_R_ge_0.80', 0)/
                                      levels[l]['attempted'] for l, n in population.items()),
        'global_no_record_stop': sum(n*levels[l]['global_kept_by_rule'].get('no_GLOBAL_STOP_rule', 0)/
                                   levels[l]['attempted'] for l, n in population.items())}


def audit(config, design, groups, tokenizer, account):
    failures, checks = [], 0
    def check(ok, name):
        nonlocal checks
        checks += 1
        if not ok:
            failures.append(name)
    check(account['confirmed_usd'] <= 5 and account['pending'] == 0, 'shared $5 cap and all calls settled')
    tokens = {}
    for name, rows in groups.items():
        check(len({r['corpus_episode_id'] for r in rows}) == len(rows), name+': no duplicate episodes')
        check({r['corpus_episode_id'] for r in rows} <= set(design['pilot_ids']), name+': exact authorized IDs only')
        stats = defaultdict(list)
        truncated = Counter()
        for row in rows:
            id = row['corpus_episode_id']
            check(row['mode'] == 'two_stage' and all(v == 0 for v in row['checks'].values()), name+id+': mode and no CHECK')
            for role, history in row['messages_by_role'].items():
                calls = [c for c in row['calls'] if c['role'] == role]
                indices = [i for i, m in enumerate(history) if m['role'] == 'assistant']
                check(len(calls) == len(indices), name+id+role+': full raw history')
                for call, index in zip(calls, indices):
                    prefix = tokenizer.apply_chat_template(call['messages'], tokenize=True, add_generation_prompt=True)
                    total = tokenizer.apply_chat_template(call['messages']+[{'role': 'assistant', 'content': call['raw']}], tokenize=True)
                    t = int(call['turn'].split(':')[0])
                    replay, compacted = fit_history(history[:index], SimpleNamespace(context_limit=8192, generation_reserve=1024),
                        tokenizer, row['actions_by_role'][role][:t-1])
                    check(replay == call['messages'] and compacted == call['history_compacted'], name+id+role+call['turn']+': exact context replay')
                    check(len(prefix) <= 7168 and len(total) <= 8192, name+id+role+call['turn']+': policy token budget')
                    check(call['messages'][0]['content'] == system_prompt(role), name+id+role+call['turn']+': same prompt')
                    check(call['max_output_tokens'] == 1024, name+id+role+call['turn']+': same output cap')
                    check(call['reasoning_effort'] == ('medium' if name.endswith('medium') else 'low'), name+id+': effort')
                    check(call['model'] == (design['model'] if name.startswith('luna') else 'gpt-6.1-sol'), name+id+': model')
                    stats[role].append(len(total))
                    truncated[role] += call['history_compacted']
        tokens[name] = {role: {'lengths': lengths(stats[role]), 'truncated': truncated[role]} for role in ROLES[:2]}
    provenance = read_json(config['paths'][PHASE+'_output']/'diagnostic_provenance.json')
    for path, expected in provenance['files'].items():
        check(file_sha(path) == expected, 'prior log unchanged: '+path)
    # Incomplete API responses never become action turns; their inputs still
    # have to obey the exact same context and model contract.
    request_count = 0
    for path in (config['paths'][PHASE+'_output']/'api/requests').glob('*.json'):
        request = read_json(path)
        request_count += 1
        length = len(tokenizer.apply_chat_template(request['messages'], tokenize=True, add_generation_prompt=True))
        check(length <= 7168 and request['max_output_tokens'] == 1024, str(path)+': API context budget')
        check(request['model'] == design['model'], str(path)+': pinned model')
        check(request['reasoning_effort'] == request['stage'].removeprefix('luna_'), str(path)+': reasoning setting')
    return {'passed': not failures, 'checks': checks, 'failures': failures, 'tokens': tokens,
            'all_api_inputs_checked': request_count}


def report(config):
    from transformers import AutoTokenizer
    design, corpus = prepare(config)
    root = config['paths'][PHASE+'_output']
    partial = read_json(root/'completed_global_evaluation.json')
    groups = {'sol_low': [read_json(config['paths']['phase7_pilot2_output']/'episodes'/(safe_id(id)+'.json'))
                           for id in design['pilot_ids']]}
    for effort in design['efforts']:
        name = 'luna_'+effort
        rows = [read_json(root/name/'episodes'/(safe_id(id)+'.json')) for id in design['pilot_ids']
                if (root/name/'episodes'/(safe_id(id)+'.json')).exists()]
        for row in rows:
            key = name+':'+row['corpus_episode_id']
            if key in partial['results']:
                row['global_only_reward'] = partial['results'][key]['global_only_reward']
        groups[name] = rows
    population = dict(Counter(r['level'] for r in corpus.values()))
    overall = {k: summary(v, corpus) for k, v in groups.items()}
    by_level = {k: {l: summary([r for r in v if r['level'] == l], corpus) for l in ('L1','L2','L3','L4')}
                for k, v in groups.items()}
    ops = sorted({r['op'] for row in corpus.values() for r in row['records']})
    by_op = {k: {op: summary([r for r in v if any(x['op'] == op for x in corpus[r['corpus_episode_id']]['records'])], corpus)
                  for op in ops} for k, v in groups.items()}
    projections = {k: projection(v, population) for k, v in by_level.items()}
    candidates = [k for k in ('luna_low', 'luna_medium') if projections[k] is not None]
    best = max(candidates, key=lambda k: (projections[k]['global']+projections[k]['korean'],
        overall[k]['rewards']['combined']['R'] if overall[k]['rewards']['combined']['R'] is not None else float('-inf'),
        -projections[k]['cost_usd'])) if candidates else None
    mixed = projection({l: by_level[best if l in ('L1','L2') else 'sol_low'][l] for l in population}, population) if best else None
    api = Phase6API(config, 0, phase=PHASE)
    account = api.accounting()
    usage, errors = defaultdict(Counter), []
    for path in (root/'api/requests').glob('*.json'):
        r = read_json(path)
        usage[r['stage']].update({k: r.get('cost', {}).get(k, 0) for k in
                                 ('input','output','reasoning','cache_read','cache_write','confirmed_usd')})
        if r['status'] != 'completed':
            errors.append({k: r.get(k) for k in ('phase_call','stage','item_id','status','error_type','http_status','usage')})
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    validation = audit(config, design, groups, tokenizer, account)
    for condition in ('luna_low','luna_medium'):
        difference = overall[condition]['cost_total']-usage[condition]['confirmed_usd']
        if abs(difference) > 1e-7:
            validation['passed'] = False
            validation['failures'].append(condition+': API cost attribution mismatch')
    fields = [role+'.'+metric for role in ROLES for metric in ('R','R_rec','R_over')]+['steps','cost']
    paired_results = {name: paired(groups[name], groups['sol_low'], fields, config) for name in ('luna_low','luna_medium')}
    result = {'design': design, 'corpus': read_json(root/'corpus_manifest.json'), 'overall': overall,
        'by_level': by_level, 'by_operator': by_op, 'population': population,
        'projections': projections, 'best_luna': best, 'mixed_projection': mixed, 'paired_completed_only': paired_results,
        'api': account, 'usage': {k: dict(v) for k, v in usage.items()}, 'api_errors': errors,
        'partial_global_evaluation': partial, 'validation': validation,
        'test_summary': (root/'tests.txt').read_text().strip().splitlines()[-1]}
    write_json(root/'metrics.json', result)
    write_json(root/'validation.json', validation)
    for name, rows in groups.items():
        write_jsonl(root/name/'evaluated_episodes.jsonl', rows)
    render(config, result)
    if not validation['passed']:
        raise ValueError('Comparison artifacts failed validation; see report')
    return result


def render(config, m):
    root = config['paths'][PHASE+'_output']
    def f(v, d=4):
        return '—' if v is None else f'{v:.{d}f}'
    def pct(v):
        return '—' if v is None else f'{100*v:.1f}%'
    def obj(v):
        return json.dumps(v, ensure_ascii=False).replace('|', '\\|')
    def table(headers, rows):
        return ['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|']+[
            '|'+'|'.join(str(v).replace('\n',' ') for v in row)+'|' for row in rows]+['']
    lines = ['# V3 Phase 7 — teacher comparison', '', '## 범위와 확정 결정', '',
        'Pilot 2의 8,192/1,024 문맥, 역할 프롬프트, 5행동마다 전체 프로필, 환경·보상은 유지했다. '
        'L_CONJ_DROP 통과 사례를 train/dev에서 하나씩 제외하고 생성·QC·재추가 진입점을 비활성화했다. '
        '시제·상 프로필 필드는 추가하지 않았다. 이전 데이터·판정·trajectory는 보존했다.', '']
    sol = m['overall']['sol_low']; low = m['overall']['luna_low']; medium = m['overall']['luna_medium']
    lines += [f"비교 결과: 환경 종료와 보상 저장까지 완료한 글은 Sol {sol['completed']}/92, "
        f"Luna low {low['completed']}/{low['attempted']}, Luna medium {medium['completed']}/{medium['attempted']}편이다. "
        f"절대 기준을 통과한 GLOBAL/KOREAN 궤적은 각각 {sol['kept']['global']}/{sol['kept']['korean']}, "
        f"{low['kept']['global']}/{low['kept']['korean']}, {medium['kept']['global']}/{medium['kept']['korean']}개다. "
        f"이번 Luna 비교의 확인 비용은 ${m['api']['confirmed_usd']:.6f}다.", '',
        'L1 하락 3건은 모두 수정할 대상의 최신 글과 프로필을 볼 수 있었던 사례다. '
        '표시 주기가 원인이라는 증거는 확인하지 못했으며 관찰 형식은 그대로 유지했다.', '']
    lines += table(['split','이전','제거','현재','WORD','SENTENCE','TEXT'], [[k,v['before']['essays'],len(v['removed_ids']),
        v['after']['essays']]+[pct(v['after']['local_shares'][l]) for l in ('WORD','SENTENCE','TEXT')] for k,v in m['corpus'].items()])
    lines += ['train의 SENTENCE 비중은 약 27.9%로 과거 균형 목표 하한 28%보다 조금 낮다. '
        '이번 지시에 따라 통과 사례 두 편만 제거했으며 추가 제거·복제·재균형은 하지 않았다.', '']
    lines += ['## L1 진단 — 저장 로그만 사용', '']
    diag = read_json(root/'l1_diagnostic.json'); s = diag['summary']
    lines += ['동일 25편/25기록. Pilot 1 보상은 새 과수정 계산식으로 다시 집계했다. 복구 판정과 API 출력은 그대로다. '
        'GPT·Bareun·채점기 호출 없이 저장 관찰과 실제 전송 메시지를 읽었다.', '']
    lines += table(['pilot','KOREAN R','R_rec','R_q','R_over','steps'], [[k]+[f(v[a]) for a in
        ('R','R_rec','R_q','R_over','R_step')] for k,v in s['roles'].items()])
    lines += [f"기록 전이: `{obj(s['transitions'])}`. 이전 원래 계산식 수치: `{obj(s['pilot1_original_formula'])}`.", '']
    lines += table(['operator','n','pilot1 recovery','pilot2 recovery'], [[op,v['n'],f(v['pilot1']),f(v['pilot2'])]
        for op,v in s['operators'].items()])
    lines += table(['pilot','수락된 대상 EDIT','직전 관찰 전체','전체 프로필 문맥 보유','대상 프로필 최신','대상 글 최신','미시도 기록','GLOBAL 뒤 대상 소실'],
        [[k]+[s[k+'_visibility'][x] for x in ('accepted_target_EDITS','immediate_observation_full','full_profile_in_context',
          'target_profile_current','target_text_current','unattempted_records','missing_site_after_global')]
          for k in ('pilot1','pilot2')])
    records = read_jsonl(root/'l1_per_record.jsonl')
    def edits(v):
        return '; '.join(f"t{e['t']}{'' if e['valid'] else '!'}:직전{'전체' if e['visibility']['immediate_observation_full'] else '부분'}/"
            f"전체보유={e['visibility']['full_profile_in_context']}/최신={e['visibility']['target_profile_current']}" for e in v['edits']) or '없음'
    lines += table(['episode / record','op','public sid','복구 1→2','K R 1→2','pilot1 수정 직전','pilot2 수정 직전'],
        [[r['record_id'],r['op'],','.join(r['pilot2']['public_sids']),
          f"{r['pilot1']['main_recovery']}→{r['pilot2']['main_recovery']}",f"{f(r['pilot1']['R'])}→{f(r['pilot2']['R'])}",
          edits(r['pilot1']),edits(r['pilot2'])] for r in records])
    lines += ['`!`는 거절된 시도다. “최신”은 축약 전 누적 로그에서 가장 최근에 관찰한 해당 문장 프로필/글과 실제 전송 내용이 같은지를 뜻한다. '
        '표지는 정답 여부를 뜻하지 않는다. 세 하락 사례 모두 대상 글과 프로필을 실제로 보았고, 수정 직전 문맥 축약도 없었다. '
        '현재 자료는 주기적 프로필의 누락을 원인으로 지지하지 않는다. 주의 분산 효과는 이 비통제 비교로 배제할 수 없다.', '',
        '조건부 최소 변경 제안(미적용): 별도 대조 실험에서 표시 주기의 인과 효과가 확인된다면 전체 프로필 주기만 5→3으로 줄인다. '
        '이번 비교에는 적용하지 않았으며 8,192토큰 문맥 규칙을 유지했다.', '']
    for r in diag['examples']:
        lines += [f"### 사례 {r['id']} · {r['op']}", '']
        lines += [f"원천 {sid}: {text}" for sid,text in r['source'].items()]+['']
        for k in ('pilot1','pilot2'):
            v = r[k]
            lines += [f"**{k}** · 복구 {v['main_recovery']}; K R {f(v['R'])}", '', '구조 수정 뒤 인접 문맥:', '']
            lines += [f"> **{s['sid']}** {s['text']}" for s in v['context_after_global']]+['']
            lines += [f"최종 {sid}: **{text}**" for sid,text in v['final'].items()]+['']
            for a in v['global_actions']:
                if a['action'] != 'STOP':
                    lines += [f"- GLOBAL t{a['t']} {a['action']} `{obj(a['args'])}`: {a['thought']}"]
            for e in v['edits']:
                lines += [f"- KOREAN t{e['t']} EDIT `{e['args']['target']}`: {e['thought']}; 직전 전체={e['visibility']['immediate_observation_full']}, "
                          f"전체 보유={e['visibility']['full_profile_in_context']}, 대상 최신={e['visibility']['target_profile_current']}."]
            lines += ['']
    lines += ['하락 3건 중 L_SPACING 2건은 `것 이다`가 남아서가 아니라, 넓은 문장 재작성으로 `것이다`라는 복구 문자열도 사라져 0이 된 사례다. '
        '이는 현재의 문자열 복구 조건에 따른 결과이며 두 문장의 의미 보존이나 품질이 검증됐다는 뜻은 아니다. '
        'L_CONJ 1건은 원천 ADVERSATIVE 대신 RESTATEMENT를 선택했다. 두 실행은 GLOBAL 단계 뒤 선행 문장도 다르므로 '
        '프로필 표시 빈도만으로 이 차이를 설명할 수 없다. 보상이나 프롬프트는 고치지 않았다.', '', '## 표지 변화 알림과 후속 수정', '',
        '분모는 수락된 GLOBAL MOVE/문장 삽입/문장 삭제 행동이다. 한 행동이 같은 유형의 알림을 여러 개 만들어도 한 번 센다. '
        '후속 수정률의 분모는 해당 유형의 알림을 낸 행동이며, 알림 대상 문장 하나 이상에 KOREAN의 수락된 EDIT가 있었는지를 센다. '
        'UNDO로 되돌린 수정도 “수정 시도 실행”에는 포함한다. 알림 발생은 손상 판정이 아니며 후속 EDIT는 복구 성공 판정이 아니다.', '']
    disturbance = read_json(root/'marker_disturbance.json')
    lines += table(['cohort','essays','구조 행동','알림 행동','비율','이후 K EDIT','후속 비율','거절 행동'],
        [[k,v['episodes'],v['accepted']['actions'],v['accepted']['with_notice'],pct(v['accepted']['notice_share']),
          v['accepted']['later_korean_edited_actions'],pct(v['accepted']['later_edit_share_among_notice_actions']),v['rejected_structural_actions']]
          for k,v in disturbance.items()])
    lines += table(['cohort','notice type','알림 행동/전체','알림률','후속 K EDIT/알림 행동','후속률','수정 대상문장쌍/알림 문장쌍'],
        [[k,t,f"{v['with_notice']}/{v['actions']}",pct(v['notice_share']),
          f"{v['later_korean_edited_actions']}/{v['with_notice']}",pct(v['later_edit_share_among_notice_actions']),
          f"{v['later_edited_action_sentence_pairs']}/{v['action_sentence_pairs']}"]
          for k,values in disturbance.items() if k in ('pilot1','pilot2','phase6_real') for t,v in values['by_type'].items()])
    lines += table(['cohort','action','수락','알림','알림률','후속 K EDIT','후속률'],
        [[k,t,v['actions'],v['with_notice'],pct(v['notice_share']),v['later_korean_edited_actions'],
          pct(v['later_edit_share_among_notice_actions'])] for k,values in disturbance.items()
          if k in ('pilot1','pilot2','phase6_real') for t,v in values['by_action'].items()])
    lines += ['## Teacher 비교', '',
        f"Phase 4의 정확 ID `{m['design']['model']}`를 재사용했다. 당시 날짜 snapshot은 없었다. Sol low는 Pilot 2의 동일 92편을 재실행 없이 재사용했다. "
        'Luna low/medium을 같은 essay 순서로 교차 실행했다. 최대 동시 요청 4, 공유 $5 비용 장부, `--max-api-calls 8000`, 출력 한도 1,024이며 '
        '환경·시스템 프롬프트·문맥 규칙 hash가 Pilot 2와 같은지 검사했다.', '',
        '완료는 환경 종료 및 보상 저장이다. 행동 오류 누적 종료와 STOP을 별도로 센다. API 중도 종료는 미완료이며 성공으로 바꾸지 않았다. '
        'KOREAN 호출이 실패했어도 이미 종료한 GLOBAL 단계는 별도 로컬 채점으로 역할 보상을 계산했다. '
        '완료된 5개 기준 에피소드에서 이 계산이 저장 GLOBAL 보상과 같은지 확인했다. KOREAN/combined 평균에는 보상이 없는 미완료 글을 넣지 않는다. '
        '조건별 평균의 n을 함께 보아야 한다. 절대 기준 통과율과 비용 추정에서는 미완료 역할도 시도 횟수 분모에 포함한다.', '']
    lines += table(['setting','attempted/92','completed','completion','두 역할 STOP','G invalid/actions','K invalid/actions','G steps','K steps','USD/episode'],
        [[k,f"{v['attempted']}/92",v['completed'],pct(v['completion_rate']),v['both_stopped'],
          f"{v['roles']['global']['invalid']}/{v['roles']['global']['actions']} ({pct(v['roles']['global']['invalid_rate'])})",
          f"{v['roles']['korean']['invalid']}/{v['roles']['korean']['actions']} ({pct(v['roles']['korean']['invalid_rate'])})",
          f(v['roles']['global']['steps_per_attempted_episode']),f(v['roles']['korean']['steps_per_attempted_episode']),f(v['cost_per_episode'],6)] for k,v in m['overall'].items()])
    lines += table(['setting','role','reward n','R','R_rec','R_q','R_over','형태소','순서','종료 사유'],
        [[k,r,v['rewards'][r]['n']]+[f(v['rewards'][r][x]) for x in ('R','R_rec','R_q','R_over','morpheme','order')]+
          [obj(v['roles'][r]['terminations']) if r != 'combined' else '—'] for k,v in m['overall'].items() for r in ROLES])
    lines += ['두 조건 모두 완료한 동일 essay들만의 paired 비교도 제시한다. 이는 실패율을 제외한 부분집합 비교이므로 '
              '전체 92편의 운영 성능과 함께 읽어야 한다.', '']
    lines += table(['Luna setting','metric','paired n','Luna','Sol','Luna−Sol','bootstrap 95% CI'],
        [[setting,metric,v['n'],f(v.get('left_mean')),f(v.get('right_mean')),f(v['difference']),
          '['+', '.join(f(x) for x in v['ci95'])+']' if v['ci95'] else '—']
          for setting,values in m['paired_completed_only'].items() for metric,v in values.items()
          if metric in ('global.R','korean.R','combined.R','combined.R_over','steps','cost')])
    comparison = m['paired_completed_only']['luna_low']['combined.R']
    if comparison['n'] and low['cost_per_episode']:
        lines += [f"Luna low의 편당 확인 비용은 Sol의 약 1/{sol['cost_per_episode']/low['cost_per_episode']:.1f}이다. "
            f"두 조건 모두 완료한 {comparison['n']}편에서 combined R 차이는 {comparison['difference']:.4f}, "
            f"95% 구간은 [{comparison['ci95'][0]:.4f}, {comparison['ci95'][1]:.4f}]이다. "
            '총보상에는 과수정 비용도 포함되므로 연산자별 복구율을 함께 보아야 한다.', '']
    def grouped_table(values, label):
        return table(['setting',label,'attempted','reward n G/K/C','G R','K R','combined R','R_over','형태소','순서'],
            [[k,g,v['attempted'],'/'.join(str(v['rewards'][r]['n']) for r in ROLES)]+[f(v['rewards'][r]['R']) for r in ROLES]+
             [f(v['rewards']['combined'][x]) for x in ('R_over','morpheme','order')]
             for k,groups in values.items() for g,v in groups.items()])
    lines += ['### L1–L4', '']+grouped_table(m['by_level'],'level')
    lines += ['### 연산자별', '', '복합 손상 글은 여러 operator 행에 포함된다. 해당 operator의 인과 효과나 독립 실험은 아니다. '
        '동일 92편에는 L_POLARITY 사례가 없으므로 이 연산자의 teacher 성능은 이번 비교로 추정하지 않는다.']+['']+grouped_table(m['by_operator'],'operator')
    lines += table(['setting','operator','완료 기록 수','main recovery','coupled 포함'],
        [[k,op,v['n'],f(v['main']),f(v['recovery'])] for k,info in m['overall'].items() for op,v in sorted(info['record_recovery'].items())])
    lines += ['## 절대 기준과 코퍼스 전체 추정', '',
        'GLOBAL 기록이 있으면 GLOBAL R≥0.80. 없으면 STOP≤2, R_over=0, MOVE/문장 삽입/삭제 시도 없음. KOREAN R≥0.80. '
        'percentile은 사용하지 않았다. 아래는 보상 기준을 통과하는 역할 trajectory 수이며 SFT 학습 실행이나 의미 보존 검증은 아니다.', '']
    lines += table(['setting','G GLOBAL 기준','G no-GLOBAL STOP','G 합계','K 합계'],
        [[k,v['global_kept_by_rule'].get('GLOBAL_R_ge_0.80',0),v['global_kept_by_rule'].get('no_GLOBAL_STOP_rule',0),v['kept']['global'],v['kept']['korean']]
         for k,v in m['overall'].items()])
    lines += [f"사전 고정한 best Luna 기준: `{m['design']['best_luna_rule']}`. 선택된 설정: **{m['best_luna']}**. "
        f"전체 활성 train level 수는 `{obj(m['population'])}`. level별 편당 비용/통과율에 실제 level 수를 곱해 합산했다.", '']
    plans = {'all Sol': m['projections']['sol_low']}
    if m['best_luna']:
        plans['all Luna ('+m['best_luna']+')'] = m['projections'][m['best_luna']]
        plans['Luna L1–L2 + Sol L3–L4'] = m['mixed_projection']
    lines += table(['계획','예상 USD','예상 G 합계','G GLOBAL 기준','G no-GLOBAL STOP','예상 K'],
        [[k]+[f(v[x],2) for x in ('cost_usd','global','global_record_threshold','global_no_record_stop','korean')]
         for k,v in plans.items() if v is not None])
    lines += ['추정치는 반올림 전 기대 개수이며 확정적인 최종 데이터 수가 아니다. level별 19–25편의 작은 표본, '
        '낮은 빈도의 연산자, 캐시 적중률에 영향을 받는다. 두 Luna 설정의 교차 실행 중 공유된 prefix 캐시도 비용에 반영돼 있다. '
        'L1–L2 혼합안은 두 단계 모두 Luna, L3–L4는 두 단계 모두 Sol을 쓴다는 뜻이다. bulk 실행은 하지 않았다.', '',
        '## 문맥·실패·비용 검증', '']
    lines += table(['setting','role','turns','median','p90','max','>8192','축약'],
        [[k,r,v['lengths']['n'],f(v['lengths']['median'],1),f(v['lengths']['p90'],1),v['lengths']['max'],
          v['lengths']['above_8192'],v['truncated']] for k,roles in m['validation']['tokens'].items() for r,v in roles.items()])
    lines += table(['setting','input','output','reasoning(출력 포함)','cache read','cache write','USD'],
        [[k]+[v.get(x,0) for x in ('input','output','reasoning','cache_read','cache_write')]+[f(v.get('confirmed_usd'),6)]
         for k,v in m['usage'].items()])
    incomplete_usage = Counter((r['stage'], (r.get('usage') or {}).get('output_tokens')) for r in m['api_errors'])
    lines += [f"공유 ledger: `{obj(m['api'])}`. API 비완료 응답 {len(m['api_errors'])}건은 `api_errors`에 보존했다. "
        '비용에는 실패 응답의 확인된 usage도 들어간다. Phase 4와 같은 단가(Luna input/read/write/output: $0.10/$0.01/$0.125/$0.50 per M)를 '
        '확인된 usage에 적용했으며 청구서 대조는 아니다. Sol 기존 비용은 이번 $5 예산에 다시 포함하지 않았다.', '',
        '비완료 응답의 setting/output token/count: `'+obj([{'setting':k[0], 'output_tokens':k[1], 'n':v}
            for k,v in incomplete_usage.items()])+'`. 0토큰 응답의 원인은 이 기록으로 확정하지 않는다. '
        'invalid-action 비율은 환경에 제출된 행동 중 거절 비율이며 API 미완료는 완료율과 이 표에 별도로 반영했다.', '',
        'generation 한도 1,024는 API의 내부 reasoning 토큰도 포함한다. 따라서 medium의 미완료 응답은 '
        '현재 한도 아래의 운영 결과로 해석해야 하며 모델 자체의 최대 추론 능력 평가로 일반화할 수 없다. '
        '출력 한도를 늘리거나 실패 응답을 재생성하지 않았다.', '',
        f"완료 GLOBAL 별도 평가: {m['partial_global_evaluation']['completed_stages']}개; 기준 일치 검사 "
        f"{len(m['partial_global_evaluation']['reference_checks'])}개. 검증: passed={m['validation']['passed']}, "
        f"checks={m['validation']['checks']}, failures=`{obj(m['validation']['failures'])}`.", '',
        f"테스트: **{m['test_summary']}**. 최초 sandbox에서는 기존 웹 테스트 4개가 localhost 소켓 권한으로 실패하여 "
        '권한을 허용한 동일 테스트를 재실행했다. 원래 로그는 `tests_sandbox.txt`에 보존했다.', '',
        f"활성 코퍼스: `{config['paths']['active_corrupt']}`. 상세 per-record/visibility/marker 자료, 실제 요청·응답·장부, "
        f"부분 GLOBAL 평가, 비교 metrics/validation: `{root}`.", '',
        '프로필/표시주기/복구 정의를 변경하지 않았다. Bulk teacher generation, SFT, RFT는 실행하지 않았다. 이 보고서에서 중단한다.', '']
    (config['paths']['repo']/'imple/reports/V3_PHASE_7_TEACHER.md').write_text('\n'.join(lines), encoding='utf-8')
