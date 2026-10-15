"""Comparable reward accounting and policy-token audits for the second pilot."""
from collections import Counter, defaultdict
from statistics import mean
import json

from ..common import read_json, write_json, file_sha
from ..phase2 import read_jsonl, write_jsonl
from ..eval.api import Phase6API
from ..eval.summary import paired
from ..agent.runner import fit_history
from .pilot import safe_id
from .pilot2 import prepare, recompute_old
from .pilot2_data import PHASE, audit
from .pilot_report import action_type
from .formatting import select_roles, export, lengths

ROLES = ('global', 'korean', 'combined')


def summarize_group(rows):
    complete = [r for r in rows if r.get('completed') and r.get('reward')]
    result = {'n': len(rows), 'completed': len(complete), 'rewards': {}, 'actions': {},
              'termination': {}, 'recovery': {}, 'cost_per_episode': mean(r['cost_usd'] for r in rows) if rows else None}
    for role in ROLES:
        result['rewards'][role] = {key: mean(r['reward'][role][key] for r in complete) if complete else None
            for key in ('R', 'R_rec', 'R_q', 'R_over', 'R_step')}
        for key in ('morpheme', 'order'):
            values = [r['reward'][role]['overedit'].get(key) for r in complete]
            result['rewards'][role][key] = mean(values) if values and None not in values else None
    for role in ROLES[:2]:
        actions = [a for r in rows for a in r['actions_by_role'].get(role, [])]
        result['actions'][role] = {group: dict(Counter(action_type(a) for a in actions if group == 'attempted' or a['valid']))
            for group in ('attempted', 'accepted')}
        result['actions'][role]['invalid'] = sum(not a['valid'] for a in actions)
        result['termination'][role] = dict(Counter(r['termination'].get(role, 'runtime_error') for r in rows))
    result['non_stop_terminations'] = [{'id': r['corpus_episode_id'], 'role': role, 'reason': reason,
        'errors': [a.get('error_code') for a in r['actions_by_role'].get(role, []) if not a['valid']]}
        for r in rows for role, reason in r['termination'].items() if reason != 'STOP']
    ops = defaultdict(list)
    for r in complete:
        for record in r['reward']['combined']['per_record']:
            ops[record['op']].append(record)
    for op, records in sorted(ops.items()):
        result['recovery'][op] = {'n': len(records), 'main': mean(r['main'] for r in records),
                                  'recovery': mean(r['recovery'] for r in records)}
    return result


def runtime_lengths(rows, tokenizer):
    result = {}
    for role in ROLES[:2]:
        calls = [c for r in rows for c in r['calls'] if c['role'] == role]
        prompt, total = [], []
        for c in calls:
            prompt.append(len(tokenizer.apply_chat_template(c['messages'], tokenize=True, add_generation_prompt=True)))
            total.append(len(tokenizer.apply_chat_template(c['messages']+[{'role': 'assistant', 'content': c['raw']}], tokenize=True)))
        result[role] = {'prompt': lengths(prompt), 'prompt_plus_target': lengths(total),
            'truncation_calls': sum(c.get('history_compacted', False) for c in calls),
            'calls': len(calls),
            'truncation_episodes': sum(any(c.get('history_compacted', False) for c in r['calls'] if c['role'] == role) for r in rows)}
    return result


def validate(config, design, rows, corpus, tokenizer, formatting):
    from types import SimpleNamespace
    checks = []
    def check(name, valid):
        checks.append({'name': name, 'passed': bool(valid)})
    check('only approved pilot IDs', {r['corpus_episode_id'] for r in rows} == set(design['pilot_ids']))
    check('all episodes complete', all(r['completed'] and r['reward'] for r in rows))
    for row in rows:
        id = row['corpus_episode_id']
        check(id+': no deleted-support records', all(r['op'] != 'G_DELETE_SUPPORT' for r in corpus[id]['records']))
        check(id+': no CHECK', all(n == 0 for n in row['checks'].values()))
        for role in ROLES[:2]:
            history = row['messages_by_role'][role]
            assistant_indices = [i for i, m in enumerate(history) if m['role'] == 'assistant']
            role_calls = [c for c in row['calls'] if c['role'] == role]
            check(id+': complete raw history '+role, len(assistant_indices) == len(role_calls))
            for i, call in zip(assistant_indices, role_calls):
                t = int(call['turn'].split(':')[0])
                expected, compacted = fit_history(history[:i], SimpleNamespace(name='policy', context_limit=8192),
                    tokenizer, row['actions_by_role'][role][:t-1])
                check(id+': actual target preserved '+role+call['turn'], history[i]['content'] == call['raw'])
                check(id+': latest-profile context replay '+role+call['turn'], expected == call['messages'])
                check(id+': truncation flag replay '+role+call['turn'], compacted == call['history_compacted'])
        for c in row['calls']:
            prefix = tokenizer.apply_chat_template(c['messages'], tokenize=True, add_generation_prompt=True)
            check(id+': prompt <=7168 '+c['role']+c['turn'], len(prefix) <= 8192-1024)
            complete = tokenizer.apply_chat_template(c['messages']+
                [{'role': 'assistant', 'content': c['raw']}], tokenize=True)
            check(id+': actual prompt and target <=8192 '+c['role']+c['turn'], len(complete) <= 8192)
            check(id+': generation reserve '+c['role']+c['turn'], c['max_output_tokens'] == 1024)
            check(id+': teacher low '+c['role']+c['turn'], c['reasoning_effort'] == 'low')
            t = int(c['turn'].split(':')[0])
            context, _ = fit_history(c['messages'], SimpleNamespace(name='policy', context_limit=8192), tokenizer,
                row['actions_by_role'][c['role']][:t-1])
            check(id+': policy receives identical messages '+c['role']+c['turn'], context == c['messages'])
    for role, values in formatting['roles'].items():
        check(role+': SFT formatting successful', not values['format_errors'])
        check(role+': no inference sample above8192', values['inference_turns']['above_8192'] == 0)
    audit = read_json(config['paths'][PHASE+'_output']/'reward_audit/source_recovery_audit.json')
    check('source recovery and order audit', not audit['source_recovery_failures'] and
          audit['source_overedit_failures'] == 0 and audit['unchanged_order_overedit_failures'] == 0)
    result = {'passed': all(c['passed'] for c in checks), 'count': len(checks),
              'failed': [c for c in checks if not c['passed']]}
    write_json(config['paths'][PHASE+'_output']/'validation.json', result)
    return result


def summarize(config):
    from transformers import AutoTokenizer
    design, corpus, _ = prepare(config)
    root = config['paths'][PHASE+'_output']
    data_validation = audit(config)
    old_root = config['paths']['phase7_pilot_output']
    old_design = read_json(old_root/'design.json')
    old_corpus = {r['episode_id']: r for r in read_jsonl(config['paths']['previous_corrupt']/'agent_train.jsonl')}
    old_rows = [read_json(old_root/'episodes'/(safe_id(id)+'.json')) for id in old_design['pilot_ids']]
    updated_old = [recompute_old(row, old_corpus[row['corpus_episode_id']]) for row in old_rows]
    write_jsonl(root/'previous_pilot_new_reward.jsonl', updated_old)
    rows = [read_json(root/'episodes'/(safe_id(id)+'.json')) for id in design['pilot_ids']
            if (root/'episodes'/(safe_id(id)+'.json')).exists()]
    groups = {'old100_original_reward': old_rows, 'old100_new_reward': updated_old,
        'old_paired_new_reward': [r for r in updated_old if r['corpus_episode_id'] in design['retained_pilot_ids']],
        'new_paired': [r for r in rows if r['corpus_episode_id'] in design['retained_pilot_ids']],
        'new_drop': [r for r in rows if r['corpus_episode_id'] in design['new_drop_ids']], 'new_all': rows}
    summaries = {k: summarize_group(v) for k, v in groups.items()}
    by_level = {key: {level: summarize_group([r for r in values if r['level'] == level])
                      for level in ('L1', 'L2', 'L3', 'L4')}
                for key, values in groups.items() if key in {'old_paired_new_reward', 'new_paired', 'new_all'}}
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    selection = select_roles(rows, corpus)
    write_json(root/'selection.json', selection)
    formatting = export(config, rows, selection, tokenizer, output_key=PHASE+'_output')
    token_stats = {name: runtime_lengths(groups[name], tokenizer) for name in ('old_paired_new_reward', 'new_paired', 'new_all')}
    validation = validate(config, design, rows, corpus, tokenizer, formatting)
    api = Phase6API(config, 0, phase=PHASE)  # Read-only accounting; report cannot dispatch calls.
    account = api.accounting()
    usages = defaultdict(Counter)
    for path in (root/'api/requests').glob('*.json'):
        c = read_json(path)
        usages[c['stage']].update({k: c.get('cost', {}).get(k, 0) for k in
            ('input', 'output', 'reasoning', 'cache_read', 'cache_write', 'confirmed_usd')})
    population = Counter(r['level'] for r in corpus.values())
    level_costs = {level: by_level['new_all'][level]['cost_per_episode'] for level in population}
    projection = sum(population[l]*level_costs[l] for l in population) if all(level_costs.values()) else None
    fields = [role+'.'+metric for role in ROLES for metric in ('R', 'R_rec', 'R_over')]+['steps', 'cost']
    differences = paired(groups['new_paired'], groups['old_paired_new_reward'], fields, config)
    result = {'design': design, 'corpus': read_json(root/'corpus_manifest.json'), 'groups': summaries,
        'by_level': by_level, 'paired_differences': differences, 'tokens': token_stats,
        'previous_selected_formatting': read_json(old_root/'metrics.json')['formatting'],
        'previous_api_tokens': read_json(old_root/'metrics.json')['api_tokens'],
        'selection': selection, 'formatting': formatting, 'validation': validation,
        'api': account, 'usage_by_stage': {k: dict(v) for k, v in usages.items()},
        'projected_train_cost_by_level': projection, 'train_counts_by_level': dict(population),
        'diagnostic': read_json(root/'ending_diagnostic.json'), 'data_validation': data_validation,
        'test_summary': (root/'tests.txt').read_text(encoding='utf-8').strip().splitlines()[-1]}
    write_json(root/'metrics.json', result)
    write_jsonl(root/'episodes.jsonl', rows)
    render(config, result)
    if not validation['passed']:
        raise ValueError('Pilot artifacts failed validation; report written with failures')
    return result


def render(config, result):
    root = config['paths'][PHASE+'_output']
    def f(value, digits=5):
        return '—' if value is None else f'{value:.{digits}f}'
    def obj(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True).replace('|', '\\|')
    def table(headers, rows):
        return ['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|']+[
            '|'+'|'.join(map(str, r))+'|' for r in rows]+['']
    d, groups = result['design'], result['groups']
    delta = result['paired_differences']['combined.R']
    lines = ['# V3 Phase 7 Pilot 2', '',
        f"재실행 {groups['new_all']['completed']}/{len(d['pilot_ids'])}편. 기존 100편 중 삭제 근거가 있는 {len(d['removed_pilot_ids'])}편을 제외하고 "
        f"{len(d['retained_pilot_ids'])}편을 재실행했으며, 새 L_CONJ_DROP {len(d['new_drop_ids'])}편을 추가했다.", '',
        f"새 사례는 목표 20편보다 {d['new_shortfall']}편 부족하다. 사용자의 후속 결정에 따라 QC 통과 사례만 사용했고 추가 후보는 생성하지 않았다.", '',
        f"동일한 92편의 combined R은 {f(delta['right_mean'])} → {f(delta['left_mean'])}; "
        f"차이 {f(delta['difference'])}, bootstrap 95% CI [{f(delta['ci95'][0])}, {f(delta['ci95'][1])}]다. "
        '이번 표본에서 보상 향상은 확인되지 않았다. 문맥·프롬프트·출력 예산을 함께 바꿨으므로 축약만의 효과로 분리할 수 없다.', '',
        f"동일 표본의 combined R_over는 {f(groups['old_paired_new_reward']['rewards']['combined']['R_over'])} → "
        f"{f(groups['new_paired']['rewards']['combined']['R_over'])}로 증가했다. 문맥 상한 준수와 과수정 억제는 따로 평가해야 한다.", '',
        '## 구현과 실행 계약', '',
        '- 전체 글·Korean document profile: 단계 시작, 매 5번째 행동 뒤. 그 외 EDIT/MOVE/UNDO는 바뀐 문단 + 앞뒤 이웃 한 문장, 표지 변화 알림, 삭제 ID/문단 순서만 갱신한다.',
        '- Teacher/SFT/추론 공통 8,192토큰, 생성 1,024 예약. 넘치면 system → KOREAN 인계 → 최신 전체 프로필 → 작업 일지 → 들어가는 최근 턴. 첫 관찰을 최신 상태 대신 사용하지 않는다. 필수 내용이 넘치면 오류로 보존한다.',
        '- gpt-6.1-sol / low / two_stage / CHECK 없음. QC만 high. Bareun 분석기·프로필 필드·고정 Kanana 채점기는 변경하지 않았다.',
        '- 접속어로 기존 문장의 관계를 명료화하는 수정은 허용. 새 사실·이유·사례를 만드는 것은 계속 금지.',
        '- R_over = 0.5×형태소 거리 + 0.5×순서 거리. 레코드 직접/연동 대상은 제외한다. 남은 문장의 문단이 달라지면 1, 아니면 해당 문장의 역전 쌍 비율을 계산하여 평균한다. 순서만 다르면 정규화 Kendall 거리다. 삭제는 형태소 항에서 계산한다.',
        '- 역할 순서 비용은 해당 역할의 유효 MOVE가 새로 만든 역전에만 부과한다. 복원·UNDO·이전 역할에서 물려받은 순서 비용은 부과하지 않는다.',
        '- GLOBAL 레코드 없는 SFT 후보: STOP≤2, R_over=0과 함께 MOVE/문장 삽입/문장 삭제 시도가 없어야 한다. 실행 거절 또는 UNDO도 해당 행동으로 센다.', '',
        '## 코퍼스와 LLM-verified QC', '',
        '고정 시드 73. Section 4.3의 기존 점수·길이·장르 기준에 맞는 서로 다른 source train 300편/dev 80편. L1은 DROP만, L2는 서로 다른 문장에 DROP+WORD+TEXT를 적용했다. '
        'Phase 2c closed set을 그대로 사용하므로 예시의 `실제로`는 후보에 포함하지 않았다(기존 closed set에 없음). 모든 바뀐 문장을 Bareun으로 재분석하고 역복원 검증했다.', '']
    rows = []
    for split, info in result['corpus'].items():
        for stage in ('before', 'after_removal', 'kept_new', 'final'):
            v = info[stage]
            rows.append([split, stage, v['essays'], v['sources']]+[f(v['local_shares'][l]*100, 2)+'%' for l in ('WORD', 'SENTENCE', 'TEXT')])
    lines += table(['split', '단계', 'essays', 'sources', 'WORD', 'SENTENCE', 'TEXT'], rows)
    for split, info in result['corpus'].items():
        outside = {level: share for level, share in info['final']['local_shares'].items() if not .28 <= share <= .38}
        if outside:
            lines += [f"{split}: 최종 비율 중 이전 28–38% 구간 밖 항목은 `{obj(outside)}`다. "
                      '이번 요청의 삭제·통과 사례 추가를 그대로 적용한 결과이며, 추가 재표집은 하지 않았다.', '']
    rows = []
    for split, info in result['corpus'].items():
        for op, v in info['qc_by_operator'].items():
            rows.append([split, op, v['judged'], f(v['damage_real']/v['judged']), f(v['original_is_fix']/v['judged']), f"{v['both']}/{v['judged']}"])
        lines.append(f"{split} essay yield: **{info['kept_new']['essays']}/{info['candidates']['essays']}**. 최종 L1–L4: `{obj(info['final']['levels'])}`; 장르: `{obj(info['final']['genres'])}`; 연산자: `{obj(info['final']['operators'])}`.\n")
    lines += table(['split', 'op', 'records judged', 'damage_real', 'original_is_fix', 'both true'], rows)
    for split in result['corpus']:
        build = read_json(root/f'build_{split}.json')
        lines += [f"{split} Bareun 적용 시도: `{obj(build['attempts'])}`; 폐기·재시도 사유: `{obj(build['failures'])}`.", '']
    lines += ['## 보상 비교', '',
        '공정한 paired 비교는 동일한 92편이며 이전 trajectory의 형태소 항·복구·점수는 보존하고 새 순서 항만 추가하여 재계산했다. '
        '이전 100편의 원래 숫자는 별도 행이다. 새 전체 집계에는 추가 DROP 사례가 섞이므로 직접 paired 비교가 아니다.', '']
    rows = []
    for name, g in groups.items():
        for role in ROLES:
            v = g['rewards'][role]
            rows.append([name, g['completed'], role]+[f(v[k]) for k in ('R', 'R_rec', 'R_q', 'R_over', 'morpheme', 'order', 'R_step')])
    lines += table(['group', 'n', 'role', 'R', 'R_rec', 'R_q', 'R_over', '형태소 항', '순서 항', 'steps'], rows)
    lines += table(['paired metric', '새−이전', 'bootstrap 95% CI'], [[k, f(v['difference']),
        '['+', '.join(f(x) for x in v['ci95'])+']'] for k, v in result['paired_differences'].items()])
    rows = []
    for level in ('L1', 'L2', 'L3', 'L4'):
        for role in ROLES:
            a = result['by_level']['old_paired_new_reward'][level]
            b = result['by_level']['new_paired'][level]
            rows.append([level, b['completed'], role, f(a['rewards'][role]['R']), f(b['rewards'][role]['R']),
                f(a['rewards'][role]['R_over']), f(b['rewards'][role]['R_over']),
                f(a['rewards'][role]['morpheme']), f(b['rewards'][role]['morpheme']),
                f(a['rewards'][role]['order']), f(b['rewards'][role]['order'])])
    lines += table(['level', 'n', 'role', '이전 R', '새 R', '이전 R_over', '새 R_over',
                    '이전 형태소', '새 형태소', '이전 순서', '새 순서'], rows)
    ops = sorted({op for g in groups.values() for op in g['recovery']})
    columns = ['old100_original_reward', 'old_paired_new_reward', 'new_paired', 'new_drop']
    rows = []
    for op in ops:
        cells = [op]
        for name in columns:
            v = groups[name]['recovery'].get(op)
            cells.append(f"n={v['n']}; main={f(v['main'])}; coupled 포함={f(v['recovery'])}" if v else '—')
        rows.append(cells)
    lines += table(['operator']+columns, rows)
    if groups['new_drop']['n'] < 20:
        lines += [f"새 L_CONJ_DROP teacher 결과는 {groups['new_drop']['n']}편뿐이므로 일반적인 복구 성능을 추정할 표본으로 보기 어렵다.", '']
    lines += ['## 행동과 종료', '']
    rows = []
    for name in ('old100_original_reward', 'old_paired_new_reward', 'new_paired', 'new_all'):
        for role in ROLES[:2]:
            a = groups[name]['actions'][role]
            rows.append([name, role]+[a['accepted'].get(k, 0) for k in ('MOVE', 'sentence_insert', 'sentence_delete', 'in_sentence_EDIT', 'UNDO', 'STOP')]+[a['invalid']])
    lines += table(['group', 'role', 'MOVE', '문장 삽입', '문장 삭제', '문장내 EDIT', 'UNDO', 'STOP', '거절'], rows)
    lines += [f"새 종료 사유: `{obj(groups['new_all']['termination'])}`.", '',
        f"STOP 외 종료의 ID·역할·오류 코드: `{obj(groups['new_all']['non_stop_terminations'])}`. "
        '환경의 종료와 보상 저장 여부인 completed와 STOP 종료를 구분한다.', '',
        '## 문맥 길이와 SFT 후보', '',
        '정책 Kanana tokenizer/chat template로 센 실제 입력+현재 assistant target 길이. 새 실행의 입력은 최대 7,168토큰이며 출력용 1,024토큰을 별도로 예약한다. '
        '전체 누적 trajectory 파일은 감사용이며 실제 SFT 단위는 아래 turn별 문맥이다.', '']
    rows = []
    for group, roles in result['tokens'].items():
        for role, info in roles.items():
            v = info['prompt_plus_target']
            rows.append([group, role, v['n'], f(v['median'], 1), f(v['p90'], 1), v['max'], v['above_8192'],
                         f"{info['truncation_calls']}/{info['calls']}", info['truncation_episodes']])
    lines += table(['모든 실행 turn', 'role', 'n', 'median', 'p90', 'max', '>8192', '축약 호출', '축약 episode'], rows)
    rows = []
    for name, fmt in [('이전 selected', result['previous_selected_formatting']), ('새 selected', result['formatting'])]:
        for role, info in fmt['roles'].items():
            v = info['inference_turns']
            rows.append([name, role, info['kept_trajectories'], v['n'], f(v['median'], 1), f(v['p90'], 1), v['max'], v['above_8192']])
    lines += table(['SFT 선택 후', 'role', 'trajectories', 'turns', 'median', 'p90', 'max', '>8192'], rows)
    lines += table(['새 SFT role', 'JSON parse 오류 target', 'protocol 오류 target', '실행 거절 행동', 'format errors'],
        [[role, v['selected_json_parse_error_turns'], v['selected_protocol_invalid_turns'],
          v['selected_invalid_actions'], len(v['format_errors'])] for role, v in result['formatting']['roles'].items()])
    lines += [f"새 선택 기준 p60: `{obj(result['selection']['thresholds'])}`. GLOBAL 사유별 선택: `{obj(result['selection']['global_reason_counts'])}`. "
              'assistant target에만 loss가 있으며 teacher에 실제 전송한 메시지가 SFT prefix와 동일한지 검증했다. '
              '선택된 trajectory의 형식 오류·거절 행동도 원시 기록대로 포함되어 있다. 이 파일은 요청된 형식 검사 산출물이며 SFT 학습을 실행한 결과가 아니다.', '',
              '## Phase 6 실제 글 진단', '']
    diagnostic = result['diagnostic']
    lines += [f"기존 real 30편의 수락된 EDIT {diagnostic['accepted_EDIT_actions']}개를 캐시된 Bareun 태그와 전후 surface diff로 검사했다. "
              '변하지 않은 표면에서 태그만 달라진 사례와 문체·추측일 수 있는 `-겠-` 단독 변화는 제외했다. '
              '이 수치는 표지 변화 횟수이며 의미 오류율이 아니다. 새 프로필 필드나 API 판정을 추가하지 않았다.', '']
    lines += table(['유형', 'EDIT 수', '문장 수', 'essay 수'], [[k, v['edit_actions'], v['unique_sentences'], v['essays']] for k, v in diagnostic['counts'].items()])
    for id, sid in [('valid:5384', 'S4'), ('valid:4990', 'S3'), ('valid:4721', 'S8')]:
        value = next(r for r in diagnostic['transitions'] if r['essay_id'] == id and r['sid'] == sid)
        lines += [f"**{id} {sid}** · {value['role']} step {value['step']}", '',
            '- 전: '+value['before'], '- 후: '+value['after'], '- thought: '+value['thought'],
            '- 바뀐 시제·상 표지: `'+obj(value['changed_tense_before'])+' → '+obj(value['changed_tense_after'])+'`',
            '- 대명사→명사구: `'+obj(value['pronoun_replacements'])+'`', '']
    lines += ['## 비용·검증·산출물', '']
    lines += table(['stage', 'input', 'output', 'reasoning(출력에 포함)', 'cache read', 'cache write', '확인 비용 USD'],
        [[stage]+[v.get(k, 0) for k in ('input', 'output', 'reasoning', 'cache_read', 'cache_write')]+[f(v.get('confirmed_usd'), 6)] for stage, v in result['usage_by_stage'].items()])
    lines += ['비용은 확인된 usage에 기존 장부 단가(input $2/M, cache read $0.10/M, cache write $2.50/M, output $10/M)를 적용했다. '
              'reasoning은 output에 포함되며 중복 계산하지 않았다. 청구서 대조는 하지 않았다.', '']
    previous_tokens = result['previous_api_tokens']
    new_tokens = result['usage_by_stage'].get('teacher_pilot2', {})
    lines += [f"Teacher 입력 cache-read 비율: 이전 {f(100*previous_tokens.get('cache_read',0)/max(1,previous_tokens.get('input',0)),2)}% → "
        f"새 {f(100*new_tokens.get('cache_read',0)/max(1,new_tokens.get('input',0)),2)}%. "
        '최신 상태를 앞으로 옮기는 문맥 축약은 이전의 긴 append-only 접두부 캐시를 계속 재사용하지 않는다. '
        '실제 비용은 입력 길이뿐 아니라 캐시 토큰과 행동 수도 함께 보아야 한다.', '']
    lines += [f"공유 ledger: `{obj(result['api'])}`. QC와 재실행 합계이며 이전 파일럿 비용은 포함하지 않는다.", '',
        f"paired 비용/편: 이전 ${f(groups['old_paired_new_reward']['cost_per_episode'], 6)} → 새 ${f(groups['new_paired']['cost_per_episode'], 6)}. "
        f"새 전체 비용/편 ${f(groups['new_all']['cost_per_episode'], 6)}. 활성 train {d['corpus_count']}편의 L1–L4 가중 추정 비용 **${f(result['projected_train_cost_by_level'], 2)}**. "
        '추정은 bulk 실행 승인이 아니며 작은 추가 DROP 표본의 불확실성이 있다.', '',
        f"산출물 검증: `{obj(result['validation'])}`. 데이터 출처·문항 분리·train 해시 배제 검증: `{obj(result['data_validation'])}`. "
        f"전체 테스트 결과: **{result['test_summary']}** (`{root/'tests.txt'}`). dev reward sanity audit: `{root/'reward_audit/source_recovery_audit.json'}`.", '',
        f"활성 코퍼스: `{config['paths']['active_corrupt']}/agent_train.jsonl`, `agent_dev.jsonl`. 이전 파일은 `{config['paths']['previous_corrupt']}`에 보존.",
        f"설계·QC·API 장부·episodes·SFT 형식 샘플·상세 진단·재계산 이전 보상: `{root}`.", '',
        'Bulk teacher generation, SFT 학습, RFT는 실행하지 않았다. 이 보고서에서 중단하고 다음 승인을 기다린다.', '']
    path = config['paths']['repo']/'imple/reports/V3_PHASE_7_PILOT2.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
