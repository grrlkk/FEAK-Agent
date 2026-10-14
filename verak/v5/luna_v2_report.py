"""Render preparation evidence and machine/human evaluation status honestly."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import statistics

import numpy as np

from verak.v4.common import REPO, read_json, write_json, file_sha
from .luna_v2_data import ROOT, CONDITIONS
from .luna_v2_env import PHI
from .luna_v2_eval import edit_cases, paired


def accounting():
    path = ROOT/'api/ledger.sqlite'
    if not path.exists(): return {'confirmed':0,'reserved':0,'calls':0}
    db = sqlite3.connect(path)
    rows = db.execute('SELECT kind,status,confirmed,reserved FROM calls').fetchall(); db.close()
    return {'confirmed':sum(r[2] for r in rows),'reserved':sum(r[3] for r in rows),'calls':len(rows),
            'providers':{k:{'calls':sum(r[0]==k for r in rows),'confirmed':sum(r[2] for r in rows if r[0]==k)} for k in ['luna','claude']}}


def estimate(records, selection):
    # Projection, not accounting. Reserve actual requests separately in the
    # durable ledger. Explicit assumed Claude output sizes avoid false precision.
    if not records: return {}
    ratio = statistics.mean(x['characters'] for x in selection['eval50'])/statistics.mean(x['characters'] for x in selection['prep10'])
    costs = [sum(c['cost_usd'] for c in r['calls']) for r in records]
    generator = statistics.mean(costs)*ratio*50*3*2
    baselines = sum((x['characters']*1.3+700)*.05/1e6+(x['characters']*1.3)*.25/1e6 for x in selection['eval50'])*2
    feedback = 0.; essay = 0.
    for meta in selection['eval50']:
        row = read_json(ROOT/'sources'/(meta['source_id'].replace(':','_')+'.json'))
        feedback += ((len(row['text'])+len(row['feedback']))*1.3+500)/1e6+700*5/1e6
        essay += 5*((2*len(row['text'])*1.3+800)/1e6+600*5/1e6)
    changes = [len([c for c in edit_cases(r['original'],r['final'])[0] if c['judgable']]) for r in records]
    n_links = int(np.ceil(statistics.mean(changes)*ratio*250))
    local_context_chars = []
    for r in records:
        local_context_chars.extend(len(c['prompt']) for c in edit_cases(r['original'],r['final'])[0] if c['judgable'])
    link = n_links*((statistics.mean(local_context_chars)*1.3 if local_context_chars else 1000)/1e6+100*5/1e6)
    return {'usd':{'generation_BCD_including_one_retry':generator,'generation_A_Aplus_batch':baselines,
                   'Claude_feedback_batch':feedback,'Claude_essay_batch':essay,'Claude_link_batch':link,
                   'projected_total_new_calls':generator+baselines+feedback+essay+link},
            'assumptions':{'eval_prep_length_ratio':ratio,'estimated_link_requests':n_links,
                           'Claude_tokens_per_character':1.3,'Claude_output_tokens_feedback':700,
                           'Claude_output_tokens_essay':600,'Claude_output_tokens_link':100},
            'not_a_guarantee':'Durable reservations enforce the $5 cap even if this projection underestimates usage.'}


def report():
    sel = read_json(ROOT/'selection.json')
    prep = read_json(ROOT/'preparation.json') if (ROOT/'preparation.json').exists() else None
    records = [read_json(p) for p in sorted((ROOT/'prep_discovery/B').glob('*/result.json'))]
    counts, actions, parses, reasons, calls = Counter(), Counter(), Counter(), Counter(), defaultdict(list)
    for r in records:
        counts.update(r['notices_by_type']); actions.update(r['korean_actions'])
        for c in r['calls']: calls[c['role']].append(c)
        for line in (ROOT/'prep_discovery/B'/r['essay_id'].replace(':','_')/'events.jsonl').read_text().splitlines():
            e = json.loads(line)
            if e['kind']=='plan_parse': parses['success' if e['passed'] else 'failure'] += 1
            if e['kind']=='attempt': reasons.update(k for k,_ in e['reasons'])
    money = accounting(); projection = estimate(records,sel)
    write_json(ROOT/'cost_projection.json',projection)
    body = ['# Luna 프롬프트 전용 구조 비교 — v2', '',
            '## 현재 상태', '',
            f"준비 탐색 {len(records)}/10편 완료. 본실험 완료 {len(list((ROOT/'main').glob('*/*/result.json')))}/250개 출력. 학습은 실행하지 않았고 최종 테스트 원문은 열지 않았다.", '',
            '준비 탐색은 δ와 수정량 상한을 측정하기 위해 품질 검사 결과를 기록하면서 첫 후보를 채택했다. 이를 검사 있는 B의 성능 결과로 해석하지 않는다.', '',
            '## 사전 고정 설계', '',
            'A: 한 번에 수정; A+: 연결 확인 지시 추가; B: 전체 구조; C: 한국어 호출과 대상 선정은 유지하고 알림 내용 제거; D: 품질 탈락·재시도 제거. N=1, k=1, 최대 3문단. B의 원문 문단 유지율이 50%를 넘으면 N=3으로 B/C/D를 추가 실행한다. 추가 실행도 별도 승인이 없으면 총 $5 안에서만 수행한다.', '',
            'C는 동일한 현재 상태에서 B와 완전히 같은 문장·문맥을 받고 알림만 빠진다. 서로 다른 수리나 채택으로 상태가 달라진 뒤에는 후속 문맥도 달라질 수 있다. 이 경우 모든 호출의 문맥이 동일하다고 주장하지 않는다.', '',
            '## 데이터', '',
            f"train 64,017편에서 준비 10편, valid 8,000편에서 평가 50편을 선정했다. 문단 있는 valid에서 기존 학습 ID·해시와 겹치는 {sel['counts']['valid_excluded_training_overlap']}편을 제외했다. 남은 후보는 {sum(sel['eligible']['valid'].values()):,}편이다. 원래 줄바꿈만 문단 경계로 사용하며 새 문단을 추정하지 않는다.", '',
            '| 장르 | 준비 | 평가 |', '|---|---:|---:|']
    for genre in ['explanatory','argumentative','emotional']:
        body.append(f"| {genre} | {sum(x['genre']==genre for x in sel['prep10'])} | {sum(x['genre']==genre for x in sel['eval50'])} |")
    body += ['',f"평가 문항 학습 중복 {sum(x['question_seen_in_training'] for x in sel['eval50'])}/50. 비중복은 {sum(not x['question_seen_in_training'] for x in sel['eval50'])}편이므로 독립적인 일반화 근거로 삼기 어렵다. 이후 8B 학습 제외 목록은 `verak/v5/evaluation_holdouts/luna_upper_bound_v2.json`에 보존한다.", '',
             '피드백은 저장된 `assistant` 필드를 사용한다. 사람·전문가 작성 여부는 확인되지 않았으며 사용자 승인을 받아 이 한계를 명시한다. 피드백과 문제 목록은 어떤 Luna 관찰에도 제공하지 않는다.', '',
             '## 준비 실험', '', '| 알림 | 건수 |','|---|---:|']
    body += [f'| {k} | {counts[k]} |' for k in PHI]
    body += ['',f"한국어 행동: {dict(actions)}. 총괄 계획 JSON 파싱 성공 {parses['success']}, 실패 {parses['failure']}.", '',
             f"검사에서 발견한 사유(탐색에서는 탈락시키지 않음): {dict(reasons)}.", '']
    if prep:
        body += [f"표현만 바꾼 문단의 채점 변화 절댓값으로 δ={prep['delta']:.6f}을 정했다(유효 {prep['delta_sample_count']}/10, 90백분위수). 수정량 상한은 문단 내 수정·삭제 문장 비율 {prep['modification_limit']:.3f}(의미 검사를 통과한 {prep['modification_calibration_samples']}개 후보의 90백분위수)이다.",'']
    body += ['다섯 유형을 확인하는 합성 환경 검증과 실제 준비 글에서 발생한 알림은 분리해 집계했다. φ는 기존 바른 기반 코드를 호출한다. 생략 주어의 ‘확실’은 앞 문장 주제어를 추정할 수 있다는 뜻이며 지시 대상을 확정하지 않는다. 인용 안 종결어미는 기존 인용 인식 코드를 따른다.', '',
             '사람 확인: '+('완료' if (ROOT/'human_preparation_review.json').exists() else '**대기 중**')+'. [전체 준비 로그](luna_upper_bound_prep_review.md). v2 §5.3의 사람 확인 없이 본실험이 실행되지 않도록 했다.', '',
             '## 호출·토큰·시간·비용', '', '| 역할 | 논리 호출 | 입력 토큰 | 출력 토큰 |', '|---|---:|---:|---:|']
    for role, rows in calls.items():
        body.append(f"| {role} | {len(rows)} | {sum(c['usage']['input_tokens'] for c in rows)} | {sum(c['usage']['output_tokens'] for c in rows)} |")
    body += ['',f"원자료 선정: {sel['wall_seconds']:.2f}초. 준비 글별 실행시간 합계: {sum(r['wall_seconds'] for r in records):.2f}초. "+(f"기준값 보정 포함 준비 단계 전체: {prep['wall_seconds']:.2f}초." if prep else '기준값 보정은 실행 중이다.'), '',
             f"실제 API 확정 사용액 **${money['confirmed']:.6f}**, 미확정 예약액 **${money['reserved']:.6f}**, 총 상한 **$5.00**. 동일 관찰의 저장 응답을 재사용한 논리 호출은 실제 결제 요청과 구분한다.", '']
    if projection:
        body += ['| 본실험 비용 추정 항목 | USD |','|---|---:|']+[f'| {k} | ${v:.4f} |' for k,v in projection['usd'].items()]
        body += ['', '이는 준비 글의 실제 Luna 토큰·수정 수와 평가 글 길이로 추정한 값이며 청구액이 아니다. Claude는 문자당 1.3토큰, 출력은 문제 목록 700/글 평가 600/연결 평가 100토큰을 가정한다. 실제 요청은 최대 출력까지 예산을 예약하며 상한을 넘기기 전에 중단한다. Batch 개별 처리시간은 제공되지 않으므로 배치 전체 대기·완료 시간만 보고한다.','']
    body += ['## 실행 전 정한 해석 및 제한', '',
             '- B−A, B−A+, B−C, B−D를 같은 글끼리 비교하고 Wilcoxon, 차이 중앙값, rank-biserial을 보고한다. 유의하지 않다는 이유로 동등하다고 결론 내리지 않는다.',
             '- 연결 오류율은 MOVE/INSERT/DELETE 편집과 전체 편집을 나눠 계산한다. 이전 21.7% 결과는 후보·분모·판정 방식 차이가 기록돼 있어 직접적인 재현 수치라고 부르지 않는다.',
             '- 수정 문장 비율은 수정량이다. 비율이 크다는 것만으로 불필요한 과잉 수정으로 판정하지 않는다.',
             '- Sonnet 5.5의 미지원 `temperature=0`은 사용자 승인으로 생략했다. `between_tools`, effort `medium`으로 고정한다. [공식 API 변경 문서](https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide).',
             '- P5에는 2절 관찰 표에서 요구한 현재 문단을 추가했다. φ_r는 새 문체 불일치뿐 아니라 불일치가 해소된 실제 문체 변화도 사실 알림으로 남긴다.',
             '- 실제 사람의 10편×5설정=50개 익명 판정이 들어오기 전에는 사람 대조가 끝났다고 표시하지 않는다.', '',
             '## 산출물', '', f'실험 원자료: `{ROOT}`. 코드 작업 브랜치: `feat/feak-luna-v2-20261014`. 평가 출력과 판정이 완료되면 설정별 지표·짝비교·좋은/나쁜 예시를 이 보고서에 추가한다.']
    path = REPO/'reports/luna_upper_bound.md'; path.write_text('\n'.join(body)+'\n')
    print(json.dumps({'report':str(path),'prepared':len(records),'cost':money,'projection':projection},ensure_ascii=False))


if __name__=='__main__': report()
