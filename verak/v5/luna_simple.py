"""Train10 only: three Luna roles without notices, quality gates or retries.

No feedback enters generation or judging. Independent Claude judgments never
affect candidates. Outputs live in the existing bounded v2 experiment ledger.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from verak.v4.common import REPO, read_json, write_json, atomic_new, sha_text, file_sha
from .api import PaidAPI
from .luna_v2_data import ROOT
from .luna_v2_r2_env import character_edits
from .luna_v2_run import parse_json, score_text

HOME = ROOT / 'simple_train10'
PROMPTS_PATH = Path(__file__).with_name('luna_simple_prompts.json')
PROMPTS = read_json(PROMPTS_PATH)
SOURCE_IDS = ['train:41378', 'train:25214', 'train:9168', 'train:40272',
              'train:47713', 'train:41891', 'train:32440', 'train:38914',
              'train:34369', 'train:32890']
LINK_TYPES = {'connective_ending', 'initial_conjunction', 'omitted_subject',
              'particle', 'reference', 'style', 'fragment', 'other'}
STAGES = ['one_pass', 'revision', 'three_agent']
LABELS = {'one_pass': 'Luna 한 번', 'revision': '총괄+글 수정 (중간본)',
          'three_agent': '총괄+글 수정+한국어'}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def frozen(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Frozen artifact differs: '+str(path))
    else:
        atomic_new(path, value)
    return value


def source(source_id):
    if source_id not in SOURCE_IDS:
        raise ValueError('Only the explicitly frozen preparation train10 is allowed')
    sid = source_id.replace(':', '_')
    row = read_json(ROOT/'sources'/(sid+'.json'))
    previous = read_json(ROOT/'prep_r2_checked/B'/sid/'result.json')
    assert previous['essay_id'] == source_id
    # Exact same canonical original and cached automatic scores for both arms.
    # Only allowlisted fields survive; feedback/human scores are never exposed.
    return {'source_id': source_id, 'question': row['question'],
            'original': previous['original'], 'scores': previous['scores_before']}


def generation_messages(role, row, *, plan=None, revised=None):
    observation = {'문항': row['question'], '원문': row['original']}
    if role in {'one_pass', 'plan'}:
        observation['참고용 자동 채점 점수 (9점 만점)'] = score_text(row['scores'])
    if role == 'revision':
        # Writer notes are deliberately withheld from the writer.
        observation['수정 계획'] = {k: plan[k] for k in ('thesis_quote', 'priorities')}
    if role == 'korean':
        observation['현재 수정본'] = revised
    return [{'role': 'system', 'content': PROMPTS['common']+'\n\n'+PROMPTS[role]},
            {'role': 'user', 'content': json.dumps(observation, ensure_ascii=False)}]


def validate_action(role, raw):
    try:
        value = parse_json(raw)
    except json.JSONDecodeError as exc:
        if 'Invalid control character' not in str(exc): raise
        # Decode a saved JSON string's literal newline without changing its text.
        value = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()), strict=False)
        if not isinstance(value, dict): raise ValueError('JSON object required')
    if role == 'plan':
        if not isinstance(value.get('thesis_quote'), str): raise ValueError('Missing thesis quote')
        items = value.get('priorities')
        if not isinstance(items, list) or len(items) > 3: raise ValueError('Invalid plan items')
        if any(not isinstance(x, dict) or any(not isinstance(x.get(k), str) for k in ('problem', 'instruction')) for x in items):
            raise ValueError('Invalid plan schema')
        if not isinstance(value.get('writer_notes'), list) or any(not isinstance(x, str) for x in value['writer_notes']):
            raise ValueError('Invalid writer notes')
    elif not isinstance(value.get('essay'), str) or not value['essay'].strip():
        raise ValueError('Empty/non-string essay')
    return value


def initialize():
    cfg = {'experiment': 'luna_simple_train10_v1', 'source_ids': SOURCE_IDS,
           'scope': 'repeated preparation essays; not a held-out estimate',
           'generation': {'model': 'gpt-6-luna', 'reasoning_effort': 'none', 'temperature': 0,
                          'arms': {'one_pass': ['one_pass'], 'three_agent': ['plan', 'revision', 'korean']},
                          'rewrite_scope': 'whole essay in one call', 'candidates': 1, 'retries': 0,
                          'notices': False, 'quality_gates': False, 'score_acceptance': False,
                          'environment_analysis': False, 'search': False, 'feedback_input': False,
                          'format_validation_only': True, 'writer_notes_to_revision': False},
           'judge': {'model': 'claude-sonnet-5-5', 'temperature': 'omitted as approved',
                     'comparisons': [['one_pass', 'three_agent'], ['revision', 'three_agent']],
                     'orders': 'both', 'blinded': True, 'feedback_input': False,
                     'primary_safe_improvement': 'both order judgments: improved AND meaning_preserved AND empty new_content'},
           'max_workers': 4, 'shared_total_cap_usd': 5, 'training': False,
           'eval50_authorized': False, 'test_opened': False,
           'prompts_sha256': file_sha(PROMPTS_PATH),
           'inputs': {sid: {'original_sha256': sha_text(source(sid)['original']),
                            'question_sha256': sha_text(source(sid)['question'])} for sid in SOURCE_IDS}}
    contract = read_json(ROOT/'api_contract.json')
    assert contract['total_cap_usd'] == 5
    assert contract['luna']['model'] == cfg['generation']['model']
    assert contract['luna']['reasoning_effort'] == 'none' and contract['luna']['temperature'] == 0
    assert contract['claude']['model'] == cfg['judge']['model']
    assert 'temperature' not in contract['claude'].get('generation_options', {})
    frozen(HOME/'config.json', cfg)
    frozen(HOME/'prompts.json', PROMPTS)
    if not (HOME/'start.json').exists():
        atomic_new(HOME/'start.json', {'utc': stamp(), 'prior_accounting': read_json(ROOT/'api/accounting.json')})
    return cfg


def episode(source_id, arm, api_factory=PaidAPI, directory=HOME):
    path = directory/'episodes'/arm/source_id.replace(':', '_')
    if (path/'result.json').exists(): return read_json(path/'result.json')
    row = source(source_id)
    api = api_factory('luna', root=ROOT)
    started = time.monotonic()
    calls = []
    path.mkdir(parents=True, exist_ok=True)

    def call(role, **kwargs):
        messages = generation_messages(role, row, **kwargs)
        frozen(path/(role+'_observation.json'), messages)
        response = api.request(messages, item_id='simple-v1:'+source_id+':'+role,
                               max_output=1536 if role == 'plan' else 4096)
        # Save the raw response BEFORE parsing; malformed output is not rolled
        # back to the original or retried under another ID.
        frozen(path/(role+'_response.json'), response)
        value = validate_action(role, response['raw'])
        try:
            parse_json(response['raw'])
            recovered = False
        except json.JSONDecodeError:
            recovered = True
        calls.append({'role': role, 'fingerprint': response['fingerprint'],
                      'usage': response['usage'], 'cost_usd': response['confirmed_usd'],
                      'dispatch_seconds': response['elapsed_s'], 'json_control_character_recovered': recovered})
        return value

    try:
        if arm == 'one_pass':
            result = {'final': call('one_pass')['essay']}
        elif arm == 'three_agent':
            plan = call('plan')
            revision = call('revision', plan=plan)['essay']
            final = call('korean', revised=revision)['essay']
            result = {'plan': plan, 'revision': revision, 'final': final,
                      'thesis_quote_is_literal': plan['thesis_quote'] in row['original'],
                      'korean_edit': character_edits(revision, final)}
        else:
            raise ValueError('Unknown arm')
        result.update(source_id=source_id, arm=arm, original=row['original'], question=row['question'],
                      calls=calls, elapsed_seconds=time.monotonic()-started,
                      character_edit=character_edits(row['original'], result['final']))
        atomic_new(path/'result.json', result)
        return result
    finally:
        api.close()


def generation():
    cfg = initialize()
    started = time.monotonic()
    tasks = [(sid, arm) for sid in SOURCE_IDS for arm in ('one_pass', 'three_agent')]
    with ThreadPoolExecutor(max_workers=cfg['max_workers']) as pool:
        futures = {pool.submit(episode, sid, arm): (sid, arm) for sid, arm in tasks}
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps({'completed': futures[future], 'calls': len(result['calls'])}), flush=True)
    frozen(HOME/'generation_complete.json', {'episodes': len(tasks), 'sources': SOURCE_IDS})
    if not (HOME/'generation_wall.json').exists():
        first_observation = min(p.stat().st_mtime for p in (HOME/'episodes').glob('*/*/*_observation.json'))
        atomic_new(HOME/'generation_wall.json', {
            'wall_seconds': time.time()-first_observation,
            'definition': 'first persisted observation to completion, including parser recovery pause',
            'last_invocation_seconds': time.monotonic()-started, 'completed_utc': stamp()})


def results(sid):
    name = sid.replace(':', '_')
    one = read_json(HOME/'episodes/one_pass'/name/'result.json')
    three = read_json(HOME/'episodes/three_agent'/name/'result.json')
    assert one['original'] == three['original']
    return one, three, {'one_pass': one['final'], 'revision': three['revision'], 'three_agent': three['final']}


def judge_messages(row, left, right):
    # No condition/role, source ID, plan, score, feedback or provenance exposed.
    observation = {'문항': row['question'], '원문': row['original'], 'left': left, 'right': right}
    return [{'role': 'system', 'content': PROMPTS['judge']},
            {'role': 'user', 'content': json.dumps(observation, ensure_ascii=False)}]


def validate_judgment(raw, texts):
    value = parse_json(raw)
    if value.get('preference') not in {'left', 'right', 'tie'} or not isinstance(value.get('reason'), str):
        raise ValueError('Invalid preference')
    for side in ('left', 'right'):
        item = value[side]
        if any(type(item.get(k)) is not bool for k in ('improved', 'meaning_preserved')):
            raise ValueError('Invalid quality booleans')
        if not isinstance(item.get('reason'), str): raise ValueError('Missing judgment reason')
        for field in ('new_content', 'new_link_errors'):
            if not isinstance(item.get(field), list): raise ValueError('Missing issue list')
            for issue in item[field]:
                if not issue.get('quote') or issue['quote'] not in texts[side]: raise ValueError('Nonliteral evidence')
                if not isinstance(issue.get('reason'), str): raise ValueError('Missing issue reason')
                if field == 'new_link_errors' and issue.get('type') not in LINK_TYPES: raise ValueError('Unknown link type')
    return value


def judge_one(sid, comparison, order):
    _, _, texts = results(sid)
    labels = comparison if order == 0 else list(reversed(comparison))
    key = [sid, comparison, order]
    anonymous_id = sha_text(json.dumps(key))[:24]
    path = HOME/'judgments'/anonymous_id
    path.mkdir(parents=True, exist_ok=True)
    frozen(path/'mapping.json', {'source_id': sid, 'comparison': comparison, 'order': order,
                               'left': labels[0], 'right': labels[1]})
    if (path/'judgment.json').exists(): return read_json(path/'judgment.json')
    side_texts = {side: texts[label] for side, label in zip(('left', 'right'), labels)}
    messages = judge_messages(source(sid), side_texts['left'], side_texts['right'])
    frozen(path/'observation.json', messages)
    api = PaidAPI('claude', root=ROOT)
    try:
        response = api.request(messages, item_id=anonymous_id, max_output=4096)
        frozen(path/'response.json', response)
        accepted = 'response.json'
        try:
            value = validate_judgment(response['raw'], side_texts)
        except (ValueError, KeyError, TypeError) as exc:
            # One evaluation-only format recovery. Identical blinded inputs;
            # no criterion involving preference/quality can enable this retry.
            frozen(path/'format_retry.json', {'reason': type(exc).__name__+': '+str(exc),
                                             'max_retries': 1, 'original_response': 'response.json'})
            response = api.request(messages, item_id=anonymous_id+'-format1', max_output=4096)
            accepted = 'response_format_retry.json'
            frozen(path/accepted, response)
            value = validate_judgment(response['raw'], side_texts)
        frozen(path/'accepted_response.json', {'filename': accepted})
        atomic_new(path/'judgment.json', value)
        return value
    finally:
        api.close()


def judging():
    cfg = initialize()
    if not (HOME/'generation_complete.json').exists(): raise RuntimeError('Generation incomplete')
    started = time.monotonic()
    tasks = [(sid, pair, order) for sid in SOURCE_IDS for pair in cfg['judge']['comparisons'] for order in (0, 1)]
    with ThreadPoolExecutor(max_workers=cfg['max_workers']) as pool:
        futures = {pool.submit(judge_one, *task): task for task in tasks}
        for future in as_completed(futures):
            future.result()
            print(json.dumps({'judged': futures[future]}), flush=True)
    frozen(HOME/'judging_complete.json', {'requests': len(tasks)})
    if not (HOME/'judging_wall.json').exists():
        first_observation = min(p.stat().st_mtime for p in (HOME/'judgments').glob('*/observation.json'))
        atomic_new(HOME/'judging_wall.json', {
            'wall_seconds': time.time()-first_observation,
            'definition': 'first persisted judgment observation to completion, including evidence validation recovery pause',
            'last_invocation_seconds': time.monotonic()-started, 'completed_utc': stamp()})


def safe(item):
    return item['improved'] and item['meaning_preserved'] and not item['new_content']


def load_judgments():
    rows = []
    for path in sorted((HOME/'judgments').glob('*/judgment.json')):
        accepted = path.parent/'accepted_response.json'
        response_name = read_json(accepted)['filename'] if accepted.exists() else 'response.json'
        rows.append({'mapping': read_json(path.parent/'mapping.json'), 'judgment': read_json(path),
                     'response': read_json(path.parent/response_name)})
    return rows


def evaluations(judgments, sid, comparison, stage):
    values = []
    for row in judgments:
        m = row['mapping']
        if m['source_id'] == sid and m['comparison'] == comparison:
            side = 'left' if m['left'] == stage else 'right'
            values.append(row['judgment'][side])
    if len(values) != 2: raise ValueError('Both orders required')
    return values


def preference(judgments, sid, comparison):
    votes = []
    for row in judgments:
        m, j = row['mapping'], row['judgment']
        if m['source_id'] == sid and m['comparison'] == comparison:
            votes.append('tie' if j['preference'] == 'tie' else m[j['preference']])
    if len(votes) != 2: raise ValueError('Both preferences required')
    return votes[0] if votes[0] == votes[1] else 'disagree'


def report():
    cfg = initialize()
    assert (HOME/'judging_complete.json').exists()
    # Refresh accounting after every worker has closed, avoiding stale snapshots.
    accounting = PaidAPI('luna', root=ROOT)
    accounting.close()
    judges = load_judgments()
    assert len(judges) == 40
    metrics, per_essay = {}, {}
    for stage in STAGES:
        comparison = ['revision', 'three_agent'] if stage == 'revision' else ['one_pass', 'three_agent']
        data = []
        for sid in SOURCE_IDS:
            one, three, texts = results(sid)
            values = evaluations(judges, sid, comparison, stage)
            calls = one['calls'] if stage == 'one_pass' else three['calls'][:2] if stage == 'revision' else three['calls']
            record = {'safe_both': all(safe(v) for v in values),
                      'improved_both': all(v['improved'] for v in values),
                      'preserved_both': all(v['meaning_preserved'] and not v['new_content'] for v in values),
                      'new_content_any': any(v['new_content'] for v in values),
                      'new_link_any': any(v['new_link_errors'] for v in values),
                      'edit_ratio': character_edits(one['original'], texts[stage])['ratio'],
                      'unchanged': one['original'] == texts[stage], 'calls': len(calls),
                      'input_tokens': sum(x['usage']['input_tokens'] for x in calls),
                      'output_tokens': sum(x['usage']['output_tokens'] for x in calls),
                      'dispatch_seconds': sum(x['dispatch_seconds'] for x in calls),
                      'cost_usd': sum(x['cost_usd'] for x in calls)}
            data.append(record)
            per_essay.setdefault(sid, {})[stage] = record
        metrics[stage] = {k: sum(d[k] for d in data) / (10 if k in {'edit_ratio', 'calls', 'input_tokens', 'output_tokens', 'dispatch_seconds'} else 1) for k in data[0]}
    comparisons = {','.join(pair): dict(Counter(preference(judges, sid, pair) for sid in SOURCE_IDS)) for pair in cfg['judge']['comparisons']}
    secondary_final = {sid: all(safe(x) for x in evaluations(judges, sid, ['revision', 'three_agent'], 'three_agent')) for sid in SOURCE_IDS}
    context_flips = [sid for sid in SOURCE_IDS if secondary_final[sid] != per_essay[sid]['three_agent']['safe_both']]
    role_costs = {}
    for role in ('one_pass', 'plan', 'revision', 'korean'):
        calls = [c for sid in SOURCE_IDS for r in results(sid)[:2] for c in r['calls'] if c['role'] == role]
        role_costs[role] = {'calls': len(calls), 'cost_usd': sum(c['cost_usd'] for c in calls),
                            'dispatch_seconds': sum(c['dispatch_seconds'] for c in calls),
                            'input_tokens': sum(c['usage']['input_tokens'] for c in calls),
                            'output_tokens': sum(c['usage']['output_tokens'] for c in calls)}
    judge_responses = [read_json(p) for p in (HOME/'judgments').glob('*/response*.json')]
    judge_cost = sum(j['confirmed_usd'] for j in judge_responses)
    cost = sum(x['cost_usd'] for x in role_costs.values()) + judge_cost
    summary = {'metrics': metrics, 'per_essay': per_essay, 'preferences_both_orders': comparisons,
               'same_final_safe_judgment_context_flips': context_flips,
               'secondary_final_safe_both': sum(secondary_final.values()),
               'role_usage': role_costs, 'judge_cost_usd': judge_cost, 'experiment_cost_usd': cost,
               'judge_billed_calls': len(judge_responses), 'judge_format_retries': len(judge_responses)-40,
               'generation_wall': read_json(HOME/'generation_wall.json'), 'judging_wall': read_json(HOME/'judging_wall.json'),
               'accounting': read_json(ROOT/'api/accounting.json'),
               'format_recoveries': sum(c.get('json_control_character_recovered', False) for sid in SOURCE_IDS for r in results(sid)[:2] for c in r['calls']),
               'literal_thesis_quotes': sum(results(sid)[1]['thesis_quote_is_literal'] for sid in SOURCE_IDS),
               'korean_identical': sum(results(s)[1]['revision'] == results(s)[1]['final'] for s in SOURCE_IDS)}
    write_json(HOME/'summary.json', summary)
    qualitative_path = HOME / 'qualitative_review.md'
    qualitative = qualitative_path.read_text().splitlines() if qualitative_path.exists() else []
    lines = ['# Luna 세 에이전트 단순화 실험 — 준비용 10편', '',
             '알림·환경 분석·품질 탈락·재시도를 제거하고 총괄 → 글 수정 → 한국어를 모두 Luna로 실행했다. '
             '이 결과는 반복해서 살펴본 train 준비용 10편의 탐색 결과다. 평가 50편은 실행하지 않았고, 최종 테스트 파일은 열지 않았다. 학습도 없다.', '',
             '## 먼저 읽을 결론', '',
             f'세 에이전트가 한 번 수정보다 낫다는 근거를 얻지 못했다. 양쪽 제시 순서가 일치한 선호는 '
             f'한 번 수정 {comparisons["one_pass,three_agent"].get("one_pass",0)}편, 세 에이전트 {comparisons["one_pass,three_agent"].get("three_agent",0)}편, '
             f'순서 간 불일치 {comparisons["one_pass,three_agent"].get("disagree",0)}편이다. '
             f'한국어 단계는 {summary["korean_identical"]}/10편을 문자 그대로 유지했고, 추가 개선의 일관된 선호는 얻지 못했다.', '',
             f'아래 절대 품질 수치는 **자동 판정 원집계이며 검증된 오류율이 아니다**. 같은 최종 글인데 비교 상대를 바꾸자 '
             f'안전한 개선 판정이 {len(context_flips)}/10편에서 달라졌다. 원래 있던 오류나 “새 내용은 아님”이라고 설명한 항목도 '
             '새 오류·새 내용 목록에 들어갔다. 따라서 안전한 개선 수치만으로 한국어 단계가 글을 망쳤다고 해석하면 안 된다. '
             '현재는 추가 학습이나 평가 50편 확대의 근거로 쓰지 않고, 아래 구체적 사례와 전체 글을 검토하는 데 쓴다.', '',
             '## 실행 방식', '',
             '- 한 번 고치기: 문항·원문·기존 자동 채점 점수 → 고친 글 전체 (1회).',
             '- 세 에이전트: 총괄 계획 (1회) → 글 전체 수정 (1회) → 한국어 최소 다듬기 (1회).',
             '- 한국어 단계는 현재 글 전체와 원문을 읽는다. 알림, 대상 문장 선별, φ 계산은 없다.',
             '- 원문 밖 주장·해결책·사실 금지는 프롬프트 지시로만 적용했다. JSON 형식 검증 외에 내용에 따른 탈락·재생성·원문 복원은 없다.',
             '- 수정량 상한이나 채점 향상 기준은 적용하지 않았다. 글자 편집 비율은 결과를 설명하는 용도로만 계산했다.',
             '- 기존 v2의 문단별 실행·관찰 형식과 다르다. 이번 파일럿을 8B 학습 형식 검증으로 해석하지 않는다.',
             '- GPT-6 Luna: reasoning none, temperature 0. Claude Sonnet 5.5: 승인대로 temperature 생략. 동시 작업 4개.',
             '- 두 생성 설정에 동일한 원문과 자동 점수를 사용했다. 저장 피드백·사람 점수는 생성과 평가에 모두 제공하지 않았다. 따라서 v2의 피드백 기반 문제 해결률은 이번에 측정하지 않았다.', '',
             f'총괄의 중심 내용 인용은 {summary["literal_thesis_quotes"]}/10편에서 원문과 문자 그대로 일치했다. '
             f'API 응답 {summary["format_recoveries"]}건은 JSON 문자열 안의 실제 줄바꿈을 허용해 복구했다. '
             '저장된 수정문의 문자는 바꾸지 않았고, 후보 재생성이나 추가 API 재시도는 없었다.', '',
             '## 익명 평가', '',
             'Claude가 원문과 두 후보의 전문을 보고 판정했다. 시스템명·역할·계획은 숨겼다. '
             '한 번 고치기 대 최종본, 한국어 직전 중간본 대 최종본을 각각 좌우 순서를 바꾸어 2회 평가했다 (유효 판정 40건). '
             '판정은 생성이 끝난 뒤에만 수행했으며 글을 다시 고치거나 채택하는 데 쓰지 않았다.', '',
             f'평가 형식 복구 {summary["judge_format_retries"]}회, 실제 평가 호출 {summary["judge_billed_calls"]}회. '
             '실행 중 추가한 예외 처리: 수정본에 없는 문자열을 근거로 인용한 판정은 원응답을 보존하고 동일한 익명 입력으로 한 번 재판정했다. '
             '선호나 품질 결과를 이유로 재판정하지 않았으며 두 호출의 비용을 모두 포함했다.', '',
             '안전한 개선은 **개선됨 AND 의미 보존 AND 새 내용 없음**이다. 아래 개선·보존 수치는 두 순서 판정이 모두 동의한 글 수이다. '
             '새 내용·새 연결 오류는 두 판정 중 하나라도 지적한 글 수이므로 보수적 경보이며 사람의 확정 판정이 아니다. '
             '주 표는 두 설정을 서로 비교한 동일한 평가 맥락의 결과만 표시한다.', '',
             '| 설정 | 안전한 개선 /10 | 개선 /10 | 보존 /10 | 새 내용 지적 /10 | 새 연결 오류 지적 /10 | 평균 글자 편집 비율 | 원문 유지 /10 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for stage, m in metrics.items():
        if stage == 'revision': continue
        lines.append(f'| {LABELS[stage]} | {m["safe_both"]:.0f} | {m["improved_both"]:.0f} | {m["preserved_both"]:.0f} | {m["new_content_any"]:.0f} | {m["new_link_any"]:.0f} | {m["edit_ratio"]:.1%} | {m["unchanged"]:.0f} |')
    lines += ['', '글자 편집 비율 = 공백·줄바꿈을 포함한 Unicode 문자 Levenshtein 거리 / 두 글 길이 중 큰 값. 수정량 자체를 불필요한 과잉 수정으로 판정하지 않는다.', '',
              '| 비교 | 양쪽 순서 모두 앞 설정 선호 | 양쪽 모두 뒤 설정 선호 | 양쪽 모두 동률 | 순서 간 불일치 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for pair in cfg['judge']['comparisons']:
        counts = comparisons[','.join(pair)]
        lines.append(f'| {LABELS[pair[0]]} vs {LABELS[pair[1]]} | {counts.get(pair[0], 0)} | {counts.get(pair[1], 0)} | {counts.get("tie", 0)} | {counts.get("disagree", 0)} |')
    lines += ['', f'한국어 단계가 중간본을 그대로 낸 글은 {summary["korean_identical"]}/10편이다. '
              '한 번 고치기와 세 에이전트는 호출 수와 입력·출력 토큰이 다르므로 같은 계산량의 비교는 아니다. '
              '한국어 단계의 효과는 같은 글의 실제 중간본과 최종본 비교로 별도 확인했다.', '',
              f'한국어 전후를 서로 비교한 판정에서는 안전한 개선이 중간본 {metrics["revision"]["safe_both"]:.0f}/10, '
              f'최종본 {sum(secondary_final.values())}/10이었다. 동일한 최종본을 한 번 수정과 비교한 주 표에서는 '
              f'{metrics["three_agent"]["safe_both"]:.0f}/10이다. 이는 한국어 단계에 의한 품질 변화가 아니라 판정 맥락에 따른 차이다. '
              '차이가 난 글: '+', '.join(context_flips)+'.', '',
              *qualitative,
              '## 시간·토큰·비용', '',
              f'전체 생성 wall-clock: {summary["generation_wall"]["wall_seconds"]:.1f}초. '
              f'독립 평가 wall-clock: {summary["judging_wall"]["wall_seconds"]:.1f}초. '
              '각 시간은 해당 단계의 첫 관찰 저장부터 전체 완료까지이며 형식·근거 검증 복구를 위한 중단 시간도 포함한다. '
              '병렬 실행 때문에 호출 시간 합과 wall-clock은 다르다.', '',
              '| 설정 | 글당 호출 | 글당 입력 토큰 | 글당 출력 토큰 | 글당 호출 시간 합(초) | 10편 생성 비용(USD) |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for stage, m in metrics.items():
        lines.append(f'| {LABELS[stage]} | {m["calls"]:.1f} | {m["input_tokens"]:.1f} | {m["output_tokens"]:.1f} | {m["dispatch_seconds"]:.1f} | {m["cost_usd"]:.6f} |')
    lines += ['', '중간본은 세 에이전트 실행의 앞 두 호출이며 별도 생성 비용이 아니다.', '',
              f'이번 평가 비용 ${judge_cost:.6f}; 이번 실험 전체 ${cost:.6f}. '
              f'기존 준비 실험까지 공유 원장 누적 ${summary["accounting"]["confirmed_usd"]:.6f}, '
              f'미확정 예약 ${summary["accounting"]["reserved_usd"]:.6f}; 누적 상한 $5.', '',
              '## 글별 결과와 검토 자료', '',
              '| 글 | 한 번: 안전한 개선 | 세 에이전트: 안전한 개선 | 한 번 vs 세 에이전트 | 한국어 전후 |',
              '| --- | --- | --- | --- | --- |']
    votes_ko = {**LABELS, 'tie': '동률', 'disagree': '순서 간 불일치'}
    for sid in SOURCE_IDS:
        m = per_essay[sid]
        lines.append(f'| {sid} | {"예" if m["one_pass"]["safe_both"] else "아니오"} | {"예" if m["three_agent"]["safe_both"] else "아니오"} | {votes_ko[preference(judges,sid,["one_pass","three_agent"])]} | {votes_ko[preference(judges,sid,["revision","three_agent"])]} |')
    lines += ['', '전체 10편의 문항·원문·한 번 수정본·총괄 계획·중간본·최종본 및 Claude 근거: '
              '[luna_three_agent_simple_samples.md](luna_three_agent_simple_samples.md). 사람 판정은 아직 수행하지 않았다.', '',
              '설정·프롬프트·원문 해시는 실행 전에 고정했다. 각 관찰과 API 원응답, 역할별 토큰·비용·시간, 익명 좌우 매핑과 두 순서 판정은 '
              f'`{HOME}`에 저장했다. 이전 준비 로그와 데이터는 수정하지 않았다.']
    report_dir = REPO/'reports'
    report_dir.mkdir(exist_ok=True)
    (report_dir/'luna_three_agent_simple.md').write_text('\n'.join(lines)+'\n')
    samples = ['# Luna 단순화 실험 — 준비용 10편 전체와 평가 근거', '',
               'Claude 자동 판정이며 사람의 확인 결과가 아니다. 두 순서의 판정을 모두 남겼다.']
    for sid in SOURCE_IDS:
        one, three, texts = results(sid)
        samples += ['', f'## {sid}', '', '### 문항', '', one['question'], '', '### 원문', '', one['original'],
                    '', '### Luna 한 번 수정', '', texts['one_pass'], '', '### 총괄 계획', '',
                    '원문 중심 내용 인용: '+three['plan']['thesis_quote']]
        for item in three['plan']['priorities']:
            samples += ['', '- 문제: '+item['problem']+' / 수정 지시: '+item['instruction']]
        samples += ['', '작성자에게 남긴 사항 (글 수정 입력에서 제외): '+(' / '.join(three['plan']['writer_notes']) or '없음'),
                    '', '### 글 수정 직후 (한국어 전)', '', texts['revision'], '', '### 한국어 다듬기 후 최종본', '', texts['three_agent'],
                    '', f'한국어 단계 글자 편집 비율: {three["korean_edit"]["ratio"]:.1%}.', '', '### 자동 평가 근거']
        for record in sorted((j for j in judges if j['mapping']['source_id'] == sid), key=lambda x:(x['mapping']['comparison'], x['mapping']['order'])):
            m, j = record['mapping'], record['judgment']
            samples += ['', f'**왼쪽: {LABELS[m["left"]]} / 오른쪽: {LABELS[m["right"]]}**', '',
                        '선호: '+('동률' if j['preference']=='tie' else LABELS[m[j['preference']]])+' — '+j['reason']]
            for side in ('left','right'):
                item = j[side]
                samples += ['', f'- {LABELS[m[side]]}: '+item['reason']]
                for field, name in [('new_content','새 내용 지적'),('new_link_errors','새 연결 오류 지적')]:
                    for issue in item[field]:
                        samples += ['', f'  - {name}: “{issue["quote"]}” — {issue["reason"]}']
        samples += ['', '사람 검토: ____________________']
    (report_dir/'luna_three_agent_simple_samples.md').write_text('\n'.join(samples)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['initialize', 'generate', 'judge', 'report'])
    args = parser.parse_args()
    {'initialize': initialize, 'generate': generation, 'judge': judging, 'report': report}[args.command]()


if __name__ == '__main__':
    main()
