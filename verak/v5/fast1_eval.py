"""FAST1 Job A: frozen C30 scoring and blinded Claude evaluation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

from .common import REPO, read_json, sha_text, write_json, safe_id
from .fast1_common import (evaluate_variants, normalize_whitespace, score_cached,
                           sentence_change_counts, prepare_judge_manifest,
                           validate_fast1_judgment)
from verak.v4.scale2_projection import restore_document


PIPE = REPO/'verak/v5/outputs/pipe1'
ORCH = REPO/'verak/v5/outputs/orchestrator1_chat'
COOP = REPO/'verak/v5/outputs/cooperate1'
ROOT = REPO/'verak/v5/outputs/fast1/a_eval'
REPORT = REPO/'reports/FAST1_A.md'
CAP_USD = 4.0
VARIANTS = ('onepass_8b', 'luna_alone', 'control_run1', 'orchestrator1_luna')


def _verified(ref):
    path = Path(ref['path'])
    if not path.is_absolute():
        path = REPO/path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ref['sha256']:
        raise ValueError(f'Frozen artifact hash mismatch: {path}')
    return read_json(path), str(path)


def _doc_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and {'layout', 'tokens', 'text_sha256'} <= set(value):
        return restore_document(value).text
    if isinstance(value, dict) and isinstance(value.get('text'), str):
        return value['text']
    raise ValueError('Unrecognized saved essay document')


def load_cases():
    """Load the fixed 30 ORCHESTRATOR1 C cases and verify every source artifact."""
    design, _ = _verified({'path': str(ORCH/'design.json'),
                           'sha256': hashlib.sha256((ORCH/'design.json').read_bytes()).hexdigest()})
    cases = []
    for spec in design['cases']:
        case_id, cohort = spec['case_id'], spec['cohort']
        key = safe_id(case_id)
        c_design, _ = _verified({'path': str(COOP/'design.json'),
                                 'sha256': hashlib.sha256((COOP/'design.json').read_bytes()).hexdigest()})
        c_spec = next(x for x in c_design['cases'] if x['case_id'] == case_id)
        row, row_path = _verified(c_spec['row'])
        onepass, onepass_path = _verified(c_spec['C_outputs']['onepass'])
        luna, luna_path = _verified(c_spec['C_outputs']['luna'])
        control_ref = spec['controls']['1']['episode']
        control, control_path = _verified(control_ref)
        orch_path = ORCH/'evaluation/run1'/cohort/'attempts'/(key+'.json')
        orch = read_json(orch_path)
        original = row['text']
        onepass_prefix = read_json(Path(onepass_path).parent/'onepass_call.prefix.json')
        user_content = next(m['content'] for m in onepass_prefix['messages'] if m['role'] == 'user')
        marker = '[학생 글]\n'
        if marker not in user_content:
            raise ValueError(f'One-pass raw input marker missing: {case_id}')
        raw_original = user_content.split(marker, 1)[1]
        if raw_original != row['text']:
            raise ValueError(f'One-pass prefix original differs from frozen source row: {case_id}')
        originals = [raw_original, _doc_text(luna['initial']),
                     control['initial']['text'], orch['initial']['text']]
        if any(normalize_whitespace(x) != normalize_whitespace(original) for x in originals):
            raise ValueError(f'C30 source text mismatch: {case_id}')
        variants = {
            'onepass_8b': onepass['response']['raw'],
            'luna_alone': _doc_text(luna['final']),
            'control_run1': control['final']['text'],
            'orchestrator1_luna': orch['final']['text'],
        }
        cases.append({'case_id': case_id, 'cohort': cohort, 'question': row['question'],
            'original': raw_original, 'variants': variants,
            'sources': {'public_row': row_path, 'onepass': onepass_path, 'luna_alone': luna_path,
                        'control_run1': control_path, 'orchestrator1_luna': str(orch_path)}})
    if len(cases) != 30 or Counter(c['cohort'] for c in cases) != {'smoke20': 20, 'agent_dev10': 10}:
        raise ValueError('FAST1 A requires the fixed 20 + 10 C30')
    return cases


def api_contract():
    """Create the isolated $4 Claude-only ledger contract before any paid request."""
    prior = read_json(ORCH/'api_contract.json')
    model = dict(prior['claude'])
    model['cap_usd'] = CAP_USD
    contract = {'user_approved': True,
        'authorization': 'FAST1 Job A evaluation only; delegated maximum Claude budget $4.00.',
        'scope': 'Blind evaluation of saved C30 outputs and no other API use.',
        'allowed_providers': ['claude'], 'total_cap_usd': CAP_USD, 'claude': model}
    path = ROOT/'api_contract.json'
    if path.exists() and read_json(path) != contract:
        raise ValueError('FAST1 A API contract is frozen and differs')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, contract)
    return contract


def score_a1(cases):
    """Score source and all four raw full essays after identical whitespace normalization."""
    started = time.monotonic(); rows = []; violations = []
    for case in cases:
        original = score_cached(ROOT, case['question'], case['original'])
        variant_scores = {}
        for name, revised_text in case['variants'].items():
            revised = score_cached(ROOT, case['question'], revised_text)
            if revised['fingerprint'] != original['fingerprint']:
                raise ValueError('A1 original/revision scorer fingerprint mismatch')
            same = normalize_whitespace(case['original']) == normalize_whitespace(revised_text)
            delta = float(revised['mean']) - float(original['mean'])
            violation = same and delta != 0.0
            if violation:
                violations.append({'case_id': case['case_id'], 'condition': name,
                    'original_mean': original['mean'], 'revised_mean': revised['mean'], 'delta': delta})
            variant_scores[name] = {'revised_mean': revised['mean'], 'delta': delta,
                'identical_normalized': same, 'no_change_without_judge': same,
                'score_violation': violation,
                'revised_score_sha256': sha_text(json.dumps(revised, sort_keys=True, ensure_ascii=False))}
        rows.append({'case_id': case['case_id'], 'cohort': case['cohort'],
            'original_mean': original['mean'], 'variants': variant_scores,
            'fingerprint': original['fingerprint'],
            'original_score_sha256': sha_text(json.dumps(original, sort_keys=True, ensure_ascii=False))})
    fingerprints = {row['fingerprint'] for row in rows}
    if len(fingerprints) != 1:
        raise ValueError('A1 score outputs do not share one frozen scorer fingerprint')
    result = {'rows': rows, 'scorer_fingerprint': next(iter(fingerprints)),
        'whitespace_normalization': "' '.join(text.split()); no other text transformations",
        'input_shape': 'question + one raw full-essay string; never a sentence array',
        'identical_score_delta_violations': violations,
        'all_identical_deltas_zero': not violations,
        'wall_seconds': time.monotonic()-started}
    write_json(ROOT/'a1_scores.json', result)
    return result


def score_all_variants(cases):
    rows = []
    for case in cases:
        scores = {}
        for name, text in case['variants'].items():
            scores[name] = score_cached(ROOT, case['question'], text)
            if scores[name]['fingerprint'] != score_cached(ROOT, case['question'], case['original'])['fingerprint']:
                raise ValueError('A2 variant scorer fingerprint mismatch')
        rows.append({'case_id': case['case_id'], 'cohort': case['cohort'],
                     'original': score_cached(ROOT, case['question'], case['original']),
                     'variants': scores})
    value = {'rows': rows, 'scorer_fingerprint': rows[0]['original']['fingerprint'],
             'all_variant_scores_are_GPU_reference': True}
    write_json(ROOT/'a2_scores.json', value)
    return value


def text_metrics(cases):
    rows = []
    for case in cases:
        row = {'case_id': case['case_id'], 'cohort': case['cohort']}
        for name, text in case['variants'].items():
            row[name] = sentence_change_counts(case['original'], text)
        rows.append(row)
    write_json(ROOT/'a3_sentence_counts.json', {'rows': rows,
        'segmentation': 'regex sentence boundary after . ! ? 。！？; whitespace normalized first',
        'alignment': 'SequenceMatcher over ordered sentence strings; no model calls'})
    return rows


def run_judge(cases, *, max_requests=None):
    api_contract()
    from .api import PaidAPI
    api = PaidAPI('claude', root=ROOT)
    try:
        rebuild = not any((ROOT/'judgments').glob('*.json'))
        prepare_judge_manifest(ROOT, cases, anchor='orchestrator1_luna',
            baselines=('onepass_8b', 'luna_alone', 'control_run1'), max_output=600, rebuild=rebuild)
        manifest, summary = evaluate_variants(ROOT, cases, anchor='orchestrator1_luna',
            baselines=('onepass_8b', 'luna_alone', 'control_run1'), api=api,
            max_output={'primary': 2000, 'reverse': 1200},
            retry_max_output={'primary': 2400, 'reverse': 1600},
            max_requests=max_requests)
    finally:
        api.close()
    return manifest, summary


def _judgment_summary(manifest):
    cases = {x['case_id']: x for x in manifest['cases']}
    result = []
    for case_id, spec in cases.items():
        reqs = [r for r in manifest['requests'] if r['case_id'] == case_id]
        judgments = {}
        for req in reqs:
            path = ROOT/'judgments'/(req['id']+'.json')
            if path.exists(): judgments[req['order']] = read_json(path)
        result.append({'case_id': case_id, 'aliases': spec['aliases'], 'judgments': judgments})
    return result


def aggregate_judgments(cases, manifest):
    case_specs = {x['case_id']: x for x in manifest['cases']}
    requests = {(r['case_id'], r['order']): r for r in manifest['requests']}
    counts = {name: {'improved': Counter(), 'meaning_preserved': Counter(),
                     'fabrication': Counter(), 'no_change': 0} for name in VARIANTS}
    pair_counts = {name: Counter() for name in ('onepass_8b', 'luna_alone', 'control_run1')}
    pair_order = {name: [] for name in pair_counts}
    for case in cases:
        cid = case['case_id']; spec = case_specs[cid]; alias_to_condition = {v:k for k,v in spec['aliases'].items()}
        primary = read_json(ROOT/'judgments'/(requests[(cid, 'primary')]['id']+'.json'))
        for name, text in case['variants'].items():
            if normalize_whitespace(text) == normalize_whitespace(case['original']):
                counts[name]['no_change'] += 1
                continue
            verdict = primary.get('verdict', {})
            item = next((x for x in verdict.get('absolute', []) if x['id'] == spec['aliases'][name]), None)
            if item:
                for label in ('improved', 'meaning_preserved', 'fabrication'):
                    counts[name][label][item[label]] += 1
            else:
                for label in ('improved', 'meaning_preserved', 'fabrication'):
                    counts[name][label]['unjudged'] += 1
        for index, baseline in enumerate(('onepass_8b', 'luna_alone', 'control_run1'), 1):
            observed = []
            for order in ('primary', 'reverse'):
                request = requests[(cid, order)]
                saved = read_json(ROOT/'judgments'/(request['id']+'.json'))
                pair = next((x for x in saved.get('verdict', {}).get('pairs', []) if x['id'] == f'P{index}'), None)
                if not pair:
                    result = 'tie' if normalize_whitespace(case['variants']['orchestrator1_luna']) == normalize_whitespace(case['variants'][baseline]) else 'unjudged'
                elif pair['preference'] in {'tie', 'uncertain'}:
                    result = pair['preference']
                else:
                    payload = json.loads(request['messages'][1]['content'])
                    pair_spec = next(x for x in payload['pairwise_comparisons'] if x['id'] == f'P{index}')
                    winner_alias = pair_spec['left'] if pair['preference'] == 'left' else pair_spec['right']
                    result = 'anchor' if alias_to_condition[winner_alias] == 'orchestrator1_luna' else 'baseline'
                pair_counts[baseline][result] += 1
                observed.append(result)
            pair_order[baseline].append({'case_id': cid, 'primary': observed[0], 'reverse': observed[1],
                'concordant': observed[0] == observed[1]})
    return {'absolute': {name: {key: (dict(value) if isinstance(value, Counter) else value)
                                for key, value in labels.items()} for name, labels in counts.items()},
            'pairwise': {name: {'counts': dict(value), 'both_order_concordant': sum(x['concordant'] for x in pair_order[name]),
                                'essays': len(pair_order[name]), 'per_essay': pair_order[name]}
                         for name, value in pair_counts.items()}}


def replace_interrupted_unknown():
    """Replace a request with an explicitly new item ID; preserve the unknown ledger row."""
    manifest = read_json(ROOT/'judge_manifest.json')
    from .api import PaidAPI, BudgetStop
    replaced = []
    api = PaidAPI('claude', root=ROOT)
    try:
        for request in manifest['requests']:
            path = ROOT/'judgments'/(request['id']+'.json')
            current = read_json(path) if path.exists() else {}
            if current.get('status') != 'unknown_api_outcome':
                continue
            archive = ROOT/'unknown_outcomes'/(request['id']+'.json')
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists(): write_json(archive, current)
            initial_limit = {'primary': 2000, 'reverse': 1200}[request['order']]
            retry_limit = {'primary': 2400, 'reverse': 1600}[request['order']]
            attempts = []
            verdict = None
            for number, output_limit in enumerate((initial_limit, retry_limit), 1):
                item_id = f'FAST1:A:{request["case_id"]}:{request["order"]}:replacement_after_interrupted_unknown:attempt{number}'
                try:
                    response = api.request(request['messages'], item_id=item_id, max_output=output_limit)
                except RuntimeError as exc:
                    match = re.search(r'Paid request terminal outcome: ([a-z_]+) ([0-9a-f]{64})', str(exc))
                    if number == 1 and match and match.group(1) == 'incomplete':
                        attempts.append({'status': 'incomplete', 'fingerprint': match.group(2),
                                         'max_output_tokens': output_limit})
                        continue
                    attempts.append({'status': 'error', 'error': str(exc), 'max_output_tokens': output_limit})
                    break
                attempts.append({'status': response['status'], 'fingerprint': response['fingerprint'],
                                 'max_output_tokens': output_limit})
                try:
                    verdict = validate_fast1_judgment(response['raw'], request['expected_absolute_ids'],
                                                       request['expected_pair_ids'])
                    attempts[-1]['valid'] = True
                    break
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    attempts[-1].update(valid=False, validation_error=type(exc).__name__+': '+str(exc))
            if verdict is None:
                result = {'status': 'replacement_failed', 'case_id': request['case_id'],
                    'order': request['order'], 'request_id': request['id'], 'attempts': attempts,
                    'original_unknown_preserved_at': str(archive), 'retry_forbidden_for_original': True}
            else:
                result = {'status': 'valid', 'verdict': verdict, 'case_id': request['case_id'],
                    'order': request['order'], 'request_id': request['id'], 'attempts': attempts,
                    'replaces_unknown_outcome': str(archive), 'replacement_item_id_used': True}
            write_json(path, result); replaced.append(result['status'])
    finally:
        api.close()
    return replaced


def finalize_judgments(manifest):
    statuses = Counter()
    for request in manifest['requests']:
        path = ROOT/'judgments'/(request['id']+'.json')
        status = read_json(path).get('status', 'missing') if path.exists() else 'missing'
        statuses[status] += 1
    result = {'counts': dict(statuses), 'request_count': len(manifest['requests']),
        'all_requests_saved': sum(statuses.values()) == len(manifest['requests']),
        'all_judgments_valid_or_code_only': not any(statuses[k] for k in ('unknown_api_outcome', 'replacement_failed', 'error', 'budget_stop', 'missing'))}
    write_json(ROOT/'judge_complete.json', result)
    return result


def write_report(cases, a1, a2_scores, counts, manifest, judge_summary, *, report_path=None):
    """Render the reproducible full A report; optionally target a scratch path for review."""
    conditions = list(VARIANTS)
    labels = judge_summary['absolute']
    report_path = Path(report_path) if report_path else REPORT
    n = len(cases)

    def old_quality_sources():
        coop = read_json(COOP/'metrics.json')
        orch = read_json(ORCH/'metrics.json')
        old = {
            'onepass_8b': coop['quality']['all/onepass'],
            'luna_alone': coop['quality']['all/luna'],
            'control_run1': orch['quality']['control'],
            'orchestrator1_luna': orch['quality']['luna'],
        }
        deltas = {
            'onepass_8b': coop['quality']['all/onepass']['mean_Q_delta'],
            'luna_alone': coop['quality']['all/luna']['mean_Q_delta'],
            'control_run1': orch['metrics']['control1']['mean_Q_delta'],
            'orchestrator1_luna': orch['metrics']['luna']['mean_Q_delta'],
        }
        return old, deltas, orch

    old_quality, old_deltas, old_orch = old_quality_sources()
    old_no_change = {name: sum(normalize_whitespace(case['original']) ==
                               normalize_whitespace(case['variants'][name]) for case in cases)
                     for name in conditions}
    timing_path = ROOT/'timing.json'
    timing = read_json(timing_path) if timing_path.exists() else None
    accounting = read_json(ROOT/'api/accounting.json') if (ROOT/'api/accounting.json').exists() else {}
    provider = accounting.get('by_provider', {}).get('claude', {})

    lines = ['# FAST1 Job A', '',
        f"평가 대상은 smoke20 {sum(c['cohort']=='smoke20' for c in cases)}편(train split)과 agent_dev10 {sum(c['cohort']=='agent_dev10' for c in cases)}편, 총 {n}편이다. 학습은 실행하지 않았고 frozen test cohort는 사용하지 않았다.",
        '아래 새 평가는 저장된 one-pass 8B, Luna 단독, FEAK control run1, ORCHESTRATOR1 Luna 결과를 같은 C30에서 다시 평가했다. ORCHESTRATOR1 Luna가 새 조건이며 control run1은 기존 운영 조건이다.', '',
        '## 고정 프로토콜', '',
        f"- 익명 Claude 프롬프트 SHA-256: `{manifest['prompt_sha256']}`. 조건명, 생성 행동, 점수는 API 입력에서 가렸다.",
        '- 글마다 별도의 1차 및 역순 요청을 사용했다. 역순 요청은 ORCHESTRATOR1 대조 쌍을 뒤집어 별도 평가했다.',
        '- 1차 요청은 네 익명 전체 글의 원문 대비 절대 판정과 세 쌍 비교를 담았다. 역순 요청은 세 쌍 비교만 했다.',
        '- 공백 정규화 후 원문과 동일한 글은 코드상 no-change 처리하여 절대 판정에서 제외했다. 같은 쌍은 코드상 동률이다.',
        "- 입력은 `' '.join(text.split())`만 적용한 전체 글 한 문자열이다. One-pass는 저장된 raw 전체 응답을 사용했으며 문장 배열은 보내지 않았다.",
        f"- Claude 사용량: {provider.get('calls', 0)}회, 확인 비용 ${provider.get('confirmed_usd', 0):.6f}, 미확정 예약 ${provider.get('reserved_usd', 0):.6f}.",
        '', '## A1 — 고정 GPU 기준점수', '',
        f"원문 및 네 수정본을 동일한 공백 정규화 후 GPU reference scorer `{a1['scorer_fingerprint']}`로 전체 글 채점했다. 정규화 동일 쌍은 {sum(v['identical_normalized'] for r in a1['rows'] for v in r['variants'].values())}건이며, 동일 텍스트의 비영 ΔQ 위반은 {len(a1['identical_score_delta_violations'])}건이다.",
        '| 조건 | 평균 ΔQ 대 원문 | 원문 동일 (건/n) | 비율 |', '|---|---:|---:|---:|']
    for name in conditions:
        unchanged = sum(r['variants'][name]['identical_normalized'] for r in a1['rows'])
        mean_delta = statistics.mean(r['variants'][name]['delta'] for r in a1['rows'])
        lines.append(f"| {name} | {mean_delta:+.6f} | {unchanged}/{n} | {unchanged/n:.1%} |")
    lines += ['', '글별 점수·동일성·위반은 `a1_scores.json`에 있다.', '',
        '## A2 — 절대 판정', '',
        '각 조건은 전체 n편을 분모로 한다. 공백 정규화 후 원문과 동일한 수정은 코드상 개선=no, 의미 보존=yes, 새 사실 없음으로 포함한다. no-change 열은 이 코드 도출 수를 별도로 보여주며 앞의 라벨과 중복된다.',
        '| 조건 | 개선 yes/no/불확실/미판정 | 의미 yes/no/불확실/미판정 | 새 사실 없음/있음/불확실/미판정 | no-change 건수 / 비율 |',
        '|---|---:|---:|---:|---:|']
    for name in conditions:
        absolute = labels[name]
        unchanged = absolute['no_change']
        improved = absolute['improved']
        meaning = absolute['meaning_preserved']
        fabrication = absolute['fabrication']
        improved_no = improved.get('no', 0) + unchanged
        meaning_yes = meaning.get('yes', 0) + unchanged
        fabrication_absent = fabrication.get('absent', 0) + unchanged
        lines.append(f"| {name} | {improved.get('yes',0)}/{improved_no}/{improved.get('uncertain',0)}/{improved.get('unjudged',0)} | {meaning_yes}/{meaning.get('no',0)}/{meaning.get('uncertain',0)}/{meaning.get('unjudged',0)} | {fabrication_absent}/{fabrication.get('present',0)}/{fabrication.get('uncertain',0)}/{fabrication.get('unjudged',0)} | {unchanged}/{n} ({unchanged/n:.1%}) |")

    lines += ['', '## 저장된 과거 결과와 이번 FAST1 비교', '',
        '기존 값은 `cooperate1/metrics.json`의 onepass/Luna 단독과 `orchestrator1_chat/metrics.json`의 control1/총괄 Luna 집계다. 과거 judge 프로토콜·scorer가 달라 전후 차이는 통제된 인과효과가 아니다. 이전 절대 라벨은 저장 집계의 yes 및 의미 보존 yes만 표기한다. 기존 no-change는 저장된 원문·수정본의 정규화 동일성으로 계산했다.',
        '| 조건 | 기존 ΔQ | 기존 개선 yes | 기존 의미 yes | 기존 no-change | FAST1 ΔQ | FAST1 개선 yes/no/불확실/미판정 | FAST1 의미 yes/no/불확실/미판정 | FAST1 fabrication 없음/있음/불확실/미판정 | FAST1 no-change 건수/비율 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in conditions:
        old = old_quality[name]
        qold = old.get('global_quality', old)
        old_improved = qold.get('better_than_original', {})
        old_meaning = qold.get('meaning_preserved', {})
        absolute = labels[name]
        unchanged = absolute['no_change']
        improved = absolute['improved']; meaning = absolute['meaning_preserved']; fabrication = absolute['fabrication']
        lines.append(f"| {name} | {old_deltas[name]:+.6f} | {old_improved.get('yes',0)}/{old.get('judged',old.get('n',n))} | {old_meaning.get('yes',0)}/{old.get('judged',old.get('n',n))} | {old_no_change[name]}/{n} | {statistics.mean(r['variants'][name]['delta'] for r in a1['rows']):+.6f} | {improved.get('yes',0)}/{improved.get('no',0)+unchanged}/{improved.get('uncertain',0)}/{improved.get('unjudged',0)} | {meaning.get('yes',0)+unchanged}/{meaning.get('no',0)}/{meaning.get('uncertain',0)}/{meaning.get('unjudged',0)} | {fabrication.get('absent',0)+unchanged}/{fabrication.get('present',0)}/{fabrication.get('uncertain',0)}/{fabrication.get('unjudged',0)} | {unchanged}/{n} ({unchanged/n:.1%}) |")
    lines += ['', '이전 저장물의 추가 claim/fabrication 라벨과 FAST1 claim 판정:',
        'COOPERATE1은 fabrication yes/no 라벨을 저장했다. ORCHESTRATOR1은 new-specific-claim 출처 상태(none/supported/contradicted/unknown)를 저장했다. 이 서로 다른 ORCHESTRATOR1 라벨은 FAST1 fabrication 판정으로 변환할 수 없어 원래 스키마 그대로 나란히 표시한다.',
        '| 조건 | 이전 라벨 스키마 | 저장된 이전 라벨 수 | FAST1 fabrication 없음/있음/불확실/미판정 (n편; no-change는 없음에 포함) |',
        '|---|---|---|---:|']
    for name in conditions:
        old = old_quality[name]
        prior = old.get('global_quality', old)
        if 'new_specific_claims' in prior:
            schema = 'ORCHESTRATOR1 new_specific_claims'
            prior_counts = prior['new_specific_claims']
            prior_display = ', '.join(f"{k}:{prior_counts.get(k,0)}" for k in ('none','supported','contradicted','unknown'))
        else:
            schema = 'COOPERATE1 fabrication'
            prior_counts = prior.get('fabrication', {})
            prior_display = ', '.join(f"{k}:{prior_counts.get(k,0)}" for k in ('yes','no','uncertain','unjudged'))
        fabric = labels[name]['fabrication']; unchanged = labels[name]['no_change']
        new_display = '/'.join(str(v) for v in (fabric.get('absent',0)+unchanged,
            fabric.get('present',0),fabric.get('uncertain',0),fabric.get('unjudged',0)))
        lines.append(f"| {name} | {schema} | {prior_display} | {new_display} |")
    lines += ['', '## A2 쌍 비교 — 순서별', '',
        '결과는 ORCHESTRATOR1 Luna 우세 / baseline 우세 / 동률 / 불확실 / 미판정 순이다. FAST1 두 순서는 별도 요청으로 집계했다. 저장된 기존 순서별 비교는 control1 대 Luna에 대해서만 있었다.',
        '| 대조 조건 | 기존 1차 | 기존 역순 | FAST1 1차 | FAST1 역순 |', '|---|---|---|---|---|']
    old_primary = old_orch.get('primary_preference', {})
    old_reverse = old_orch.get('reverse_preference', {})
    def old_pair_order(data):
        return f"{data.get('luna',0)} / {data.get('control',0)} / {data.get('tie',0)} / {data.get('uncertain',0)} / {data.get('unjudged',0)}"
    pairwise = judge_summary['pairwise']
    for name in ('control_run1', 'onepass_8b', 'luna_alone'):
        per_essay = pairwise[name]['per_essay']
        order_counts = {}
        for order in ('primary', 'reverse'):
            order_counts[order] = Counter(row[order] for row in per_essay)
        def fmt(counts):
            return ' / '.join(str(counts.get(k,0)) for k in ('anchor','baseline','tie','uncertain','unjudged'))
        if name == 'control_run1':
            old1, old2 = old_pair_order(old_primary), old_pair_order(old_reverse)
        else:
            old1 = old2 = '저장 비교 없음'
        lines.append(f"| {name} | {old1} | {old2} | {fmt(order_counts['primary'])} | {fmt(order_counts['reverse'])} |")
    agree_names = []
    for name in ('onepass_8b','luna_alone','control_run1'):
        per = pairwise[name]['per_essay']
        agree_names.append(f"{name} {sum(x['concordant'] for x in per)}/{len(per)}")
    lines += ['', '두 순서에서 선호 결과가 일치한 글: '+', '.join(agree_names)+'. 같은 Claude 모델의 순서 민감도 검사이며 독립 평가자가 아니다.', '',
        '## A3 — 문장 변경량', '',
        '공백 정규화한 문장 문자열을 순서 정렬로 대응했다. 변경 수는 수정량 기술이며 품질 점수가 아니다.',
        '| 조건 | 평균 동일 문장 비율 | 평균 변경 원문 문장 수 | 평균 추가·변경 수정 문장 수 |', '|---|---:|---:|---:|']
    for name in conditions:
        metrics = [r[name] for r in counts]
        lines.append(f"| {name} | {statistics.mean(x['unchanged_share'] or 0 for x in metrics):.3f} | {statistics.mean(x['changed_source_sentences'] for x in metrics):.2f} | {statistics.mean(x['inserted_or_changed_revised_sentences'] for x in metrics):.2f} |")
    lines += ['', '## 비용 및 경과 시간', '']
    if timing:
        window = timing['measurement_window']; judge_window = timing['judge_window']; a1_time = timing['a1_scorer']
        lines.append(f"- 측정 구간: **{window['wall_seconds']:.1f}초** ({window['start_utc']}–{window['end_utc']}); {window['scope']}")
        lines.append(f"- Judge 원장 구간: **{judge_window['wall_seconds']:.1f}초** ({judge_window['start_utc']}–{judge_window['end_utc']}); 이는 전체 구간의 일부다.")
        lines.append(f"- A1 scorer: **{a1_time['wall_seconds']:.1f}초**; 전체 측정 구간과 겹치므로 합산하지 않는다.")
    else:
        lines.append('- `timing.json`이 없어 실행 경과 시간은 보고하지 않았다.')
    lines.append(f"- Claude 확인 비용 **${provider.get('confirmed_usd',0):.6f}**, 미확정 예약 **${provider.get('reserved_usd',0):.6f}**; 확인+예약은 승인 상한 내 원장 기록이다.")
    lines += ['', '결과는 저장된 진단 표본이며 held-out test 결과가 아니다. Claude 판정은 사람 정답이 아니다. 새 사실 판정은 제공된 원문에 대한 상대 평가이며 외부 사실 검증은 하지 않았다.', '']
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    return report_path

def run_all(*, judge=True):
    ROOT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    frozen = {'case_ids': [c['case_id'] for c in cases],
        'source_hashes': {c['case_id']: c['sources'] for c in cases},
        'conditions': list(VARIANTS), 'test_cohort_touched': False}
    freeze_path = ROOT/'a_design.json'
    if freeze_path.exists() and read_json(freeze_path) != frozen:
        raise ValueError('FAST1 A frozen source design differs')
    if not freeze_path.exists(): write_json(freeze_path, frozen)
    a1 = score_a1(cases)
    a2_scores = score_all_variants(cases)
    counts = text_metrics(cases)
    if judge:
        manifest, _ = run_judge(cases)
        replace_interrupted_unknown()
        finalize_judgments(manifest)
    else:
        manifest = None
    if manifest:
        aggregate = aggregate_judgments(cases, manifest)
        write_json(ROOT/'judge_aggregate.json', aggregate)
        write_report(cases, a1, a2_scores, counts, manifest, aggregate)
    return {'cases': len(cases), 'a1': a1, 'judge_manifest': manifest,
            'a2_scores': a2_scores, 'a3_counts': counts}


if __name__ == '__main__':
    result = run_all()
    print(json.dumps({'cases': result['cases'], 'scorer_fingerprint': result['a1']['scorer_fingerprint']}, indent=2))
