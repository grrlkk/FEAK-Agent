"""Run the bounded checkpoint-selection diagnostic, with durable resumable outputs."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from .common import WORKTREE, PORT, tokenizer, write_json, file_sha, verified
from .api import PaidAPI, BudgetStop
from .selection1 import (ROOT, REPO, MAX_CONTEXT, LOCAL_OUTPUT, SELECTOR_PROMPT, JUDGE_PROMPT,
    atomic_new, read_json, ref, safe_id, sha_text, prepare, messages, validate_selection,
    validate_judgment, reference_consensus, decision_metrics, order_comparison, repair_dangling_quote)


def cohort(manifest, phase):
    return [c for c in manifest['cases'] if phase == 'full' or c['case_id'] in manifest['pilot_ids']]


def output_path(kind, case, order, arm=None):
    folder = ROOT / kind
    if arm:
        folder /= arm
    return folder / (safe_id(case['case_id']) + f'_{order["permutation"]}.json')


def preflight():
    manifest = prepare(); tok = tokenizer(); rows = []
    for case in manifest['cases']:
        for order in case['orders']:
            request = messages(SELECTOR_PROMPT, order['payload'])
            tokens = len(tok.apply_chat_template(request, tokenize=True, add_generation_prompt=True))
            rows.append({'case_id': case['case_id'], 'order': order['permutation'],
                         'input_tokens': tokens, 'fits': tokens + LOCAL_OUTPUT <= MAX_CONTEXT})
    result = {'manifest': ref(ROOT/'manifest.json'), 'token_counts': rows,
              'max_input_tokens': max(r['input_tokens'] for r in rows),
              'context_failures': [r for r in rows if not r['fits']],
              'cases': len(manifest['cases']), 'pilot_ids': manifest['pilot_ids'],
              'versions': sum(len(c['candidates']) for c in manifest['cases']),
              'model_unchanged': True, 'new_training': False, 'new_essay_generation': False,
              'proposed_API_cap_usd': 3, 'new_paid_calls': 0}
    write_json(ROOT/'preflight.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('token_counts', 'manifest')}, ensure_ascii=False), flush=True)
    return result


def local(phase):
    import httpx
    manifest = prepare(); frozen = read_json(ROOT/'preflight.json')
    verified(frozen['manifest'])
    if file_sha(Path(manifest['adapter']['path'])) != manifest['adapter']['sha256']:
        raise ValueError('Frozen orchestrator adapter changed')
    limits = {(r['case_id'], r['order']): r for r in frozen['token_counts']}
    completed = Counter(); start = time.monotonic()
    with httpx.Client(timeout=240) as client:
        for case in cohort(manifest, phase):
            for arm, model in [('orchestrator', 'v5-orchestrator'), ('base', 'kanana-v5')]:
                for order in case['orders']:
                    path = output_path('local', case, order, arm)
                    if path.exists():
                        completed[read_json(path)['status']] += 1; continue
                    if len(case['candidates']) == 1:
                        result = {'status': 'single_candidate', 'selected_key': case['original_key'], 'new_call': False}
                    elif not limits[(case['case_id'], order['permutation'])]['fits']:
                        result = {'status': 'context_overflow', 'new_call': False}
                    else:
                        request = {'model': model, 'messages': messages(SELECTOR_PROMPT, order['payload']),
                                   'temperature': 0., 'top_p': 1., 'seed': 47, 'max_tokens': LOCAL_OUTPUT}
                        request_path = output_path('local_requests', case, order, arm)
                        if not request_path.exists(): atomic_new(request_path, request)
                        begin = time.monotonic()
                        try:
                            response = client.post(f'http://127.0.0.1:{PORT}/v1/chat/completions', json=request)
                            response.raise_for_status(); response = response.json()
                            raw = response['choices'][0]['message']['content']
                            result = {'status': 'invalid', 'raw': raw, 'usage': response.get('usage'),
                                      'response_model': response.get('model'),
                                      'finish_reason': response['choices'][0]['finish_reason'],
                                      'elapsed_s': time.monotonic()-begin, 'new_call': True,
                                      'request': ref(request_path)}
                            if result['finish_reason'] != 'stop':
                                result['error'] = 'Generation did not finish'
                            else:
                                try: result.update(validate_selection(raw, order), status='valid')
                                except (ValueError, KeyError, TypeError) as exc: result['error'] = str(exc)
                        except Exception as exc:
                            result = {'status': 'transport_error', 'error_type': type(exc).__name__,
                                      'new_call': True, 'elapsed_s': time.monotonic()-begin}
                    atomic_new(path, result); completed[result['status']] += 1
            print('local', case['case_id'], flush=True)
    write_json(ROOT/f'local_{phase}_complete.json', {'counts': dict(completed), 'seconds': time.monotonic()-start})


def gpu(phase):
    from verak.v4.full_run import idle
    from .run import server_command
    import httpx
    if not (ROOT/'preflight.json').exists(): preflight()
    if phase == 'full' and not read_json(ROOT/'metrics_pilot.json')['expand']:
        raise RuntimeError('Frozen pilot expansion rule did not pass')
    hardware = idle((0,))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', PORT))
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'VLLM_NO_USAGE_STATS': '1',
           'DO_NOT_TRACK': '1', 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '8'}
    process = None
    try:
        with (ROOT/f'vllm_{phase}.log').open('a') as log:
            process = subprocess.Popen(server_command(), cwd=WORKTREE, env=env, stdout=log, stderr=subprocess.STDOUT)
            write_json(ROOT/f'server_{phase}.json', {'pid': process.pid, 'hardware': hardware, 'at': time.time()})
            deadline = time.monotonic()+600
            while time.monotonic() < deadline:
                if process.poll() is not None: raise RuntimeError('vLLM failed; see owned server log')
                try:
                    if httpx.get(f'http://127.0.0.1:{PORT}/health', timeout=2).status_code == 200: break
                except httpx.HTTPError: pass
                time.sleep(2)
            else: raise RuntimeError('vLLM startup timeout')
            local(phase)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
        write_json(ROOT/f'server_{phase}_stopped.json', {'pid': process.pid if process else None, 'at': time.time()})


def judge(phase):
    manifest = prepare()
    if phase == 'full' and not read_json(ROOT/'metrics_pilot.json')['expand']:
        raise RuntimeError('Frozen pilot expansion rule did not pass')
    if read_json(ROOT/'api_contract.json').get('user_approved') is not True:
        raise PermissionError('SELECTION1 scoped API approval missing')
    jobs = [(c, o) for c in cohort(manifest, phase) for o in c['orders'] if len(c['candidates']) > 1]
    def one(job):
        case, order = job; path = output_path('judgments', case, order)
        if path.exists(): return read_json(path)['status']
        api = PaidAPI('claude', root=ROOT)
        try:
            prompt = messages(JUDGE_PROMPT, order['payload'])
            for attempt in (1, 2):
                response = api.request(prompt, item_id=f'SELECTION1:{case["case_id"]}:{order["permutation"]}:{attempt}', max_output=4096)
                try:
                    verdict = validate_judgment(response['raw'], order)
                    atomic_new(path, {'status': 'valid', 'verdict': verdict, 'response': response})
                    return 'valid'
                except (ValueError, KeyError, TypeError) as exc:
                    atomic_new(ROOT/'invalid_judgments'/f'{safe_id(case["case_id"])}_{order["permutation"]}_{attempt}.json',
                               {'response': response, 'error': str(exc)})
            atomic_new(path, {'status': 'invalid'}); return 'invalid'
        except BudgetStop: return 'budget_not_dispatched'
        except Exception as exc:
            write_json(output_path('judge_failures', case, order), {'error': str(exc)})
            return 'api_error'
        finally: api.close()
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        counts = Counter(pool.map(one, jobs))
    write_json(ROOT/f'judge_{phase}_complete.json', {'counts': dict(counts), 'seconds': time.monotonic()-start})
    print('judgments', dict(counts), flush=True)


def recover_syntax(phase):
    """Retain primary invalids; create a separately labelled sensitivity analysis."""
    repaired = []; failed = []
    for case in cohort(prepare(), phase):
        for order in case['orders']:
            path = output_path('judgments', case, order)
            if not path.exists() or read_json(path)['status'] != 'invalid': continue
            source = ROOT/'invalid_judgments'/f'{safe_id(case["case_id"])}_{order["permutation"]}_2.json'
            value = read_json(source)
            raw, edits = repair_dangling_quote(value['response']['raw'])
            try:
                verdict = validate_judgment(raw, order)
                if not edits: raise ValueError('No eligible syntax repair')
                target = output_path('recovered_judgments', case, order)
                if not target.exists():
                    atomic_new(target, {'status': 'syntax_recovered', 'verdict': verdict,
                        'source': ref(source), 'syntax_edits': edits, 'new_API_calls': 0,
                        'note': 'Supplement only. No string content, labels, or ranking changed.'})
                repaired.append(str(target))
            except (ValueError, KeyError, TypeError) as exc:
                failed.append({'source': str(source), 'error': str(exc)})
    write_json(ROOT/'syntax_recovery_audit.json', {'repaired': repaired, 'unrecovered': failed,
        'primary_judgments_unchanged': True, 'expansion_still_requires_primary_rule': True, 'new_API_calls': 0})
    print(json.dumps({'repaired': len(repaired), 'unrecovered': failed}, ensure_ascii=False), flush=True)


def report(phase, include_recovered=False):
    manifest = prepare(); rows = []; missing = []
    for case in cohort(manifest, phase):
        if len(case['candidates']) == 1:
            reference = {'useful': [], 'harmful': [], 'best': [case['original_key']],
                         'best_sets_agree': True, 'useful_sets_agree': True, 'harmful_sets_agree': True,
                         'final_excluded_both': False, 'rankings': [{case['original_key']: 0}]*2}
        else:
            paths = [output_path('judgments', case, o) for o in case['orders']]
            if include_recovered:
                paths = [p if p.exists() and read_json(p)['status'] == 'valid'
                         else output_path('recovered_judgments', case, o) for p, o in zip(paths, case['orders'])]
            if any(not p.exists() or read_json(p)['status'] not in ('valid', 'syntax_recovered') for p in paths):
                missing.append(case['case_id']); continue
            reference = reference_consensus(case, [read_json(p)['verdict'] for p in paths])
        row = {'case_id': case['case_id'], 'genre': case['genre'], 'cohort': case['cohort'],
               'candidate_count': len(case['candidates']), 'reference': reference,
               'useful_available': bool(reference['useful']), 'best_resolved': bool(reference['best']),
               'useful_intermediate_better_than_final': reference['final_excluded_both'] and bool(
                   (set(reference['best']) & set(reference['useful'])) - {case['final_key'], case['original_key']}),
               'arms': {}}
        for arm in ('orchestrator', 'base'):
            paths = [output_path('local', case, o, arm) for o in case['orders']]
            results = [read_json(p) if p.exists() else {'status': 'missing'} for p in paths]
            primary = results[0]
            valid = all(r['status'] == 'valid' for r in results)
            row['arms'][arm] = {'primary_status': primary['status'],
                'primary_key': primary.get('selected_key'),
                'order_stable': results[0].get('selected_key') == results[1].get('selected_key') if valid else None,
                'order_quality_comparison': order_comparison(results[0]['selected_key'], results[1]['selected_key'], reference) if valid else None,
                'orders': [{k:v for k,v in r.items() if k in ('status','selected_key','reason','error')} for r in results],
                **decision_metrics(primary.get('selected_key'), reference)}
        for arm, key in [('always_original', case['original_key']), ('always_final', case['final_key'])]:
            row['arms'][arm] = {'primary_status': 'baseline', 'primary_key': key, **decision_metrics(key, reference)}
        rows.append(row)
    aggregate = {}
    for arm in ('orchestrator', 'base', 'always_original', 'always_final'):
        values = [r['arms'][arm] for r in rows]
        aggregate[arm] = {'n': len(values),
            'valid_primary': sum(v['primary_status'] in ('valid', 'single_candidate', 'baseline') for v in values),
            'useful_selected': sum(v['useful'] for v in values),
            'harmful_selected': sum(v['harmful'] for v in values),
            'reference_best_selected': sum(v['reference_best'] is True for v in values),
            'reference_best_resolved': sum(v['reference_best'] is not None for v in values),
            'order_stable': sum(v.get('order_stable') is True for v in values),
            'order_comparable': sum(v.get('order_stable') is not None for v in values),
            'order_quality_comparison': dict(Counter(v['order_quality_comparison'] for v in values if v.get('order_quality_comparison')))}
    diagnostics = {'reference_best_resolved': sum(r['best_resolved'] for r in rows),
        'reference_best_sets_agree': sum(r['reference']['best_sets_agree'] for r in rows),
        'useful_candidate_cases': sum(r['useful_available'] for r in rows),
        'final_excluded_from_best_in_both_orders': sum(r['reference']['final_excluded_both'] for r in rows),
        'useful_intermediate_best_cases': sum(r['useful_intermediate_better_than_final'] for r in rows)}
    gate = manifest['design']['expansion']
    expand = not include_recovered and phase == 'pilot' and not missing and len(rows) == 10 and all([
        diagnostics['reference_best_resolved'] >= gate['reference_best_resolved_at_least'],
        diagnostics['useful_candidate_cases'] >= gate['useful_candidate_cases_at_least'],
        diagnostics['final_excluded_from_best_in_both_orders'] >= gate['final_excluded_from_best_in_both_orders_at_least']])
    accounting = read_json(ROOT/'api/accounting.json') if (ROOT/'api/accounting.json').exists() else {'confirmed_usd': 0, 'reserved_usd': 0}
    result = {'phase': phase, 'rows': rows, 'missing_reference': missing, 'aggregate': aggregate,
              'diagnostics': diagnostics, 'expand': expand, 'expansion_rule': gate, 'accounting': accounting,
              'manifest': ref(ROOT/'manifest.json'), 'default_agents_changed': False}
    suffix = '_syntax_supplement' if include_recovered else ''
    result['syntax_recovery_supplement'] = include_recovered
    write_json(ROOT/f'metrics_{phase}{suffix}.json', result)
    lines = ['# V5_SELECTION1 — 저장된 수정 후보 선택 실험', '',
             'RECOVERY1의 원문·작업자 단계 종료본·최종본을 고정해 비교했다. 새 글 생성과 재학습은 없다.',
             '동일한 글은 합쳤다. 총괄 adapter와 학습 전 base에 새로운 선택 프롬프트를 사용했다.',
             '기존 총괄의 실행 프로토콜 자체를 평가한 것은 아니다. 형식 실패와 품질 판단을 구분한다.', '',
             f'현재 단계: {phase}; 평가 집계 {len(rows)}편; 참조 평가 미완료 {len(missing)}편.', '',
             '| 선택 방식 | 유효 선택 | 유익한 수정 선택 | 새 결함·의미 훼손·악화 선택 | 참조 최선 선택 | 순서 변경에도 같은 선택 |',
             '|---|---:|---:|---:|---:|---:|']
    for arm, v in aggregate.items():
        lines.append(f'| {arm} | {v["valid_primary"]}/{v["n"]} | {v["useful_selected"]} | {v["harmful_selected"]} | '
                     f'{v["reference_best_selected"]}/{v["reference_best_resolved"]} | {v["order_stable"]}/{v["order_comparable"]} |')
    lines += ['', '두 순서의 Claude 평가가 모두 인정한 개선/악화만 해당 항목에 센다. 유익하지 않다고 해서 악화라는 뜻은 아니다.',
              '참조 최선은 두 평가의 최상위 동률 그룹 교집합이다. 교집합이 비면 미확정이며 최선 선택 분모에서 제외한다.',
              '모델의 형식 실패는 유익/악화 선택에 포함되지 않으므로 반드시 유효 선택 수와 함께 해석한다.', '',
              '새 결함·의미 훼손·악화는 종합 악화, 의미 훼손, 새 주요 결함, 출처 불일치 중 하나라도 두 평가가 인정한 경우다. 글 전체의 순위와는 구별한다.', '',
              '진단: `'+json.dumps(diagnostics, ensure_ascii=False)+'`',
              f'10→30편 확대 기준 통과: {expand}' if phase == 'pilot' else '30편 진단 집계 완료 여부는 미완료 수를 확인한다.',
              f'Claude 비용: ${accounting["confirmed_usd"]:.6f}; 미확정 예약 ${accounting["reserved_usd"]:.6f}.', '',
              '제약: 재사용 진단 자료이며 사람 정답이 아니다. 동일 평가 모델의 두 순서는 독립적인 두 평가자가 아니다.',
              '후보 풀에서의 사후 선택 결과이며 실제 협업 루프의 품질 향상이나 복구 성능을 증명하지 않는다.',
              f'상세 자료: `{ROOT}`.']
    if include_recovered:
        lines.insert(2, '보조 집계: 두 평가 응답의 JSON 문법만 복구했다. 판정 문자열과 순위는 수정하지 않았고 원래 실패 응답을 보존했다. 이 표로 30편 확대 여부를 바꾸지 않는다.\n')
    report_path = REPO/('reports/V5_SELECTION1_SYNTAX_SUPPLEMENT.md' if include_recovered else 'reports/V5_SELECTION1.md')
    report_path.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'phase': phase, 'aggregate': aggregate, 'diagnostics': diagnostics, 'expand': expand,
                      'missing': missing, 'cost': accounting['confirmed_usd']}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=['prepare', 'gpu', 'local', 'judge', 'report', 'recover_syntax'])
    parser.add_argument('--phase', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--include-syntax-recovery', action='store_true')
    args = parser.parse_args()
    if args.part == 'prepare': preflight()
    elif args.part == 'report': report(args.phase, args.include_syntax_recovery)
    else: globals()[args.part](args.phase)
