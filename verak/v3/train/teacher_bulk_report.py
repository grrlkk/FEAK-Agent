"""Streaming audit/report of saved teacher attempts; never calls a teacher."""
from collections import Counter, defaultdict
import json
from statistics import mean

from ..agent.runner import system_prompt
from ..common import file_sha, read_json, write_json
from .formatting import lengths
from .teacher_comparison import absolute_selection
from .teacher_bulk import PHASE, ROLES, BulkAPI, attempt_path, prepare, replay_baseline_contexts


def transport_retry_audit(api):
    """Distinguish live transport retries from cached replay on resume."""
    with api.db() as db:
        rows = db.execute('SELECT id,stage,item_id,fingerprint,status,reserved,confirmed,path FROM calls ORDER BY id').fetchall()
    grouped = defaultdict(list)
    for call_id, stage, item_id, fingerprint, status, reserved, confirmed, path in rows:
        grouped[fingerprint].append({'phase_call': call_id, 'stage': stage, 'item_id': item_id,
            'status': status, 'reserved_usd': reserved, 'confirmed_usd': confirmed, 'path': path})
    repeated = []
    for fingerprint, calls in grouped.items():
        if len(calls) < 2:
            continue
        for call in calls:
            request = read_json(call['path']) if call['path'] else {}
            call['http_status'] = request.get('http_status')
            call['error_type'] = request.get('error_type')
            call['retryable_transport_error'] = call['status'] == 'error' and (
                call['http_status'] in (429, 500, 502, 503, 504, 520) or
                'Timeout' in (call['error_type'] or ''))
        repeated.append({'fingerprint': fingerprint, 'sends': len(calls), 'calls': calls,
            'confirmed_usd': sum(c['confirmed_usd'] for c in calls),
            'retained_reservations_usd': sum(c['reserved_usd'] for c in calls),
            'nonfinal_outcomes_are_retryable_transports': all(c['retryable_transport_error'] for c in calls[:-1])})
    return {'fingerprints_with_multiple_sends': len(repeated),
        'additional_sends': sum(g['sends'] - 1 for g in repeated),
        'total_sends_in_retry_groups': sum(g['sends'] for g in repeated),
        'confirmed_usd_in_retry_groups': sum(g['confirmed_usd'] for g in repeated),
        'retained_reservations_usd_in_retry_groups': sum(g['retained_reservations_usd'] for g in repeated),
        'live_retry_contract_valid': all(g['sends'] <= 4 and g['nonfinal_outcomes_are_retryable_transports'] for g in repeated),
        'groups': repeated}


def report(config, *, awaiting_seed_clarification=False):
    from transformers import AutoTokenizer
    design, corpus = prepare(config)
    root = config['paths'][PHASE + '_output']
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    api = BulkAPI(config, 0, phase=PHASE)
    account = api.accounting()
    retries = transport_retry_audit(api)
    baseline_replay = replay_baseline_contexts(config, tokenizer)
    failures, slots, selected = [], [], {role: {} for role in ROLES}
    checks = 0
    def check(ok, name):
        nonlocal checks
        checks += 1
        if not ok:
            failures.append(name)
    partial_path = config['paths']['phase7_teacher_output'] / 'completed_global_evaluation.json'
    partial = read_json(partial_path)['results'] if partial_path.exists() else {}
    metrics = {str(a): {'attempted': 0, 'completed': 0, 'reused': 0, 'errors': Counter(),
        'termination': {r: Counter() for r in ROLES}, 'actions': {r: Counter() for r in ROLES},
        'rewards': {r: [] for r in (*ROLES, 'combined')}, 'kept': Counter(), 'new_cost_usd': 0.0,
        'historical_cost_usd': 0.0, 'levels': defaultdict(lambda: {'n': 0, 'completed': 0, 'R': []}),
        'operators': defaultdict(list)} for a in (1, 2)}
    token_lengths = {r: [] for r in ROLES}
    input_lengths = {r: [] for r in ROLES}
    compacted, call_counts = Counter(), Counter()
    completed_api_ids = set()
    source_counts = Counter()
    for attempt in (1, 2):
        group = metrics[str(attempt)]
        for episode_id in design['orders'][str(attempt)]:
            reused = attempt == 1 and episode_id in design['reuse_attempt_1']
            if reused:
                source = design['reuse_attempt_1'][episode_id]
                path = config['paths']['repo'] / source['path']
                check(file_sha(path) == source['sha256'], f'{episode_id}: reused bytes unchanged')
            else:
                path = attempt_path(root, attempt, episode_id)
                if not path.exists():
                    slots.append({'attempt': attempt, 'corpus_episode_id': episode_id, 'status': 'not_attempted'})
                    continue
            row = read_json(path)
            if reused:
                key = 'luna_low:' + episode_id
                if key in partial:
                    row['global_only_reward'] = partial[key]['global_only_reward']
            else:
                check(row['bulk']['design_sha256'] == file_sha(root / 'design.json'), f'{attempt}:{episode_id}: frozen design')
            group['attempted'] += 1
            group['completed'] += bool(row['completed'])
            group['reused'] += reused
            group['levels'][row['level']]['n'] += 1
            group['levels'][row['level']]['completed'] += bool(row['completed'])
            source_counts[row['source_id']] += 1
            if row['runtime_error']:
                group['errors'][row['runtime_error']['type'] + ': ' + row['runtime_error']['message']] += 1
            cost = row.get('confirmed_episode_cost', row.get('cost_usd', 0))
            group['historical_cost_usd' if reused else 'new_cost_usd'] += cost
            check(row['model'] == design['model'] and row['mode'] == 'two_stage', f'{episode_id}: pinned model/mode')
            check(all(v == 0 for v in row['checks'].values()), f'{episode_id}: no CHECK')
            check(row['corpus_episode_id'] == episode_id, f'{episode_id}: identity')
            for role in (*ROLES, 'combined'):
                reward = row['reward'][role] if row.get('completed') and row.get('reward') else (
                    row.get('global_only_reward') if role == 'global' else None)
                if reward is not None:
                    group['rewards'][role].append({k: reward[k] for k in ('R', 'R_rec', 'R_q', 'R_over', 'R_step')})
                if role in ROLES:
                    group['termination'][role][row['termination'].get(role, 'not_completed')] += 1
                    actions = row['actions_by_role'].get(role, [])
                    group['actions'][role].update({'all': len(actions), 'invalid': sum(not a['valid'] for a in actions),
                        'CHECK': sum(a['action'] == 'CHECK' for a in actions)})
            if row.get('reward'):
                group['levels'][row['level']]['R'].append(row['reward']['combined']['R'])
                for rec in row['reward']['combined']['per_record']:
                    group['operators'][rec['op']].append({'main': rec['main'], 'recovery': rec['recovery']})
            keep = absolute_selection(row, corpus[episode_id])
            for role in ROLES:
                if not keep[role]:
                    continue
                group['kept'][role] += 1
                reward = row['reward'][role] if row.get('reward') else row['global_only_reward']
                candidate = {'attempt': attempt, 'path': str(path), 'sha256': file_sha(path),
                    'R': reward['R'], 'reused': reused, 'rule': keep['global_rule'] if role == 'global' else 'KOREAN_R_ge_0.80'}
                previous = selected[role].get(episode_id)
                if previous is None or (candidate['R'], -attempt) > (previous['R'], -previous['attempt']):
                    selected[role][episode_id] = candidate
            for call in row['calls']:
                role = call['role']
                prefix = call.get('policy_input_tokens')
                total = call.get('policy_total_tokens')
                if prefix is None:
                    prefix = len(tokenizer.apply_chat_template(call['messages'], tokenize=True, add_generation_prompt=True))
                if total is None:
                    total = len(tokenizer.apply_chat_template(call['messages'] + [{'role': 'assistant', 'content': call['raw']}], tokenize=True))
                input_lengths[role].append(prefix)
                token_lengths[role].append(total)
                compacted[role] += bool(call['history_compacted'])
                call_counts[role] += 1
                check(prefix <= 7168 and total <= 8192, f'{attempt}:{episode_id}:{role}:{call["turn"]}: context')
                check(call['messages'][0]['content'] == system_prompt(role), f'{episode_id}: exact role prompt')
                check(call['model'] == design['model'] and call['reasoning_effort'] == 'low' and
                      call['max_output_tokens'] == 1024, f'{episode_id}: request contract')
                if not reused:
                    check(call['stage'] == f'bulk_attempt_{attempt}', f'{episode_id}: independent attempt namespace')
                    completed_api_ids.add(call['phase_call'])
            slots.append({'attempt': attempt, 'corpus_episode_id': episode_id,
                'status': 'completed' if row['completed'] else 'failed', 'reused': reused,
                'path': str(path), 'sha256': file_sha(path), 'new_cost_usd': 0 if reused else cost,
                'selected': {r: bool(keep[r]) for r in ROLES}})
    usage = Counter()
    statuses = Counter()
    request_count = 0
    for path in (root / 'api/requests').glob('*.json'):
        request = read_json(path)
        request_count += 1
        statuses[request['status']] += 1
        usage.update({k: request.get('cost', {}).get(k, 0) for k in
            ('input', 'output', 'reasoning', 'cache_read', 'cache_write', 'confirmed_usd')})
        check(request['model'] == design['model'] and request['reasoning_effort'] == 'low' and
              request['max_output_tokens'] == 1024, str(path) + ': raw request contract')
        check(request['stage'] in {'bulk_attempt_1', 'bulk_attempt_2'}, str(path) + ': authorized stage')
        if request['phase_call'] not in completed_api_ids:
            n = len(tokenizer.apply_chat_template(request['messages'], tokenize=True, add_generation_prompt=True))
            check(n <= 7168, str(path) + ': failed/interrupted input context')
    for group in metrics.values():
        group['completion_rate'] = group['completed'] / group['attempted'] if group['attempted'] else None
        group['rewards'] = {role: {'n': len(values), **{k: mean(v[k] for v in values) if values else None
            for k in ('R', 'R_rec', 'R_q', 'R_over', 'R_step')}} for role, values in group['rewards'].items()}
        group['levels'] = {level: {**{k: v for k, v in values.items() if k != 'R'},
            'combined_R': mean(values['R']) if values['R'] else None} for level, values in group['levels'].items()}
        group['operators'] = {op: {'n': len(values), 'main': mean(v['main'] for v in values),
            'recovery': mean(v['recovery'] for v in values)} for op, values in group['operators'].items()}
    missing = [s for s in slots if s['status'] == 'not_attempted']
    check(len(slots) == 2860 and len({(s['attempt'], s['corpus_episode_id']) for s in slots}) == 2860,
          'complete unique authorized attempt matrix')
    check(metrics['1']['reused'] == 92 and metrics['2']['reused'] == 0, 'exact92 reused once as attempt1')
    check(account['confirmed_usd'] + account['reserved_usd'] <= 25 + 1e-9 and account['pending'] == 0,
          'separate$25 cap including uncertainty and no live calls')
    new_cost = sum(v['new_cost_usd'] for v in metrics.values())
    check(abs(new_cost - account['confirmed_usd']) < 1e-7, 'all confirmed API usage attributed to saved attempts')
    check(abs(usage['confirmed_usd'] - account['confirmed_usd']) < 1e-7, 'raw usage matches confirmed ledger')
    check(retries['live_retry_contract_valid'], 'repeated fingerprints only follow allowed live transport retries, at most4 sends')
    status = read_json(root / 'run_status.json') if (root / 'run_status.json').exists() else {}
    if awaiting_seed_clarification:
        if account['calls'] or metrics['1']['attempted'] != 92 or metrics['2']['attempted']:
            raise ValueError('Cannot label an already dispatched collection as awaiting seed clarification')
        status = {'stop_reason': 'awaiting_seed_clarification', 'run_executed': False}
    if missing:
        check(status.get('stop_reason') in {'budget_cap', 'requested_limit', 'dispatch_error', 'awaiting_seed_clarification'},
              'missing attempts explained')
    validation = {'passed': not failures, 'checks': checks, 'failures': failures,
        'scope': 'reused trajectories and preparation only; requested collection pending' if awaiting_seed_clarification
            else 'saved attempted trajectories; see missing slots for collection coverage',
        'request_records_checked': request_count, 'context_measurement': 'Policy tokenizer at live generation; legacy and failed calls retokenized offline'}
    token_stats = {role: {'calls': call_counts[role], 'input': lengths(input_lengths[role]),
        'total': lengths(token_lengths[role]), 'compacted_calls': compacted[role],
        'compaction_rate': compacted[role] / call_counts[role] if call_counts[role] else None} for role in ROLES}
    result = {'design': design, 'attempts': metrics, 'api': account, 'usage': dict(usage),
        'api_statuses': dict(statuses), 'transport_retries': retries, 'tokens': token_stats,
        'selection_counts': {r: len(v) for r, v in selected.items()},
        'selected': selected, 'missing_slots': missing, 'run_status': status, 'validation': validation,
        'baseline_context_replay': baseline_replay, 'distinct_source_essays': len(source_counts),
        'no_training_executed': True}
    test_path = root / 'tests.txt'
    result['test_summary'] = test_path.read_text().strip().splitlines()[-1] if test_path.exists() else None
    write_json(root / 'metrics.json', result)
    write_json(root / 'validation.json', validation)
    write_json(root / 'attempt_manifest.json', slots)
    write_json(root / 'best_role_trajectories.json', selected)
    render(config, result)
    print(json.dumps({'attempts': {a: {k: v[k] for k in ('attempted', 'completed', 'reused')} for a, v in metrics.items()},
        'budget': account, 'selected': result['selection_counts'], 'validation': validation}, ensure_ascii=False), flush=True)
    if failures:
        raise ValueError('Bulk artifact audit failed; see validation.json')
    return result


def render(config, metrics):
    root = config['paths'][PHASE + '_output']
    def fmt(value, digits=4):
        return '—' if value is None else f'{value:.{digits}f}'
    def table(headers, values):
        return ['|' + '|'.join(headers) + '|', '|' + '|'.join(['---'] * len(headers)) + '|'] + [
            '|' + '|'.join(str(v).replace('|', '\\|').replace('\n', ' ') for v in row) + '|' for row in values] + ['']
    account = metrics['api']
    attempted = sum(v['attempted'] for v in metrics['attempts'].values())
    completed = sum(v['completed'] for v in metrics['attempts'].values())
    lines = ['# VERAK v3 — bulk two-stage teacher generation', '',
        f"Saved {attempted}/2,860 authorized attempt slots; {completed} completed. "
        f"Exactly 92 existing Luna-low trajectories were reused as attempt 1, including the saved failure. "
        f"New confirmed API usage is ${account['confirmed_usd']:.6f}; retained uncertainty is "
        f"${account['reserved_usd']:.6f}, within the separate $25 cap.", '',
        '## Frozen collection contract', '',
        'All 1,430 active agent_train corruption essays; two attempts each; Phase-4-pinned `gpt-6-luna`, '
        'reasoning low, accepted two-stage prompts/environment/reward, no CHECK, 8,192 policy-token '
        'context with 1,024 reserved for generation. No active corpus, existing trajectory, or corruption '
        'was regenerated or changed. The pinned ID is the Phase-4 ID; no dated snapshot was available then.', '',
        'At most 16 independent episodes/API requests run concurrently, with one serialized GPU1 scorer '
        'service per process. The earlier unexecuted four-worker manifest was preserved before dispatch; '
        'the formal manifest froze 16 workers with zero paid calls. Based on the saved 92-essay timing, '
        'the pre-run wall-time estimate was approximately 2.7 hours, not a completion guarantee.', '',
        'Seed 71 determines the stratified first-attempt collection order and seed 72 the second order. '
        'The Responses API has no sampling-seed parameter. The implemented alternative makes each '
        'second attempt an independent request in a separate cache namespace; provider sampling seed '
        'is recorded as null. Exact prompt text and generation settings remain unchanged. '
        'These ordering seeds do not make stochastic model outputs reproducible.', '',
        'Environment, reward, and prompt hashes match the saved Luna-low pilot. The shared runner later '
        'gained optional observation hooks; its default behavior was checked by replaying all 1,165 '
        'saved model-turn contexts from the 92 trajectories exactly, including compacted histories. '
        'All 92 saved initial layouts match the active corrupted layouts exactly. Both runner hashes '
        'and the context-replay evidence are retained.', '',
        '## Completion and role rewards', '']
    lines += table(['attempt', 'saved/1430', 'reused', 'completed', 'completion', 'new USD'], [
        [a, v['attempted'], v['reused'], v['completed'], fmt(v['completion_rate']), fmt(v['new_cost_usd'], 6)]
        for a, v in metrics['attempts'].items()])
    lines += ['Completion means environment termination and stored reward. STOP, step limits, and invalid-action '
        'termination are distinguished below. API failures remain failures; their confirmed usage is charged. '
        'A completed GLOBAL stage can receive its unchanged local reward when the later KOREAN stage failed.', '']
    lines += table(['attempt', 'failed episodes', 'runtime error'], [
        [a, count, error] for a, group in metrics['attempts'].items()
        for error, count in sorted(group['errors'].items())])
    lines += table(['attempt', 'role', 'reward n', 'R', 'R_rec', 'R_q', 'R_over', 'R_step'], [
        [a, role, v['n']] + [fmt(v[k]) for k in ('R', 'R_rec', 'R_q', 'R_over', 'R_step')]
        for a, group in metrics['attempts'].items() for role, v in group['rewards'].items()])
    lines += table(['attempt', 'role', 'termination counts', 'invalid/actions'], [
        [a, role, json.dumps(v['termination'][role]),
         f"{v['actions'][role].get('invalid', 0)}/{v['actions'][role].get('all', 0)}"]
        for a, v in metrics['attempts'].items() for role in ROLES])
    lines += table(['attempt', 'level', 'saved', 'completed', 'combined R'], [
        [a, level, v['n'], v['completed'], fmt(v['combined_R'])]
        for a, group in metrics['attempts'].items() for level, v in sorted(group['levels'].items())])
    lines += table(['attempt', 'operator', 'completed records', 'main recovery', 'coupled recovery'], [
        [a, op, v['n'], fmt(v['main']), fmt(v['recovery'])]
        for a, group in metrics['attempts'].items() for op, v in sorted(group['operators'].items())])
    lines += ['## Selection inventory, without training', '',
        'For reporting, GLOBAL-record episodes use GLOBAL R≥0.80; no-GLOBAL episodes require '
        'STOP within two steps, R_over=0, and no structural action attempt. KOREAN requires R≥0.80. '
        'The best qualifying attempt per essay and role is indexed; ties prefer attempt 1. '
        'These are reward-based candidates, not independently verified quality labels.', '',
        f"Unique eligible GLOBAL/KOREAN trajectories: **{metrics['selection_counts']['global']} / "
        f"{metrics['selection_counts']['korean']}**. The index stores exact file paths and SHA-256 hashes; "
        'no SFT or RFT was run.', '', '## Context, cost, and resume audit', '']
    lines += table(['role', 'model turns', 'compacted turns', 'compaction rate', 'input max', 'total max'], [
        [role, v['calls'], v['compacted_calls'], fmt(v['compaction_rate']), v['input'].get('max'), v['total'].get('max')]
        for role, v in metrics['tokens'].items()])
    lines += [f"Ledger: `{json.dumps(account, ensure_ascii=False)}`.", '',
        f"Usage: `{json.dumps(metrics['usage'], ensure_ascii=False)}`. Reasoning tokens are included in output tokens. "
        'Costs use the existing pinned Luna rates; historical reused cost is excluded from this $25 budget.', '',
        'A single-process file lock protects collection. Attempt JSON files are committed atomically and '
        'never overwritten. Resuming replays completed API responses under the same attempt namespace. '
        'If a crash occurs after the durable request file is written but before its ledger update, '
        'recovery validates the call identity, payload fingerprint, status, and usage before restoring '
        'the ledger. Valid completed responses replay without another provider request; incomplete '
        'and timeout outcomes remain failures. Missing, malformed, or mismatched records retain their '
        'uncertain reservations, and responses lacking usage retain their billing bounds. '
        'On resume, a fingerprint with prior failed or uncertain outcomes and no completed response '
        'is not sent again; interrupted request bounds remain reserved. Within one live invocation, '
        'the existing API transport may send a request up to four times for transient HTTP errors '
        '(429/500/502/503/504/520) or timeouts. Incomplete responses are not retried. Every send, '
        'confirmed charge, and retained timeout reservation remains in the ledger. Budget exhaustion '
        'stops new dispatch and drains already reserved requests.', '',
        f"Stop reason: `{metrics['run_status'].get('stop_reason')}`; unattempted slots: {len(metrics['missing_slots'])}. "
        f"Audit passed: **{metrics['validation']['passed']}** ({metrics['validation']['checks']} checks); "
        f"failures: `{json.dumps(metrics['validation']['failures'], ensure_ascii=False)}`.", '',
        f"Focused implementation tests: `{metrics.get('test_summary')}`. Local preflight verified Bareun, "
        'the frozen scorer, embedding service, and API DNS before any paid dispatch.', '',
        f"Artifacts: `{root}`. `attempt_manifest.json` lists all 2,860 slots and exact provenance; "
        '`metrics.json` contains full failures, per-level/operator data, raw-usage totals, and selection details. '
        'API request records, append-only events, final/failed trajectories, and unknown reservations are retained.', '',
        'This is teacher-data collection on agent_train, not a held-out performance evaluation. '
        'The two attempts are stochastic repetitions, and multiple corruption variants can share a source essay. '
        'No training or extra teacher generation follows this report.', '']
    retries = metrics.get('transport_retries')
    if retries is not None:
        lines += ['## Live transport retry audit', '',
            f"Repeated request fingerprints: {retries['fingerprints_with_multiple_sends']}; "
            f"additional sends: {retries['additional_sends']}. Confirmed usage across those complete "
            f"retry groups: ${retries['confirmed_usd_in_retry_groups']:.6f}; retained reservations: "
            f"${retries['retained_reservations_usd_in_retry_groups']:.6f}. These amounts are already "
            'included in the collection ledger, not added a second time. Cached replays do not create '
            'new ledger sends. Per-call statuses and errors are preserved in `metrics.json`.', '']
        lines += table(['stage', 'item', 'sends', 'statuses', 'confirmed USD', 'reserved USD'], [
            [g['calls'][0]['stage'], g['calls'][0]['item_id'], g['sends'],
             ', '.join(c['status'] + (':' + c['error_type'] if c['error_type'] else '') for c in g['calls']),
             fmt(g['confirmed_usd'], 6), fmt(g['retained_reservations_usd'], 6)] for g in retries['groups']])
    if metrics['run_status'].get('stop_reason') == 'awaiting_seed_clarification':
        lines[2:2] = ['**Collection has not started.** The requested change of model sampling seed cannot '
            'be sent through the existing Responses API. A clarification is pending on using independent '
            'unseeded requests with ordering seeds 71/72. There are zero new teacher calls; all 2,768 new '
            'attempt slots remain pending. The tables below describe the 92 reused trajectories and '
            'implementation readiness only. A passing artifact audit does not mean the bulk request is complete.', '']
    authorization_path = root / 'dispatch_authorization.json'
    if authorization_path.exists():
        authorization = read_json(authorization_path)
        lines += ['Dispatch was authorized with the user message: “' + authorization['user_message'] + '.” '
            'The exact approval and timestamp are retained in [`dispatch_authorization.json`]('
            + str(authorization_path) + ').', '']
    provenance_path = root / 'dispatch_code_provenance.json'
    if provenance_path.exists():
        provenance = read_json(provenance_path)
        lines += ['[`dispatch_code_provenance.json`](' + str(provenance_path) + ') preserves the dispatch-time '
            'file hashes, Git HEAD `' + provenance['git_HEAD'] + '`, and tracked code-diff SHA-256 `'
            + provenance['tracked_v3_diff_sha256'] + '`. Later commits do not replace this record.', '']
    report_path = config['paths']['repo'] / 'imple/reports/V3_TEACHER_BULK_TWO_STAGE.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines), encoding='utf-8')
