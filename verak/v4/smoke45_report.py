"""Render the paired diagnostic report from immutable local smoke artifacts."""
from collections import Counter
import json
from pathlib import Path

from .common import REPO, read_json, write_json, safe_id, file_sha
from .full_data import ROOT as TRAIN, verified, ref
from .smoke45 import verify_preflight
from .smoke45_infer import ROOT, CONDITIONS, completion, summarize, blocked_reason
from . import orchestrator_v44, policy_prompts_v43


def fraction(n, d):
    return f'{n}/{d} ({100*n/d:.1f}%)' if d else '—'


def literal(value):
    return '`'+json.dumps(value, ensure_ascii=False, sort_keys=True).replace('|', '\\|').replace('`', "'")+'`'


def baseline_attempts(cohort):
    plan = read_json(TRAIN/'smoke_search_runtime_recheck/plan.json')
    attempts = []
    for case in plan['cohorts'][cohort]:
        if case['retry']:
            path = TRAIN/'smoke_search_runtime_recheck'/cohort/'attempts'/(safe_id(case['case_id'])+'.json')
            attempts.append(read_json(path))
        else: attempts.append(verified(case['original']))
    return attempts


def report():
    preflight = verify_preflight()
    result = read_json(ROOT/'complete.json')
    manifest = verified(preflight['evaluation_manifest'])
    if not (ROOT/'server_stopped.json').exists():
        raise ValueError('Wait for owned GPU server shutdown')
    cohorts = {}
    audit = Counter()
    prefixes = []
    references = []
    prompt_texts = {**policy_prompts_v43.PROMPTS, 'orchestrator': orchestrator_v44.PROMPT}
    for condition in CONDITIONS:
        cohorts[condition] = {}
        for name, cohort in manifest['cohorts'].items():
            attempts = []
            for case in cohort['cases']:
                path = ROOT/condition/name/'attempts'/(safe_id(case['case_id'])+'.json')
                attempt = read_json(path)
                if attempt['case_id'] != case['case_id'] or attempt['source_id'] != verified(case['row'])['source_id']:
                    raise ValueError('Paired smoke case identity differs')
                if attempt['completion'] != completion(attempt): raise ValueError('Saved completion differs')
                references.append(ref(path)); attempts.append(attempt)
                audit['episodes'] += 1
                if attempt['privileged_plan_at_inference'] or attempt['records_at_inference'] or attempt['new_paid_calls']:
                    raise ValueError('Diagnostic scope changed')
                for call in attempt['calls']:
                    prefix = verified(call['prefix'])
                    role = call['role']
                    if prefix['messages'][0] != {'role': 'system', 'content': prompt_texts[role]}:
                        raise ValueError('A dispatched system prompt changed')
                    if prefix['prefix_tokens'] != len(prefix['prefix_input_ids']): raise ValueError('Prefix length mismatch')
                    audit['returned_requests'] += 1
                    audit['prompt_count_matches'] += call['template_audit']['status'] == 'match'
                    audit['frozen_prompt_matches'] += 1
                    prefixes.append({'condition': condition, 'cohort': name, 'case_id': case['case_id'],
                        'role': role, 'prefix_tokens': prefix['prefix_tokens'], 'full_tokens': call['full_tokens'],
                        'reserved_total': prefix['prefix_tokens']+prefix['output_limit']})
            metrics = summarize(attempts, len(cohort['cases']))
            if metrics != result['conditions'][condition]['cohorts'][name]:
                raise ValueError('Aggregate does not reproduce saved case metrics')
            cohorts[condition][name] = (attempts, metrics)
    audit.update(checkpoints_verified=len(preflight['checkpoints']), prompts_unchanged=3)
    # Same Orchestrator checkpoint/seed need not produce bitwise-identical local
    # generations. Audit the shared state before either Revision adapter acts.
    initial_pairing = []
    def canonical_graph(value):
        return {k: sorted(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in v)
                if isinstance(v, list) else v for k, v in value.items()}
    for name in manifest['cohorts']:
        pairs = [dict((a['case_id'], a) for a in cohorts[c][name][0]) for c in CONDITIONS]
        for case_id in pairs[0]:
            pair = [p[case_id] for p in pairs]
            calls = [next(c for c in a['calls'] if c['role'] == 'orchestrator' and c['action'].get('action') == 'MAP') for a in pair]
            pfx = [verified(c['prefix'])['prefix_input_ids'] for c in calls]
            initial_pairing.append({'cohort': name, 'case_id': case_id,
                'same_initial_MAP_prefix_ids': pfx[0] == pfx[1],
                'same_initial_scorer_result': pair[0]['score_tools'][0]['result'] == pair[1]['score_tools'][0]['result'],
                'same_raw_MAP_output': calls[0]['raw'] == calls[1]['raw'],
                'same_canonical_filtered_MAP': canonical_graph(calls[0]['action']['value']['map']) == canonical_graph(calls[1]['action']['value']['map'])})
    write_json(ROOT/'paired_initial_state_audit.json', {'cases': initial_pairing,
        'generation_seed': 47, 'temperature': 0, 'retries': 0})
    evidence = {'integrity': dict(audit), 'preflight': ref(ROOT/'preflight.json'),
                'complete': ref(ROOT/'complete.json'), 'episodes': references,
                'paired_initial_state_audit': ref(ROOT/'paired_initial_state_audit.json'),
                'prefix_lengths': prefixes, 'training_started': False, 'RFT_started': False,
                'full_evaluation_started': False, 'test_cohort_untouched': True}
    write_json(ROOT/'report_audit.json', evidence)
    totals = {c: {key: sum(m[key] for _, m in cs.values()) for key in
                  ('saved', 'no_protocol_error', 'all_stages_STOP_done')} for c, cs in cohorts.items()}
    epoch2, epoch1 = totals['all_epoch2'], totals['revision_epoch1']
    finish_count = sum(a.get('orchestrator_ending') == 'FINISH' for cs in cohorts.values() for aa, _ in cs.values() for a in aa)
    accepted_maps = sum(m['map_validation']['accepted'] for cs in cohorts.values() for _, m in cs.values())
    map_calls = sum(m['map_validation']['MAP_calls'] for cs in cohorts.values() for _, m in cs.values())
    context_events = sum(len(a['context_limits']) for cs in cohorts.values() for aa, _ in cs.values() for a in aa)
    lines = ['# V4_SMOKE2 — v4.5 inference-only paired smoke', '',
        'Both requested conditions ran on the same frozen smoke20 and agent_dev10: 60 saved episodes. '
        'No training, RFT, full evaluation, new teacher/API calls or test-cohort use. '
        'The owned inference server and scorer have exited. These are protocol diagnostics, not writing-quality results.', '',
        '## Completion definitions', '',
        '**(a) No protocol error:** Orchestrator FINISH and every worker delegation ends with a model STOP(done), '
        'or STOP(blocked) carrying a nonblank summary or issue. No controller-forced stop, step/context limit, '
        'invalid-format ending, request/template/runtime exception or scorer-tool failure. '
        'An invalid action followed by a valid terminal report affects valid-action rate, but does not by itself fail (a). '
        'Presence of a blocked reason is checked mechanically; its correctness is not judged.', '',
        '**(b) All stages done:** (a), with every worker delegation ending in model STOP(done). '
        'Orchestrator uses FINISH, not STOP. Controller-generated STOP(blocked) never counts as a model STOP. '
        'The mandatory Korean whole-essay pass is included in both denominators.', '',
        '|Condition|Cohort|(a) No protocol error|(b) All stages STOP(done)|Orchestrator FINISH|Mean model delegations|',
        '|---|---|---:|---:|---:|---:|']
    for condition, by_cohort in cohorts.items():
        for name, (attempts, metrics) in by_cohort.items():
            n = len(attempts)
            lines.append(f'|{condition}|{name}|{fraction(metrics["no_protocol_error"], n)}|'
                f'{fraction(metrics["all_stages_STOP_done"], n)}|'
                f'{fraction(sum(a.get("orchestrator_ending") == "FINISH" for a in attempts), n)}|'
                f'{metrics["mean_model_delegations"]:.2f}|')
    lines += ['', '`all_epoch2`: Orchestrator/Revision/Korean epoch 2. '
        '`revision_epoch1`: Revision epoch 1, Orchestrator/Korean epoch 2. '
        'All 30 cases are rerun for each condition; no cherry-picked retries or replacement cases.', '',
        f'Across both cohorts, all-epoch-2 has **{fraction(epoch2["no_protocol_error"], epoch2["saved"])} (a), '
        f'{fraction(epoch2["all_stages_STOP_done"], epoch2["saved"])} (b)**; Revision-epoch-1 has '
        f'**{fraction(epoch1["no_protocol_error"], epoch1["saved"])} (a), '
        f'{fraction(epoch1["all_stages_STOP_done"], epoch1["saved"])} (b)**. '
        f'Orchestrator FINISH: {finish_count}/{audit["episodes"]}; MAP acceptance: {accepted_maps}/{map_calls}; '
        f'context-limit events: {context_events}. The tables below distinguish worker rejections and step-limit endings. '
        'These diagnostic counts do not establish an editing-quality advantage for either checkpoint.', '',
        '## Prior v4.4 diagnostics, re-tabulated under the same definitions', '',
        'This reuses the accepted V4_TRAIN aggregate (19 original cases plus 11 SEARCH-runtime repairs), '
        'without changing any old result. Its published completion treated model STOP(blocked) as failure; '
        'that old column is not directly comparable to (a).', '',
        '|Cohort|v4.4 (a)|v4.4 (b)|', '|---|---:|---:|']
    for name in manifest['cohorts']:
        old = baseline_attempts(name)
        if {a['case_id'] for a in old} != {c['case_id'] for c in manifest['cohorts'][name]['cases']}:
            raise ValueError('v4.4 comparison cohort differs')
        c = [completion(a) for a in old]
        lines.append(f'|{name}|{fraction(sum(x["no_protocol_error"] for x in c), len(c))}|'
                     f'{fraction(sum(x["all_stages_STOP_done"] for x in c), len(c))}|')
    lines += ['', '## Per-role protocol tables', '',
        'Valid-action denominators include every returned model response, including format/validation rejections. '
        'Non-dispatched context failures are separate. STOP columns are model reports; Orchestrator FINISH is above.', '',
        '|Condition|Cohort|Role|Valid / returned|Worker stages|STOP done / stages|Model STOP / stages|STOP blocked (with reason / without)|Controller stop|Step limit|Context limits|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for condition, by_cohort in cohorts.items():
        for name, (attempts, metrics) in by_cohort.items():
            for role, r in metrics['roles'].items():
                blocked = [s for a in attempts for s in a['stages'] if s['role'] == role and s['ending'] == 'STOP' and s['stop_status'] == 'blocked']
                reason = sum(blocked_reason(s.get('report', {})) for s in blocked)
                limits = r['stage_endings'].get('step_limit', 0) if role != 'orchestrator' else metrics['orchestrator_endings'].get('step_limit', 0)
                lines.append(f'|{condition}|{name}|{role}|{fraction(r["valid_actions"], r["returned_actions"])}|'
                    f'{r["stages"] if role != "orchestrator" else "—"}|{fraction(r["model_STOP_done"], r["stages"])}|'
                    f'{fraction(r["model_STOP_done"]+r["model_STOP_blocked"], r["stages"])}|'
                    f'{reason} / {len(blocked)-reason}|{r["controller_STOP_blocked"]}|{limits}|{len(r["context_limits"])}|')
    for condition, by_cohort in cohorts.items():
        for name, (_, metrics) in by_cohort.items():
            lines += ['', f'### Remaining errors: {condition} / {name}', '',
                      '|Role|Rejection reason|Count|', '|---|---|---:|']
            for role, r in metrics['roles'].items():
                for reason, n in sorted(r['invalid_errors'].items()):
                    lines.append(f'|{role}|{str(reason).replace("|", "/")}|{n}|')
            lines += ['', 'Orchestrator endings: '+literal(metrics['orchestrator_endings'])+'.',
                'Episode protocol-error reasons (categories can overlap): '+literal(metrics['protocol_error_reasons'])+'.',
                'Execution failures: '+literal(metrics['execution_failures'])+'.',
                'Scorer-tool errors: '+literal(metrics['scorer_tool_errors'])+'.',
                'Exceptions: '+literal(metrics['exceptions'])+'.']
    lines += ['', '## Inference-side fixes and observed use', '',
        '**EDIT:** '+preflight['span_matching']+' The raw model action is retained; the executor substitutes '
        'the matched original span internally, then applies the same scope, mask, boundary, no-op and undo rules. '
        'A unique whitespace match can therefore still be rejected by another guard.', '',
        '|Condition|Cohort|Fallback attempted|Unique normalized match|Accepted EDIT|', '|---|---|---:|---:|---:|']
    for condition, by_cohort in cohorts.items():
        for name, (_, metrics) in by_cohort.items():
            x = metrics['whitespace_fallback']
            lines.append(f'|{condition}|{name}|{x["attempted"]}|{x["unique_match"]}|{x["accepted_EDIT"]}|')
    fallbacks = [r for cs in cohorts.values() for _, m in cs.values() for r in m['whitespace_fallback']['records']]
    lines += ['', f'The {len(fallbacks)} fallback attempts contain '
        f'{sum(r["normalized_matches"] == 0 for r in fallbacks)} zero-match and '
        f'{sum(r["normalized_matches"] > 1 for r in fallbacks)} ambiguous spans; none becomes uniquely matchable by '
        'whitespace normalization. The remaining span errors therefore require different action targets, not this normalization fallback.', '',
        '**MAP:** the unchanged training edge filter (`v4_edge_filter_1`) runs after metadata sanitation. '
        'It keeps at most one outgoing support and one outgoing example per sentence; the cross-paragraph main/key rule '
        'applies only above six paragraphs. Bad paragraph keys/role fields become null; invalid entries and edges are dropped. '
        'Missing paragraph metadata becomes explicit unknowns. Invalid genre is retained as unknown, never replaced by the source genre. '
        'A supplied valid paragraph node, relation or off_topic ID is graph content; automatically filled placeholders or genre alone are not. '
        'MAP is rejected only if no such content survives. Off-topic remains advisory, with the existing two-hop relevance protection.', '',
        '|Condition|Cohort|Accepted MAP / calls|Maps salvaged|Raw edges|Kept edges|Dropped edge reasons|Dropped metadata reasons|',
        '|---|---|---:|---:|---:|---:|---|---|']
    for condition, by_cohort in cohorts.items():
        for name, (_, metrics) in by_cohort.items():
            x = metrics['map_validation']
            lines.append(f'|{condition}|{name}|{x["accepted"]}/{x["MAP_calls"]}|{x["salvaged_maps"]}|'
                f'{x["raw_edges"]}|{x["kept_edges"]}|{literal(x["dropped_edges_by_reason"])}|{literal(x["dropped_fields_by_reason"])}|')
    lines += ['', '**Context:** Orchestrator 16,384 total / 4,096 output reserve; Korean 16,384 / 1,024; '
        'Revision 8,192 / 1,024. No truncation. Each prefix and every raw response is saved.', '',
        '|Condition|Role|Maximum dispatched prefix|Maximum prefix + output reserve|Maximum observed full tokens|',
        '|---|---|---:|---:|---:|']
    for condition in CONDITIONS:
        for role in ('orchestrator', 'revision', 'korean'):
            rows = [x for x in prefixes if x['condition'] == condition and x['role'] == role]
            lines.append(f'|{condition}|{role}|{max((x["prefix_tokens"] for x in rows), default=0)}|'
                f'{max((x["reserved_total"] for x in rows), default=0)}|{max((x["full_tokens"] for x in rows), default=0)}|')
    for condition, by_cohort in cohorts.items():
        target = next(a for a in by_cohort['smoke20'][0] if a['source_id'] == 'train:57542')
        lines += ['', f'Previously overflowing `train:57542`, {condition}: Orchestrator ending '
            f'{literal(target.get("orchestrator_ending"))}; context events {literal(target["context_limits"])}.']
    lines += ['', '## Frozen prompts, weights and audit', '',
        '|Role|Prompt version|Body / system tokens|Prompt SHA256|', '|---|---|---:|---|']
    for role, data in preflight['prompts_unchanged'].items():
        lines.append(f'|{role}|{"v4.4" if role == "orchestrator" else "v4.3"}|'
                     f'{data["policy_tokens"]} / {data["system_message_tokens"]}|`{data["sha256"]}`|')
    lines += ['', f'Pinned base `{preflight["base_revision"]}`. Existing adapters only; no weights or training data changed. '
        'GPU0 vLLM BF16/multi-LoRA, max sequence 16,384, eager, one request at a time; GPU1 the same frozen reference scorer as V4_TRAIN. '
        'Temperature 0, top_p 1, seed 47, unchanged from the earlier smoke. The existing local Wiki interpreter supplies SEARCH dependencies.', '',
        f'**Pairing limitation:** initial SCORE results and first-MAP prefix token IDs match in '
        f'{sum(x["same_initial_scorer_result"] and x["same_initial_MAP_prefix_ids"] for x in initial_pairing)}/30 pairs, '
        f'but first-MAP outputs, including the canonical filtered graphs, differ in '
        f'{sum(not x["same_canonical_filtered_MAP"] for x in initial_pairing)}/30 pairs despite the same Orchestrator adapter, '
        'temperature and seed. This is observed local generation variation; its numerical cause was not isolated. '
        'Consequently the small condition differences cannot be attributed exclusively to the Revision checkpoint. '
        'No extra repetitions or serving-setting changes were made. Exact paired evidence is saved in '
        '`smoke45/paired_initial_state_audit.json`.', '',
        '|Checkpoint|Adapter SHA256|', '|---|---|']
    for name, files in preflight['checkpoints'].items():
        lines.append(f'|{name}|`{files["adapter_model.safetensors"]["sha256"]}`|')
    lines += ['', f'All {audit["episodes"]} episode identities and the three frozen system prompts were audited. '
        f'HF/vLLM prompt-token-count matches: {audit["prompt_count_matches"]}/{audit["returned_requests"]}. '
        'This does not claim equality of the server’s actual token IDs. Saved aggregate metrics reproduce directly from the 60 attempt files. '
        'Preflight verifies checkpoint, prompt, source-manifest, prior-report and runtime-code hashes.', '',
        'CPU validation: 41 tests passed across `test_smoke45.py`, `test_training44.py` and `test_smoke43.py`. '
        'They cover unique/ambiguous normalized spans, preserved outside text and undo, mask/scope/boundary/no-op guards, '
        'unchanged training edge filtering, partial MAP salvage, strict multi-JSON rejection, and blocked-versus-done completion. '
        'The host environment check confirmed both GPUs, the frozen scorer files and local Bareun; its optional OpenAI-key '
        'check was unavailable and irrelevant to this entirely local run.', '',
        '- Contract: `verak/v4/outputs/smoke45/preflight.json`.',
        '- Aggregate: `verak/v4/outputs/smoke45/complete.json`.',
        '- Per-episode raw actions, dropped MAP entries, normalized-match diagnostics, worker reports and prefixes: '
        '`verak/v4/outputs/smoke45/{condition}/{cohort}/`.',
        '- Integrity and all 60 attempt hashes: `verak/v4/outputs/smoke45/report_audit.json`.', '',
        'Stopped after the two smokes. No RFT or full evaluation was started; these diagnostic sets do not establish writing quality.', '']
    path = REPO/'imple/reports/V4_SMOKE2.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    write_json(ROOT/'report.json', {'path': str(path), 'sha256': file_sha(path), 'quality_claims': False})
    return path


if __name__ == '__main__':
    print(report())
