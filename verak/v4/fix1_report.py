"""Saved-output-only fix1 report, including dev corruption mechanics, no models."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, stdev

from .common import REPO, file_sha, read_json, write_json
from .fix1_taxonomy import ROOT, TYPES, rejection_rows
from .full_data import ref, verified
from .fix1_infer import completion, blocked_reason
from .scale2_projection import (restore_document, project_document, private_document, ProjectionUnknown)
from verak.v3.reward.recovery import recover_records


def spread(values, *, percent=False):
    factor = 100 if percent else 1
    values = [v*factor for v in values]
    return {'mean': mean(values), 'sd': stdev(values) if len(values)>1 else None,
            'min': min(values), 'max': max(values), 'runs': len(values)}


def fmt(values, percent=False):
    v = spread(values, percent=percent)
    return f"{v['mean']:.2f}" + (f" ± {v['sd']:.2f} [{v['min']:.2f}, {v['max']:.2f}]" if len(values)>1 else '')


def stage_protocol(stage):
    return (stage.get('ending') == 'STOP' and (stage.get('stop_status') == 'done' or
            stage.get('stop_status') == 'blocked' and blocked_reason(stage.get('report', {}))))


def recovery(case, attempt):
    # This function runs only after all policy generations are saved. These
    # references, source answers and records never enter model observations.
    prepared = verified(case['prepared']); row = verified(case['row'])
    source = restore_document(prepared['reward_source'])
    initial = restore_document(prepared['reward_initial'])
    if initial.text != attempt['initial']['text']:
        raise ValueError('Initial normalization/reward text differs')
    records = prepared['corpus']['records']
    result = {'case_id': case['case_id'], 'records': records,
        'initial': recover_records(source, initial, records, corrupted=initial), 'final': None, 'unknown': None}
    try:
        public = restore_document(attempt['final_token_document'])
        final = private_document(project_document(public, row['normalization'], attempt['final']['lineage']),
                                 prepared.get('stable_id_aliases'))
        result['final'] = recover_records(source, final, records, corrupted=initial)
    except ProjectionUnknown as exc:
        result['unknown'] = str(exc)
    return result


def report():
    from .fix1_run import verify_preflight
    preflight = verify_preflight()
    results = read_json(ROOT/'evaluation/complete.json')
    taxonomy = read_json(ROOT/'taxonomy.json')
    data = read_json(ROOT/'data/manifest.json')
    manifest = verified(preflight['evaluation_manifest'])
    cases = {c['case_id']: c for cohort in manifest['cohorts'].values() for c in cohort['cases']}
    inventory = []; cells = []; requests = Counter()
    for cell in results['cells']:
        attempts = []; rejects = []; mechanics = []
        for path in sorted((Path(cell['directory'])/'attempts').glob('*.json')):
            attempt = read_json(path)
            assert completion(attempt) == attempt['completion']
            assert not attempt['privileged_plan_at_inference'] and not attempt['records_at_inference']
            inventory.append(ref(path)); attempts.append(attempt)
            rejects.extend(rejection_rows(attempt, path, near=False))
            for call in attempt['calls']:
                requests[call['template_audit']['status']] += 1
            if cell['cohort'] == 'agent_dev10':
                mechanics.append(recovery(cases[attempt['case_id']], attempt))
        if len(attempts) != (20 if cell['cohort'] == 'smoke20' else 10):
            raise ValueError('An evaluation cell is incomplete')
        measured = {**cell, 'rejections': rejects, 'recovery': mechanics, 'extra': {}}
        for role in ('revision', 'korean'):
            stages = [s for a in attempts for s in a['stages'] if s['role'] == role]
            measured['extra'][role] = {'stages': len(stages), 'protocol_stages': sum(stage_protocol(s) for s in stages),
                'STOP_done': sum(s['ending'] == 'STOP' and s['stop_status'] == 'done' for s in stages),
                'rejection_types': dict(Counter(r['type'] for r in rejects if r['role'] == role))}
        cells.append(measured)
    if len(inventory) != 240:
        raise ValueError('Expected 240 saved episodes')
    value = {'cells': cells, 'attempt_inventory': inventory, 'request_audits': dict(requests),
             'recovery_definition': 'v1 main per-record recovery at final endpoint; normalization baseline separate; unknown projections not zero',
             'test_cohort_untouched': True, 'API_calls': 0}
    write_json(ROOT/'report_metrics.json', value)
    lines = ['# V4_FIX1 — targeted worker corrections', '',
        'Completed the failure audit, v4.6 exception, deterministic SFT-data fixes, one continuation epoch per worker, '
        'and 240 diagnostic executions. No external API calls, test-cohort evaluation, Orchestrator training, or large RFT.', '',
        'Both old and new workers run in v4.6: old Revision epoch 1 / Korean epoch 2; new continuations from those exact checkpoints. '
        'The Orchestrator remains epoch 2 with its frozen prompt. The comparison measures the combined data corrections and continued SFT; '
        'it does not isolate the three data fixes or extra training alone.', '',
        '## 1. Saved failure taxonomy', '', taxonomy['scope'], '',
        f"Audited {taxonomy['episodes']} saved executions and every one of {taxonomy['rejections']} rejected worker actions. "
        'Historical SEARCH dependency errors remain visible as other/search_dependency, rather than model editing errors.', '',
        '|Role|Type|Count|Near-copy / short ambiguous|Next action or stage ending|', '|---|---|---:|---|---|']
    for role, kinds in taxonomy['summary'].items():
        for kind, stats in kinds.items():
            lines.append(f"|{role}|{kind}|{stats['count']}|{stats['near_copy']} / {stats['short_ambiguous']}|`{json.dumps(stats['next'], ensure_ascii=False)}`|")
    lines += ['', 'Near-copy definition: '+taxonomy['near_copy_policy'],
        'Short ambiguous means at most one whitespace-delimited eojeol. Longer repeated phrases also count as ambiguous. '
        'Character deletions/substitutions are shown directly; particle/ending identity is not claimed from edit distance alone.', '',
        '### Five examples per type (all available if fewer)', '']
    for kind in TYPES:
        examples = taxonomy['examples'][kind]
        lines += [f'#### {kind}', '']
        if not examples:
            lines += ['No saved instance of this exact type; no examples invented.', '']; continue
        for i, example in enumerate(examples, 1):
            lines += [f"{i}. `{example['case_id']}` / {example['role']} / next `{example['next']}`", '',
                'Sentence: '+str(example['sentence_text']), '',
                'Action: `'+json.dumps(example['action'], ensure_ascii=False)+'`', '',
                'Error: '+example['error'], '']
            if example.get('nearest'):
                lines += ['Nearest real span: `'+json.dumps(example['nearest'], ensure_ascii=False)+'`', '']
    lines += ['## 2. v4.6 and frozen contracts', '',
        'The only new acceptance rule is after == before + one of . ? !, with a nonempty before containing no unprotected '
        'terminal punctuation. Wording changes plus punctuation, punctuation inside a sentence, repeated marks, splits and mask '
        'changes do not qualify. Decimal/URL/mask protection follows v4.3; all other v4.5 scope, matching, step and controller rules remain.', '',
        '|Role|Prompt SHA256|Body / system tokens|', '|---|---|---:|']
    for role, item in preflight['prompts_unchanged'].items():
        lines.append(f"|{role}|`{item['sha256']}`|{item['policy_tokens']} / {item['system_message_tokens']}|")
    lines += ['', '## 3. Deterministic training data', '',
        'Only the original SFT training partition supplies examples. Source holdouts and validation bytes are unchanged. '
        'No diagnostic failure text becomes a training example. EDIT targets cover whole eojeols with one occurrence, with identical '
        'prefix/suffix additions to old/new and an assertion that the resulting sentence is unchanged. Widening replaces targets; '
        'it adds zero states. Synthetic rejected attempts leave text/profile/handoff unchanged, consume one step, update the public journal '
        'and exact rejection notice, and expose the correct teacher action or existing STOP(done) as the only loss target. '
        'Observations, notices and unsuccessful actions remain masked.', '',
        '|Role|Original states|EDIT targets widened / EDIT targets|Recovery not-found / ambiguous / no-op|Extra STOP(done) copies|Final states|STOP(done) share before → after|',
        '|---|---:|---:|---|---:|---:|---:|']
    for role, item in data['roles'].items():
        c = item['counts']
        lines.append(f"|{role}|{c['base_examples']}|{c['EDIT_widened']} / {c['EDIT_targets']}|"
            f"{c['recovery_span_not_found']} / {c['recovery_span_ambiguous']} / {c['recovery_no_op_edit']}|"
            f"{c['STOP_done_extra_copy']}|{item['train']['turns']}|{item['STOP_done_share_before']:.2%} → {item['STOP_done_share_after']:.2%}|")
    lines += ['', 'Full per-role counts, including STOP targets within recovery augmentations and context exclusions: '
              '`fix1/data/manifest.json`. Frozen ordering and quotas are recorded before training.', '',
        '## 4. Continued SFT', '',
        'Pinned Kanana snapshot c963a5f4f6496c749f94064a20b33028b0db9f19; QLoRA r16/alpha32/dropout .05, '
        'NF4 double quantization/BF16, lr 5e-5, one epoch, 8192 context, gradient checkpointing, two-GPU data parallelism, '
        'effective batch 8, cosine/warmup .03. Fresh optimizer with saved adapter weights; target-token-normalized CE. '
        'Orchestrator weights and all three prompts are unchanged.', '',
        '|Role|Initial checkpoint|Initial validation NLL|After one epoch NLL|Training seconds|New adapter SHA256|',
        '|---|---|---:|---:|---:|---|']
    for role in ('korean', 'revision'):
        path = ROOT/'adapters'/role
        initial = read_json(path/'initial_validation.json')
        done = read_json(path/'complete.json')
        loss = [x['eval_loss'] for x in done['log_history'] if 'eval_loss' in x][-1]
        lines.append(f"|{role}|epoch {2 if role == 'korean' else 1}|{initial['eval_loss']:.6f}|{loss:.6f}|"
                     f"{done['elapsed_s']:.1f}|`{file_sha(path/'epoch_1/adapter_model.safetensors')}`|")
    lines += ['', '## 5. Deterministic comparison — three repeated runs', '',
        'Temperature 0 / top_p 1; request seeds 47, 73, 101. Seeds do not create stochastic sampling at temperature zero. '
        'Mean ± sample SD [min, max] measures the observed local serving nondeterminism floor, not uncertainty across new essays. '
        'Conditions alternate first/second by case and repetition. Orchestrator plans afresh from public input; no saved Sol plans or records.', '',
        '(a) No protocol error: Orchestrator FINISH and every worker stage has model STOP(done) or STOP(blocked) with a reason, '
        'without controller forced stop, step/context limits, invalid ending, runtime or scorer failure. '
        '(b) additionally every worker stage is STOP(done). Valid STOP(done) is a protocol report, not proof that every content task is fixed.', '',
        '|Cohort|Workers|(a) % mean ± SD [range]|(b) % mean ± SD [range]|Per-run (a), (b) counts|',
        '|---|---|---:|---:|---|']
    deterministic = [c for c in cells if c['setting']['name'] == 'deterministic']
    for cohort in ('smoke20', 'agent_dev10'):
        for condition in ('old', 'new'):
            subset = [c for c in deterministic if c['cohort'] == cohort and c['condition'] == condition]
            a = [c['metrics']['no_protocol_error']/c['metrics']['planned'] for c in subset]
            b = [c['metrics']['all_stages_STOP_done']/c['metrics']['planned'] for c in subset]
            counts = [(c['metrics']['no_protocol_error'], c['metrics']['all_stages_STOP_done']) for c in subset]
            lines.append(f'|{cohort}|{condition}|{fmt(a, True)}|{fmt(b, True)}|`{counts}`|')
    lines += ['', '|Cohort|Workers|Role|STOP(done) % mean ± SD [range]|Valid action % mean ± SD [range]|Stages across runs|',
              '|---|---|---|---:|---:|---:|']
    for cohort in ('smoke20', 'agent_dev10'):
        for condition in ('old', 'new'):
            subset = [c for c in deterministic if c['cohort'] == cohort and c['condition'] == condition]
            for role in ('revision', 'korean'):
                stop = [c['extra'][role]['STOP_done']/c['extra'][role]['stages'] for c in subset]
                valid = [c['metrics']['roles'][role]['valid_action_rate'] for c in subset]
                lines.append(f"|{cohort}|{condition}|{role}|{fmt(stop, True)}|{fmt(valid, True)}|{sum(c['extra'][role]['stages'] for c in subset)}|")
    lines += ['', '### Rejected actions by the original taxonomy', '',
              '|Cohort|Workers|Role|Type|Counts in runs 1 / 2 / 3|Mean ± SD [range]|', '|---|---|---|---|---|---:|']
    for cohort in ('smoke20', 'agent_dev10'):
        for condition in ('old', 'new'):
            subset = [c for c in deterministic if c['cohort'] == cohort and c['condition'] == condition]
            for role in ('revision', 'korean'):
                for kind in TYPES:
                    counts = [c['extra'][role]['rejection_types'].get(kind, 0) for c in subset]
                    lines.append(f'|{cohort}|{condition}|{role}|{kind}|`{counts}`|{fmt(counts)}|')
    lines += ['', '### agent_dev10 corruption recovery (diagnostic only)', '',
        'Existing v1 record-level main recovery at the final essay is used unchanged. Initial normalization recovery is shown '
        'separately. Unknown exact projections remain unknown and retain their count; they are not assigned zero or silently dropped.', '',
        '|Workers|Operator|Distinct records|Initially full|Final full / observed record-runs|Unknown record-runs|Mean main recovery per run|',
        '|---|---|---:|---:|---:|---:|---|']
    for condition in ('old', 'new'):
        subset = [c for c in deterministic if c['cohort'] == 'agent_dev10' and c['condition'] == condition]
        ops = sorted({r['op'] for c in subset for e in c['recovery'] for r in e['initial']})
        for op in ops:
            initial = [r for e in subset[0]['recovery'] for r in e['initial'] if r['op'] == op]
            observed = []; unknown = 0; run_means = []
            for c in subset:
                run = [r for e in c['recovery'] for r in (e['final'] or []) if r['op'] == op]
                observed += run
                unknown += sum(r['op'] == op for e in c['recovery'] if e['final'] is None for r in e['initial'])
                run_means.append(mean(r['main'] for r in run) if run else None)
            lines.append(f"|{condition}|{op}|{len(initial)}|{sum(r['main'] == 1 for r in initial)}|"
                f"{sum(r['main'] == 1 for r in observed)} / {len(observed)}|{unknown}|`{run_means}`|")
    lines += ['', '## 6. Sampling preview — separate from deterministic results', '',
        'One pass per condition at temperature .7 / top_p .95 / seed 173, for all three roles. '
        'This is only a preview of rollout protocol reliability; no rollout is selected for RFT or trained on.', '',
        '|Cohort|Workers|Episodes without protocol error|Revision stages without protocol error|Korean stages without protocol error|',
        '|---|---|---:|---:|---:|']
    for c in cells:
        if c['setting']['name'] != 'sampling_preview':
            continue
        m = c['metrics']; r = c['extra']['revision']; k = c['extra']['korean']
        def fraction(n, d):
            return f'{n}/{d} ({n/d:.2%})' if d else '0/0 (undefined)'
        lines.append(f"|{c['cohort']}|{c['condition']}|{fraction(m['no_protocol_error'], m['planned'])}|"
                     f"{fraction(r['protocol_stages'], r['stages'])}|{fraction(k['protocol_stages'], k['stages'])}|")
    lines += ['', '## 7. Audit and stopping point', '',
        f"Saved episodes: {len(inventory)}. HF/vLLM prompt-token-count audit: `{dict(requests)}`. "
        'This is not a server token-ID equality claim. Every original checkpoint, source export, prompt, cohort and protected report '
        'was checked against its frozen hash. Q comes only from the unchanged GPU reference SCORE tool; no CPU Q or combined reward is reported.', '',
        '- Frozen contract: `verak/v4/outputs/fix1/preflight.json`.',
        '- All original rejected actions and examples: `fix1/taxonomy.json`.',
        '- Data counts, source partition and audit: `fix1/data/manifest.json`.',
        '- Full paired metrics and every attempt hash: `fix1/report_metrics.json`.',
        '- Raw requests and actions: `fix1/evaluation/{setting}/{run}/{condition}/{cohort}/`.', '',
        'Stopped after the requested targeted experiment. No full evaluation, test-cohort use or large RFT.']
    path = REPO/'imple/reports/V4_FIX1.md'
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(ROOT/'report_complete.json', {'report': ref(path), 'metrics': ref(ROOT/'report_metrics.json'), 'episodes': 240})
    print(str(path))


if __name__ == '__main__':
    report()
