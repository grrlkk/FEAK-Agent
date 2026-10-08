"""Saved-evidence SFT evaluation summaries and the requested phase report."""
from collections import Counter
from statistics import mean
import json
from pathlib import Path

from ..common import file_sha, read_json, write_json
from ..agentic.preservation import restore_final
from ..eval.measurement import changes
from ..eval.resources import Resources
from ..eval.summary import distribution, fmt, table
from ..reward.overedit import overedit
from .pilot import safe_id
from .sft_data import PHASE, ROLES
from .sft_composition import export_composition, global_inaction
from .sft_eval import CONDITIONS, action_kind, prepare, saved_rows
from .teacher_bulk import load_environment


def measure_real(config):
    """Terminal local diagnostics only; no policy/API calls and no real reward."""
    load_environment(config)
    _, _, examples = prepare(config)
    root = config['paths'][PHASE + '_output']
    resources = Resources(config, examples, output_key=PHASE + '_output')
    failures = []
    try:
        for condition in CONDITIONS:
            for row, source, _ in saved_rows(config, condition):
                if row['cohort'] != 'real':
                    continue
                path = root / 'real_measurements' / condition / (safe_id(row['corpus_episode_id']) + '.json')
                if path.exists():
                    if read_json(path)['episode_sha256'] != file_sha(source):
                        raise ValueError('Measured real episode changed')
                    continue
                try:
                    state = resources.worker()
                    # Keep stable units from the saved layout. Bareun's isolated
                    # sentence split can differ from the original paragraph split.
                    original_text = resources.source(row['source_id']).text
                    before = restore_final(row['initial_layout'], original_text, state.analysis)
                    after = restore_final(row['final_layout'], row['final_text'], state.analysis)
                    actions = [a for values in row['actions_by_role'].values() for a in values]
                    result = {'episode_id': row['corpus_episode_id'], 'condition': condition,
                        'episode_sha256': file_sha(source), 'completed': row['completed'],
                        'measurement': changes(before, after, actions),
                        'R_over_all_source': overedit(before, before, after, [])['value'],
                        'R_over_interpretation': 'descriptive edit distance over all original sentences; '
                            'not evidence that these edits were unnecessary', 'quality_measurement': None}
                    if row['completed']:
                        if row.get('quality_measurement'):
                            result['quality_measurement'] = row['quality_measurement']
                        else:
                            question = examples[row['source_id']].question
                            initial = resources.score(question, before.text)
                            final = resources.score(question, after.text)
                            result['quality_measurement'] = {'before': initial, 'after': final,
                                                            'delta_q': final['mean'] - initial['mean']}
                    write_json(path, result)
                except Exception as exc:
                    failures.append({'condition': condition, 'episode_id': row['corpus_episode_id'],
                                     'error': type(exc).__name__ + ': ' + str(exc)})
    finally:
        resources.close()
    write_json(root / 'real_measurement_status.json', {'errors': failures, 'api_calls': 0, 'policy_calls': 0})
    if failures:
        raise RuntimeError('Real measurement failures preserved; inspect before final reporting')


def ratio(n, d):
    return n / d if d else None


def summarize(rows, *, intended=None):
    intended = len(rows) if intended is None else intended
    completed = [r for r in rows if r.get('completed')]
    rewarded = [r for r in completed if r.get('reward')]
    actions = [a for r in rows for values in r.get('actions_by_role', {}).values() for a in values]
    calls = [c for r in rows for c in r.get('calls', [])]
    result = {'intended': intended, 'attempted': len(rows), 'unattempted': intended - len(rows),
        'completed': len(completed), 'completion_rate': ratio(len(completed), intended),
        'model_turns': len(calls), 'protocol_valid_turns': sum(c['valid_json_action'] for c in calls),
        'protocol_valid_rate': ratio(sum(c['valid_json_action'] for c in calls), len(calls)),
        'executed_action_attempts': len(actions), 'valid_actions': sum(a['valid'] for a in actions),
        'valid_action_rate': ratio(sum(a['valid'] for a in actions), len(actions)),
        'steps_all_attempted': distribution(sum(r['steps'].values()) for r in rows),
        'steps_completed': distribution(sum(r['steps'].values()) for r in completed),
        'action_counts_attempted': dict(Counter(action_kind(a) for a in actions)),
        'action_counts_valid': dict(Counter(action_kind(a) for a in actions if a['valid'])),
        'reward': {}, 'roles': {}, 'recovery_by_operator': {}, 'recovery_by_record_level': {},
        'errors': [{'id': r['corpus_episode_id'], 'error': r.get('runtime_error')} for r in rows if r.get('runtime_error')]}
    for role in ROLES:
        own = [a for r in rows for a in r.get('actions_by_role', {}).get(role, [])]
        termination = Counter(r.get('termination', {}).get(role, 'not_completed') for r in rows)
        result['roles'][role] = {'STOP': termination['STOP'], 'STOP_rate': ratio(termination['STOP'], intended),
            'terminations': dict(termination), 'valid_action_rate': ratio(sum(a['valid'] for a in own), len(own)),
            'action_count': len(own), 'steps': distribution(r['steps'].get(role, 0) for r in rows),
            'action_counts': dict(Counter(action_kind(a) for a in own))}
    result['both_roles_STOP_rate'] = ratio(sum(all(r.get('termination', {}).get(role) == 'STOP' for role in ROLES)
                                                 for r in rows), intended)
    for role in (*ROLES, 'combined'):
        result['reward'][role] = {k: distribution(r['reward'][role][k] for r in rewarded)
                                  for k in ('R', 'R_rec', 'R_q', 'R_over', 'R_step')}
    records = [record for r in rewarded for record in r['reward']['combined']['per_record']]
    for field, target in [('op', 'recovery_by_operator'), ('level', 'recovery_by_record_level')]:
        for key in sorted({r[field] for r in records}):
            own = [r for r in records if r[field] == key]
            result[target][key] = {'n': len(own), 'main': distribution(r['main'] for r in own),
                'recovery': distribution(r['recovery'] for r in own),
                'coupled': distribution(r['coupled'] for r in own if r.get('coupled') is not None)}
    return result


def accounting(root, suffix):
    path = root / suffix / 'api/accounting.json'
    return read_json(path) if path.exists() else {'calls': 0, 'confirmed_usd': 0., 'reserved_usd': 0., 'pending': 0}


def report(config):
    from .sft_markers import summary as marker_summary
    design, corpus, examples = prepare(config)
    root = config['paths'][PHASE + '_output']
    data = read_json(root / 'data/manifest.json')
    composition = export_composition(root, config['paths']['active_corrupt'] / 'agent_train.jsonl',
        partial_path=config['paths']['phase7_teacher_output'] / 'completed_global_evaluation.json')
    metrics = {'design': design, 'data': data, 'data_composition': composition['roles'],
               'training': {}, 'conditions': {},
               'markers': marker_summary(config), 'api': {s: accounting(root, s) for s in ('luna', 'sol')}}
    for role in ROLES:
        path = root / 'adapters' / role
        metrics['training'][role] = {'recipe': read_json(path / 'recipe.json'),
            'initial_validation': read_json(path / 'initial_validation.json'),
            'complete': read_json(path / 'complete.json'),
            'epochs': {str(e): read_json(path / f'epoch_{e}/provenance.json') for e in (1, 2)}}
    for condition in CONDITIONS:
        rows = [r for r, _, _ in saved_rows(config, condition)]
        for row in rows:
            row['seen_by_scorer'] = examples[row['source_id']].seen_by_scorer
        dev = [r for r in rows if r['cohort'] == 'dev']
        real = [r for r in rows if r['cohort'] == 'real']
        stats = summarize(dev, intended=100)
        stats['by_level'] = {level: summarize([r for r in dev if r['level'] == level], intended=25)
                             for level in ('L1', 'L2', 'L3', 'L4')}
        stats['global_inaction'] = global_inaction(dev, corpus, design['contract']['dev_ids'])
        for level, group in stats['by_level'].items():
            group['global_inaction'] = global_inaction(dev, corpus,
                [i for i in design['contract']['dev_ids'] if corpus[i]['level'] == level])
        stats['by_operator'] = {op: summarize([r for r in dev if any(rec['op'] == op for rec in corpus[r['corpus_episode_id']]['records'])],
            intended=sum(any(rec['op'] == op for rec in corpus[i]['records']) for i in design['contract']['dev_ids']))
                                for op in sorted({rec['op'] for i in design['contract']['dev_ids'] for rec in corpus[i]['records']})}
        stats['by_damage_level'] = {level: summarize([r for r in dev if any(rec['level'] == level for rec in corpus[r['corpus_episode_id']]['records'])],
            intended=sum(any(rec['level'] == level for rec in corpus[i]['records']) for i in design['contract']['dev_ids']))
            for level in ('GLOBAL', 'WORD', 'SENTENCE', 'TEXT')}
        stats['by_seen_question'] = {str(seen): summarize([r for r in dev if r['seen_by_scorer'] == seen]) for seen in (False, True)}
        stats['by_genre'] = {genre: summarize([r for r in dev if r['genre'] == genre]) for genre in sorted({r['genre'] for r in dev})}
        real_stats = summarize(real, intended=30)
        measured = [read_json(p) for p in (root / 'real_measurements' / condition).glob('*.json')]
        good = [r for r in measured if r['completed']]
        real_stats['measurement_n'] = len(good)
        real_stats['measurement'] = {key: distribution(r['measurement'][key] for r in good)
            for key in ('source_sentences', 'changed_share', 'text_changed_share', 'deleted', 'inserted', 'moved')}
        real_stats['R_over_all_source'] = distribution(r['R_over_all_source'] for r in good)
        real_stats['delta_q'] = distribution(r['quality_measurement']['delta_q'] for r in good if r['quality_measurement'])
        real_stats['changed_share_micro'] = ratio(sum(r['measurement']['sentences_changed'] for r in good),
                                                 sum(r['measurement']['source_sentences'] for r in good))
        metrics['conditions'][condition] = {'dev': stats, 'real': real_stats}
    write_json(root / 'metrics.json', metrics)
    lines = ['# VERAK v3 Phase 7 — two-stage warm-start SFT', '',
        'Warm-start SFT only. GLOBAL was trained first, then KOREAN on the saved teacher hand-offs. '
        'The future Orchestrator rule is recorded in addendum Section 10; no Orchestrator or RFT training was run.', '',
        '## Data and training', '',
        'The accepted 810 GLOBAL / 903 KOREAN best-attempt selections were exported exactly. Each sample uses '
        'the actual inference context followed by that turn’s saved action. Only current action content and its '
        'end-of-turn token receive loss; system messages, profiles, observations, tool outputs, notices, '
        'hand-offs and historical actions are masked. Each saved action is targeted once per epoch. '
        'No packing, document truncation or hidden context substitution was used.', '',
        'Five percent of unique source essays per role, rounded up, were held out with seed 71. '
        'All corruption variants of a source stay together, and validation sources are excluded from both roles’ training sets. '
        'The dev and real evaluation sources are disjoint from all SFT sources.', '']
    lines += [table(['role', 'train trajectories / sources / turns', 'validation trajectories / sources / turns',
                     'max tokens', 'train target tokens'], [[role,
        ' / '.join(str(data['roles'][role]['train'][key]) for key in ('trajectories', 'sources', 'turns')),
        ' / '.join(str(data['roles'][role]['validation'][key]) for key in ('trajectories', 'sources', 'turns')),
        max(data['roles'][role][part]['lengths']['max'] for part in ('train', 'validation')),
        data['roles'][role]['train']['target_tokens']] for role in ROLES]), '',
        'Pinned base: `c963a5f4f6496c749f94064a20b33028b0db9f19`. GPU0, NF4 double-quantized base, BF16 compute, '
        'all-linear LoRA r=16/alpha=32/dropout=.05, learning rate 1e-4, two epochs, gradient checkpointing. '
        'Micro-batch 1, gradient accumulation 8, paged AdamW 8-bit, cosine schedule with 3% warmup, '
        'weight decay 0, gradient clipping 1. The base and scorer remain frozen. Epoch checkpoints retain '
        'adapter weights and optimizer/trainer state. Both roles start independently from the pinned base.', '',
        'The loss projects only the hidden states needed to predict supervised action tokens. A numerical unit test '
        'checks equality of loss and gradients against full masked causal-LM loss. Validation is token-weighted '
        'action negative log likelihood, not an episode reward or a quality judgment.', '',
        '[PEFT quantization recipe](https://huggingface.co/docs/peft/v0.18.0/developer_guides/quantization); '
        '[TRL SFT trainer](https://huggingface.co/docs/trl/v0.23.1/sft_trainer).', '']
    loss_rows = []
    for role in ROLES:
        train = metrics['training'][role]
        values = [entry for entry in train['complete']['log_history'] if 'eval_loss' in entry and entry.get('epoch') is not None]
        loss_rows.append([role, fmt(train['initial_validation']['eval_loss']),
            *[fmt(next((e['eval_loss'] for e in values if round(e['epoch']) == epoch), None)) for epoch in (1, 2)],
            fmt(train['complete']['elapsed_s'] / 3600), fmt(train['complete']['max_cuda_allocated_bytes'] / 2**30)])
    lines += [table(['role', 'initial validation loss', 'epoch 1', 'epoch 2', 'GPU0 elapsed hours', 'peak allocated GiB'], loss_rows), '',
        'Selected trajectories retain their original malformed or rejected action targets; eligibility was not silently '
        'redefined during export. Counts and compaction rates:', '',
        table(['role', 'partition', 'compacted / turns', 'JSON-invalid targets', 'protocol-invalid targets'],
            [[r, p, f"{data['roles'][r][p]['compacted_turns']}/{data['roles'][r][p]['turns']}",
              data['roles'][r][p]['json_invalid_targets'], data['roles'][r][p]['protocol_invalid_targets']]
             for r in ROLES for p in ('train', 'validation')]), '',
        '### Exported trajectory composition', '',
        'Structural actions are MOVE or sentence insertion/deletion. Attempts include rejected actions; '
        'accepted actions are also counted separately. STOP-only means all attempted role actions are STOP '
        '(including rejected STOP retries), with a terminal STOP. The frozen selections and running training '
        'were not changed for this diagnostic.', '',
        table(['GLOBAL partition', 'trajectories', 'structural attempt', 'accepted structural action',
               'STOP-only', 'other', 'STOP-only with GLOBAL records'],
            [[p, (s := composition['roles']['global'][p])['trajectories'], s['with_structural_attempt'],
              s['with_accepted_structural_action'], s['STOP_only'], s['other_than_structural_or_STOP_only'],
              s['STOP_only_with_global_records']] for p in ('all', 'train', 'validation')]), '',
        'Operator counts below count selected trajectories, not individual actions or records. '
        'Full recovery means saved main recovery equals 1; GLOBAL is measured after GLOBAL, '
        'and KOREAN local records at the final state. Any positive recovery includes partial recovery. '
        'A trajectory with several operators appears in several rows. Per-record counts and trajectory IDs '
        'are retained in `export_composition.json`.', '',
        table(['role', 'operator', 'contains operator', 'at least one fully recovered', 'all fully recovered',
               'any positive recovery', 'train / validation fully recovered'],
            [[r, op, (s := composition['roles'][r]['all']['operators'][op])['selected_trajectories_with_operator'],
              s['at_least_one_fully_recovered'], s['all_fully_recovered'], s['any_positive_recovery'],
              f"{composition['roles'][r]['train']['operators'][op]['all_fully_recovered']} / "
              f"{composition['roles'][r]['validation']['operators'][op]['all_fully_recovered']}"]
             for r in ROLES for op in composition['roles'][r]['all']['operators']]), '',
        '## Evaluation contract', '',
        f"Seed 73 selected 100 active agent_dev corruption instances, 25 per L1–L4, from {design['contract']['dev_unique_sources']} "
        'distinct source essays. The same 30 Phase-6 real essays were used in all conditions. These are development '
        'comparisons; the test split was not opened. Epoch 1 pairs the GLOBAL and KOREAN epoch-1 adapters; '
        'epoch 2 pairs both epoch-2 adapters.', '',
        'Base and both trained conditions use the same GPU0 vLLM 0.8.5 server, BF16 pinned base, multi-LoRA, '
        '8,192 context / 1,024 generation reserve, temperature 0, top_p 1 and seed 47; no JSON-constrained decoder. '
        'Four episode workers share GPU1’s frozen scorer. '
        '[vLLM multi-LoRA serving](https://docs.vllm.ai/en/v0.8.5/features/lora.html).', '',
        f"Luna reuses {sum(i in design['reuse'] for i in design['contract']['dev_ids'])} saved dev runs and all 30 saved "
        'setting-(a) real runs, including failures. Remaining dev runs use pinned Luna low with independent unseeded '
        'requests and the unchanged two-stage environment. No saved comparison was regenerated.', '',
        'Completion includes normal STOP or environment budget/error termination, but excludes runtime/API failures. '
        'Valid-action rate is accepted environment actions / all attempted actions; protocol-valid rate additionally '
        'reports model turns including JSON retries. STOP rates use all intended essays per role. Reward means use '
        'completed corruption episodes only; unavailable runs are never assigned zero reward.', '',
        '## Development results', '']
    lines += [table(['condition', 'completed / 100', 'valid action', 'protocol valid', 'GLOBAL STOP', 'KOREAN STOP',
        'combined R', 'R_rec', 'R_over', 'steps / attempted'], [[condition,
        f"{(s := metrics['conditions'][condition]['dev'])['completed']}/100", fmt(s['valid_action_rate']),
        fmt(s['protocol_valid_rate']), fmt(s['roles']['global']['STOP_rate']), fmt(s['roles']['korean']['STOP_rate']),
        *[fmt(s['reward']['combined'][k]['mean']) for k in ('R', 'R_rec', 'R_over')],
        fmt(s['steps_all_attempted']['mean'])] for condition in CONDITIONS]), '',
        table(['condition', 'role', 'reward n', 'R', 'R_rec', 'R_q', 'R_over', 'R_step'],
            [[c, r, metrics['conditions'][c]['dev']['reward'][r]['R']['n'],
              *[fmt(metrics['conditions'][c]['dev']['reward'][r][k]['mean']) for k in ('R', 'R_rec', 'R_q', 'R_over', 'R_step')]]
             for c in CONDITIONS for r in (*ROLES, 'combined')]), '',
        '### GLOBAL STOP without structural action when GLOBAL damage is present', '',
        'The denominator is all selected dev essays with at least one GLOBAL corruption record, '
        'including failed episodes. The numerator requires an observed GLOBAL STOP. '
        'Both no structural attempt (including rejected attempts) and no accepted structural action '
        'are shown. A completed GLOBAL stage remains observable if KOREAN later fails. '
        'Missing GLOBAL decisions stay unknown; a point rate is withheld until all such decisions are known. '
        'The no-record STOP teaching examples are excluded from this denominator.', '',
        table(['condition', 'GLOBAL-record essays', 'unknown GLOBAL decisions',
               'STOP / no structural attempt', 'share', 'STOP / no accepted structural action', 'share'],
            [[c, (s := metrics['conditions'][c]['dev']['global_inaction'])['intended_with_global_records'],
              s['unknown'], s['STOP_without_structural_attempt']['n'],
              fmt(s['STOP_without_structural_attempt']['rate']), s['STOP_without_accepted_structural_action']['n'],
              fmt(s['STOP_without_accepted_structural_action']['rate'])] for c in CONDITIONS]), '',
        'Counts, bounds, essay IDs and L1–L4 breakdowns are retained in `metrics.json`.', '',
        '### By curriculum level', '',
        table(['condition', 'level', 'complete / 25', 'GLOBAL R', 'KOREAN R', 'combined R', 'R_over', 'steps', 'valid action'],
            [[c, level, f"{(s := metrics['conditions'][c]['dev']['by_level'][level])['completed']}/25",
              *[fmt(s['reward'][r]['R']['mean']) for r in (*ROLES, 'combined')],
              fmt(s['reward']['combined']['R_over']['mean']), fmt(s['steps_all_attempted']['mean']), fmt(s['valid_action_rate'])]
             for c in CONDITIONS for level in ('L1', 'L2', 'L3', 'L4')]), '',
        '### By corruption operator', '',
        'Role/combined rewards below are episode means among completed essays containing the operator. '
        'An essay with several operators contributes to several rows. Record recovery is shown separately.', '',
        table(['condition', 'operator', 'reward n', 'GLOBAL R', 'KOREAN R', 'combined R', 'R_over', 'record recovery'],
            [[c, op, (s := metrics['conditions'][c]['dev']['by_operator'][op])['reward']['combined']['R']['n'],
              *[fmt(s['reward'][r]['R']['mean']) for r in (*ROLES, 'combined')],
              fmt(s['reward']['combined']['R_over']['mean']),
              fmt(metrics['conditions'][c]['dev']['recovery_by_operator'].get(op, {}).get('recovery', {}).get('mean'))]
             for c in CONDITIONS for op in metrics['conditions'][c]['dev']['by_operator']]), '',
        '### By damage level', '',
        'These episode groups overlap when an essay contains several damage levels.', '',
        table(['condition', 'damage level', 'complete / intended', 'GLOBAL R', 'KOREAN R', 'combined R', 'R_over', 'steps'],
            [[c, level, f"{(s := metrics['conditions'][c]['dev']['by_damage_level'][level])['completed']}/{s['intended']}",
              *[fmt(s['reward'][r]['R']['mean']) for r in (*ROLES, 'combined')],
              fmt(s['reward']['combined']['R_over']['mean']), fmt(s['steps_all_attempted']['mean'])]
             for c in CONDITIONS for level in ('GLOBAL', 'WORD', 'SENTENCE', 'TEXT')]), '',
        'Full completion, validity, STOP rates, reward components, record recovery, steps and action counts for each '
        'curriculum/damage/operator group, plus genre and scorer-question exposure breakdowns, are retained in `metrics.json`.', '',
        '### Action counts', '',
        table(['condition', 'cohort', 'attempted action counts', 'accepted action counts'],
            [[c, cohort, json.dumps(metrics['conditions'][c][cohort]['action_counts_attempted'], sort_keys=True),
              json.dumps(metrics['conditions'][c][cohort]['action_counts_valid'], sort_keys=True)]
             for c in CONDITIONS for cohort in ('dev', 'real')]), '', '## Real essays', '',
        'No answer-based recovery reward is defined for real essays. Change shares include rewritten, deleted or '
        'moved original sentences; inserted sentences are reported separately. The all-source R_over is a descriptive '
        'morpheme/order distance, not a judgment that revisions are unnecessary.', '',
        table(['condition', 'complete / 30', 'valid action', 'GLOBAL STOP', 'KOREAN STOP', 'steps',
               'changed share', 'text-changed share', 'deletions / essay', 'all-source R_over', 'scorer delta Q'],
            [[c, f"{(s := metrics['conditions'][c]['real'])['completed']}/30", fmt(s['valid_action_rate']),
              fmt(s['roles']['global']['STOP_rate']), fmt(s['roles']['korean']['STOP_rate']),
              fmt(s['steps_all_attempted']['mean']), fmt(s['measurement']['changed_share']['mean']),
              fmt(s['measurement']['text_changed_share']['mean']), fmt(s['measurement']['deleted']['mean']),
              fmt(s['R_over_all_source']['mean']), fmt(s['delta_q']['mean'])] for c in CONDITIONS]), '',
        '### Residual marker misfit — same Sol check', '',
        'Sol reads only the final affected sentence and its predecessor, with the accepted conservative marker-fit '
        'prompt and field schema. Sites are selected after valid GLOBAL structural actions from changed adjacency, '
        'insertions and internal Bareun notices. Every accepted structural action stays in the denominator. '
        'Incomplete episodes and unjudged surviving sites remain unknown; unknown is never a pass. Exact saved '
        'Luna judgments are reused. Identical final-text/schema requests share a single judgment.', '',
        table(['condition', 'structural actions', 'flagged', 'unknown', 'misfit lower–upper', 'judged / eligible cases'],
            [[c, (s := metrics['markers']['rates'][c])['structural_actions'], s['flagged_actions'], s['unknown_actions'],
              f"{fmt(s['lower_bound'])}–{fmt(s['upper_bound'])}", f"{s['judged_cases']}/{s['eligible_cases']}"] for c in CONDITIONS]), '',
        'These automated checks measure the specified final marker sites, not whole-essay factual correctness. '
        'Scorer gains and Sol residual misfit are separate measures.', '', '## Costs, failures and artifacts', '',
        table(['API stage', 'new confirmed USD / cap', 'reserved USD', 'pending', 'calls'],
            [[s, f"{metrics['api'][s]['confirmed_usd']:.6f} / 3", metrics['api'][s]['reserved_usd'],
              metrics['api'][s]['pending'], metrics['api'][s]['calls']] for s in ('luna', 'sol')]), '',
        '```json', json.dumps({c: {cohort: metrics['conditions'][c][cohort]['errors'] for cohort in ('dev', 'real')}
                              for c in CONDITIONS}, ensure_ascii=False, indent=2), '```', '',
        f'Artifacts: `{root}`. Export manifest, source-group splits, original trajectory hashes, epoch adapters, '
        'optimizer checkpoints, losses, policy requests, event logs, API ledgers and marker judgments are retained locally. '
        'No data or model weights are uploaded.', '',
        'Validation evidence and interpretation are recorded below after the completed runs are audited. '
        'No RFT was run. Stop after this phase report.', '']
    target = config['paths']['repo'] / 'imple/reports/V3_PHASE_7_SFT.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    return metrics
