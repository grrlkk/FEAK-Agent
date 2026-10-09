"""Evidence-only A/C reports; no promotion to another RFT round."""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from ..common import file_sha, read_json, write_json
from ..eval.summary import distribution, fmt, paired, table
from ..train.sft_composition import global_inaction
from ..train.sft_eval import eval_config, saved_rows as sft_rows
from ..train.sft_report import accounting, summarize
from .config import PHASE, ROLES
from .context_audit import audit
from .evaluate import inputs, saved_rows
from .markers import summary as marker_summary


def summarize_oneshot(rows, *, intended):
    complete = [r for r in rows if r['completed']]
    rewarded = [r for r in complete if r.get('reward')]
    records = [v for r in rewarded for v in r['reward']['combined']['per_record']]
    result = {'intended': intended, 'attempted': len(rows), 'completed': len(complete),
        'completion_rate': len(complete) / intended if intended else None,
        'model_turns': sum('generation' in r for r in rows), 'protocol_valid_rate': None,
        'valid_action_rate': None, 'executed_action_attempts': None, 'valid_actions': None,
        'steps_all_attempted': distribution(sum(r['steps'].values()) for r in rows),
        'steps_completed': distribution(sum(r['steps'].values()) for r in complete),
        'action_counts_attempted': {}, 'action_counts_valid': {}, 'roles': {}, 'both_roles_STOP_rate': None,
        'termination': dict(Counter(r.get('termination', {}).get('single', 'not_generated') for r in rows)),
        'reward': {role: {key: distribution(r['reward'][role][key] for r in rewarded if r['reward'].get(role))
                         for key in ('R', 'R_rec', 'R_q', 'R_over', 'R_step')}
                   for role in (*ROLES, 'combined')},
        'metric_note': 'Tool action validity, role STOP, and intermediate role rewards are not applicable to whole-text generation',
        'errors': [{'id': r['corpus_episode_id'], 'error': r['runtime_error']} for r in rows if r.get('runtime_error')]}
    for field, name in [('op', 'recovery_by_operator'), ('level', 'recovery_by_record_level')]:
        result[name] = {value: {'n': sum(rec[field] == value for rec in records),
            'main': distribution(rec['main'] for rec in records if rec[field] == value),
            'recovery': distribution(rec['recovery'] for rec in records if rec[field] == value),
            'coupled': distribution(rec['coupled'] for rec in records if rec[field] == value and rec.get('coupled') is not None)}
            for value in sorted({rec[field] for rec in records})}
    result['alignment'] = {key: sum(r.get('alignment', {}).get(key, 0) for r in complete)
                           for key in ('matched', 'new_units', 'unmatched_input')}
    result['alignment']['matched_below_0_6'] = sum(s['lexical_similarity'] is not None and s['lexical_similarity'] < .6
        for r in complete for s in r.get('alignment', {}).get('sentences', []))
    return result


def current_metrics(config, condition, rows, corpus, design):
    # Diagnostics can encounter a rejected non-object args value. The original
    # trajectory remains immutable; only dictionary access in old summaries is adapted.
    rows = [{**r, 'actions_by_role': {role: [{**a,
        'action': a.get('action') if isinstance(a.get('action'), str) else 'INVALID',
        'args': a.get('args') if isinstance(a.get('args'), dict) else {}}
        for a in values] for role, values in r.get('actions_by_role', {}).items()}} for r in rows]
    summarize_fn = summarize_oneshot if condition == 'oneshot' else summarize
    dev = [r for r in rows if r['cohort'] == 'dev']
    real = [r for r in rows if r['cohort'] == 'real']
    stats = summarize_fn(dev, intended=100)
    stats['by_level'] = {level: summarize_fn([r for r in dev if r['level'] == level], intended=25)
                         for level in ('L1', 'L2', 'L3', 'L4')}
    ids = design['contract']['dev_ids']
    operators = sorted({r['op'] for eid in ids for r in corpus[eid]['records']})
    stats['by_operator'] = {op: summarize_fn(
        [r for r in dev if any(v['op'] == op for v in corpus[r['corpus_episode_id']]['records'])],
        intended=sum(any(v['op'] == op for v in corpus[eid]['records']) for eid in ids)) for op in operators}
    if condition == 'rft1':
        stats['global_inaction'] = global_inaction(dev, corpus, ids)
        for level in stats['by_level']:
            stats['by_level'][level]['global_inaction'] = global_inaction(dev, corpus, [i for i in ids if corpus[i]['level'] == level])
    real_stats = summarize_fn(real, intended=30)
    good = [r for r in real if r['completed']]
    if any('measurement' not in r or 'R_over_all_source' not in r or 'quality_measurement' not in r for r in good):
        raise ValueError('Completed real outputs require terminal measurements before reporting')
    real_stats['measurement_n'] = len(good)
    real_stats['measurement'] = {key: distribution(r['measurement'][key] for r in good)
        for key in ('source_sentences', 'changed_share', 'text_changed_share', 'deleted', 'inserted', 'moved')}
    real_stats['R_over_all_source'] = distribution(r['R_over_all_source'] for r in good)
    real_stats['delta_q'] = distribution(r['quality_measurement']['delta_q'] for r in good)
    denominator = sum(r['measurement']['source_sentences'] for r in good)
    real_stats['changed_share_micro'] = sum(r['measurement']['sentences_changed'] for r in good) / denominator if denominator else None
    return {'dev': stats, 'real': real_stats}


def comparison_lines(conditions):
    lines = ['## Frozen dev evaluation', '',
        'The same 100 active dev episodes (25 per L1–L4, seed73) and 30 Phase6 real essays are used. '
        'SFT epoch2 and Luna are reused saved runs. Completion includes valid environment termination at an '
        'error or step limit; STOP is reported separately. Rewards only summarize completed, rewarded episodes.', '',
        table(['condition', 'complete /100', 'valid actions', 'G STOP', 'K STOP', 'G R', 'K R', 'combined R', 'R_over', 'steps'],
            [[name, m['dev']['completed'], fmt(m['dev']['valid_action_rate']),
              fmt(m['dev']['roles'].get('global', {}).get('STOP_rate')), fmt(m['dev']['roles'].get('korean', {}).get('STOP_rate')),
              *[fmt(m['dev']['reward'][role]['R']['mean']) for role in (*ROLES, 'combined')],
              fmt(m['dev']['reward']['combined']['R_over']['mean']), fmt(m['dev']['steps_all_attempted']['mean'])]
             for name, m in conditions.items()]), '']
    for level in ('L1', 'L2', 'L3', 'L4'):
        lines += [f'### {level}', '', table(['condition', 'complete /25', 'valid actions', 'G R', 'K R', 'combined R', 'R_over', 'steps'],
            [[name, (s := m['dev']['by_level'][level])['completed'], fmt(s['valid_action_rate']),
              *[fmt(s['reward'][role]['R']['mean']) for role in (*ROLES, 'combined')],
              fmt(s['reward']['combined']['R_over']['mean']), fmt(s['steps_all_attempted']['mean'])]
             for name, m in conditions.items()]), '']
    lines += ['### GLOBAL stop without a structural action', '',
        'The denominator contains only dev essays with GLOBAL corruption records. Attempts count even '
        'if rejected; unknown/incomplete GLOBAL stages remain explicit.', '',
        table(['condition', 'level', 'STOP without structural attempt / GLOBAL-record essays', 'share', 'unknown'],
            [[name, level, f"{(g := s['global_inaction'])['STOP_without_structural_attempt']['n']} / "
                          f"{g['intended_with_global_records']}",
              fmt(g['STOP_without_structural_attempt']['rate']), g['unknown']]
             for name, m in conditions.items()
             for level, s in [('all', m['dev']), *list(m['dev']['by_level'].items())]
             if 'global_inaction' in s]), '']
    ops = sorted({op for m in conditions.values() for op in m['dev']['recovery_by_operator']})
    lines += ['### Operator recovery and episode rewards', '',
        'Operator recovery is measured at the final essay. Episode rewards below are for all essays '
        'containing that operator; an essay with several operators contributes to several rows.', '',
        table(['condition', 'operator', 'records', 'main recovery', 'recovery incl coupled', 'G R', 'K R', 'combined R', 'R_over'],
            [[name, op, (v := m['dev']['recovery_by_operator'].get(op, {})).get('n', 0),
              fmt(v.get('main', {}).get('mean')), fmt(v.get('recovery', {}).get('mean')),
              *[fmt(m['dev']['by_operator'].get(op, {}).get('reward', {}).get(role, {}).get('R', {}).get('mean'))
                for role in (*ROLES, 'combined')],
              fmt(m['dev']['by_operator'].get(op, {}).get('reward', {}).get('combined', {}).get('R_over', {}).get('mean'))]
             for name, m in conditions.items() for op in ops]), '',
        '### Actions and termination', '',
        table(['condition', 'cohort', 'model turns', 'protocol-valid turns', 'valid / attempted actions', 'action counts', 'KOREAN endings'],
            [[name, cohort, (s := m[cohort])['model_turns'], fmt(s['protocol_valid_rate']),
              f"{s['valid_actions']} / {s['executed_action_attempts']}", s['action_counts_attempted'],
              s['roles'].get('korean', {}).get('terminations', 'not applicable')]
             for name, m in conditions.items() for cohort in ('dev', 'real')]), '',
        table(['condition', 'cohort', 'KOREAN STOP', 'KOREAN step limit', 'other endings'],
            [[name, cohort, (t := m[cohort]['roles']['korean']['terminations']).get('STOP', 0),
              t.get('max_steps', 0), {key: value for key, value in t.items() if key not in {'STOP', 'max_steps'}}]
             for name, m in conditions.items() for cohort in ('dev', 'real') if 'korean' in m[cohort]['roles']]), '',
        '## Real essays', '',
        'Real essays have no synthetic answer records, so recovery/role rewards are not invented. '
        'R_over below is descriptive change over all source sentences, not proof of unnecessary editing. '
        'Failed essays remain in completion denominators; change metrics use completed outputs.', '',
        table(['condition', 'complete /30', 'sentence-change share (macro)', 'sentence-change share (micro)',
               'deletions /essay', 'insertions /essay', 'R_over all source', 'delta Q', 'steps'],
            [[name, (s := m['real'])['completed'], fmt(s['measurement']['changed_share']['mean']),
              fmt(s['changed_share_micro']), fmt(s['measurement']['deleted']['mean']), fmt(s['measurement']['inserted']['mean']),
              fmt(s['R_over_all_source']['mean']), fmt(s['delta_q']['mean']), fmt(s['steps_all_attempted']['mean'])]
             for name, m in conditions.items()]), '']
    if 'oneshot' in conditions:
        lines += ['One-shot generation endings (separate from successful scoring/completion):', '',
            table(['cohort', 'endings'], [[cohort, conditions['oneshot'][cohort]['termination']]
                                         for cohort in ('dev', 'real')]), '']
    return lines


def write_report(config, condition='rft1'):
    phase = PHASE if condition == 'rft1' else 'oneshot_baseline'
    root = config['paths']['repo'] / 'verak/v3/outputs' / phase
    status = read_json(root / 'evaluation' / condition / 'status.json')
    if not status['all_attempted'] or status['errors']:
        raise RuntimeError('Finish all authorized evaluation attempts before reporting')
    design, corpus, _ = inputs(config)
    old = read_json(config['paths']['phase7_sft_output'] / 'metrics.json')
    rows = [r for r, _, _ in saved_rows(config, condition)]
    if len(rows) != 130:
        raise ValueError('Expected exactly 100 dev and30 real output records')
    conditions = {k: deepcopy(old['conditions'][k]) for k in ('epoch_2', 'luna_low')}
    if condition == 'oneshot':
        conditions['rft1'] = read_json(config['paths'][PHASE + '_output'] / 'metrics.json')['conditions']['rft1']
    conditions[condition] = current_metrics(config, condition, rows, corpus, design)
    markers = marker_summary(config, condition)
    metrics = {'condition': condition, 'conditions': conditions, 'markers': markers,
        'baseline_markers': {name: old['markers']['rates'][name] for name in ('epoch_2', 'luna_low')},
        'sol_shared_A_C': accounting(config['paths'][PHASE + '_output'], 'sol'),
        'data': read_json(root / 'data/manifest.json'), 'training': {}, 'paired': {}}
    if condition == 'oneshot':
        prior = read_json(config['paths'][PHASE + '_output'] / 'metrics.json')['markers']
        metrics['baseline_markers']['rft1'] = {'structural_actions': prior['units'],
            'flagged_actions': prior['flagged_units'], 'unknown_actions': prior['unknown_units'],
            'share': prior['share'], 'judged_cases': prior['judged_cases']}
    sft_config = eval_config()
    for baseline in ('epoch_2', 'luna_low'):
        baseline_rows = [r for r, _, _ in sft_rows(sft_config, baseline)]
        metrics['paired'][baseline] = paired(rows, baseline_rows,
            ['combined.R', 'combined.R_rec', 'combined.R_over', 'steps'], config)
    if condition == 'oneshot':
        metrics['paired']['rft1'] = paired(rows, [r for r, _, _ in saved_rows(config)],
            ['combined.R', 'combined.R_rec', 'combined.R_over', 'steps'], config)
    for role in (ROLES if condition == 'rft1' else ('oneshot',)):
        directory = root / 'adapters' / role
        metrics['training'][role] = {name: read_json(directory / (name + '.json'))
                                    for name in ('recipe', 'initial_validation', 'complete')}
    if condition == 'rft1':
        selection = read_json(root / 'selection.json')
        metrics['selection'] = {key: value for key, value in selection.items() if key not in ('selections', 'rollout_slots')}
        metrics['context_audit'] = audit(config)
        rollout = read_json(root / 'rollout_status.json')
        lines = ['# VERAK v3 Phase8 — RFT round1', '',
            'One authorized rejection-sampling round from the accepted SFT epoch2 pair. No round2, DPO, '
            'v2 training or Orchestrator training follows this report.', '', '## Rollouts and selection', '',
            f"Saved {rollout['saved']}/{rollout['requested']} samples across all 1,430 active train episodes, "
            'four per episode, temperature0.7/top_p0.95. GPU0 vLLM used the paired epoch2 editors; GPU1 '
            'used the frozen scorer. The two-stage v1 environment, prompts, tools, rewards, 8192 context '
            'and 1024 output reserve were unchanged. Per-sample seeds and raw calls are preserved; '
            'model failures are retained and completed samples are never regenerated.', '',
            'Select the highest eligible role R per essay, then merge accepted teacher selections by higher role R '
            '(teacher wins ties). GLOBAL-record examples and KOREAN require R>=0.80. As explicitly confirmed, '
            'no-GLOBAL examples retain STOP-within2, R_over=0 and no structural-action attempt instead of '
            'the R threshold. Every teacher and rollout selection additionally requires terminal valid STOP '
            'and at most one rejected action. These added terminal/rejection gates are stricter than the '
            'historical SFT export. No held-out SFT source is used for gradient training.', '',
            'The revised round1 design combines student rollouts with extra v1 GLOBAL teacher data '
            'when those additions were ready at the rollout boundary. Its gains cannot then be '
            'attributed to student rollouts alone. Only GPU-reference rewards enter training. '
            'G_DEL_LINK v2 additions remain separate for round2; KOREAN receives no extra teacher data.', '',
            f"Extra-teacher merge: `{selection.get('extra_teacher', {})}`.", '',
            table(['role', 'eligible rollout best', 'merged', 'train unique', 'train weighted', 'validation', 'STOP-only train share'],
                [[role, (s := selection['summaries'][role])['rollout_best']['unique_trajectories'],
                  s['merged']['unique_trajectories'], s['train']['unique_trajectories'],
                  s['train']['weighted_trajectories'], s['validation']['unique_trajectories'],
                  fmt(s['train']['STOP_only_share_unique'])] for role in ROLES]), '',
            'GLOBAL STOP-only is capped at35% before weighting and remains below35% afterwards. '
            'A fully recovered G_PARA_SWAP or G_SENT_MOVE trajectory receives weight2 only if that '
            'operator has fewer than200 fully recovered unique train trajectories after the merge. '
            'Full L_CONJ KOREAN recovery retains weight2. Several matching criteria still give '
            'total weight2 once. Validation is unweighted.', '',
            table(['role', 'operator', 'train unique', 'weighted', 'fully recovered unique', 'fully recovered weighted'],
                [[role, op, v['unique_trajectories'], v['weighted_trajectories'], v['fully_recovered_unique'],
                  v['fully_recovered_weighted']] for role in ROLES
                 for op, v in selection['summaries'][role]['train']['operators'].items()]), '']
        lines += ['## Training composition by source', '',
            table(['role', 'source', 'operator', 'unique', 'weighted', 'fully recovered unique'],
                [[role, source, op, count['unique_trajectories'], count['weighted_trajectories'],
                  count['fully_recovered_unique']]
                 for role in ROLES for op, counts in selection['summaries'][role]['train']['operators'].items()
                 for source, count in counts['by_source'].items()]), '',
            'Source tags are sft_teacher, extra_teacher and rft_rollout; every exported action target '
            'retains its trajectory source tag, source essay and GPU score provenance.', '']
    else:
        data = metrics['data']
        lines = ['# VERAK v3 — one-shot trained baseline', '',
            'C began after RFT1 training, evaluation and report A. Targets are saved teacher final essays; '
            'there were no new teacher calls. This is a whole-text distillation comparison, not an agent rollout.', '',
            '## Data difference', '',
            'For each essay in the union of SFT selections, choose the complete teacher attempt with the '
            'higher combined R first. Keep it only if the same attempt passes GLOBAL R>=0.80 when GLOBAL '
            'records exist and KOREAN R>=0.80. There is no fallback to a lower combined-R attempt. '
            'All SFT validation source essays are excluded from gradient training and retained only as '
            'validation when they pass the same filter. Input is only question plus corrupted essay; '
            'loss covers the final revised essay and EOT, with all input tokens masked.', '',
            table(['set', 'train episodes', 'train sources'],
                [[f'agent SFT {role}', v['episodes'], v['sources']] for role, v in data['agent_SFT_by_role'].items()] +
                [['agent SFT union', data['SFT_train_union']['episodes'], data['SFT_train_union']['sources']],
                 ['one-shot kept', data['selection_counts']['train']['episodes'], data['selection_counts']['train']['sources']]]), '',
            f"Exclusions from train candidates: `{data['excluded_counts']['train']}`. "
            f"Validation examples: {data['selection_counts']['validation']['episodes']}.", '',
            table(['operator', 'agent SFT G', 'agent SFT K', 'SFT union', 'one-shot kept'],
                [[op, *[data['agent_SFT_by_role'][role]['operator_trajectories'].get(op, 0) for role in ROLES],
                  data['SFT_train_union']['operator_trajectories'].get(op, 0),
                  data['selection_counts']['train']['operator_trajectories'].get(op, 0)]
                 for op in sorted(data['SFT_train_union']['operator_trajectories'])]), '',
            'The one-shot policy uses the existing Phase6 whole-essay prompt at training and inference. '
            'Context remains8192; a whole-essay output allows up to4096 tokens (bounded by remaining '
            'context), matching the existing local one-shot baseline convention. Agent actions retain '
            'their1024 limit. No input or target is truncated. Temperature0/top_p1/seed47 and the pinned '
            'BF16 vLLM base are unchanged. Raw output is not cleaned with another model.', '',
            'Final sentences are aligned to corrupted input only using the existing Hungarian lexical '
            'alignment (threshold0.35), without the clean source or record answers. Combined R uses '
            'the existing one-shot convention: one ONE_SHOT step. GLOBAL/KOREAN role R, tool validity '
            'and tool STOP are not applicable because there is no intermediate stage/action protocol. '
            'As in the existing Phase6 baseline, a nonempty output with successful terminal measurements '
            'counts as completed; EOS versus output-limit endings are reported separately, and bounded '
            'outputs are scored exactly as generated. '
            'Recovery by operator/record level, combined R, R_over, completion and observed changes '
            'remain comparable; alignment uncertainty and the different data size limit interpretation.', '']
    lines += ['## Training', '',
        'Pinned base snapshot `c963a5f4f6496c749f94064a20b33028b0db9f19`, QLoRA r16/alpha32/dropout.05, '
        'all-linear, NF4 double quantization/BF16 compute,8192 context, gradient checkpointing. '
        'One epoch with both GPUs in data parallelism, micro-batch1 per rank and accumulation4 '
        '(global effective batch8). Policy serving and scorer were stopped during training. '
        'Loss is globally normalized over target tokens; a numerical test verifies accumulated '
        'two-rank gradients against the full target-token mean. '
        '[DDP gradient averaging](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).', '',
        ('RFT continues epoch2 adapter weights at lr5e-5 with a new optimizer/schedule.' if condition == 'rft1' else
         'One-shot starts a fresh LoRA from the same base, using the SFT lr1e-4 recipe.'), '',
        table(['role', 'train turns', 'initial validation NLL', 'final validation NLL', 'elapsed hours', 'peak per-GPU GiB'],
            [[role, metrics['data']['roles'][role]['train']['turns'], fmt(v['initial_validation']['eval_loss']),
              fmt(next((x['eval_loss'] for x in reversed(v['complete']['log_history']) if 'eval_loss' in x), None)),
              fmt(v['complete']['elapsed_s'] / 3600), fmt(v['complete']['max_cuda_allocated_bytes'] / 2**30)]
             for role, v in metrics['training'].items()]), '']
    lines += comparison_lines(conditions)
    lines += ['## Paired comparison', '',
        table(['new condition minus baseline', 'metric', 'paired n', 'mean difference', '95% CI'],
            [[condition + ' - ' + baseline, field, v['n'], fmt(v['difference']), v.get('ci95')]
             for baseline, fields in metrics['paired'].items() for field, v in fields.items()]), '',
        '## Residual marker misfit', '',
        'The accepted Sol/high final previous-sentence/affected-sentence prompt and closed fields are reused. '
        'Identical cached judgments are reused by contract hash; missing judgments remain unknown. '
        'RFT uses final surviving sites from actual accepted GLOBAL structural actions. One-shot uses '
        'observed predecessor/marker changes in the aligned final rewrite and a per-rewrite denominator; '
        'its rate is not relabeled as a per-action rate.', '',
        table(['condition', 'denominator', 'flagged', 'unknown', 'misfit share', 'judged cases'],
            [[name, v['structural_actions'], v['flagged_actions'], v['unknown_actions'], fmt(v['share']), v['judged_cases']]
             for name, v in metrics['baseline_markers'].items()] +
            [[condition, f"{markers['units']} {markers['unit']}", markers['flagged_units'], markers['unknown_units'],
              fmt(markers['share']), markers['judged_cases']]]), '',
        f"New-condition flagged final pairs: {markers['flagged_cases']}/{markers['judged_cases']} judged; "
        f"essays with at least one confirmed residual misfit: {markers['flagged_essays']}. "
        f"Bounds including unknown units: {markers['lower_bound']}–{markers['upper_bound']}.", '',
        f"Shared A/C Sol ledger at this report: `${metrics['sol_shared_A_C']['confirmed_usd']:.6f}` confirmed, "
        f"`${metrics['sol_shared_A_C'].get('reserved_usd', 0):.6f}` reserved/unknown; cap$3. "
        'B has its independent$10 ledger. Local GPU generation creates no teacher API charge.', '']
    if condition == 'rft1':
        contexts = metrics['context_audit']['essays']
        lines += ['## Context overflow audit', '',
            table(['real essay', 'characters', 'essay tokens', 'sentences', 'paragraphs'],
                [[eid, v['characters'], v['essay_tokens'], v['sentences'], v['paragraphs']] for eid, v in contexts.items()]), '',
            table(['essay', 'condition', 'role', 'initial prompt', 'mandatory compacted', 'mandatory + last update', 'prompt cap'],
                [[eid, name, role, h['initial_prompt_tokens'], h['mandatory_compaction_tokens'],
                  h['mandatory_plus_last_observation_tokens'], 7168]
                 for eid, v in contexts.items() for name, record in v['conditions'].items()
                 if name in ('epoch_2', 'rft1') for role, h in record['histories'].items()]), '',
            metrics['context_audit']['proposal'], '',
            'All four original SFT/Luna conditions and the RFT histories, component token counts, '
            'episode hashes and exact failure reasons are retained in `context_audit.json`. '
            'The proposal does not change this evaluation or silently exclude these essays.', '',
            'GLOBAL-record stop-without-structural-action shares, per-level action counts, KOREAN '
            'STOP/step-limit/error endings, per-operator role rewards and all paired metrics are '
            'available in `metrics.json`. These are development comparisons, not held-out test claims.', '']
    else:
        lines += [f"Alignment totals (completed dev): `{conditions['oneshot']['dev']['alignment']}`.", '',
            'This authorized one-shot comparison is complete. No further training, RFT round or new '
            'teacher generation is started.', '']
    write_json(root / 'metrics.json', metrics)
    name = 'V3_PHASE_8_RFT1.md' if condition == 'rft1' else 'V3_ONESHOT_BASELINE.md'
    destination = config['paths']['repo'] / 'imple/reports' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(root / 'report_status.json', {'completed': True, 'report': str(destination),
        'report_sha256': file_sha(destination), 'metrics_sha256': file_sha(root / 'metrics.json')})
    return {'report': str(destination), 'condition': condition}
