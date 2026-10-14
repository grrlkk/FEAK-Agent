"""File-only V4_TRAIN report; no quality conclusions, new calls or model execution."""
from collections import Counter
import json
from pathlib import Path

from .common import REPO, file_sha, read_json, write_json
from .full_data import ROOT, ROLES
from . import orchestrator_v44 as orch


def report():
    if not (ROOT/'execution_complete.json').exists() and not (ROOT/'execution_failure.json').exists():
        return {'status': 'pending', 'report_written': False}
    data = read_json(ROOT/'data/manifest.json')
    frozen = read_json(orch.FROZEN)
    outcome = {r: read_json(ROOT/'adapters'/r/'complete.json') if (ROOT/'adapters'/r/'complete.json').exists()
               else None for r in ROLES}
    smoke = read_json(ROOT/'smoke/complete.json') if (ROOT/'smoke/complete.json').exists() else None
    lines = ['# V4 full SFT and integrated protocol smoke', '',
        'Training uses the frozen v4.3 worker demonstrations. Only inference uses v4.4. '
        'This report measures optimization loss and execution failures; it makes no writing-quality or recovery claim. '
        'RFT, the full benchmark, one-shot evaluation and the held-out test cohort were not run.', '',
        '## Frozen contracts', '',
        f"Orchestrator `{frozen['version']}`: **{frozen['policy_tokens']} body / {frozen['system_message_tokens']} system tokens**, SHA256 `{frozen['sha256']}`. "
        'The same text and observation/action renderer are used for every Orchestrator target and integrated inference.', '',
        '```text', frozen['text'], '```', '',
        'Worker system prompts and all original teacher messages/action targets remain v4.3. The Orchestrator uses '
        'structured public `DELEGATE.items` to retain locations, action types and needs_search; their union is the enforced scope. '
        'Revision receives 1–2 items; Korean can receive up to four plus its whole-essay form pass.', '',
        '|Inference rule|Implementation and reporting|', '|---|---|',
        '|No-op EDIT|Rejection adds `고칠 것이 없으면 STOP 한다.`|',
        '|Three consecutive rejections|Controller ends that stage as STOP(blocked); failure, never a valid model STOP. A valid action or a new stage resets the streak.|',
        '|Masked sentence MOVE/DELETE|Whole sentences may move or be removed. An EDIT still cannot alter or remove a mask token.|',
        '|Korean context|16,384 total / 1,024 output reserve; Revision stays 8,192 / 1,024. No document truncation. Orchestrator uses 8,192 / 4,096 for MAP.|',
        '|Before FINISH|Korean runs over the whole essay if no such pass has occurred since the last Revision delegation. The pass also runs after an Orchestrator protocol stop.|',
        '|Strict student parser|One JSON object. Teacher-only trimming is not used.|', '',
        '## Training data and shared validation', '',
        'Validation holds out exactly ceil(5%) of source groups per role. Identical normalized original essays share a group, '
        'including variants, both attempts, map/plan targets, both workers and one-shot. All four roles use one shared partition. '
        'The smoke20/dev10 sources and the frozen test questions are excluded. Training on earlier small diagnostic adapters is not continued.', '',
        '|Role|Training targets / sources|Validation targets / sources|Training groups / held groups|Training target tokens|',
        '|---|---:|---:|---:|---:|']
    for role in ROLES:
        t, v = (data['roles'][role][p] for p in ('train', 'validation'))
        lines.append(f"|{role}|{t['turns']} / {t['sources']}|{v['turns']} / {v['sources']}|{t['source_groups']} / {v['source_groups']}|{t['target_tokens']}|")
    lines += ['', '|Role / partition|Target counts by source|Action counts|', '|---|---|---|']
    for role in ROLES:
        for part, value in data['roles'][role].items():
            lines.append(f"|{role} / {part}|`{json.dumps(value['kinds'], sort_keys=True)}`|`{json.dumps(value['actions'], sort_keys=True)}`|")
    lines += ['', 'Loss covers the current action/final-essay target and end-of-turn token only. '
        'Observations, scores, tool outputs, search passages, notices and previous reports are masked. '
        'No sequence packing or truncation is used. Worker exported tokens/labels are audited against the original v4.3 exports.', '',
        'Orchestrator supervision combines the 1,000 B2 public plans, B3 record-derived public tasks, and the valid saved Luna consensus maps. '
        'Raw feedback, 1–9 LLM scores, human grades, corruption records, original answer text and private IDs never appear in its observations. '
        'These are partially supervised decisions, using actual saved worker states and status-only reports. '
        'Missing SCORE responses and MAPs are represented as unavailable; no successful tool/worker execution is invented. '
        'In two B3 cases a prior Revision deleted a later Korean target. The now-impossible item is omitted from the '
        'new Orchestrator target, preserving other items, the whole-essay pass and the actual blocked worker report; '
        'this is not treated as successful recovery. The original worker demonstrations are unchanged. '
        'The saved 2,000-map cohort has 1,997 valid pairs; three invalid pairs are retained as exclusions, with no new extraction.', '',
        f"Export notes: `{json.dumps(dict(Counter(n['kind'] for n in data['notes'])), sort_keys=True)}`. "
        f"Excluded action targets: `{json.dumps(dict(Counter((n['role']+'/'+n['reason']) for n in data['excluded_targets'])), sort_keys=True)}`. "
        'Exact source IDs, reasons, lengths and source hashes are preserved in `train44/data/manifest.json`.', '',
        'One-shot uses B2 quality-kept final attempts, retaining title/passage-id citations. For B3, it chooses the higher-combined-R '
        'completed attempt before applying all relevant role SFT conditions; it does not switch to a lower-R attempt if the winner fails. '
        'The pool is restricted to practices represented in the worker selections. This preserves the agreed baseline filter; '
        'role-specific agent selections and final-essay selections can therefore differ. All use the same source split.', '',
        '## Training and validation loss', '',
        'Each adapter starts from pinned base `c963a5f4f6496c749f94064a20b33028b0db9f19`. '
        'QLoRA NF4/double quantization/BF16, r=16, alpha=32, dropout=.05, all-linear, lr=1e-4, cosine schedule, '
        'warmup=.03, context 8,192, gradient checkpointing. Two-GPU data parallelism, microbatch 1/rank, '
        'four accumulation steps/rank (effective batch 8). Loss is globally normalized by supervised target tokens. '
        'Korean, Revision and Orchestrator each train two epochs; one-shot retains its prior one-epoch recipe. '
        'Each epoch is saved separately. Validation is token-weighted target NLL, not a quality result.', '',
        '|Role|Status|Epoch 1 validation loss|Epoch 2 validation loss|Elapsed training seconds|', '|---|---|---:|---:|---:|']
    checkpoints = {}
    for role, done in outcome.items():
        losses = {round(v['epoch']): v['eval_loss'] for v in (done or {}).get('log_history', [])
                  if 'eval_loss' in v and v.get('epoch', 0) > 0}
        lines.append(f"|{role}|{'complete' if done else 'not complete'}|{losses.get(1, '—')}|{losses.get(2, '—')}|{(done or {}).get('elapsed_s', '—')}|")
        checkpoints[role] = []
        if done:
            recipe = ROOT/'adapters'/role/'recipe.json'
            if file_sha(recipe) != done['recipe_sha256']: raise ValueError('Training recipe changed')
            for epoch in range(1, done['epochs']+1):
                folder = ROOT/'adapters'/role/f'epoch_{epoch}'
                proof = read_json(folder/'provenance.json')
                if file_sha(folder/'adapter_model.safetensors') != proof['adapter_sha256']:
                    raise ValueError('Saved epoch adapter changed')
                checkpoints[role].append({'epoch': epoch, 'path': str(folder), 'sha256': proof['adapter_sha256']})
    lines += ['', '## Integrated smoke', '',
        'The same20 smoke and the fixed agent_dev10 are reported separately. The trained epoch-2 Orchestrator generates MAP '
        'and worker delegations from the input question/essay. Saved Sol plans and corruption-record delegations are not loaded at inference. '
        'SCORE uses the frozen local scorer for diagnosis only; no scorer delta or quality comparison is reported. '
        'Worker STOP(blocked), controller STOP(blocked), step/context limits and exceptions are failures.', '']
    if smoke:
        for name, group in smoke['cohorts'].items():
            lines += [f'### {name}', '',
                f"Saved {group['saved']}/{group['planned']}; protocol-complete episodes {group['protocol_complete']}/{group['planned']}. "
                f"Orchestrator endings: `{json.dumps(group['orchestrator_endings'])}`; FINISH rate {100*group['orchestrator_FINISH_rate']:.2f}%. Mean model delegations per essay: {group['mean_model_delegations']:.3f}.", '',
                '|Role|Valid / returned|Valid-action rate|Stages|Model STOP(done / blocked)|Model STOP rate|Controller STOP(blocked)|Context limits|',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
            for role, r in group['roles'].items():
                valid = '—' if r['valid_action_rate'] is None else f"{100*r['valid_action_rate']:.2f}%"
                stop_rate = '—' if r['model_STOP_rate'] is None else f"{100*r['model_STOP_rate']:.2f}%"
                lines.append(f"|{role}|{r['valid_actions']} / {r['returned_actions']}|{valid}|{r['stages']}|{r['model_STOP_done']} / {r['model_STOP_blocked']}|{stop_rate}|{r['controller_STOP_blocked']}|{len(r['context_limits'])}|")
            lines += ['', '```json', json.dumps(group, ensure_ascii=False, indent=2), '```', '']
    else:
        lines += ['Integrated smoke did not complete. Preserve and inspect `failure_smoke.json`; no result is inferred.', '']
    failures = {p.name: read_json(p) for p in ROOT.glob('*failure*.json')}
    failures.update({p.name: read_json(p) for p in ROOT.glob('failure_*.json')})
    if (ROOT/'preparation_incidents.json').exists():
        failures['resolved_preparation_incidents'] = read_json(ROOT/'preparation_incidents.json')
    lines += ['## Failures, provenance and stopping point', '', '```json', json.dumps(failures, ensure_ascii=False, indent=2), '```', '',
        'Checkpoints and dataset hashes are in `train44/report_metrics.json`. All raw teacher collections, v1 rewards, '
        'prior reports and `FEAK_AGENT_METHOD.md` remain unchanged. No RFT or full evaluation follows this run.', '']
    path = REPO/'imple/reports/V4_TRAIN.md'
    path.write_text('\n'.join(lines))
    result = {'data_manifest_sha256': file_sha(ROOT/'data/manifest.json'), 'prompt_manifest_sha256': file_sha(orch.FROZEN),
        'checkpoints': checkpoints, 'training': outcome, 'smoke': smoke, 'failures': failures,
        'report': {'path': str(path), 'sha256': file_sha(path)}, 'quality_claims': False,
        'RFT_started': False, 'full_evaluation_started': False, 'new_paid_API_calls': 0}
    write_json(ROOT/'report_metrics.json', result)
    return result
