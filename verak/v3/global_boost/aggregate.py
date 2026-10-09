"""Final component artifacts across all immutable GLOBAL practice batches."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..common import file_sha, read_json, write_json
from ..train.teacher_bulk import atomic_new, collection_lock
from .config import OPERATORS, PHASE
from .expansion import batch_configs, original_teacher_running, root_for
from .measure import measured_path
from .paid import BoostAPI
from .prepare import corpus


def approved(config):
    path = root_for(config).parent / 'cpu_scorer/selection_approval.json'
    if not path.exists():
        return None
    value = read_json(path)
    if not value.get('canonical_for_selection') or not value.get('fingerprint'):
        return None
    return {**value, 'path': str(path), 'sha256': file_sha(path)}


def weighted(rows, key, count='global_reward_measured'):
    total = sum(r[count] for r in rows if r[key] is not None)
    return sum(r[key]*r[count] for r in rows if r[key] is not None)/total if total else None


def report(config, approval):
    from .report import report as batch_report
    root = root_for(config)
    batches, selected, inputs = [], {}, {}
    score_devices = Counter()
    for cfg in batch_configs(config):
        cfg[PHASE].update(score_fingerprint=approval['fingerprint'], scorer_approval_sha256=approval['sha256'])
        target = cfg['paths'][PHASE + '_output']
        batch_report(cfg)
        value = read_json(target / 'component_metrics.json')
        batches.append({'root': str(target), 'source_plan_sha256': file_sha(target / 'source_plan.json'),
                        'metrics': value})
        for key, item in value['selected'].items():
            if key in selected:
                raise ValueError('Duplicate immutable practice identity across batches')
            selected[key] = item
        inputs.update(corpus(cfg))
        score_devices.update(value['score_device_observations'])
    metrics = {}
    for operator in OPERATORS:
        rows = [b['metrics']['operators'][operator] for b in batches]
        qc = {k: sum(r['qc'][k] for r in rows) for k in ('generated', 'judged', 'passed', 'unknown')}
        qc['pass_rate_judged'] = qc['passed']/qc['judged'] if qc['judged'] else None
        counts = {k: sum(r[k] for r in rows) for k in ('generated', 'teacher_planned', 'teacher_saved',
            'generation_completed', 'global_reward_measured', 'fully_recovered_attempts', 'selected_global')}
        pairs = {k: sum(r['at_least_one_of_two'][k] for r in rows) for k in ('fully_observed_essays', 'recovered')}
        pairs['share'] = pairs['recovered']/pairs['fully_observed_essays'] if pairs['fully_observed_essays'] else None
        terminations = {}
        for role in ('global', 'korean'):
            end = Counter()
            for r in rows:
                end.update(r['termination'][role])
            terminations[role] = dict(end)
        unknown = [item for r in rows for item in r['unknown']]
        metrics[operator] = {'requested': 400, **counts, 'qc': qc,
            'not_generated': 400-counts['generated'], 'selected_korean': 0,
            'completion_rate': counts['generation_completed']/counts['teacher_saved'] if counts['teacher_saved'] else None,
            'teacher_main_recovery': weighted(rows, 'teacher_main_recovery'),
            'full_recovery_rate': counts['fully_recovered_attempts']/counts['global_reward_measured'] if counts['global_reward_measured'] else None,
            'R': weighted(rows, 'R'), 'R_over': weighted(rows, 'R_over'),
            'at_least_one_of_two': pairs,
            'observed_any_sample': {k: sum(r['observed_any_sample'][k] for r in rows) for k in ('essays', 'recovered')},
            'termination': terminations, 'unknown': unknown,
            'distinct_sources': len({r['source_id'] for r in inputs.values() if r['operator'] == operator}),
            'selected_distinct_sources': len({r['source_id'] for r in selected.values() if r['operator'] == operator})}
        hashes = [r['corrupted_hash'] for r in inputs.values() if r['operator'] == operator]
        if len(hashes) != len(set(hashes)) or counts['generated'] > 400:
            raise ValueError('Expansion exceeds its quota or repeats a corrupted practice')
    api = BoostAPI(config, 0)
    account = api.accounting()
    with api.db() as db:
        outcomes = Counter(dict(db.execute('SELECT status,COUNT(*) FROM calls GROUP BY status').fetchall()))
    api.close()
    if account['confirmed_usd']+account['reserved_usd'] > 12.+1e-8:
        raise ValueError('Shared GLOBAL boost cap exceeded')
    initial = read_json(root / 'source_plan.json')
    audit = read_json(root / 'variant_supply_validated.json')
    status = read_json(root / 'expansion_status.json')
    source_histogram = Counter(r['source_id'] for r in inputs.values())
    value = {'phase': PHASE, 'aggregate': True, 'operators': metrics,
        'selected_global': len(selected), 'selected_korean': 0, 'selected': selected,
        'api': account, 'api_terminal_status_counts': dict(outcomes),
        'distinct_new_sources': len(source_histogram), 'practices_per_source': dict(source_histogram),
        'qc_pass_distinct_sources': len({inputs[i]['source_id'] for b in batches for i in
            read_json(Path(b['root']) / 'teacher_design.json')['orders']['1']}),
        'selected_distinct_sources': len({r['source_id'] for r in selected.values()}),
        'source_inventory': initial['source_inventory']['counts'],
        'source_plan_parent_sha256': file_sha(root / 'source_plan.json'),
        'variant_supply': audit['overall'], 'variant_supply_sha256': file_sha(root / 'variant_supply_validated.json'),
        'batch_manifests': [{'root': b['root'], 'source_plan_sha256': b['source_plan_sha256']} for b in batches],
        'selection_rule': 'Existing SFT absolute_selection; best GLOBAL R per practice, not per shared source. No extra RFT-only STOP/rejection gate.',
        'source_grouping': 'These are new corruption practices on shared unused sources; any future training/validation partition must group by source.',
        'teacher_roles': ['global', 'korean'], 'selected_roles': ['global'],
        'unmeasured_reward_roles': ['korean', 'combined'],
        'score_device_observations': dict(score_devices),
        'canonical_for_selection': True, 'scorer_approval': approval,
        'cpu_score_audit': read_json(root.parent / 'cpu_scorer/calibration.json'),
        'completion': {'teacher_requested': sum(r['teacher_planned'] for r in metrics.values()),
            'teacher_saved': sum(r['teacher_saved'] for r in metrics.values()),
            'rewards_measured': sum(r['global_reward_measured'] for r in metrics.values()),
            'stop_reason': status['stop_reason'], 'pending_api': account['pending']},
        'gpu_used': False, 'training': False}
    write_json(root / 'best_global_trajectories.json', {'global': selected, 'korean': {}})
    write_json(root / 'component_metrics.json', value)
    def number(v):
        return 'NA' if v is None else f'{v:.4f}'
    lines = ['# GLOBAL data boost', '',
        f"The preserved Phase3b source filter leaves {len(source_histogram)} unused source essays. "
        f"They supply {audit['overall']['G_PARA_SWAP']['valid_distinct']} distinct valid G_PARA_SWAP "
        f"and {audit['overall']['G_SENT_MOVE']['valid_distinct']} G_SENT_MOVE variants. "
        'Multiple distinct practices can share a source; each contains one corruption record. '
        'The original 84-candidate batch is preserved unchanged.', '',
        '|Operator|Practices|Distinct sources|QC pass/judged|Teacher saved/planned|Main recovery|R_over|Selected GLOBAL|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for op, r in metrics.items():
        lines.append(f"|{op}|{r['generated']}|{r['distinct_sources']}|{r['qc']['passed']}/{r['qc']['judged']}|"
            f"{r['teacher_saved']}/{r['teacher_planned']}|{number(r['teacher_main_recovery'])}|{number(r['R_over'])}|{r['selected_global']}|")
    lines += ['', f"Stopped: `{status['stop_reason']}`. API cost ${account['confirmed_usd']:.6f}; "
        f"retained unknown reservations ${account['reserved_usd']:.6f}; live pending {account['pending']}; shared cap $12.", '',
        value['selection_rule'], '', value['source_grouping'], '',
        'The unchanged two-stage v1 teacher runs both editors, but only GLOBAL reward states and GLOBAL selections are measured here. '
        'KOREAN/combined rewards are unmeasured. No KOREAN data was selected; no GPU or training was used.', '',
        f"Selection uses the approved CPU scorer fingerprint `{approval['fingerprint']}` plus exact saved GPU-cache hits. "
        'The earlier FP32 arithmetic results are excluded from canonical selection; calibration and scorer provenance are preserved.', '',
        'Unattempted or unmeasured slots and terminal errors remain explicit in component_metrics.json; '
        'a missing result is never a successful recovery.', '']
    (root / 'component_report.md').write_text('\n'.join(lines), encoding='utf-8')
    return value


def finalize(config):
    from .measure import run as measure
    root = root_for(config)
    status_path = root / 'finalization_status.json'
    with collection_lock(root / 'finalization'):
        while True:
            expansion = read_json(root / 'expansion_status.json') if (root / 'expansion_status.json').exists() else {}
            approval = approved(config)
            if expansion.get('stage') == 'generation_finished' and not original_teacher_running(config) and approval:
                break
            write_json(status_path, {'stage': 'waiting_generation_and_scorer_approval', 'at': time.time(),
                'generation_stage': expansion.get('stage'), 'scorer_approved': bool(approval),
                'gpu_used': False, 'training': False})
            time.sleep(30)
        terminal_errors = []
        for cfg in batch_configs(config):
            cfg[PHASE].update(score_fingerprint=approval['fingerprint'], scorer_approval_sha256=approval['sha256'])
            batch = cfg['paths'][PHASE + '_output']
            write_json(status_path, {'stage': 'measure', 'batch': str(batch), 'at': time.time(),
                'scorer_approval_sha256': approval['sha256'], 'gpu_used': False, 'training': False})
            measure(cfg)
            for path in sorted(batch.glob('attempt_*/episodes/*.json')):
                attempt = int(path.parent.parent.name.split('_')[-1])
                if measured_path(cfg, attempt, path).exists():
                    continue
                error_path = batch / f'attempt_{attempt}/measurement_errors' / path.name
                error = read_json(error_path) if error_path.exists() else {}
                if error.get('type') not in {'ValueError', 'KeyError', 'TypeError'}:
                    raise RuntimeError('Canonical measurement still unavailable: '+str(path)+' '+json.dumps(error))
                terminal_errors.append({'path': str(path), 'error': error})
        if approved(config) != approval:
            raise RuntimeError('Scorer approval changed while measuring')
        result = report(config, approval)
        if result['completion']['teacher_saved'] < result['completion']['teacher_requested'] and 'budget' not in result['completion']['stop_reason']:
            raise RuntimeError('Unattempted authorized teacher slots remain without a budget stop')
        if result['api']['pending']:
            raise RuntimeError('Live paid calls remain; cannot close the GLOBAL boost')
        result['terminal_measurement_errors'] = terminal_errors
        result['measurements_finished'] = True
        write_json(root / 'component_metrics.json', result)
        marker = {'status': 'budget_stop' if 'budget' in result['completion']['stop_reason'] else 'complete',
            'metrics_path': str(root / 'component_metrics.json'), 'metrics_sha256': file_sha(root / 'component_metrics.json'),
            'report_path': str(root / 'component_report.md'), 'report_sha256': file_sha(root / 'component_report.md'),
            'scorer_approval_sha256': approval['sha256'], 'no_live_paid_calls': True,
            'measurements_finished': True, 'stopped': True, 'gpu_used': False, 'training': False,
            'terminal_measurement_error_count': len(terminal_errors), 'at': time.time()}
        if (root / 'complete.json').exists():
            raise ValueError('GLOBAL boost already finalized; preserve its completion marker')
        atomic_new(root / 'complete.json', marker)
        write_json(status_path, {'stage': 'complete', **marker})
        return marker


def launch(config):
    root = root_for(config)
    worker = root / 'finalizer_worker.json'
    if worker.exists():
        old = read_json(worker)
        process = Path(f'/proc/{old["pid"]}/cmdline')
        if process.exists() and b'verak.v3.cli.global_boost' in process.read_bytes() and b'finalize' in process.read_bytes():
            return old
    with (root / 'finalization.log').open('a') as stream:
        process = subprocess.Popen([sys.executable, '-m', 'verak.v3.cli.global_boost', 'finalize'],
            cwd=str(Path(__file__).resolve().parents[3]), stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    value = {'pid': process.pid, 'at': time.time(), 'gpu_used': False, 'training': False}
    write_json(worker, value)
    return value
