"""Join final artifacts; the watcher also dispatches authorized post-RFT scoring."""
from datetime import datetime, timezone
import fcntl
import math
from pathlib import Path
import time

from ..common import file_sha, read_json, write_json

CAPS = {'global': 40., 'insertion': 6.}


def component(root, name):
    directory = (root / name).resolve()
    marker_path = directory / 'complete.json'
    if not marker_path.exists():
        return None
    marker = read_json(marker_path)
    if marker.get('status') not in {'complete', 'budget_stop'}:
        raise ValueError(name + ': component is not finalized')
    for key in ('no_live_paid_calls', 'measurements_finished', 'stopped'):
        if marker.get(key) is not True:
            raise ValueError(name + ': unfinished ' + key)
    artifacts = {}
    for kind in ('metrics', 'report'):
        path = Path(marker[kind + '_path']).resolve()
        if not path.is_relative_to(directory) or file_sha(path) != marker[kind + '_sha256']:
            raise ValueError(name + ': changed or out-of-scope ' + kind)
        artifacts[kind] = path
    metrics = read_json(artifacts['metrics'])
    budget = metrics.get('api', metrics.get('budget'))
    if budget is None or any(not math.isfinite(budget[k]) or budget[k] < 0
                             for k in ('confirmed_usd', 'reserved_usd')):
        raise ValueError(name + ': missing or invalid budget')
    if budget['confirmed_usd'] + budget['reserved_usd'] > CAPS[name] + 1e-8:
        raise ValueError(name + ': separate budget cap exceeded')
    if metrics.get('training') is not False:
        raise ValueError(name + ': data-only resource contract violated')
    if metrics.get('score_source') != 'gpu_reference':
        raise ValueError(name + ': final rewards are not exclusively GPU reference values')
    korean = metrics.get('selected_korean', metrics.get('selected_counts', {}).get('korean'))
    if korean != 0 or metrics.get('selected', {}).get('korean'):
        raise ValueError(name + ': unauthorized KOREAN selections')
    return {'marker': marker, 'marker_sha256': file_sha(marker_path), 'metrics': metrics,
            'budget': budget, 'report': artifacts['report'].read_text(encoding='utf-8')}


def snapshot_section(snapshot):
    def rate(value):
        return 'NA' if value is None else f'{value:.2%}'
    lines = ['## RFT recovery from the saved snapshot', '',
        f"Frozen at {snapshot['at']}: {snapshot['saved_samples']} saved samples; "
        f"{snapshot['essays_with_four_saved_samples']} essays with all four samples. "
        'No new model, scorer, analyzer, or API calls were made for this snapshot.', '',
        snapshot['definition'], snapshot['main_definition'], snapshot['unknown_policy'], '',
        '| Operator | Essays with 4 samples | Main full recovery | Full record incl. coupled | Main unknown | Main upper bound |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for op, value in snapshot['operators'].items():
        main, full = value['role_main'], value['combined_full_record']
        n = value['essays_with_four_saved_samples']
        lines.append(f"| {op} | {n} | {main['successful_essays']}/{n} ({rate(main['share'])}) | "
                     f"{full['successful_essays']}/{n} ({rate(full['share'])}) | "
                     f"{main['unresolved_essays']} | {rate(main['upper_share_if_unknown_success'])} |")
    lines += ['', snapshot['caveat'], 'L_POLARITY has only one essay in this snapshot.', '']
    return '\n'.join(lines)


def demote(text):
    lines = text.splitlines()
    if lines and lines[0].startswith('# '):
        lines = lines[1:]
    return '\n'.join('#' + line if line.startswith('#') else line for line in lines).strip()


def finalize(repo):
    repo = Path(repo).resolve()
    root = repo / 'verak/v3/outputs/data_boost'
    contract = read_json(root / 'task_contract.json')
    components = {name: component(root, name) for name in CAPS}
    if any(value is None for value in components.values()):
        return None
    approval_path = root / 'cpu_scorer/selection_approval.json'
    approval = read_json(approval_path)
    approval_sha = file_sha(approval_path)
    if approval.get('canonical_for_selection') is not True or approval.get('score_source') != 'gpu_reference':
        raise ValueError('GPU reference rescoring remains unapproved')
    if any(value['marker'].get('scorer_approval_sha256') != approval_sha
           for value in components.values()):
        raise ValueError('A component used a superseded scorer approval')
    audit_path = Path(approval['calibration_path']).resolve()
    if not audit_path.is_relative_to((root / 'cpu_scorer').resolve()) or file_sha(audit_path) != approval['calibration_sha256']:
        raise ValueError('CPU/GPU audit path or digest changed')
    audit = read_json(audit_path)
    if audit.get('unique_source_essays', 0) < 200 or audit.get('status') != 'complete':
        raise ValueError('The required 200-source-essay CPU/GPU comparison is unfinished')
    hold = read_json(repo / 'verak/v3/outputs/phase8_rft1/oneshot_hold.json')
    if type(hold.get('hold')) is not bool:
        raise ValueError('Missing explicit one-shot hold/resumption state')
    rft_design = repo / 'verak/v3/outputs/phase8_rft1/rollout_design.json'
    if rft_design.exists():
        design = read_json(rft_design)
        if file_sha(design['corpus']) != design['corpus_sha256']:
            raise ValueError('Active RFT corpus changed during the data additions')
        for relative, expected in design['v1_runtime_sha256'].items():
            if file_sha(repo / relative) != expected:
                raise ValueError('Accepted v1 runtime changed: ' + relative)
    snapshot_path = Path(contract['snapshot'])
    snapshot = read_json(snapshot_path)
    if snapshot['new_calls'] != 0 or snapshot['gpu_used']:
        raise ValueError('Saved-rollout snapshot made new calls')
    statuses = {name: value['marker']['status'] for name, value in components.items()}
    finality = 'Final data-preparation report; budget stops and unknown outcomes are disclosed below.'
    lines = ['# VERAK v3 — GLOBAL data boost', '', '**Status: final.** ' + finality, '',
        'RFT rollouts retained their original data, environment and sampling recipe. '
        + ('One-shot baseline C remains on hold until explicit user resumption. ' if hold['hold'] else
           'One-shot baseline C was held for this task and its hold has subsequently been released. ')
        + 'Only GPU-rescored v1 GLOBAL additions available before training may join RFT1; '
        'G_DEL_LINK stays separate for round2 and no round2 training is authorized. '
        'KOREAN actions, operators, prompt and extra selected training data are unchanged. '
        'Both teacher roles run, but only GLOBAL is selected.', '',
        '## Decisions and costs', '',
        'For G_DEL_LINK only, the gate is QC passes / judged records >=30%. The archived retry '
        'passed96/270=35.56%; its earlier96/380 source yield remains a coverage finding. '
        'The additional batch and pooled QC rates are reported separately. See addendum Section10 '
        'items21–23. L_FUSE remains dropped.', '',
        'Weak-operator practice variants use eligible agent_train sources; newly generated variants '
        'may reuse active-corpus sources only at proved new swap/move positions. '
        'The number of distinct source essays is reported separately from the number of distinct '
        'single-record practices; multiple valid variants do not add source diversity. '
        'The approved diversified pool is400 practices per weak operator, at most4 per source/operator, '
        'subject to the cumulative$40 GLOBAL cap. Archived excess remains preserved.', '',
        '| Component | Stop status | Confirmed USD | Retained reservation USD | Cap USD |',
        '| --- | --- | ---: | ---: | ---: |']
    for name, value in components.items():
        budget = value['budget']
        lines.append(f"| {name} | {statuses[name]} | {budget['confirmed_usd']:.6f} | "
                     f"{budget['reserved_usd']:.6f} | {CAPS[name]:.2f} |")
    lines += ['', snapshot_section(snapshot), '', '## Weak GLOBAL operators', '',
              demote(components['global']['report']), '', '## G_DEL_LINK insertion', '',
              demote(components['insertion']['report']), '', '## Provenance and checks', '',
              'Original SFT eligibility is used, with the best GLOBAL attempt per practice. '
              'No later RFT-only STOP/rejection gate is added to the data-preparation selections; '
              'the RFT1 merge additionally applies its terminal-STOP/rejection rules. '
              'All final rewards and selections below use the GPU reference scorer. CPU rewards '
              'are provisional comparison values only; unmeasured values are not zero.', '',
              f"CPU/GPU audit: {audit['unique_source_essays']} distinct source essays. Full audit: `{audit_path}`. "
              'Per-rubric agreement, Q errors and role-threshold changes are disclosed in the component reports.', '',
              'The unchanged v1 replay matched all92 saved pilot trajectories (91 full rewards and '
              'the historical completed GLOBAL reward), with no new API, Bareun, scorer or GPU calls. '
              'Control/report code and experiment data are kept separate; no data, models or reports '
              'were uploaded.', '',
              f'Snapshot artifact: `{snapshot_path}`. Component artifacts and budget ledgers: `{root}`.', '']
    report = repo / 'imple/reports/V3_DATA_BOOST.md'
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(report)
    result = {'status': 'complete', 'at': datetime.now(timezone.utc).isoformat(),
              'component_statuses': statuses, 'report': str(report), 'report_sha256': file_sha(report),
              'component_marker_sha256': {name: value['marker_sha256'] for name, value in components.items()},
              'scorer_approval_sha256': approval_sha,
              'snapshot_sha256': file_sha(snapshot_path), 'gpu_used': False,
              'cpu_gpu_audit_sha256': file_sha(audit_path), 'score_source': 'gpu_reference',
              'paid_calls': 0, 'training': False, 'oneshot_hold': hold['hold']}
    write_json(root / 'report_complete.json', result)
    return result


def watch(repo):
    """Wait for data, dispatch deferred reference scoring, then finish the report."""
    repo = Path(repo).resolve()
    root = repo / 'verak/v3/outputs/data_boost'
    control = root / 'report_controller'
    control.mkdir(parents=True, exist_ok=True)
    with (control / 'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                # No GPU work can be scheduled until the entire RFT A report is
                # complete; ready components already scored in its earlier gap
                # are skipped. Every GPU call runs in a separate owned child.
                rft_done = repo / 'verak/v3/outputs/phase8_rft1/a_complete.json'
                if rft_done.exists():
                    from ..rft1.config import config_for
                    from .rescore import service_deferred
                    service_deferred(config_for())
                result = finalize(repo)
            except Exception as exc:
                write_json(root / 'report_controller/status.json', {'stage': 'failed',
                    'type': type(exc).__name__, 'message': str(exc), 'at': time.time()})
                raise
            if result is not None:
                write_json(root / 'report_controller/status.json', {'stage': 'complete', **result})
                return result
            write_json(root / 'report_controller/status.json', {'stage': 'waiting_for_components',
                'at': time.time(), 'gpu_used': False, 'paid_calls': 0, 'training': False})
            time.sleep(30)
