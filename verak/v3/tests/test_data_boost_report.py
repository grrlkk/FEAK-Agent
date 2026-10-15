from pathlib import Path

import pytest

from verak.v3.common import file_sha, read_json, write_json
from verak.v3.data_boost.report import component, finalize


def fixture(tmp_path):
    root = tmp_path / 'verak/v3/outputs/data_boost'
    root.mkdir(parents=True)
    snapshot = root / 'snapshot.json'
    write_json(snapshot, {'at': 'frozen', 'saved_samples': 4, 'essays_with_four_saved_samples': 1,
        'definition': 'same-sample all-record recovery', 'main_definition': 'role main',
        'unknown_policy': 'unknown retained', 'operators': {}, 'caveat': 'partial cohort',
        'new_calls': 0, 'gpu_used': False})
    write_json(root / 'task_contract.json', {'snapshot': str(snapshot)})
    write_json(root.parent / 'phase8_rft1/oneshot_hold.json', {'hold': True})
    approval = root / 'cpu_scorer/selection_approval.json'
    audit = root / 'cpu_scorer/audit_200/calibration.json'
    write_json(audit, {'status': 'complete', 'unique_source_essays': 200})
    write_json(approval, {'canonical_for_selection': True, 'fingerprint': 'frozen-gpu', 'score_source': 'gpu_reference',
                         'calibration_path': str(audit), 'calibration_sha256': file_sha(audit)})
    for name in ('global', 'insertion'):
        directory = root / name
        metrics, report = directory / 'metrics.json', directory / 'component_report.md'
        write_json(metrics, {'api': {'confirmed_usd': 1., 'reserved_usd': 0.},
                            'selected_korean': 0, 'gpu_used': True, 'training': False, 'score_source': 'gpu_reference'})
        report.write_text('# Component\n\nActual component results.\n')
        write_json(directory / 'complete.json', {'status': 'complete', 'no_live_paid_calls': True,
            'measurements_finished': True, 'stopped': True,
            'metrics_path': str(metrics), 'metrics_sha256': file_sha(metrics),
            'report_path': str(report), 'report_sha256': file_sha(report),
            'scorer_approval_sha256': file_sha(approval)})
    return root


def test_finalizer_waits_for_both_components_and_writes_no_provisional_final(tmp_path):
    root = tmp_path / 'verak/v3/outputs/data_boost'
    root.mkdir(parents=True)
    write_json(root / 'task_contract.json', {})
    assert finalize(tmp_path) is None
    assert not (tmp_path / 'imple/reports/V3_DATA_BOOST.md').exists()


def test_finalizer_requires_immutable_data_only_component_and_cap(tmp_path):
    root = fixture(tmp_path)
    metrics = root / 'insertion/metrics.json'
    original = read_json(metrics)
    write_json(metrics, {**original, 'training': True})
    with pytest.raises(ValueError, match='changed'):
        component(root, 'insertion')
    marker = root / 'insertion/complete.json'
    value = read_json(marker)
    write_json(marker, {**value, 'metrics_sha256': file_sha(metrics)})
    with pytest.raises(ValueError, match='resource contract'):
        component(root, 'insertion')
    write_json(metrics, {**original, 'api': {'confirmed_usd': 5.9, 'reserved_usd': .2}})
    write_json(marker, {**value, 'metrics_sha256': file_sha(metrics)})
    with pytest.raises(ValueError, match='budget cap'):
        component(root, 'insertion')


def test_finalizer_rejects_superseded_scorer_and_preserves_live_hold_state(tmp_path):
    root = fixture(tmp_path)
    approval = root / 'cpu_scorer/selection_approval.json'
    write_json(approval, {**read_json(approval), 'fingerprint': 'different'})
    with pytest.raises(ValueError, match='superseded'):
        finalize(tmp_path)
    for name in ('global', 'insertion'):
        marker = root / name / 'complete.json'
        write_json(marker, {**read_json(marker), 'scorer_approval_sha256': file_sha(approval)})
    result = finalize(tmp_path)
    assert result['oneshot_hold'] is True
    assert not result['gpu_used'] and result['paid_calls'] == 0 and not result['training']
    assert Path(result['report']).is_file()
    assert read_json(root.parent / 'phase8_rft1/oneshot_hold.json') == {'hold': True}
