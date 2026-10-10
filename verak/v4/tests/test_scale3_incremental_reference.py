from copy import deepcopy
from pathlib import Path

import pytest

from verak.v3.common import file_sha, pair_key, read_json, write_json
from verak.v3.corrupt.document import Document, Paragraph, Unit
from verak.v4.scale2_projection import save_document
from verak.v4 import scale3_incremental_reference as inc


def score(key, value=2.):
    return {'cache_key': key, 'expected': [value]*8, 'integers': [2]*8,
            'mean': value, 'cache_hit': False, 'score_line': '2 2 2 2 2 2 2 2',
            'input_tokens': 10, 'genre': '설명'}


def response(key):
    return {'fingerprint': 'reference', 'execution_fingerprint': 'gpu0-config',
            'execution_device': 'gpu_reference', 'physical_gpu': 0, 'local_gpu': 0,
            'cache_disabled': True, 'result': score(key)}


def test_calibration_is_exact_across_all_eight_expectations_digits_and_Q():
    manifest = {'reference_fingerprint': 'reference', 'episode_count': 20,
                'distinct_sources': 20, 'endpoint_references': 60,
                'requests': [{'key': 'key', 'expected': score('key')}]}
    assert inc.calibration_summary(manifest, {'key': response('key')})['passed']
    for field in ('expected', 'integers', 'mean', 'score_line', 'input_tokens'):
        changed = response('key')
        if field == 'expected':
            changed['result'][field][7] += 1e-13
        elif field == 'integers':
            changed['result'][field][7] = 3
        elif field == 'mean':
            changed['result'][field] += 1e-13
        else:
            changed['result'][field] = 'different' if field == 'score_line' else 11
        assert not inc.calibration_summary(manifest, {'key': changed})['passed']
    cached = response('key')
    cached['result']['cache_hit'] = True
    with pytest.raises(ValueError, match='fresh'):
        inc.calibration_summary(manifest, {'key': cached})


def files(tmp_path):
    root, repo = tmp_path/'scale3', tmp_path/'repo'
    output = root/'B3/gpu_rescore/incremental'
    sample = root/'B3/sample.json'
    write_json(sample, {'episode_ids': ['practice:1'], 'reference_gpu_fingerprint': 'reference'})
    contract_path = root/'B3/contract.json'
    write_json(contract_path, {'prompt_version': 'v4.3_editors_20261010'})
    prepared = root/'B3/essays/practice_1.json'
    write_json(prepared, {'corpus': {'question': '문항', 'episode_id': 'practice:1',
                                   'source_id': 'source:1', 'scorer_fingerprint': 'reference'}})
    doc = save_document(Document([Paragraph('P1', [Unit('S1', 'B3글', [])])], ['']))
    raw = root/'B3/content/attempts/practice_1_a1.json'
    write_json(raw, {'completed': True, 'status': 'completed', 'corpus_episode_id': 'practice:1',
                    'source_id': 'source:1', 'attempt': 1, 'prompt_version': 'v4.3_editors_20261010',
                    'prepared_path': str(prepared), 'prepared_sha256': file_sha(prepared),
                    'states': {k: {'document': doc} for k in ('initial', 'stage1', 'final')},
                    'reward_projection': {'states': {k: {} for k in ('initial', 'stage1', 'final')}, 'unknown': {}}})
    key = pair_key('문항', 'RFT글')
    calibration = {'reference_fingerprint': 'reference', 'episode_count': 20, 'distinct_sources': 20,
        'endpoint_references': 60, 'requests': [{'key': key, 'question': '문항', 'text': 'RFT글', 'expected': score(key)}],
        'bindings': {}, 'episodes': []}
    write_json(output/'calibration/manifest.json', calibration)
    contract = {'reference_fingerprint': 'reference', 'bindings': {str(sample): file_sha(sample),
        str(contract_path): file_sha(contract_path), str(output/'calibration/manifest.json'): file_sha(output/'calibration/manifest.json')}}
    write_json(output/'contract.json', contract)
    config = {'paths': {'repo': repo}}
    return root, config, raw, prepared


class FakeScorer:
    calls = []
    closed = 0
    mismatch = False
    fingerprint = 'reference'
    execution_fingerprint = 'gpu0-config'
    device = {'physical_gpu': 0, 'local_gpu': 0, 'name': 'test-double'}

    def __init__(self, config, contract):
        pass

    def score(self, question, text):
        self.calls.append(text)
        return score(pair_key(question, text), 2.1 if self.mismatch and text == 'RFT글' else 2.)

    def close(self):
        type(self).closed += 1


def fake_gpu(monkeypatch):
    FakeScorer.calls = []
    FakeScorer.closed = 0
    FakeScorer.mismatch = False
    monkeypatch.setattr(inc, 'GPU0Scorer', FakeScorer)
    monkeypatch.setattr(inc, 'ensure_gpu0_only', lambda: None)


def test_failed_calibration_never_scores_B3_or_retries_even_after_resume(tmp_path, monkeypatch):
    root, config, _, _ = files(tmp_path)
    fake_gpu(monkeypatch)
    FakeScorer.mismatch = True
    with pytest.raises(RuntimeError, match='calibration failed'):
        inc.score_batch(config, root)
    assert FakeScorer.calls == ['RFT글']
    assert not (root/'B3/gpu_rescore/responses').exists()
    with pytest.raises(RuntimeError, match='calibration failed'):
        inc.score_batch(config, root)
    assert FakeScorer.calls == ['RFT글']


def test_snapshots_only_completed_attempts_and_detect_later_mutations(tmp_path):
    root, _, raw, _ = files(tmp_path)
    value = read_json(raw)
    write_json(raw, {**value, 'completed': False})
    assert inc.endpoint_plan(raw, root) is None
    write_json(raw, value)
    plan = inc.endpoint_plan(raw, root)
    assert plan['quality_keys']['initial'] == pair_key('문항', 'B3글')
    assert Path(plan['raw']['snapshot_path']).read_bytes() == raw.read_bytes()
    write_json(raw, {**value, 'extra': 'changed'})
    with pytest.raises(ValueError, match='snapshotted'):
        inc.endpoint_plan(raw, root)


def final_manifest(root, config, raw, prepared):
    sample, contract = root/'B3/sample.json', root/'B3/contract.json'
    key = pair_key('문항', 'B3글')
    manifest = {'component': 'B3', 'version': 'v4.3', 'teacher_collection_finished': True,
        'no_live_paid_calls': True, 'reference_gpu_fingerprint': 'reference', 'api': {'pending': 0},
        'sample_path': str(sample), 'sample_sha256': file_sha(sample),
        'contract_path': str(contract), 'contract_sha256': file_sha(contract),
        'episodes': [{'episode_id': 'practice:1', 'attempt': 1, 'raw_path': str(raw), 'raw_sha256': file_sha(raw),
            'prepared_path': str(prepared), 'prepared_sha256': file_sha(prepared),
            'quality_keys': {k: key for k in ('initial', 'stage1', 'final')}}],
        'requests': [{'key': key, 'question': '문항', 'text': 'B3글'}]}
    path = root/'B3/gpu_manifest.json'
    write_json(path, manifest)
    write_json(root/'B3/gpu_ready.json', {'manifest_path': str(path), 'manifest_sha256': file_sha(path),
                                       'teacher_collection_finished': True, 'no_live_paid_calls': True})
    report = config['paths']['repo']/'report.md'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('Completed one-shot.')
    write_json(config['paths']['repo']/'verak/v3/outputs/oneshot_baseline/complete.json', {
        'completed': True, 'report': {'completed': True, 'report': str(report), 'report_sha256': file_sha(report)}})


def test_incremental_resume_and_final_existing_handoff_need_no_new_gpu_calls(tmp_path, monkeypatch):
    root, config, raw, prepared = files(tmp_path)
    fake_gpu(monkeypatch)
    first = inc.score_batch(config, root)
    assert FakeScorer.calls == ['RFT글', 'B3글'] and first['new'] == 1
    assert first['physical_gpu'] == 0 and first['fingerprint'] == 'reference'
    assert first['execution_fingerprint'] == 'gpu0-config'
    original = file_sha(raw)
    def forbidden(*args, **kwargs):
        raise AssertionError('A fully scored batch must not load or probe GPUs')
    monkeypatch.setattr(inc, 'ensure_gpu0_only', forbidden)
    monkeypatch.setattr(inc, 'GPU0Scorer', forbidden)
    again = inc.score_batch(config, root)
    assert again['new'] == 0
    final_manifest(root, config, raw, prepared)
    final = inc.score_batch(config, root)
    assert final['status'] == 'complete'
    assert final['slot'] == 'post_oneshot_evaluation' and final['scheduled_by'] == 'root'
    assert final['manifest_sha256'] == file_sha(root/'B3/gpu_manifest.json')
    assert inc.score_batch(config, root) == final
    assert file_sha(raw) == original


def test_final_handoff_rejects_changed_or_swapped_endpoint_even_with_new_manifest_sha(tmp_path, monkeypatch):
    root, config, raw, prepared = files(tmp_path)
    fake_gpu(monkeypatch)
    inc.score_batch(config, root)
    final_manifest(root, config, raw, prepared)
    path = root/'B3/gpu_manifest.json'
    manifest = read_json(path)
    manifest['episodes'][0]['quality_keys']['initial'] = pair_key('문항', 'wrong')
    manifest['requests'].append({'key': pair_key('문항', 'wrong'), 'question': '문항', 'text': 'wrong'})
    write_json(path, manifest)
    ready = read_json(root/'B3/gpu_ready.json')
    write_json(root/'B3/gpu_ready.json', {**ready, 'manifest_sha256': file_sha(path)})
    with pytest.raises(ValueError, match='saved question and endpoint'):
        inc.score_batch(config, root)
    assert FakeScorer.calls == ['RFT글', 'B3글']


def test_gpu0_check_does_not_require_gpu1_idle(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return type('Result', (), {'stdout': ''})()
    monkeypatch.setattr(inc.subprocess, 'run', run)
    inc.ensure_gpu0_only()
    assert commands == [['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader,nounits']]
    def output(value):
        monkeypatch.setattr(inc.subprocess, 'run', lambda *a, **k: type('Result', (), {'stdout': value})())
    output(str(inc.os.getpid()) + '\n')
    inc.ensure_gpu0_only()  # The previous batch's own CUDA context is allowed.
    output(str(inc.os.getpid()) + '\n' + str(inc.os.getpid()+1) + '\n')
    with pytest.raises(RuntimeError, match='occupied'):
        inc.ensure_gpu0_only()
    for malformed in ('N/A\n', '12, 10MiB\n', '-1\n', '0\n', '\u0661\u0662\n'):
        output(malformed)
        with pytest.raises(RuntimeError, match='malformed'):
            inc.ensure_gpu0_only()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    with pytest.raises(RuntimeError, match='Only physical GPU0'):
        inc.ensure_gpu0_only()


def test_interruption_preserves_fresh_calibration_and_resumes_without_replay(tmp_path, monkeypatch):
    root, config, _, _ = files(tmp_path)
    fake_gpu(monkeypatch)
    def stop():
        if FakeScorer.calls:
            raise InterruptedError('stop after calibration')
    with pytest.raises(InterruptedError):
        inc.score_batch(config, root, stop)
    assert FakeScorer.calls == ['RFT글']
    inc.score_batch(config, root)
    assert FakeScorer.calls == ['RFT글', 'B3글']


def test_provenance_rejects_cpu_uncalibrated_and_wrong_device_values():
    value = {**response('key'), 'calibration_sha256': 'cal'}
    contract = {'reference_fingerprint': 'reference'}
    assert inc.validate_incremental_response(value, 'key', contract, 'cal', 'gpu0-config')
    for change in ({'execution_device': 'cpu'}, {'calibration_sha256': 'other'},
                   {'physical_gpu': 1}, {'execution_fingerprint': 'hidden-remap'}):
        with pytest.raises(ValueError):
            inc.validate_incremental_response({**value, **change}, 'key', contract, 'cal', 'gpu0-config')
