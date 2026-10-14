from pathlib import Path

import pytest

from verak.v3.common import file_sha, pair_key, read_json, write_json
from verak.v3.corrupt.document import Document, Paragraph, Unit
from verak.v4.scale2_projection import save_document
from verak.v4.scale3_reference import run, validate_manifest, validate_response


def ready_files(tmp_path):
    root, repo = tmp_path / 'scale3', tmp_path / 'repo'
    key = pair_key('문항', '글')
    sample, contract, raw, prepared = [root / (name + '.json') for name in ('sample', 'contract', 'raw', 'prepared')]
    for path in (sample, contract, raw, prepared):
        write_json(path, {'artifact': path.name})
    write_json(raw, {'states': {'initial': {'document': save_document(
        Document([Paragraph('P1', [Unit('S1', '글', [])])], ['']))}}})
    write_json(prepared, {'corpus': {'question': '문항'}})
    manifest = {'component': 'B3', 'version': 'v4.3', 'teacher_collection_finished': True,
        'no_live_paid_calls': True, 'reference_gpu_fingerprint': 'reference', 'api': {'pending': 0},
        'sample_path': str(sample), 'sample_sha256': file_sha(sample),
        'contract_path': str(contract), 'contract_sha256': file_sha(contract),
        'episodes': [{'raw_path': str(raw), 'raw_sha256': file_sha(raw),
            'prepared_path': str(prepared), 'prepared_sha256': file_sha(prepared), 'quality_keys': {'initial': key}}],
        'requests': [{'key': key, 'question': '문항', 'text': '글'}]}
    path = root / 'B3/gpu_manifest.json'
    write_json(path, manifest)
    write_json(root / 'B3/gpu_ready.json', {'manifest_path': str(path), 'manifest_sha256': file_sha(path),
        'teacher_collection_finished': True, 'no_live_paid_calls': True})
    return root, repo, key


def finish_oneshot(repo):
    report = repo / 'report.md'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('One-shot completed.')
    write_json(repo / 'verak/v3/outputs/oneshot_baseline/complete.json', {'completed': True,
        'report': {'completed': True, 'report': str(report), 'report_sha256': file_sha(report)}})


def response(key):
    return {'execution_device': 'gpu_reference', 'fingerprint': 'reference', 'gpu_id': 1,
            'result': {'cache_key': key, 'mean': 2., 'integers': [2] * 8, 'expected': [2.] * 8}}


def test_gpu_manifest_waits_for_completed_oneshot_and_detects_changed_evidence(tmp_path):
    root, repo, _ = ready_files(tmp_path)
    with pytest.raises(FileNotFoundError):
        validate_manifest(root, repo)
    finish_oneshot(repo)
    _, identity = validate_manifest(root, repo)
    assert identity['slot'] == 'post_oneshot_evaluation'
    write_json(root / 'raw.json', {'changed': True})
    with pytest.raises(ValueError, match='episode changed'):
        validate_manifest(root, repo)


def test_cpu_wrong_input_and_missing_rubric_scores_are_never_accepted():
    for change in ({'execution_device': 'cpu'}, {'fingerprint': 'other'}):
        with pytest.raises(ValueError):
            validate_response({**response('key'), **change}, 'key', 'reference')
    with pytest.raises(ValueError):
        validate_response(response('other-input'), 'key', 'reference')
    value = response('key')
    value['result']['expected'] = [2.] * 7
    with pytest.raises(ValueError):
        validate_response(value, 'key', 'reference')
    for expectations, mean in (([2.] * 8, 8.), ([10.] * 8, 10.)):
        value = response('key')
        value['result'].update(expected=expectations, mean=mean)
        with pytest.raises(ValueError):
            validate_response(value, 'key', 'reference')


def test_swapped_quality_endpoint_is_rejected_even_with_valid_manifest_hashes(tmp_path):
    root, repo, key = ready_files(tmp_path)
    finish_oneshot(repo)
    raw = read_json(root / 'raw.json')
    raw['states']['stage1'] = {'document': save_document(
        Document([Paragraph('P1', [Unit('S1', '수정된 글', [])])], ['']))}
    write_json(root / 'raw.json', raw)
    path = root / 'B3/gpu_manifest.json'
    manifest = read_json(path)
    second = pair_key('문항', '수정된 글')
    manifest['requests'].append({'key': second, 'question': '문항', 'text': '수정된 글'})
    manifest['episodes'][0].update(raw_sha256=file_sha(root / 'raw.json'),
        quality_keys={'initial': second, 'stage1': key})
    write_json(path, manifest)
    ready = read_json(root / 'B3/gpu_ready.json')
    write_json(root / 'B3/gpu_ready.json', {**ready, 'manifest_sha256': file_sha(path)})
    with pytest.raises(ValueError, match='saved question and endpoint'):
        validate_manifest(root, repo)


def test_reference_reuse_and_resume_need_no_model_load(tmp_path, monkeypatch):
    root, repo, key = ready_files(tmp_path)
    finish_oneshot(repo)
    prior = repo / 'verak/v3/outputs/data_boost/gpu_rescore/responses' / (key + '.json')
    write_json(prior, response(key))
    from verak.v3.rft1 import service
    from verak.v3.score import kanana
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    monkeypatch.setattr(service, 'idle_gpus', lambda: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('Cached GPU reference must not reload a model')
    monkeypatch.setattr(kanana, 'KananaScorer', forbidden)
    config = {'paths': {'repo': repo}, 'scorer': {'gpu': 1}}
    first = run(config, root)
    assert first['new'] == 0 and first['reused'] == 1 and first['errors'] == []
    assert first['CPU_scoring_performed'] is False and first['training'] is False
    assert run(config, root) == first
