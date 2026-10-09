import pytest

from verak.v3.common import file_sha, pair_key, read_json, write_json


def test_200_audit_samples_distinct_sources_without_cpu_outcomes():
    from verak.v3.insertion_boost.calibrate_200 import sample_sources
    frame = [{'source_id': f'source:{i}', 'genre': ('설명', '정서', '논증')[i % 3],
              'near_threshold': i < 62} for i in range(741)]
    chosen = sample_sources(frame)
    assert chosen == sample_sources(list(reversed(frame)))
    assert len(chosen) == len({r['source_id'] for r in chosen}) == 200
    assert sum(r['near_threshold'] for r in chosen) == 62
    with pytest.raises(ValueError, match='distinct source'):
        sample_sources(frame[:199])


def test_audit_distinguishes_generated_digits_and_argmax_and_role_gate():
    from verak.v3.insertion_boost.calibrate_200 import compare_episode
    config = {'reward': {'w_q': .3, 'quality': {'noise_floor': 0., 'noise_floor_by_genre': {}}}}
    gpu = {'mean': 5., 'integers': [5]*8, 'score_line': '5 5 5 5 5 5 5 5'}
    episode = {'source_id': 's', 'episode_id': 'e', 'genre': '설명', 'has_global_records': True,
        'states': {k: {'request_key': k, 'gpu_score': gpu} for k in ('initial', 'middle', 'final')},
        'gpu_reward': {r: {'R': .799, 'R_q': 0} for r in ('global', 'korean')}}
    scores = {name: {**gpu, 'scorer_fingerprint': 'fp32'} for name in episode['states']}
    scores['middle'] = {**scores['middle'], 'mean': 5.02, 'integers': [4]+[5]*7}
    result = compare_episode(config, episode, scores)
    middle = result['comparisons'][1]
    assert all(middle['generated_digit_agreement'])
    assert middle['digit_agreement'] == [False]+[True]*7
    assert result['decisions'][0]['threshold_changed']
    assert not result['decisions'][1]['threshold_changed']


def test_gpu_reference_reader_refuses_cpu_or_changed_recipe(tmp_path, monkeypatch):
    from verak.v3.insertion_boost import cpu_score
    config = {'paths': {'repo': tmp_path}}
    monkeypatch.setattr(cpu_score, 'gpu_reference_fingerprint', lambda cfg: 'reference')
    path = tmp_path / 'verak/v3/outputs/data_boost/gpu_rescore/responses' / (pair_key('q', 'text')+'.json')
    row = {'result': {'mean': 5, 'cache_key': pair_key('q', 'text')},
           'fingerprint': 'reference', 'execution_device': 'cpu'}
    write_json(path, row)
    with pytest.raises(ValueError, match='GPU reference'):
        cpu_score.gpu_reference_score(config, 'q', 'text')
    row['execution_device'] = 'gpu_reference'
    row['fingerprint'] = 'wrong'
    write_json(path, row)
    with pytest.raises(ValueError, match='identity changed'):
        cpu_score.gpu_reference_score(config, 'q', 'text')
    row['fingerprint'] = 'reference'
    write_json(path, row)
    assert cpu_score.gpu_reference_score(config, 'q', 'text')['execution_device'] == 'gpu_reference'


def test_gpu_finalizer_redoes_selection_and_never_falls_back(tmp_path, monkeypatch):
    from verak.v3.insertion_boost import gpu_handoff as module
    root = tmp_path / 'insertion'
    config = {'paths': {'v2_ops_output': root, 'repo': tmp_path}}
    shared = tmp_path / 'shared'
    monkeypatch.setattr(module, 'shared_root', lambda cfg: shared)
    monkeypatch.setattr(module, 'gpu_reference_fingerprint', lambda cfg: 'gpu-frozen')
    corpus = {f'e{i}': {'episode_id': f'e{i}', 'records': [{'level': 'GLOBAL'}]} for i in (1, 2, 3)}
    monkeypatch.setattr(module, 'prepare', lambda cfg: ({}, corpus))
    monkeypatch.setattr(module, 'Resources', lambda cfg: object())
    rows = []
    for i in (1, 2, 3):
        path = root / f'raw_{i}.json'
        write_json(path, {'completed': True})
        rows.append({'raw_path': str(path), 'raw_sha256': file_sha(path), 'episode_id': f'e{i}',
                     'source_id': f's{i}', 'attempt': 1, 'provisional_R': .85})
    manifest_path = root / 'gpu_rescore_manifest.json'
    write_json(manifest_path, {'episodes': rows})
    write_json(shared / 'gpu_rescore/insertion_complete.json', {'status': 'complete',
        'manifest_sha256': file_sha(manifest_path), 'fingerprint': 'gpu-frozen'})
    def reward(cfg, candidate, raw, resources, score, *, identity_allowed):
        assert not identity_allowed
        if candidate['episode_id'] == 'e3':
            raise ValueError('GPU score missing')
        result = .7 if candidate['episode_id'] == 'e1' else .9
        return {'R': result}, [{'execution_device': 'gpu_reference', 'scorer_fingerprint': 'gpu-frozen'}]*2, None
    monkeypatch.setattr(module, 'global_reward', reward)
    result = module.finalize(config)
    assert result['selected_counts'] == {'global': 1, 'korean': 0}
    assert result['eligibility_flips'] == 1
    assert result['cpu_only_eligible'] == 1
    assert result['unmeasured'][0]['selection'] == 'excluded_unmeasured_no_CPU_fallback'
    saved = read_json(root / 'gpu_selection.json')
    assert list(saved['selected']['global']) == ['e2']
    assert saved['score_source'] == 'gpu_reference'
    digest = file_sha(root / 'gpu_selection.json')
    monkeypatch.setattr(module, 'global_reward', lambda *a, **k: pytest.fail('Completed GPU selection was re-run'))
    assert module.finalize(config) == result
    assert file_sha(root / 'gpu_selection.json') == digest


def test_approval_rejects_old_small_or_mixed_calibration(tmp_path, monkeypatch):
    from verak.v3.insertion_boost import gpu_handoff as module
    monkeypatch.setattr(module, 'shared_root', lambda cfg: tmp_path)
    monkeypatch.setattr(module, 'gpu_reference_fingerprint', lambda cfg: 'gpu')
    root = tmp_path / 'cpu_scorer'
    write_json(root / 'provisional_contract.json', {'fingerprint': 'fp32'})
    write_json(tmp_path / 'gpu_rescore/complete.json', {'status': 'complete'})
    for name in ('insertion', 'global'):
        write_json(tmp_path / f'gpu_rescore/{name}_complete.json', {'fingerprint': 'gpu'})
    value = {'status': 'complete', 'source_essays': 5, 'unique_source_essays': 5, 'fingerprints': ['fp32']}
    write_json(root / 'audit_200/calibration.json', value)
    assert not module.publish_approval({})['published']
    value.update(source_essays=200, unique_source_essays=200, fingerprints=['fp32', 'bf16'])
    write_json(root / 'audit_200/calibration.json', value)
    assert not module.publish_approval({})['published']
    value['fingerprints'] = ['fp32']
    write_json(root / 'audit_200/calibration.json', value)
    assert module.publish_approval({})['published']
    approval = read_json(root / 'selection_approval.json')
    assert approval['score_source'] == 'gpu_reference'
    assert approval['calibration_essays'] == 200
    assert approval['calibration_sha256'] == file_sha(root / 'audit_200/calibration.json')
