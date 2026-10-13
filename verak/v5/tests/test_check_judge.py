from copy import deepcopy
import json

import pytest

from verak.v5 import api, check_judge as check
from verak.v5.common import write_json


def test_local_context_has_only_predecessor_target_and_fixed_c_prompt():
    original = [{'id': 'P1', 'sentences': [{'id': f'S{i}', 'text': f'{i}번째.'} for i in range(1, 5)]}]
    revised = deepcopy(original)
    revised[0]['sentences'][2]['text'] = '바뀐 문장이다.'
    notice = {'id': 'hidden_record_id', 'sid': 'S3', 'type': 'ec_category',
              'before': 'CAUSE', 'after': 'ADVERSATIVE', 'scope': ['S3', 'S2'],
              'label': 'known_broken_record', 'gold': 'never sent'}
    local = check.make_view(original, revised, '문항', notice, 'local')
    full = check.make_view(original, revised, '문항', notice, 'full')
    assert [s['id'] for p in local['original'] for s in p['sentences']] == ['S2', 'S3']
    assert [s['id'] for p in local['revised'] for s in p['sentences']] == ['S2', 'S3']
    assert local['notices'] == full['notices']
    assert local['notices'][0]['id'] == 'N1'
    assert local['variants'][0]['paragraphs'] == local['revised']
    assert len(full['original'][0]['sentences']) == 4
    assert not any(x in json.dumps(local) for x in ('label', 'gold', 'hidden_record_id'))
    assert len(original[0]['sentences']) == 4


def result_rows(local_yes, full_yes):
    return [{'context': context, 'operator': 'L_CONN', 'label': label, 'valid': True,
             'answer': 'yes' if i < count else 'uncertain' if label == 'known_broken_record' else 'no'}
            for context, count in [('local', local_yes), ('full', full_yes)]
            for label in ('known_broken_record', 'known_fine_unchanged')
            for i in range(5)]


def test_gate_uses_full_at_threshold_and_local_only_on_full_failure():
    r = check.gate_from_results(result_rows(5, 4), expected_per_context=10)
    assert r['continue'] and r['validated_setting'] == 'full'
    assert r['tables']['full']['ALL']['recall'] == .8
    r = check.gate_from_results(result_rows(4, 3), expected_per_context=10)
    assert r['continue'] and r['validated_setting'] == 'local' and r['C_underestimation_warning']
    r = check.gate_from_results(result_rows(3, 3), expected_per_context=10)
    assert not r['continue'] and r['stop_reason'] == 'both recalls below .8'


def test_missing_judgment_cannot_pass_even_when_all_observed_positive():
    rows = result_rows(5, 5)
    rows[0].update(valid=False, answer=None)
    rows[10].update(valid=False, answer=None)
    result = check.gate_from_results(rows, expected_per_context=10)
    assert not result['continue'] and not result['all_judgments_valid']
    assert result['tables']['local']['ALL']['recall'] is None
    assert result['tables']['local']['ALL']['positive_uncertain'] == 0


def test_invalid_control_is_explicit_and_does_not_change_recall_gate():
    rows = result_rows(5, 5)
    rows[-1].update(valid=False, answer=None)
    result = check.gate_from_results(rows, expected_per_context=10)
    assert result['continue'] and result['validated_setting'] == 'full'
    assert not result['all_judgments_valid']
    assert result['tables']['full']['ALL']['control_unjudged'] == 1
    assert result['tables']['full']['ALL']['false_positive_bounds'] == [.8, 1.]


def test_clean_first_pass_can_be_finalized_without_format_retry(tmp_path):
    rows = result_rows(5, 5)*12
    write_json(tmp_path/'validation/result_rows.json', rows)
    write_json(tmp_path/'validation/complete.json', {'wall_seconds': 1.})
    value = check.finalize_validation(root=tmp_path)
    assert value['continue'] and value['validated_setting'] == 'full'
    # Once finalized, later auxiliary files cannot rewrite the chosen gate.
    write_json(tmp_path/'validation/complete.json', {'changed': True})
    assert check.finalize_validation(root=tmp_path) == value


def test_new_budget_root_does_not_open_or_modify_pipe1_ledger(tmp_path, monkeypatch):
    prior, new = tmp_path/'pipe1', tmp_path/'check'
    prior.mkdir(); new.mkdir()
    (prior/'api_contract.json').write_text('do not read')
    monkeypatch.setattr(api, 'ROOT', prior)
    spec = {'model': 'test-model', 'cap_usd': 1., 'input_per_million': 1., 'output_per_million': 1.}
    write_json(new/'api_contract.json', {'user_approved': True, 'total_cap_usd': 2., 'claude': spec})
    client = api.PaidAPI('claude', root=new)
    try:
        assert client.directory == new/'api'
        assert not (prior/'api').exists()
    finally:
        client.close()


def test_source_first_sampling_deduplicates_same_sentence_across_variants():
    import random
    candidates = [{'source_id': 'a', 'unique_key': 'a:S1'},
                  {'source_id': 'a', 'unique_key': 'a:S1'},
                  {'source_id': 'a', 'unique_key': 'a:S2'},
                  {'source_id': 'b', 'unique_key': 'b:S1'}]
    got = check.diversified_sample(candidates, 2, random.Random(1))
    assert {r['source_id'] for r in got} == {'a', 'b'}
    with pytest.raises(ValueError, match='Insufficient'):
        check.diversified_sample(candidates, 4, random.Random(1))
