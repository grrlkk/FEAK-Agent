import json

import pytest

from verak.v5.fast1_link_audit import collect_transitions, evaluate, _validate


def episode(case, run, rev_initial, rev_final, korean=None):
    stages = [{'role': 'revision', 'number': 1, 'item': {'id': 'I1'},
               'initial_text': rev_initial, 'final_text': rev_final,
               'ending': 'four_actions'}]
    if korean is not None:
        stages.append({'role': 'korean', 'number': 2, 'reason': 'after_revision_notices',
                       'initial_text': rev_final, 'final_text': korean, 'ending': 'STOP'})
    return {'case_id': case, 'run': run, 'question': '문항', 'stages': stages}


def test_collect_keeps_every_revision_and_deduplicates_normalized_triples():
    episodes = [episode('a', 1, '처음\n글', '바뀐 글', '고친 글'),
                episode('a', 2, '처음 글', '바뀐   글', '고친 글'),
                episode('b', 1, '같음', '같음')]
    data = collect_transitions(episodes)
    assert data['denominator'] == 3
    assert data['unique_count'] == 2
    assert len(data['unique_triples'][0]['aliases']) == 2
    assert data['rows'][0]['status'] == 'judge_transition'
    assert data['rows'][2]['status'] == 'derived_revision_unchanged'
    assert data['rows'][2]['revision_created_broken_link'] == 'no'
    assert data['rows'][2]['korean_effect'] == 'not_run'


def test_revision_without_immediate_korean_is_retained_and_marked_not_run():
    row = collect_transitions([episode('x', 1, '원문', '수정')])['rows'][0]
    assert row['status'] == 'judge_revision_only'
    assert row['immediate_korean_stage'] is None
    assert row['korean_effect'] == 'not_run'


def test_label_validation_requires_all_ids_once_and_closed_label_sets():
    raw = json.dumps({'judgments': [
        {'id': 'T1', 'revision_created_broken_link': 'yes',
         'korean_effect': 'fixed_link', 'reason': '근거'}]})
    assert _validate(raw, ['T1'])[0]['korean_effect'] == 'fixed_link'
    with pytest.raises(ValueError):
        _validate(raw, ['T1', 'T2'])
    bad = json.dumps({'judgments': [
        {'id': 'T1', 'revision_created_broken_link': 'yes',
         'korean_effect': 'fixed', 'reason': ''}]})
    with pytest.raises(ValueError):
        _validate(bad, ['T1'])
    empty_reason = json.dumps({'judgments': [
        {'id': 'T1', 'revision_created_broken_link': 'no',
         'korean_effect': 'no_broken_link', 'reason': '  '}]})
    with pytest.raises(ValueError):
        _validate(empty_reason, ['T1'])


def test_evaluate_uses_injected_api_and_reports_stage_denominator(tmp_path):
    class API:
        def request(self, messages, *, item_id, max_output):
            payload = json.loads(messages[1]['content'])
            labels = [{'id': row['id'], 'revision_created_broken_link': 'yes',
                       'korean_effect': ('not_run' if row['immediate_korean_final'] is None else 'fixed_link'),
                       'reason': 'sample'} for row in payload['transitions']]
            return {'raw': json.dumps({'judgments': labels})}

    result = evaluate([episode('a', 1, '원문', '수정', '한국어 최종'),
                       episode('b', 1, '문장', '문장')], api=API(), output_dir=tmp_path)
    assert result['summary']['revision_stage_denominator'] == 2
    assert result['summary']['revision_created_broken_link'] == {'yes': 1, 'no': 1}
    assert result['summary']['korean_effect']['fixed_link'] == 1
    assert result['summary']['korean_effect']['not_run'] == 1


def test_missing_final_text_is_explicit_unknown_and_never_judged(tmp_path):
    class NoCalls:
        def request(self, *args, **kwargs):
            raise AssertionError('missing text must not be sent for judgment')

    broken = episode('x', 1, '원문', '수정', '한국어')
    broken['stages'][1].pop('final_text')
    result = evaluate([broken], api=NoCalls(), output_dir=tmp_path)
    assert result['summary']['revision_stage_denominator'] == 1
    assert result['rows'][0]['status'] == 'missing_stage_text'
    assert result['rows'][0]['revision_created_broken_link'] == 'not_judged'
    assert result['rows'][0]['korean_effect'] == 'not_judged'


def test_stale_batch_cache_is_rejected_before_api_call(tmp_path):
    class NoCalls:
        def request(self, *args, **kwargs):
            raise AssertionError('stale cache must fail before any request')

    batch = tmp_path / '000.json'
    batch.write_text(json.dumps({'status': 'valid', 'labels': [],
                                 'request_fingerprint': 'old-input'}), encoding='utf-8')
    with pytest.raises(ValueError, match='Stale link-audit cache'):
        evaluate([episode('x', 1, '원문', '수정', '최종')], api=NoCalls(), output_dir=tmp_path)


def test_unknown_delivery_is_saved_and_not_retried(tmp_path):
    class Unknown:
        def __init__(self):
            self.calls = 0
        def request(self, *args, **kwargs):
            self.calls += 1
            raise TimeoutError('response delivery unknown')

    api = Unknown()
    result = evaluate([episode('x', 1, '원문', '수정', '최종')], api=api, output_dir=tmp_path)
    saved = json.loads((tmp_path / '000.json').read_text(encoding='utf-8'))
    assert api.calls == 1
    assert saved['status'] == 'request_error_no_retry'
    assert saved['delivery_unknown'] is True
    assert result['rows'][0]['status'] == 'evaluation_unavailable'
