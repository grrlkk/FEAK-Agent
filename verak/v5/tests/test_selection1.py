import json

import pytest

from verak.v5.selection1 import (candidates, decision_metrics, payloads, reference_consensus,
                                sha_text, snapshot_text, validate_judgment, validate_selection,
                                order_comparison, repair_dangling_quote)


def snapshot(text):
    return {'layout': {'gaps': ['\n'], 'tail': '\n', 'paragraphs': [
        {'pid': 'P1', 'units': [{'sid': 'S1', 'text': text, 'leading': ' '}]}]},
        'text_sha256': sha_text('\n '+text+'\n')}


def case():
    episode = {'initial_token_document': snapshot('원문이다.'),
               'final_token_document': snapshot('최종본이다.'),
               'initial': {'text': '\n 원문이다.\n'}, 'final': {'text': '\n 최종본이다.\n'},
               'stages': [{'role': 'revision', 'final_token_document': snapshot('중간본이다.')},
                          {'role': 'korean', 'final_token_document': snapshot('최종본이다.')} ]}
    result = {'case_id': 'test:1', 'question': '문항', 'original': episode['initial']['text'],
              'original_key': sha_text(episode['initial']['text']),
              'final_key': sha_text(episode['final']['text']),
              'candidates': candidates(episode), 'retrieved_passages': []}
    result['orders'] = payloads(result)
    return result


def judgment(c, order):
    ratings = []
    best = next(x['key'] for x in c['candidates'] if 'stage_1_revision' in x['origins'])
    for candidate in order['payload']['candidates']:
        key = order['aliases'][candidate['id']]
        ratings.append({'id': candidate['id'], 'overall_vs_original':
            'equal' if key == c['original_key'] else 'better' if key == best else 'worse',
            'meaning_preserved': 'yes', 'introduced_major_defect': 'no',
            'new_specific_claims': 'none', 'reason': '완성된 글의 비교 근거'})
    ids = [next(a for a, k in order['aliases'].items() if k == key)
           for key in (best, c['original_key'], c['final_key'])]
    return {'variants': ratings, 'ranking': [[x] for x in ids], 'ranking_reason': '문장 연결 개선'}


def test_snapshot_is_lossless_and_checks_hash():
    value = snapshot('문장이다.')
    assert snapshot_text(value) == '\n 문장이다.\n'
    value['layout']['paragraphs'][0]['units'][0]['text'] = '다르다.'
    with pytest.raises(ValueError, match='round-trip'): snapshot_text(value)


def test_final_checkpoint_deduplicates_without_losing_provenance():
    c = case()
    assert len(c['candidates']) == 3
    final = next(x for x in c['candidates'] if x['key'] == c['final_key'])
    assert final['origins'] == ['stage_2_korean', 'final']


def test_reverse_order_relabels_and_hides_provenance():
    c = case(); a, b = c['orders']
    assert list(a['aliases'].values()) == list(reversed(list(b['aliases'].values())))
    assert all(set(x) == {'id', 'text'} for x in a['payload']['candidates'])
    assert set(a['payload']) == {'question', 'original', 'candidates', 'retrieved_passages'}


def test_wrong_action_protocol_is_not_a_quality_decision():
    c = case()
    with pytest.raises(ValueError, match='selected'):
        validate_selection('{"action":"SCORE"}', c['orders'][0])


def test_judge_cannot_omit_or_duplicate_candidates():
    c = case(); o = c['orders'][0]; j = judgment(c, o)
    j['ranking'][1] = j['ranking'][0]
    with pytest.raises(ValueError, match='ranked'):
        validate_judgment(json.dumps(j), o)


def test_exact_original_is_not_counted_as_an_improvement():
    c = case(); o = c['orders'][0]; j = judgment(c, o)
    label = next(a for a, k in o['aliases'].items() if k == c['original_key'])
    next(v for v in j['variants'] if v['id'] == label)['overall_vs_original'] = 'better'
    with pytest.raises(ValueError, match='Original-as-candidate'):
        validate_judgment(json.dumps(j), o)


def test_consensus_uses_text_identity_across_relabelled_orders():
    c = case(); js = [judgment(c, o) for o in c['orders']]
    for o, j in zip(c['orders'], js): validate_judgment(json.dumps(j), o)
    consensus = reference_consensus(c, js)
    assert len(consensus['useful']) == 1
    assert consensus['harmful'] == [c['final_key']]
    assert consensus['best_sets_agree'] and consensus['final_excluded_both']
    assert not decision_metrics(c['original_key'], consensus)['useful']
    assert decision_metrics(c['final_key'], consensus)['harmful']


def test_disagreeing_reference_is_unresolved_not_a_selector_error():
    c = case(); js = [judgment(c, o) for o in c['orders']]
    js[1]['ranking'].reverse()
    consensus = reference_consensus(c, js)
    assert consensus['best'] == []
    assert decision_metrics(c['original_key'], consensus)['reference_best'] is None


def test_syntax_recovery_leaves_string_content_untouched():
    raw = '{"reason":"설명이다.","},{"id":"B"}'
    fixed, edits = repair_dangling_quote(raw)
    assert fixed == '{"reason":"설명이다."},{"id":"B"}'
    assert len(edits) == 1
    valid = json.dumps({'reason': '원문에 있는 ,"} 문자열'}, ensure_ascii=False)
    assert repair_dangling_quote(valid) == (valid, [])


def test_tied_candidates_are_not_counted_as_material_order_sensitivity():
    r = {'rankings': [{'a': 0, 'b': 0, 'c': 1}, {'a': 0, 'b': 0, 'c': 1}]}
    assert order_comparison('a', 'b', r) == 'reference_tie'
    assert order_comparison('a', 'c', r) == 'first_order_preferred'


def test_local_runner_checks_binary_adapter_without_decoding_it(tmp_path, monkeypatch):
    from verak.v5 import selection1_run as runner
    from verak.v5.common import ref, write_json
    c = case(); adapter = tmp_path/'adapter.safetensors'; adapter.write_bytes(b'\xff\x00binary')
    manifest = {'cases': [c], 'pilot_ids': [c['case_id']], 'adapter': ref(adapter)}
    write_json(tmp_path/'manifest.json', manifest)
    write_json(tmp_path/'preflight.json', {'manifest': ref(tmp_path/'manifest.json'),
        'token_counts': [{'case_id': c['case_id'], 'order': i, 'fits': True} for i in (0, 1)]})
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'prepare', lambda: manifest)
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {'model': 'fake', 'choices': [{'message': {'content': '{"selected":"A","reason":"문장 연결 개선"}'}, 'finish_reason': 'stop'}]}
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, *args, **kwargs): return Response()
    monkeypatch.setattr('httpx.Client', Client)
    runner.local('pilot')
    saved = [json.loads(p.read_text()) for p in (tmp_path/'local').glob('*/*.json')]
    assert len(saved) == 4 and all(x['status'] == 'valid' for x in saved)
