from copy import deepcopy
import json

import pytest

from verak.v4.search_v43 import SearchSession, verify_citation_proof
from verak.v4.support_judge_v43 import (
    extend_schema, payload_for_attempt, strip_support, support_selection,
    summarize_support, validate_support,
)


PASSAGE = {
    'title': '태양 에너지', 'section': '활용',
    'text': '태양열은 물을 데우거나 건물을 난방하는 데 쓰인다.',
    'passage_id': 'kowiki:20261001:123:456:0123456789abcdef0123',
    'url': 'https://ko.wikipedia.org/wiki/%ED%83%9C%EC%96%91_%EC%97%90%EB%84%88%EC%A7%80',
    'license': 'CC BY-SA 4.0', 'license_url': 'https://creativecommons.org/licenses/by-sa/4.0/',
}
SOURCE = {k: PASSAGE[k] for k in ('title', 'passage_id')}


def session():
    result = SearchSession(lambda query: [deepcopy(PASSAGE)])
    result.activate([{'item_id': 'D1T1', 'needs_search': 'yes', 'owner': 'revision'}], 1)
    result.search({'action': 'SEARCH', 'item_id': 'D1T1', 'query': '태양열 활용'})
    return result


def final_attempt():
    search = session()
    search.commit_insert('S2a', SOURCE, '태양열을 이용해 난방하거나 물을 데울 수 있다.')
    current = '태양열을 이용해 건물을 난방하거나 물을 데울 수 있다.'
    return {'attempt': 1, **search.export({'S2a': current}),
            'final_paragraphs': [{'id': 'P1', 'sentences': [{'id': 'S2a', 'text': current}]}]}


def verdict(*, supported='yes', paraphrased='yes'):
    return {'attempt': 1, 'source_support_applicable': 'yes', 'source_support': [
        {'sentence_id': 'S2a', 'supported_by_cited_passage': supported,
         'paraphrased': paraphrased, 'reason': '검색 문단의 활용 사례와 일치한다.'}]}


def test_search_permission_and_public_aliases_never_call_backend_when_denied():
    calls = []
    search = SearchSession(lambda query: calls.append(query) or [])
    search.activate([{'item_id': 'D1T1', 'needs_search': 'no'}], 1)
    action = {'action': 'SEARCH', 'item_id': 'D1T1', 'query': '태양열'}
    with pytest.raises(ValueError, match='needs_search'):
        search.search(action)
    search.activate([{'item_id': 'D1T1', 'needs_search': 'yes'}], 1)
    with pytest.raises(ValueError, match='KOREAN'):
        search.search(action, role='korean')
    with pytest.raises(ValueError, match='public task'):
        search.activate([{'item_id': 'train:123:D3I1', 'needs_search': 'yes'}], 1)
    assert calls == []
    assert search.search(action)['status'] == 'no_results'
    assert calls == ['태양열']
    assert search.successful_sourced_inserts == 0
    with pytest.raises(ValueError, match='not retrieved'):
        search.validate_source(SOURCE)


def test_citation_requires_current_exact_retrieval_and_success_count_survives_undo():
    search = session()
    # Neither altered titles nor additional URLs/fields are accepted.
    for forged in ({**SOURCE, 'title': '다른 기사'}, {**SOURCE, 'passage_id': 'forged'},
                   {**SOURCE, 'url': 'https://example.com'}):
        with pytest.raises(ValueError):
            search.validate_source(forged)
    assert search.validate_source(SOURCE) == PASSAGE
    public = search.observation({})
    public['search_history'][0]['passages'][0]['text'] = 'tampered'
    assert search.validate_source(SOURCE)['text'] == PASSAGE['text']
    search.activate([{'item_id': 'D2T1', 'needs_search': 'yes'}], 2)
    with pytest.raises(ValueError, match='currently delegated'):
        search.validate_source(SOURCE)
    search.search({'action': 'SEARCH', 'item_id': 'D2T1', 'query': '태양열'})
    before = search.snapshot()
    search.commit_insert('S2a', SOURCE, '태양열로 건물을 난방할 수 있다.')
    search.restore(before)
    assert search.citations == {}
    assert len(search.history) == 2
    assert search.successful_sourced_inserts == 1
    with pytest.raises(ValueError, match='UNDO does not refund'):
        search.validate_source(SOURCE)
    assert search.validate_source(None) is None


def test_sentence_sources_survive_korean_edit_split_delete_undo_and_tamper_is_detected():
    search = session()
    search.commit_insert('S2a', SOURCE, '태양열은 물을 대우는 데 쓰인다.')
    before = search.snapshot()
    search.inherit('S2a', ['S2b'])
    current = {'S2a': '태양열은 물을 데우는 데 쓰인다.', 'S2b': '또 건물을 난방한다.'}
    attempt = {'attempt': 1, **search.export(current), 'final_paragraphs': [
        {'id': 'P1', 'sentences': [{'id': k, 'text': v} for k, v in current.items()]}]}
    assert len(verify_citation_proof(attempt)) == 2
    assert attempt['source_citations'][0]['inserted_text'] != attempt['source_citations'][0]['text']
    assert attempt['source_citations'][1]['derived_from_sid'] == 'S2a'
    search.prune([])
    assert search.export({})['source_citations'] == []
    search.restore(before)
    assert list(search.citations) == ['S2a']
    for field in ('text', 'passage'):
        broken = deepcopy(attempt)
        if field == 'text':
            broken['source_citations'][0]['text'] = '한국어 수정 전에 있던 오래된 문장'
        else:
            broken['source_citations'][0]['passage']['text'] = '판정을 조작하기 위한 새 사실'
        with pytest.raises(ValueError):
            verify_citation_proof(broken)


def test_support_gate_none_is_separate_and_unsupported_copy_missing_fail_closed():
    attempt = final_attempt()
    assert payload_for_attempt(attempt)['source_citations'][0]['passage'] == PASSAGE
    good = verdict()
    assert support_selection(good, attempt)['support_keep']
    assert not support_selection(verdict(supported='no'), attempt)['support_keep']
    assert not support_selection(verdict(paraphrased='no'), attempt)['support_keep']
    assert not support_selection({'attempt': 1}, attempt)['support_keep']
    none = {'attempt': 2, 'source_citations': [], 'search_history': [], 'final_paragraphs': []}
    no_verdict = {'attempt': 2, 'source_support_applicable': 'none', 'source_support': []}
    assert support_selection(no_verdict, none)['support_keep']
    complete = {'attempts': [good, no_verdict], 'preferred': '1', 'reason': '테스트'}
    assert validate_support(complete, [attempt, none]) == complete
    base = strip_support(complete)
    assert set(base['attempts'][0]) == {'attempt'}
    assert 'source_support' in complete['attempts'][0]
    bad = deepcopy(complete)
    bad['attempts'][0]['source_support'] *= 2
    with pytest.raises(ValueError, match='duplicate'):
        validate_support(bad, [attempt, none])
    metrics = summarize_support([
        {'attempt': attempt, 'verdict': good, 'genre': 'argumentative'},
        {'attempt': none, 'verdict': no_verdict, 'genre': 'emotional'}])
    assert metrics['applicable_attempts'] == metrics['none_attempts'] == 1
    assert metrics['applicable_support_keep_rate'] == 1
    assert metrics['by_genre']['emotional']['applicable_support_keep_rate'] is None


def test_support_schema_does_not_modify_original_and_requires_both_provenance_gates():
    schema = {'properties': {'attempts': {'items': {
        'type': 'object', 'properties': {'attempt': {'type': 'integer'}},
        'required': ['attempt'], 'additionalProperties': False}}}}
    original = json.dumps(schema, sort_keys=True)
    extended = extend_schema(schema)
    assert json.dumps(schema, sort_keys=True) == original
    item = extended['properties']['attempts']['items']
    assert 'source_support' in item['required']
    assert 'source_support_applicable' in item['required']
    assert set(item['properties']['source_support']['items']['properties']) == {
        'sentence_id', 'supported_by_cited_passage', 'paraphrased', 'reason'}
    with pytest.raises(ValueError, match='already extended'):
        extend_schema(extended)
