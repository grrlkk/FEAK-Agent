import bz2
import json
import sqlite3

from verak.v4.wiki_local import (KiwiTokenizer, LocalWikiSearch, insertion_citation_contract,
                                 iter_pages, page_filter, passage_id)


def page(**changes):
    return {"namespace": 0, "title": "지구", "wikitext": "지구는 행성이다.", **changes}


def test_drop_page_categories_and_precedence():
    assert page_filter(page(redirect=True)) == "redirect"
    assert page_filter(page(wikitext="#넘겨주기 [[지구]]")) == "redirect"
    assert page_filter(page(wikitext="{{동음이의어}} 여러 뜻")) == "disambiguation"
    assert page_filter(page(title="행성 목록")) == "list_page"
    assert page_filter(page(namespace=14)) == "non_article_namespace"
    assert page_filter(page(wikitext="지구는 행성이다. [[분류:행성]]")) is None


def test_stream_xml_keeps_revision_and_skips_redirect(tmp_path):
    xml = b'<mediawiki xmlns="urn:wiki"><page><title>Earth</title><ns>0</ns><id>4</id><redirect title="World"/><revision><id>9</id><timestamp>2026-10-01</timestamp><text>Text</text></revision></page></mediawiki>'
    path = tmp_path / "articles.xml.bz2"
    path.write_bytes(bz2.compress(xml))
    values = list(iter_pages(path))
    assert len(values) == 1
    assert values[0]["page_id"] == "4" and values[0]["revision_id"] == "9"
    assert page_filter(values[0]) == "redirect"


def test_passage_identity_is_stable_and_content_sensitive():
    key = ("20261001", "4", "9", "역사", 2, "지구의 역사다.")
    assert passage_id(*key) == passage_id(*key)
    assert passage_id(*key) != passage_id(*key[:-1], "다른 문장이다.")


def test_dump_discovery_accepts_completed_recombined_aggregate():
    from verak.v4.wiki_build import completed_articles_entry
    filename = 'kowiki-20261001-pages-articles.xml.bz2'
    entry = {'url': '/kowiki/20261001/' + filename, 'size': 42, 'sha1': 'abc'}
    metadata = {'jobs': {'articlesdump': {'status': 'done', 'files': {}},
                         'articlesdumprecombine': {'status': 'done', 'files': {filename: entry}}}}
    assert completed_articles_entry(metadata, '20261001') == ('articlesdumprecombine', entry)
    metadata['jobs']['articlesdumprecombine']['status'] = 'in-progress'
    assert completed_articles_entry(metadata, '20261001') is None


def test_content_token_filter():
    from types import SimpleNamespace as Token
    tokens = [Token(form="지구", tag="NNP"), Token(form="는", tag="JX"),
              Token(form="행성", tag="NNG"), Token(form="이", tag="VCP"),
              Token(form="다", tag="EF"), Token(form="Earth", tag="SL")]
    assert KiwiTokenizer.terms(tokens) == ["지구", "행성", "earth"]


def test_local_search_top_three_and_metadata_without_network(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network forbidden")))
    class FakeIndex:
        vocab_dict = {"지구": 0}
        scores = {"num_docs": 4}
        def retrieve(self, query, **kwargs):
            assert query == [["지구"]] and kwargs["k"] == 3
            return [[1, 0, 2]], [[3., 2., 1.]]
    engine = LocalWikiSearch.__new__(LocalWikiSearch)
    engine.index = FakeIndex()
    engine.tokenizer = lambda query: query.split()
    engine.db = sqlite3.connect(":memory:")
    engine.db.execute("CREATE TABLE passages(doc_id INTEGER PRIMARY KEY,payload TEXT)")
    for i in range(4):
        payload = {"passage_id": f"p{i}", "title": "지구", "section": "서론", "text": "문장",
                   "url": "https://ko.wikipedia.org/wiki/Earth", "license": "CC BY-SA 4.0",
                   "license_url": "https://creativecommons.org/licenses/by-sa/4.0/"}
        engine.db.execute("INSERT INTO passages VALUES (?,?)", (i, json.dumps(payload)))
    results = engine.search("지구")
    assert [x["passage_id"] for x in results] == ["p1", "p0", "p2"]
    assert insertion_citation_contract(results[0])["passage_id"] == "p1"
    assert engine.search("unknown") == []
    engine.close()


def test_eval_excerpt_never_contains_all_source_sentences():
    from verak.v4.wiki_eval import excerpt
    row = {'paragraphs': [{'id': 'P1', 'sentences': [{'id': 'S1', 'text': '첫 문장이다.'}, {'id': 'S2', 'text': '둘째 문장이다.'}]}]}
    value = excerpt(row, {'location': ['P1']})
    assert value['located_excerpt'] == 'S1: 첫 문장이다.'
    row['paragraphs'][0]['sentences'] = row['paragraphs'][0]['sentences'][:1]
    assert excerpt(row, {'location': ['essay']})['located_excerpt'] == ''


def test_paid_evaluation_requires_environment_freeze_and_respects_cap(tmp_path):
    import pytest
    from verak.v4.wiki_eval import api_for
    with pytest.raises(FileNotFoundError):
        api_for(tmp_path)
    from verak.v3.eval.api import Phase6API
    from feak_tc.runtime.openai import CallBudgetExceeded
    config = {'search_test': {'model': 'gpt-6.1-sol', 'max_cost_usd': 2.,
                             'max_concurrent_requests': 2, 'phase_api_ceiling': 110},
              'paths': {'search_test_output': tmp_path / 'test_ledger'}}
    api = Phase6API(config, 110, phase='search_test')
    assert api.reserve('query', 'i1', 'fingerprint1', 1.2) == 1
    with pytest.raises(CallBudgetExceeded):
        api.reserve('judge', 'i2', 'fingerprint2', 1.)
    assert api.accounting()['reserved_usd'] == 1.2
    assert api.accounting()['confirmed_usd'] == 0


def test_real_local_pipeline_without_network(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip('kiwipiepy'); pytest.importorskip('bm25s'); pytest.importorskip('mwparserfromhell')
    import socket
    from xml.sax.saxutils import escape
    from verak.v4.wiki_build import constrain, digest, extract, index, tokenize, write
    constrain()
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: (_ for _ in ()).throw(AssertionError('network forbidden')))
    articles = [
        ('지구', '{{정보상자|이름=지구}} 지구는 태양계의 행성이며 우리가 살고 있는 천체이다. [[분류:행성]]\n\n== 기후 ==\n지구의 대기와 바다는 생명체가 살아갈 수 있는 환경을 만든다.<ref>제외할 각주</ref>\n\n== 외부 링크 ==\n제외할 내용이다.'),
        ('금성', '금성은 태양계의 행성이며 대기가 매우 두껍고 표면 온도가 높은 천체이다.'),
        ('달', '달은 지구를 도는 자연 위성으로 밤하늘에서 관측되는 천체이다.'),
        ('행성 목록', '태양계 행성의 여러 가지 이름을 정리하여 소개하는 목록 문서이다.'),
        ('지구 (동음이의)', '{{동음이의어}} 지구라는 말은 행성과 다른 뜻을 함께 가질 수 있다.')]
    nodes = [f'<page><title>{title}</title><ns>0</ns><id>{i}</id><revision><id>{100+i}</id><text>{escape(text)}</text></revision></page>'
             for i, (title, text) in enumerate(articles, 1)]
    path = tmp_path / 'fixture.xml.bz2'; path.write_bytes(bz2.compress(('<mediawiki>' + ''.join(nodes) + '</mediawiki>').encode()))
    write(tmp_path / 'dump/verified.json', {'sha256': digest(path), 'local_path': str(path), 'dump_date': '20261001'})
    extracted = extract(tmp_path)
    assert extracted['retained_articles'] == 3 and extracted['passages'] == 4
    assert extracted['drops']['list_page'] == 1 and extracted['drops']['disambiguation'] == 1
    tokenize(tmp_path); built = index(tmp_path)
    assert built['passages'] == 4
    engine = LocalWikiSearch(tmp_path)
    try:
        hits = engine.search('지구 대기 생명체')
        assert 1 <= len(hits) <= 3 and hits[0]['title'] == '지구'
        assert all(x['license'] == 'CC BY-SA 4.0' for x in hits)
        assert all('제외할' not in x['text'] for x in hits)
        assert engine.search('') == []
    finally:
        engine.close()
