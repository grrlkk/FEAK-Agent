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
