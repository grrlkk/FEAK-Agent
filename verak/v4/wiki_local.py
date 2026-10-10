"""Local Korean Wikipedia passage extraction and Kiwi/BM25 retrieval.

This module has no network client. Dump download and the explicitly authorized
offline Sol evaluation are deliberately kept outside this interface. It is not
connected to the editor environment or teacher runs.
"""
from __future__ import annotations

import bz2
import hashlib
import html
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
from urllib.parse import quote
import xml.etree.ElementTree as ET

LICENSE = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
EXTRACTOR_VERSION = "kowiki_paragraphs_v1"
TOKENIZER_VERSION = "kiwi_content_morphemes_v1"
DISAMBIG = {"동음이의", "동음이의어", "동명이인", "동음이의어 문서", "disambig", "disambiguation", "hndis"}
SKIP_SECTIONS = {"각주", "주석", "참고", "참고 문헌", "참고문헌", "외부 링크", "외부링크", "같이 보기", "같이보기", "관련 항목", "관련항목", "더 보기"}
HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)


def normalize(text):
    return unicodedata.normalize("NFC", text)


def page_filter(page):
    """Return one primary exclusion reason; record counts rather than hide loss."""
    text = page["wikitext"]
    if page["namespace"] != 0:
        return "non_article_namespace"
    if page.get("redirect") or re.match(r"^\s*#(?:redirect|넘겨주기)\b", text, re.I):
        return "redirect"
    names = {re.sub(r"\s+", " ", x.strip().replace("_", " ")).lower()
             for x in re.findall(r"\{\{\s*([^|{}\n]+)", text)}
    if names & DISAMBIG or "(동음이의)" in page["title"] or re.search(r"\[\[분류:[^\]]*동음이의", text):
        return "disambiguation"
    if ("목록" in page["title"] or re.search(r"\blist of\b", page["title"], re.I)
            or re.search(r"\[\[분류:[^\]|]*(?:목록|리스트)(?:\||\]\])", text)
            or names & {"목록 문서", "목록", "list"}):
        return "list_page"
    return None


def iter_pages(path):
    """Stream XML without materializing or extracting the full decompressed dump."""
    opener = bz2.open if str(path).endswith(".bz2") else open
    with opener(path, "rb") as stream:
        iterator = ET.iterparse(stream, events=("start", "end"))
        _, root = next(iterator)
        ns = root.tag.partition("}")[0] + "}" if "}" in root.tag else ""
        for event, node in iterator:
            if event != "end" or node.tag != ns + "page":
                continue
            rev = node.find(ns + "revision")
            if rev is not None:
                yield {"page_id": node.findtext(ns + "id"),
                       "title": normalize(node.findtext(ns + "title") or ""),
                       "namespace": int(node.findtext(ns + "ns") or 0),
                       "redirect": node.find(ns + "redirect") is not None,
                       "revision_id": rev.findtext(ns + "id"),
                       "revision_timestamp": rev.findtext(ns + "timestamp"),
                       "wikitext": rev.findtext(ns + "text") or ""}
            root.clear()


def plain_text(wikitext):
    """Conservatively remove non-prose markup; do not expand templates online."""
    import mwparserfromhell

    text = re.sub(r"\{\|.*?\|\}", "\n", wikitext, flags=re.S)
    text = re.sub(r"^\s*[*#;:|!].*$", "", text, flags=re.M)
    code = mwparserfromhell.parse(text)
    for node in list(code.filter_templates(recursive=False)):
        code.remove(node)
    for node in list(code.filter_tags(recursive=False)):
        name = str(node.tag).strip().lower()
        if name in {"ref", "references", "gallery", "math", "score", "timeline", "table", "syntaxhighlight", "source"}:
            code.remove(node)
    for node in list(code.filter_wikilinks(recursive=False)):
        title = str(node.title).strip().lstrip(":").lower()
        if re.match(r"(?:파일|그림|분류|file|image|category):", title):
            code.remove(node)
    return normalize(html.unescape(code.strip_code(normalize=True, collapse=True)))


def passage_id(dump_date, page_id, revision_id, section, ordinal, text):
    payload = json.dumps([section, ordinal, text], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:20]
    return f"kowiki:{dump_date}:{page_id}:{revision_id}:{digest}"


def passages(page, dump_date, *, min_chars=1):
    """Preserve paragraph boundaries and their hierarchical section labels."""
    if page_filter(page):
        return
    raw = page["wikitext"]
    headings = list(HEADING.finditer(raw))
    blocks = [(None, raw[:headings[0].start()] if headings else raw)]
    blocks += [(h, raw[h.end():headings[i + 1].start() if i + 1 < len(headings) else len(raw)])
               for i, h in enumerate(headings)]
    stack = []
    ordinal = 0
    for heading, block in blocks:
        if heading is not None:
            level, label = len(heading.group(1)), plain_text(heading.group(2)).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, label))
        if any(label in SKIP_SECTIONS for _, label in stack):
            continue
        section = " > ".join(label for _, label in stack) or "서론"
        for paragraph in re.split(r"\n\s*\n", plain_text(block)):
            text = re.sub(r"\s+", " ", paragraph).strip()
            if len(text) < min_chars or not re.search(r"\w", text):
                continue
            ordinal += 1
            yield {"passage_id": passage_id(dump_date, page["page_id"], page["revision_id"], section, ordinal, text),
                   "title": page["title"], "section": section, "text": text,
                   "url": "https://ko.wikipedia.org/wiki/" + quote(page["title"].replace(" ", "_"), safe=""),
                   "revision_url": f"https://ko.wikipedia.org/w/index.php?oldid={page['revision_id']}",
                   "page_id": page["page_id"], "revision_id": page["revision_id"],
                   "revision_timestamp": page.get("revision_timestamp"), "paragraph_ordinal": ordinal,
                   "dump_date": dump_date, "license": LICENSE, "license_url": LICENSE_URL,
                   "modification": "Wiki markup, templates, tables, lists and references removed; paragraph prose retained."}


class KiwiTokenizer:
    def __init__(self, workers=1):
        from kiwipiepy import Kiwi
        self.kiwi = Kiwi(num_workers=workers)

    @staticmethod
    def terms(tokens):
        return [normalize(t.form).lower() for t in tokens
                if t.tag.split("-")[0] in {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "XR", "SL", "SN"}]

    def __call__(self, text):
        return self.terms(self.kiwi.tokenize(text))

    def batch(self, texts):
        for tokens in self.kiwi.tokenize(texts):
            yield self.terms(tokens)


class LocalWikiSearch:
    """A read-only local SEARCH(query) interface, always at most three passages."""
    def __init__(self, root):
        import bm25s
        self.root = Path(root)
        self.tokenizer = KiwiTokenizer(workers=1)
        self.index = bm25s.BM25.load(str(self.root / "bm25"), mmap=True, load_corpus=False)
        self.db = sqlite3.connect(f"file:{self.root / 'passages.sqlite'}?mode=ro", uri=True)

    def search(self, query):
        if not isinstance(query, str) or not query.strip():
            return []
        terms = [t for t in self.tokenizer(query) if t in self.index.vocab_dict]
        if not terms:
            return []
        documents, scores = self.index.retrieve([terms], k=min(3, self.index.scores["num_docs"]),
                                                show_progress=False, n_threads=1)
        results = []
        for doc_id, score in zip(documents[0], scores[0]):
            if score <= 0:
                continue
            row = self.db.execute("SELECT payload FROM passages WHERE doc_id=?", (int(doc_id),)).fetchone()
            if row is None:
                raise ValueError("Index references a missing passage")
            results.append({**json.loads(row[0]), "bm25_score": float(score)})
        return results

    def close(self):
        self.db.close()


def insertion_citation_contract(passage):
    """Metadata for future integration only; no teacher action is wired here."""
    return {"title": passage["title"], "passage_id": passage["passage_id"], "url": passage["url"],
            "license": passage["license"], "license_url": passage["license_url"],
            "instruction": "검색 문장을 그대로 복사하지 말고 바꾸어 쓰며, 문장에 사용한 출처의 title과 passage_id를 반드시 함께 인용한다."}
