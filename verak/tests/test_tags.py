"""T1: run against the real installed Bareun before extending the lexicons."""

import json
import os

import pytest

from verak.src.analyzer import BareunBackend

EXAMPLES = [
    "학생은 책을 읽는다.", "학생이 책을 읽습니다.", "비가 와서 늦어졌다.", "비가 왔고 늦어졌다.",
    "학생들은 자료를 모았다. 다음 날 발표했다.", "민수도 참석하지만 영희만 발표한다.",
    "우리는 책을 읽어요.", "학교에서는 뛰면 안 된다.", "친구가 오니까 문을 열자.", "이 길로 가십시오.",
]


@pytest.mark.skipif(os.getenv("VERAK_LIVE_TAGS") != "1", reason="Real local Bareun integration; opt in with VERAK_LIVE_TAGS=1")
def test_real_bareun_tags_and_inflected_forms():
    from dotenv import load_dotenv
    load_dotenv("/home/chanwoo/essay_scoring_llm/.env", override=False)
    backend = BareunBackend()
    observed = set()
    for text in EXAMPLES:
        data = backend.analyze(text)
        morphs = [m for sentence in data["sentences"] for word in sentence["tokens"] for m in word["morphemes"]]
        pairs = [(m["text"]["content"], m["tag"]) for m in morphs]
        observed.update(tag for _, tag in pairs)
        print(json.dumps({"text": text, "morphemes": pairs, "raw": data}, ensure_ascii=False))
    assert {"EF", "EC", "JKS", "JKO", "JX"} <= observed
