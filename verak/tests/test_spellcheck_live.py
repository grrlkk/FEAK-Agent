import json
import os

import pytest

from verak.src.run_single import ROOT, load_config
from verak.src.spellcheck import SpellChecker


@pytest.mark.skipif(os.getenv("VERAK_LIVE_SPELL") != "1", reason="Opt-in hosted spelling API probe")
def test_real_spelling_endpoint():
    cfg = load_config(ROOT / "config.yaml")
    records = []
    checker = SpellChecker({**cfg["spellcheck"], "cache": None}, on_record=records.append)
    result = checker.report("오늘은 날씨가 조타.")
    print(json.dumps({"result": result, "attempts": [
        {k:r.get(k) for k in ("attempt", "status", "error_type", "http_status", "elapsed_s")} for r in records]},ensure_ascii=False))
    assert result["status"] == "available", "Real service unavailable; offline fallback is tested separately"
