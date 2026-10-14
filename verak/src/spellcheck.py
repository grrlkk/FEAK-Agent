"""Cached spelling evidence, never an automatic acceptance/rejection rule."""

from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


def parse_bareun(text, response):
    if response.get("origin") != text or not isinstance(response.get("revised"), str):
        raise ValueError("Spellcheck response source does not match request")
    issues = []
    for block in response.get("revised_blocks", response.get("revisedBlocks", [])):
        origin = block["origin"]
        start = origin.get("begin_offset", origin.get("beginOffset", 0))
        original = origin["content"]
        end = start + len(original)
        if type(start) is not int or not 0 <= start <= end <= len(text) or text[start:end] != original:
            raise ValueError("Spellcheck returned invalid UTF32 source offsets")
        categories = set()
        def collect(value):
            categories.update(str(r.get("category", "UNKNOWN")) for r in value.get("revisions", []))
            for nested in value.get("nested", []): collect(nested)
        collect(block)
        suggestion = block["revised"]
        if suggestion != original:
            issues.append({"start": start, "end": end, "original": original,
                           "suggestion": suggestion, "category": "|".join(sorted(categories)) or "UNKNOWN"})
    # Preserve corrections missing from blocks, including whitespace cleanup.
    for tag, i, j, k, l in SequenceMatcher(None, text, response["revised"], autojunk=False).get_opcodes():
        if tag != "equal" and not any(x["start"] <= i <= j <= x["end"] for x in issues):
            issues.append({"start": i, "end": j, "original": text[i:j],
                           "suggestion": response["revised"][k:l], "category": "UNSPECIFIED"})
    return issues


class SpellChecker:
    def __init__(self, config, transport=None, on_record=None):
        self.config, self.transport = config, transport or self._post
        self.on_record = on_record or (lambda row: None)
        self.cache = {}
        self.last_call = None
        self.cache_path = Path(config["cache"]) if config.get("cache") else None
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    if row["result"]["status"] == "available": self.cache[row["key"]] = row["result"]
                except (ValueError, KeyError):
                    continue  # A truncated cache line is not evidence of no errors.

    def _post(self, text):
        key = os.getenv(self.config["api_key_env"])
        if not key:
            raise ValueError("Spellcheck API key is not configured")
        payload = {"document": {"content": text, "language": "ko_KR"}, "encoding_type": "UTF32",
                   "config": {"enable_cleanup_whitespace": False, "enable_sentence_check": False}}
        request = urllib.request.Request(self.config["endpoint"],
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"api-key": key, "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=self.config.get("timeout_s", 10)) as response:
            return json.load(response)

    def report(self, text):
        cache_key = hashlib.sha256(json.dumps([self.config["endpoint"], "p1-utf32-v1", text], ensure_ascii=False).encode()).hexdigest()
        if cache_key in self.cache:
            result = {**self.cache[cache_key], "cached": True}
            self.on_record({"stage": "spellcheck", "cached": True, "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
            return result
        for attempt in range(3):
            interval = self.config.get("min_interval_s", 1)
            if self.last_call is not None:
                time.sleep(max(0, interval - (time.monotonic() - self.last_call)))
            started = self.last_call = time.monotonic()
            record = {"stage": "spellcheck", "attempt": attempt, "cached": False,
                      "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
            try:
                response = self.transport(text)
                issues = parse_bareun(text, response)
                result = {"status": "available", "issues": issues, "cached": False}
                record.update(status="available", raw=response)
                break
            except Exception as exc:
                record.update(status="unavailable", error_type=type(exc).__name__, http_status=getattr(exc, "code", None))
                result = {"status": "unavailable", "issues": None, "cached": False,
                          "error_type": type(exc).__name__}
            finally:
                record["elapsed_s"] = time.monotonic() - started
                self.on_record(record)
        self.cache[cache_key] = result
        # Failed requests are cached only within this run, so another run may recover.
        if self.cache_path and result["status"] == "available":
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as handle:
                handle.write(json.dumps({"key": cache_key, "result": result}, ensure_ascii=False) + "\n")
                handle.flush()
        return result

    def check(self, text):
        """Return located suggestions, or None when the service is unavailable."""
        return self.report(text)["issues"]


def compare(before, after, before_report, after_report):
    if before_report["status"] != "available" or after_report["status"] != "available":
        return {"status": "unavailable", "fixed": [], "introduced": [], "persistent": []}
    blocks = SequenceMatcher(None, before, after, autojunk=False).get_matching_blocks()
    remaining = list(after_report["issues"])
    fixed, persistent = [], []
    for issue in before_report["issues"]:
        matches = []
        for block in blocks:
            if block.a <= issue["start"] <= issue["end"] <= block.a + block.size:
                start = block.b + issue["start"] - block.a
                matches += [r for r in remaining if r["start"] == start and r["end"] == start + issue["end"] - issue["start"]
                            and all(r[k] == issue[k] for k in ("original", "suggestion", "category"))]
        if matches:
            match = matches[0]
            remaining.remove(match)
            persistent.append({"before": issue, "after": match})
        else:
            fixed.append(issue)
    return {"status": "available", "fixed": fixed, "introduced": remaining, "persistent": persistent}


def attach(units, comparison):
    def overlaps(issue, span):
        if issue["start"] == issue["end"]:
            return span[0] <= issue["start"] <= span[1]
        return issue["start"] < span[1] and issue["end"] > span[0]
    for unit in units:
        unit.spelling = {"status": comparison["status"],
            "fixed": [r for r in comparison["fixed"] if overlaps(r, unit.before_span)],
            "introduced": [r for r in comparison["introduced"] if overlaps(r, unit.after_span)]}
