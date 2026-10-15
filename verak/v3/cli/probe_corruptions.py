"""Zero-GPT mechanical application audit and recorded Bareun synthetic fixtures."""

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import random

from ..common import DEFAULT_CONFIG, load_config, read_json, sha_text, write_json
from ..ko.annotation import KoreanStructure
from ..corrupt.builder import load_sources, donor_pool
from ..corrupt.document import BareunBank, Document
from ..corrupt.operators import LEVELS, apply, candidates, exact_restoration_satisfies, restore_record

SYNTHETIC = ("환경을 보호하는 일은 중요하다. 우리는 물을 아껴야 한다. 물을 낭비하면 자원이 부족해진다.\n"
             "물은 생명을 유지한다. 그래서 물을 아낄 수 있다. 환경에 관심이 있다.\n"
             "실천은 작은 행동에서 시작한다. 매일 노력해야 한다. 우리는 물을 아낀다.")
DONOR = "우주는 수많은 별과 은하로 이루어져 있다."


def capture_fixture(config):
    ko = KoreanStructure.from_config(config)
    profile = ko.analyzer.profile(SYNTHETIC)
    doc, bank = Document.from_profile(SYNTHETIC, profile), BareunBank(config)
    bank.seed(doc)
    donor = {"source_id": "synthetic:2", "question_hash": "other", "sid": "S1", "text": DONOR, "style": "한다"}
    vague = {sha_text("매일 노력해야 한다."): {"original": "매일 노력해야 한다.", "vague": "항상 실천해야 한다.", "valid": True}}
    chosen, failures = {}, {}
    for op in LEVELS:
        failures[op] = []
        for proposal in candidates(doc, op, donors=[donor], vague_cache=vague, question_hash="synthetic"):
            try:
                changed, record = apply(doc, proposal, bank)
                restored = restore_record(changed, record, bank)
                if restored.text != SYNTHETIC or not exact_restoration_satisfies(restored, record):
                    raise ValueError("Restoration failed")
                chosen[op] = asdict(proposal)
                break
            except ValueError as error:
                failures[op].append(str(error))
        if op not in chosen:
            raise ValueError(f"Synthetic operator probe failed: {op}: {failures[op]}")
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/corrupt_bareun.json"
    write_json(path, {"origin": "new synthetic unit-test text; real Bareun observations; no corpus essays",
        "text": SYNTHETIC, "profile": profile.to_dict(), "proposals": chosen,
        "tokens": {text: [asdict(t) for t in tokens] for text, tokens in bank.memory.items()}})
    print("Captured", len(chosen), "operator fixtures", flush=True)


def audit(config):
    bank = BareunBank(config)
    sources = load_sources(config, "agent_dev", bank)
    random.Random(37).shuffle(sources)
    sources = sources[:300]  # Never relax source filters to manufacture 300.
    donors = donor_pool(sources)
    vague_path = config["paths"]["phase3_output"] / "vague_cache.json"
    vague = read_json(vague_path) if vague_path.exists() else {}
    rows = []
    for i, (example, doc, _) in enumerate(sources, 1):
        dominant = doc.structure().dominant_style
        applicable_donors = [d for d in donors if d["question_hash"] != example.question_hash and
                             d["style"] == dominant][:2]
        for op in LEVELS:
            options = candidates(doc, op, donors=applicable_donors, vague_cache=vague, question_hash=example.question_hash)
            random.Random(37 + example.source_line).shuffle(options)
            if not options:
                rows.append({"source_id": example.id, "op": op, "applicable": False})
                continue
            proposal = options[0]  # First random eligible application; do not retry until success.
            try:
                changed, record = apply(doc, proposal, bank)
                restored = restore_record(changed, record, bank)
                if restored.text != doc.text or not exact_restoration_satisfies(restored, record):
                    raise ValueError("Inverse/target check failed")
                passed, reason = True, None
            except ValueError as error:
                passed, reason = False, str(error)
            rows.append({"source_id": example.id, "op": op, "applicable": True, "passed": passed, "reason": reason})
        if i % 20 == 0 or i == len(sources):
            result = {"requested_source_essays": 300, "available_source_essays": len(sources),
                      "completed": i, "seed": 37, "rows": rows, "operators": {}}
            for op in LEVELS:
                values = [r for r in rows if r["op"] == op and r["applicable"]]
                result["operators"][op] = {"applications": len(values), "passed": sum(r["passed"] for r in values),
                    "pass_rate": sum(r["passed"] for r in values)/len(values) if values else None,
                    "not_applicable": sum(r["op"] == op and not r["applicable"] for r in rows),
                    "discard_reasons": dict(Counter(r["reason"] for r in values if not r["passed"]))}
            write_json(config["paths"]["phase3_output"] / "mechanical_audit.json", result)
            print(f"Mechanical audit {i}/{len(sources)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--capture-fixture", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    capture_fixture(config) if args.capture_fixture else audit(config)


if __name__ == "__main__":
    main()
