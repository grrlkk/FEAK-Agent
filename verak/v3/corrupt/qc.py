"""Pre-registered data QC; one judgment per record, batched within each essay."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random

from ..common import read_json, sha_text, write_json
from ..phase2 import read_jsonl, write_jsonl
from .api import CorruptionAPI, QC_PROMPT, VAGUE_PROMPT
from .document import BareunBank, MARKER, source_document
from .operators import LEVELS
from .sources import select_sources


def prepare_vague(config, max_api_calls):
    output = config["paths"]["phase3_output"]
    api, bank = CorruptionAPI(config, max_api_calls), BareunBank(config)
    cache_path = output / "vague_cache.json"
    cache = read_json(cache_path) if cache_path.exists() else {}
    plan_path = output / "vague_plan.json"
    if plan_path.exists():
        plan = read_json(plan_path)
    else:
        plan = []
        for split, n in (("agent_dev", 20), ("agent_train", 4)):
            examples, _ = select_sources(config, split)
            random.Random(37).shuffle(examples)
            chosen = []
            for example in examples:
                doc = source_document(config, example, bank)
                anns = [a for a in doc.structure().annotations if not a.multi_unit and
                    a.style in {"한다", "합니다"} and 45 <= len(a.text) <= 160 and not MARKER.search(a.text)]
                if not anns:
                    continue
                # Prefer concrete example/numeric material, before seeing any GPT result.
                anns.sort(key=lambda a: (-sum(c.isdigit() for c in a.text)-3*("예를" in a.text), a.sid))
                ann = anns[0]
                chosen.append({"source_id": example.id, "split": split, "sid": ann.sid,
                    "original": ann.text, "style": ann.style, "cache_key": sha_text(ann.text)})
                if len(chosen) == n:
                    break
            plan.extend(chosen)
        write_json(plan_path, plan)
    if len(plan) > config["corruption"]["vague_generation_limit"]:
        raise ValueError("Vague plan exceeds reserved generation allowance")
    for row in plan:
        key = row["cache_key"]
        if key in cache:
            continue
        response = api.request("vague_generation", key, {"sentence": row["original"],
            "register": row["style"], "min_characters": round(len(row["original"])*.9+.5),
            "max_characters": int(len(row["original"])*1.1)})
        vague, reason = response["vague"].strip(), None
        try:
            if vague == row["original"] or not .9 <= len(vague)/len(row["original"]) <= 1.1:
                raise ValueError("identity_or_length")
            if MARKER.findall(vague) != MARKER.findall(row["original"]):
                raise ValueError("marker_changed")
            from .document import Document, Paragraph, Unit
            doc = Document([Paragraph("P1", [Unit("S1", vague, bank.tokens(vague))])], [""])
            ann = doc.structure().annotations[0]
            if ann.multi_unit or ann.style != row["style"]:
                raise ValueError("unit_count_or_register")
        except ValueError as error:
            reason = str(error)
        cache[key] = {**row, "vague": vague, "valid": reason is None, "discard_reason": reason,
                      "generation": response}
        write_json(cache_path, cache)
        print(f"Vague {len(cache)}/{len(plan)}; valid={sum(v['valid'] for v in cache.values())}; calls={api.budget.used}", flush=True)
    return cache


def select_qc(rows, n=60, seed=37):
    if len(rows) < n or any(r["split"] != "agent_dev" for r in rows):
        raise ValueError("QC needs sixty dev episodes")
    rng = random.Random(seed)
    pool = sorted(rows, key=lambda r: r["episode_id"])
    rng.shuffle(pool)
    # Coverage quotas before random filling. At most one episode per source essay.
    # Scarce operators first; this is stratified diagnostic QC, not prevalence estimation.
    available = Counter(op for row in pool for op in {r["op"] for r in row["records"]})
    picked, seen_sources, counts = [], set(), Counter()
    for op in sorted(LEVELS, key=lambda op: (available[op], op)):
        for row in pool:
            if counts[op] >= min(6, available[op]):
                break
            if row["source_id"] in seen_sources or op not in {r["op"] for r in row["records"]}:
                continue
            picked.append(row)
            seen_sources.add(row["source_id"])
            counts.update({r["op"] for r in row["records"]})
    for level in ("L1", "L2", "L3", "L4"):
        for row in pool:
            if sum(r["level"] == level for r in picked) >= 8:
                break
            if row["source_id"] not in seen_sources and row["level"] == level:
                picked.append(row)
                seen_sources.add(row["source_id"])
    for row in pool:
        if len(picked) >= n:
            break
        if row["source_id"] not in seen_sources:
            picked.append(row)
            seen_sources.add(row["source_id"])
    if len(picked) != n:
        raise ValueError(f"Coverage quotas selected {len(picked)}, expected {n}; do not trim silently")
    return picked


def snapshot_text(state):
    return "".join(gap + "".join(u["leading"]+u["text"] for u in p["units"])
                   for gap, p in zip(state["gaps"], state["paragraphs"])) + state["tail"]


def qc_payload(episode):
    records = []
    for i, record in enumerate(episode["records"]):
        after = (episode["records"][i+1]["inverse"] if i+1 < len(episode["records"])
                 else episode["corrupted_layout"])
        before_text, after_text = snapshot_text(record["inverse"]), snapshot_text(after)
        if sha_text(before_text) != record["before_sha256"] or sha_text(after_text) != record["after_sha256"]:
            raise ValueError("QC record context hashes disagree")
        records.append({"record_id": record["record_id"], "sentence_ids": record["sids"],
            "before_document": before_text, "after_document": after_text,
            "changed_before": record["original_text"], "changed_after": record["corrupted_text"]})
    return {"question": episode["question"], "genre": episode["genre"], "records": records}


def validate_judgments(row, response):
    actual = [j["record_id"] for j in response["judgments"]]
    expected = [r["record_id"] for r in row["records"]]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("QC response missing, duplicating, or inventing a record")


def qc_results(rows, calls):
    by_op = {op: [] for op in LEVELS}
    for row in rows:
        response = row.get("llm_judgment")
        if response is None:
            continue
        validate_judgments(row, response)
        ops = {r["record_id"]: r["op"] for r in row["records"]}
        for judgment in response["judgments"]:
            by_op[ops[judgment["record_id"]]].append(judgment)
    operators = {}
    for op, values in by_op.items():
        metrics = {field: sum(r[field] for r in values)/len(values) if values else None
                   for field in ("damage_real", "original_is_fix")}
        operators[op] = {"n": len(values), **metrics,
            "enabled": bool(values) and all(value >= .8 for value in metrics.values()),
            "failures": [r for r in values if not r["damage_real"] or not r["original_is_fix"]]}
    totals = {}
    for row in calls:
        if row.get("event") or not row.get("usage"):
            continue
        usage = row["usage"]
        model = row["model"]
        value = totals.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
                                          "cached_input_tokens": 0, "cache_write_tokens": 0})
        value["calls"] += 1
        value["input_tokens"] += usage.get("input_tokens", usage.get("prompt_tokens", 0))
        value["output_tokens"] += usage.get("output_tokens", usage.get("completion_tokens", 0))
        value["reasoning_tokens"] += usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
        value["cached_input_tokens"] += usage.get("input_tokens_details", {}).get("cached_tokens", 0)
        value["cache_write_tokens"] += usage.get("input_tokens_details", {}).get("cache_write_tokens", 0)
    for model, value in totals.items():
        ir, out = (.25, 2.0) if model == "gpt-5-mini" else (2.0, 10.0)
        value["cost_usd_uncached_estimate"] = (value["input_tokens"]*ir + value["output_tokens"]*out)/1e6
        cache_rate = .025 if model == "gpt-5-mini" else .1
        write_rate = ir if model == "gpt-5-mini" else 2.5
        value["cost_usd_usage_estimate"] = ((value["input_tokens"]-value["cached_input_tokens"]-
            value["cache_write_tokens"])*ir + value["cached_input_tokens"]*cache_rate +
            value["cache_write_tokens"]*write_rate + value["output_tokens"]*out)/1e6
    return {"label": "LLM-verified corruption data QC", "essays": len(rows),
        "complete": len(rows) == 60 and all(r.get("llm_judgment") is not None for r in rows),
        "operators": operators, "disabled_operators": [op for op, v in operators.items() if not v["enabled"]],
        "usage_by_model": totals, "cost_usd_uncached_estimate": sum(v["cost_usd_uncached_estimate"] for v in totals.values()),
        "cost_usd_usage_estimate": sum(v["cost_usd_usage_estimate"] for v in totals.values()),
        "seed": 37, "unit": "one judgment per record; records batched in one request per essay",
        "threshold": .8, "human_accuracy": None}


def run_qc(config, pool_path, max_api_calls, workers=4):
    output = config["paths"]["phase3_output"]
    path = output / "qc_sample.jsonl"
    if path.exists():
        rows = read_jsonl(path)
    else:
        rows = [{**row, "llm_judgment": None, "human_ok": None}
                for row in select_qc(read_jsonl(pool_path))]
        write_jsonl(path, rows)
        write_json(output / "qc_preregistration.json", {"sample_ids": [r["episode_id"] for r in rows],
            "prompt_sha256": sha_text(QC_PROMPT), "vague_prompt_sha256": sha_text(VAGUE_PROMPT),
            "seed": 37, "threshold": .8,
            "operators": dict(Counter(x["op"] for r in rows for x in r["records"])),
            "levels": dict(Counter(r["level"] for r in rows)),
            "pool_sha256": sha_text(pool_path.read_text(encoding="utf-8"))})
    api = CorruptionAPI(config, max_api_calls)
    pending = [r for r in rows if r["llm_judgment"] is None]
    if api.budget.used + len(pending) > max_api_calls:
        raise ValueError("Remaining shared budget cannot finish QC; no partial paid run")
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(api.request, "corruption_qc", r["episode_id"], qc_payload(r)): r for r in pending}
        for future in as_completed(futures):
            row = futures[future]
            try:
                response = future.result()
                # Preserve even a schema-valid response with wrong record IDs; never silently re-judge it.
                row["llm_judgment"] = response
                validate_judgments(row, response)
            except Exception as error:
                row["qc_error"] = type(error).__name__
                errors.append({"episode_id": row["episode_id"], "error": type(error).__name__})
            write_jsonl(path, rows)
            print(f"QC {sum(r['llm_judgment'] is not None for r in rows)}/60; shared calls={api.budget.used}", flush=True)
    if errors:
        write_json(output / "qc_errors.json", errors)
        raise ValueError("QC errors preserved; review raw responses before any retry")
    result = qc_results(rows, read_jsonl(output / "calls.jsonl"))
    result["api_calls_reserved"] = api.budget.used
    write_json(output / "qc_results.json", result)
    return result
