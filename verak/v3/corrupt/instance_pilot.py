"""A frozen 100-essay pilot; no operator gates and no full-corpus judge entry point."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import random

from ..api import PhaseBudget
from ..common import file_sha, read_json, sha_text, write_json
from ..phase2 import read_jsonl, write_jsonl
from .api import CorruptionAPI, QC_PROMPT
from .instance_policy import ACTIVE_LEVELS
from .qc import qc_payload, validate_judgments

SPLITS = ("agent_train", "agent_dev")
LOCAL_LEVELS = ("WORD", "SENTENCE", "TEXT")


class PilotAPI(CorruptionAPI):
    def __init__(self, config, max_api_calls):
        self.output = config["paths"]["phase3b_output"]
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = config["corruption_pilot"]
        if (self.config["judge_model"], self.config["judge_reasoning_effort"],
            self.config["phase_api_ceiling"], self.config["sample_size"]) != ("gpt-6.1-sol", "high", 100, 100):
            raise ValueError("Phase 3b pilot authorization mismatch")
        self.budget = PhaseBudget(self.output / "api_budget.json", max_api_calls,
                                 authorized_ceiling=100, phase="v3_phase3b_instance_pilot")

    def request(self, stage, sample_id, payload, **kwargs):
        if stage != "corruption_qc":
            raise ValueError("Only pilot QC is authorized; no generation calls")
        return super().request(stage, sample_id, payload, **kwargs)


def stratified_pilot(rows, n=100, seed=41):
    """SRS within (split, curriculum, rarest-present-operator) strata.

    The anchor is determined from the *unjudged* population frequencies. At most
    2*4*10=80 strata permits every nonempty stratum to be represented in 100 draws.
    Inclusion probabilities n_h/N_h are stored for corpus projections. Different
    variants of the same source are distinct candidate sampling units.
    """
    rows = sorted(rows, key=lambda r: r["episode_id"])
    if len(rows) < n or len({r["episode_id"] for r in rows}) != len(rows):
        raise ValueError("Insufficient or duplicate candidates")
    if {r["split"] for r in rows} != set(SPLITS):
        raise ValueError("Both candidate splits are required")
    frequencies = Counter(op for r in rows for op in {v["op"] for v in r["records"]})
    if set(frequencies) != set(ACTIVE_LEVELS):
        raise ValueError("Candidate pool must cover exactly the ten active operators")
    groups = defaultdict(list)
    for row in rows:
        anchor = min({r["op"] for r in row["records"]}, key=lambda op: (frequencies[op], op))
        groups[f'{row["split"]}|{row["level"]}|{anchor}'].append(row)
    if len(groups) > n:
        raise ValueError("More strata than pilot requests; cannot silently omit strata")
    # Census tiny cells; otherwise start at two if the budget permits. Allocate
    # remaining slots in proportion to cell size, independent of all judgments.
    initial = 2 if sum(min(2, len(g)) for g in groups.values()) <= n else 1
    quotas = {h: min(initial, len(g)) for h, g in groups.items()}
    while sum(quotas.values()) < n:
        h = max((h for h in sorted(groups) if quotas[h] < len(groups[h])),
                key=lambda h: len(groups[h])/(quotas[h]+1))
        quotas[h] += 1
    rng, sample, strata = random.Random(seed), [], {}
    for h in sorted(groups):
        population, quota = len(groups[h]), quotas[h]
        strata[h] = {"population": population, "sample": quota, "weight": population/quota}
        sample.extend({**r, "sample_stratum": h, "sampling_weight": population/quota,
                       "llm_judgment": None, "human_ok": None}
                      for r in rng.sample(groups[h], quota))
    rng.shuffle(sample)
    if ({v["op"] for r in sample for v in r["records"]} != set(ACTIVE_LEVELS) or
            {r["level"] for r in sample} != {"L1", "L2", "L3", "L4"}):
        raise ValueError("Fixed-seed pilot lacks required coverage; do not start judging")
    return sample, {"seed": seed, "n": n, "population": len(rows), "strata": strata,
        "method": "SRS_without_replacement_within_split_curriculum_rarest_operator_strata",
        "operator_candidate_frequencies": dict(frequencies), "anchor_ties": "lexicographic",
        "projection": "sum_h N_h * mean_h(essay_kept); level records counted only in fully kept essays",
        "source_clustering": "Candidates are units; multiple variants of one source may be sampled"}


def usage_cost(usage):
    """Reasoning is a subset of output; read/write caches are subsets of input."""
    inputs, outputs = usage["input_tokens"], usage["output_tokens"]
    detail = usage.get("input_tokens_details") or {}
    cached, written = detail.get("cached_tokens", 0), detail.get("cache_write_tokens", 0)
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
    if min(inputs, outputs, cached, written, reasoning) < 0 or cached+written > inputs or reasoning > outputs:
        raise ValueError("Unexpected usage accounting; do not invent a cost")
    return {"calls": 1, "input_tokens": inputs, "output_tokens": outputs, "reasoning_tokens": reasoning,
        "cached_input_tokens": cached, "cache_write_tokens": written,
        "cost_usd_usage_estimate": ((inputs-cached-written)*2 + cached*.1 + written*2.5 + outputs*10)/1e6,
        "cost_usd_standard_input_output": (inputs*2+outputs*10)/1e6}


def filter_row(row):
    response = row.get("llm_judgment")
    if response is None or row.get("qc_error"):
        raise ValueError("An incomplete judgment cannot enter filtering")
    validate_judgments(row, response)
    statuses = {j["record_id"]: j["damage_real"] is True and j["original_is_fix"] is True
                for j in response["judgments"]}
    return {**row, "record_kept": statuses, "essay_kept": all(statuses.values())}


def summarize(sample, plan, calls):
    if len(sample) != plan["n"] or any(not r.get("llm_judgment") for r in sample):
        raise ValueError("Pilot must be complete before projection")
    filtered = [filter_row(r) for r in sample]
    usable = [r for r in calls if not r.get("event") and r.get("usage")]
    call_ids = [c["sample_id"] for c in usable]
    expected_ids = {r["episode_id"] for r in sample}
    if len(call_ids) != len(set(call_ids)) or set(call_ids) != expected_ids:
        raise ValueError("Each pilot essay must have exactly one usage record")
    costs = {c["sample_id"]: usage_cost(c["usage"]) for c in usable}
    total = {key: sum(c[key] for c in costs.values()) for key in next(iter(costs.values()))}
    groups = {"operator": defaultdict(list), "record_level": defaultdict(list),
              "curriculum": defaultdict(list), "split": defaultdict(list)}
    essay_groups = {"curriculum": defaultdict(list), "split": defaultdict(list)}
    strata = defaultdict(list)
    for row in filtered:
        strata[row["sample_stratum"]].append(row)
        judgments = {j["record_id"]: j for j in row["llm_judgment"]["judgments"]}
        for record in row["records"]:
            result = {**judgments[record["record_id"]], "kept": row["record_kept"][record["record_id"]],
                      "in_kept_essay": row["essay_kept"]}
            for name, key in (("operator", record["op"]), ("record_level", record["level"]),
                              ("curriculum", row["level"]), ("split", row["split"])):
                groups[name][key].append(result)
        for name in essay_groups:
            essay_groups[name][row["level"] if name == "curriculum" else row[name]].append(row["essay_kept"])
    metrics = {name: {key: {"judged": len(items), "kept": sum(j["kept"] for j in items),
                "yield": sum(j["kept"] for j in items)/len(items),
                "damage_real": sum(j["damage_real"] for j in items)/len(items),
                "original_is_fix": sum(j["original_is_fix"] for j in items)/len(items),
                "records_in_kept_essays": sum(j["in_kept_essay"] for j in items)}
            for key, items in sorted(values.items())} for name, values in groups.items()}
    essays = {name: {key: {"judged": len(items), "kept": sum(items), "yield": sum(items)/len(items)}
                    for key, items in sorted(values.items())} for name, values in essay_groups.items()}
    essays["overall"] = {"judged": len(filtered), "kept": sum(r["essay_kept"] for r in filtered),
                         "yield": sum(r["essay_kept"] for r in filtered)/len(filtered)}
    projections = {split: Counter() for split in SPLITS}
    per_stratum = {}
    for h, design in plan["strata"].items():
        rows = strata[h]
        N, n = design["population"], design["sample"]
        if len(rows) != n:
            raise ValueError("Stratum sample size differs from frozen design")
        kept = sum(r["essay_kept"] for r in rows)
        cost = sum(costs[r["episode_id"]]["cost_usd_usage_estimate"] for r in rows)
        item = {**design, "judged": n, "kept": kept, "essay_yield": kept/n,
            "projected_total_kept": N*kept/n, "projected_remaining_kept": (N-n)*kept/n,
            "remaining_candidates": N-n, "projected_remaining_cost_usd": (N-n)*cost/n,
            "observed_cost_usd": cost, "candidate_count": N}
        for level in LOCAL_LEVELS:
            counts = sum(sum(rec["level"] == level for rec in r["records"])
                         for r in rows if r["essay_kept"])
            item[f"projected_{level}_records"] = N*counts/n
        per_stratum[h] = item
        projections[rows[0]["split"]].update({k: item[k] for k in (
            "judged", "kept", "projected_total_kept", "projected_remaining_kept", "remaining_candidates",
            "projected_remaining_cost_usd", "observed_cost_usd", "candidate_count",
            *[f"projected_{level}_records" for level in LOCAL_LEVELS])})
    projections["overall"] = sum(projections.values(), Counter())
    for p in projections.values():
        local_total = sum(p[f"projected_{level}_records"] for level in LOCAL_LEVELS)
        p["projected_local_shares"] = {level: p[f"projected_{level}_records"]/local_total if local_total else None
                                      for level in LOCAL_LEVELS}
        p["projected_essay_yield"] = p["projected_total_kept"]/p["candidate_count"]
    return {"label": "LLM-verified instance filtering pilot", "human_accuracy": None,
        "model": "gpt-6.1-sol", "reasoning_effort": "high", "api_usage": total,
        "record_yield": metrics, "essay_yield": essays, "projections": projections,
        "strata": per_stratum, "singleton_strata": sum(v["sample"] == 1 for v in plan["strata"].values()),
        "source_essays_sampled": len({r["source_id"] for r in sample}),
        "active_operators": sorted(ACTIVE_LEVELS), "operator_gating": False,
        "limitations": "Pilot point estimates; sparse strata and within-source dependence. No independent-record assumption.",
        "q_corrupted_computed": False, "remaining_judged": 0, "phase4_started": False}, filtered


def prepare_pilot(config):
    output = config["paths"]["phase3b_output"]
    path = output / "pilot_preregistration.json"
    hashes = {split: file_sha(output / f"candidates_{split}.jsonl") for split in SPLITS}
    validation = read_json(output / "candidate_validation.json")
    if any(validation["datasets"][split]["sha256"] != hashes[split] for split in SPLITS):
        raise ValueError("Candidate audit is stale; verify before sampling")
    if path.exists():
        plan = read_json(path)
        if plan["candidate_sha256"] != hashes or plan["prompt_sha256"] != sha_text(QC_PROMPT):
            raise ValueError("Frozen pilot population or prompt changed")
        return plan
    rows = [r for split in SPLITS for r in read_jsonl(output / f"candidates_{split}.jsonl")]
    if any(r["q_corrupted"] is not None for r in rows):
        raise ValueError("Phase 3b must not score corrupted candidates")
    sample, plan = stratified_pilot(rows)
    payloads = {r["episode_id"]: qc_payload(r) for r in sample}
    if any(len(json.dumps(p, ensure_ascii=False, sort_keys=True))+len(QC_PROMPT) > 100000 for p in payloads.values()):
        raise ValueError("A pilot request would exceed the untruncated input limit")
    plan.update(candidate_sha256=hashes, prompt_sha256=sha_text(QC_PROMPT),
        sample_ids=[r["episode_id"] for r in sample],
        payload_sha256={sid: sha_text(json.dumps(p, ensure_ascii=False, sort_keys=True)) for sid, p in payloads.items()},
        model="gpt-6.1-sol", reasoning_effort="high", filter_rule="every_record_both_fields_true",
        implementation_sha256={str(p): file_sha(p) for p in sorted((config["paths"]["repo"]/"verak/v3/corrupt").glob("*.py"))})
    write_jsonl(output / "pilot_sample.jsonl", sample)
    write_json(path, plan)
    return plan


def run_pilot(config, max_api_calls, workers=4):
    output = config["paths"]["phase3b_output"]
    output.mkdir(parents=True, exist_ok=True)
    # The shared budget prevents overrun; this additional process lock prevents
    # two runners from submitting the same sample before either saves its result.
    with (output / "pilot_run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_pilot(config, max_api_calls, workers)


def _run_pilot(config, max_api_calls, workers):
    plan = prepare_pilot(config)
    output = config["paths"]["phase3b_output"]
    path = output / "pilot_sample.jsonl"
    sample = read_jsonl(path)
    if [r["episode_id"] for r in sample] != plan["sample_ids"] or len(sample) != 100:
        raise ValueError("Pilot sample IDs changed")
    if any(file_sha(p) != digest for p, digest in plan["implementation_sha256"].items()):
        raise ValueError("QC implementation changed after preregistration")
    for r in sample:
        if sha_text(json.dumps(qc_payload(r), ensure_ascii=False, sort_keys=True)) != plan["payload_sha256"][r["episode_id"]]:
            raise ValueError("Frozen pilot payload changed")
    api = PilotAPI(config, max_api_calls)
    calls = read_jsonl(output / "calls.jsonl") if (output / "calls.jsonl").exists() else []
    pending = [r for r in sample if r.get("llm_judgment") is None]
    already_sent = {c["sample_id"] for c in calls}
    if any(r["episode_id"] in already_sent for r in pending) or any(r.get("qc_error") for r in sample):
        raise ValueError("Prior incomplete/invalid response preserved; no automatic retry")
    if api.budget.used+len(pending) > min(max_api_calls, 100):
        raise ValueError("Budget cannot finish the fixed pilot; no requests started")
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(api.request, "corruption_qc", r["episode_id"], qc_payload(r)): r for r in pending}
        for future in as_completed(futures):
            row = futures[future]
            try:
                row["llm_judgment"] = future.result()
                validate_judgments(row, row["llm_judgment"])
            except Exception as error:
                row["qc_error"] = type(error).__name__
                errors.append({"episode_id": row["episode_id"], "error": type(error).__name__})
            write_jsonl(path, sample)
            print(f"Pilot {sum(r.get('llm_judgment') is not None for r in sample)}/100; shared calls={api.budget.used}", flush=True)
    if errors:
        write_json(output / "pilot_errors.json", errors)
        raise ValueError("Pilot errors saved; no retry or remaining-corpus judgment")
    result, filtered = summarize(sample, plan, read_jsonl(output / "calls.jsonl"))
    result["api_calls_reserved"] = api.budget.used
    if api.budget.used != 100:
        raise ValueError("Expected exactly 100 accounted pilot requests")
    write_jsonl(output / "pilot_filtered.jsonl", filtered)
    write_jsonl(output / "pilot_kept.jsonl", [r for r in filtered if r["essay_kept"]])
    write_json(output / "pilot_results.json", result)
    return result
