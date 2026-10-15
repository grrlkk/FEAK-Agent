"""Phase 2c denominators, paired agreement and gates; no human accuracy claims."""

from collections import Counter

from .phase2c import COUNTS
from .phase2c_verification import FIELDS, both_ok
from .structure_metrics import usage


def pilot_projection(rows, records):
    pilot = [r for r in rows if r["pilot"]]
    if len(pilot) != 20 or any(both_ok(r) is None for r in pilot):
        raise ValueError("Finish both judgments of five items per check")
    details, projected = {}, 0
    for check, total in COUNTS.items():
        ids = {r["item_id"] for r in pilot if r["check"] == check}
        if len(ids) != 5:
            raise ValueError("Five pilot items required per check")
        data = usage([r for r in records if r.get("item_id") in ids])
        if data["calls_with_usage"] != 10 or data["calls_without_usage"]:
            raise ValueError("Pilot usage missing or duplicated; do not proceed")
        data["projected_total_usd"] = data["cost_usd"] * total / 5
        projected += data["projected_total_usd"]
        details[check] = data
    ids = {r["item_id"] for r in pilot}
    return {"by_check": details, "usage": usage([r for r in records if r.get("item_id") in ids]),
            "projected_calls": 460, "projected_total_usd": projected,
            "projection_with_10_reserve_calls_usd": projected * 470 / 460,
            "cost_limit_usd": 10, "continue_allowed": projected * 470 / 460 < 10}


def rates(rows):
    completed = [r for r in rows if both_ok(r) is not None]
    different = [r["sentence_id"] for r in completed if
                 r["llm_judgment_1"][FIELDS[r["check"]]] != r["llm_judgment_2"][FIELDS[r["check"]]]]
    n, passed = len(completed), sum(both_ok(r) for r in completed)
    return {"n_requested": len(rows), "n_completed": n, "both_true": passed,
            "pass_rate": passed / n if n else None, "agreement": (n - len(different)) / n if n else None,
            "disagreement_ids": different}


def summarize(rows, records):
    checks = {check: {"field": FIELDS[check], **rates([r for r in rows if r["check"] == check]),
                     "usage": usage([r for r in records if r.get("check") == check])} for check in COUNTS}
    conditions = {"WORD": [("word", .85)], "SENTENCE": [("omission", .90), ("conjunction", .85)],
                  "TEXT": [("style", .90)]}
    gates = {}
    for level, requirements in conditions.items():
        values = {check: {"threshold": threshold, "pass_rate": checks[check]["pass_rate"],
            "passed": checks[check]["n_completed"] == COUNTS[check] and checks[check]["pass_rate"] >= threshold}
            for check, threshold in requirements}
        gates[level] = {"checks": values, "passed": all(v["passed"] for v in values.values())}
    omissions = [r for r in rows if r["check"] == "omission" and both_ok(r) is not None]
    true_rows = [r for r in omissions if r["subject_omitted"]]
    confirmed = [r for r in true_rows if both_ok(r)]
    diagnostic = {}
    for label, group in (("predicted_true", true_rows), ("both_runs_confirmed_true", confirmed)):
        different = [r["sentence_id"] for r in group if r["llm_judgment_1"]["referent_type"] != r["llm_judgment_2"]["referent_type"]]
        diagnostic[label] = {"n": len(group), "by_run": {str(run): dict(Counter(r[f"llm_judgment_{run}"]["referent_type"] for r in group)) for run in (1, 2)},
            "consensus_counts": dict(Counter(r["llm_judgment_1"]["referent_type"] for r in group if r["sentence_id"] not in different)),
            "agreement": (len(group) - len(different)) / len(group) if group else None, "disagreement_ids": different}
    return {"label": "LLM-verified; not human accuracy", "judge_model": "gpt-6.1-sol", "reasoning_effort": "high",
        "checks": checks, "omission_subgroups": {str(value): rates([r for r in rows if r["check"] == "omission" and r["subject_omitted"] == value]) for value in (True, False)},
        "referent_type_diagnostic_only": diagnostic, "usage": usage(records), "phase3_gates": gates,
        "polarity_modality": {"source": "Phase 2b unchanged patterns.py", "both_true": 45, "n": 49, "pass_rate": 45 / 49, "passed": True},
        "level_weights": {"WORD": 1, "SENTENCE": 1, "TEXT": 1}, "phase3_started": False,
        "agreement_definition": "Boolean verdict equality; referent category agreement reported separately",
        "human_ok_filled": sum(r.get("human_ok") is not None for r in rows)}
