"""LLM-verified rates, per-field denominators and pre-registered Phase 3 gates."""

from .ec_metrics import token_usage
from .structure_verification import applicable_fields, both_ok


def usage(records):
    # The accounting implementation counts reasoning inside, not in addition to,
    # output tokens and does not assume any cached-input discount.
    return token_usage([{**r, "stage": "ec_judgment"} for r in records if r.get("stage") == "structure_judgment"])


def pilot_projection(rows, records):
    chosen = [r for r in rows if r["pilot"]]
    if len(chosen) != 15 or any(both_ok(r) is None for r in chosen):
        raise ValueError("Finish both runs of five items per check before projecting cost")
    details, projected = {}, 0
    for check, total in (("WORD", 80), ("SENTENCE", 120), ("TEXT", 80)):
        subset = [r for r in chosen if r["check"] == check]
        if len(subset) != 5:
            raise ValueError("Pilot must contain five items at each level")
        ids = {r["item_id"] for r in subset}
        data = usage([r for r in records if r.get("item_id") in ids])
        if data["calls_with_usage"] != 10 or data["calls_without_usage"]:
            raise ValueError("Pilot cost accounting incomplete or duplicated")
        data["projected_total_usd"] = data["cost_usd"] * total / 5
        projected += data["projected_total_usd"]
        details[check] = data
    ids = {r["item_id"] for r in chosen}
    return {"by_check": details, "usage": usage([r for r in records if r.get("item_id") in ids]),
            "projected_total_usd": projected, "cost_limit_usd": 15,
            "continue_allowed": projected < 15}


def rates(rows):
    completed = [r for r in rows if both_ok(r) is not None]
    fields = sorted({f for r in rows for f in applicable_fields(r)})
    details = {}
    for field in fields:
        eligible = [r for r in completed if field in applicable_fields(r)]
        different = [r["sentence_id"] for r in eligible if r["llm_judgment_1"][field] != r["llm_judgment_2"][field]]
        n = len(eligible)
        passed = sum(all(r[f"llm_judgment_{run}"][field] is True for run in (1, 2)) for r in eligible)
        details[field] = {"n": n, "both_true": passed, "pass_rate": passed / n if n else None,
                          "agreement": (n - len(different)) / n if n else None,
                          "disagreement_ids": different}
    different = [r["sentence_id"] for r in completed if any(
        r["llm_judgment_1"][f] != r["llm_judgment_2"][f] for f in applicable_fields(r))]
    n = len(completed)
    return {"n_requested": len(rows), "n_completed": n, "both_true": sum(both_ok(r) for r in completed),
            "pass_rate": sum(both_ok(r) for r in completed) / n if n else None,
            "agreement": (n - len(different)) / n if n else None,
            "disagreement_ids": different, "fields": details}


def summarize(rows, records):
    checks = {check: {**rates([r for r in rows if r["check"] == check]),
                      "usage": usage([r for r in records if r.get("check") == check])}
              for check in ("WORD", "SENTENCE", "TEXT")}
    subgroups = {group: rates([r for r in rows if r["subgroup"] == group])
                for group in ("HIGH", "LOW", "conjunction")}
    requirements = {
        "WORD": [(checks["WORD"], "relation_ok", .85), (checks["WORD"], "polarity_modality_ok", .85)],
        "SENTENCE": [(subgroups["HIGH"], "antecedent_ok", .70), (subgroups["conjunction"], "conjunction_relation_ok", .85)],
        "TEXT": [(checks["TEXT"], "style_ok", .90)],
    }
    gates = {}
    for level, requirements_for_level in requirements.items():
        fields = {}
        for data, field, threshold in requirements_for_level:
            value = data["fields"].get(field, {}).get("pass_rate")
            complete = data["n_completed"] == data["n_requested"]
            fields[field] = {"threshold": threshold, "pass_rate": value,
                "passed": complete and value is not None and value >= threshold}
        gates[level] = {"fields": fields, "passed": all(v["passed"] for v in fields.values())}
    return {"label": "LLM-verified; not human accuracy", "judge_model": "gpt-6.1-sol",
            "reasoning_effort": "high", "checks": checks, "sentence_subgroups": subgroups,
            "usage": usage(records), "phase3_gates": gates, "phase3_started": False,
            "agreement_definition": "Equality of applicable Boolean verdicts; notes/correction wording ignored",
            "human_ok_filled": sum(r.get("human_ok") is not None for r in rows)}
