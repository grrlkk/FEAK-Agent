"""LLM-verified agreement and conservative token accounting, never human accuracy."""

from .ec_verification import FIELDS, both_runs_ok, validate_judgment


def token_usage(records, *, input_rate=2.0, output_rate=10.0):
    calls = [row for row in records if row.get("stage") == "ec_judgment" and "event" not in row]
    usage = [row["usage"] for row in calls if row.get("usage") is not None]
    inputs = sum(row["input_tokens"] for row in usage)
    outputs = sum(row["output_tokens"] for row in usage)
    reasoning = sum((row.get("output_tokens_details") or {}).get("reasoning_tokens", 0) for row in usage)
    n = len(usage)
    return {"request_records": len(calls), "calls_with_usage": n, "calls_without_usage": len(calls) - n,
            "input_tokens": inputs, "output_tokens": outputs, "reasoning_tokens": reasoning,
            "mean_input_tokens_per_call": inputs / n if n else None,
            "mean_output_tokens_per_call": outputs / n if n else None,
            "mean_reasoning_tokens_per_call": reasoning / n if n else None,
            "cost_usd": (inputs * input_rate + outputs * output_rate) / 1_000_000,
            "cost_basis": "user-specified rates, no cached-input discount; reasoning is included in output",
            "input_usd_per_million": input_rate, "output_usd_per_million": output_rate}


def pilot_projection(rows, records, *, total_sentences=100, cost_limit=20.0):
    if len(rows) != 10 or any(both_runs_ok(row) is None for row in rows):
        raise ValueError("Finish both judgments of all ten pilot sentences before projecting")
    ids = {row["sentence_id"] for row in rows}
    result = token_usage([record for record in records if record.get("sentence_id") in ids])
    if result["calls_with_usage"] < 20 or result["calls_without_usage"]:
        raise ValueError("Pilot usage is incomplete; refusing to authorize the remaining calls")
    projected = result["cost_usd"] * total_sentences / len(rows)
    return {**result, "pilot_sentences": len(rows), "projected_total_sentences": total_sentences,
            "projected_total_usd": projected, "cost_limit_usd": cost_limit,
            "continue_allowed": projected < cost_limit}


def agreement(rows):
    finished = [row for row in rows if both_runs_ok(row) is not None]
    sentence_agree, token_agree, token_count, llm_ok_tokens = 0, 0, 0, 0
    field_agree = {field: 0 for field in FIELDS}
    field_both_true = {field: 0 for field in FIELDS}
    disagreements = []
    for row in finished:
        judgments = []
        for run in (1, 2):
            result = validate_judgment({"tokens": row[f"llm_judgment_{run}"]["tokens"]}, row)
            judgments.append({token.token_id: token for token in result.tokens})
        differences = []
        for token_id, a in judgments[0].items():
            b = judgments[1][token_id]
            token_count += 1
            diff = [field for field in FIELDS if getattr(a, field) != getattr(b, field)]
            token_agree += not diff
            llm_ok_tokens += all(getattr(a, field) and getattr(b, field) for field in FIELDS)
            for field in FIELDS:
                field_agree[field] += getattr(a, field) == getattr(b, field)
                field_both_true[field] += getattr(a, field) and getattr(b, field)
            if diff:
                differences.append({"token_id": token_id, "fields": diff})
        sentence_agree += not differences
        if differences:
            disagreements.append({"sentence_id": row["sentence_id"], "differences": differences})
    n = len(finished)
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    return {"label": "LLM-verified (not human accuracy)", "sentences_total": len(rows),
            "sentences_completed": n, "ec_tokens": token_count,
            "sentence_agreement_count": sentence_agree, "sentence_agreement_rate": ratio(sentence_agree, n),
            "token_agreement_count": token_agree, "token_agreement_rate": ratio(token_agree, token_count),
            "llm_ok_sentences": sum(both_runs_ok(row) for row in finished),
            "llm_ok_rate": ratio(sum(both_runs_ok(row) for row in finished), n),
            "llm_ok_tokens": llm_ok_tokens, "llm_ok_token_rate": ratio(llm_ok_tokens, token_count),
            "fields": {field: {"token_agreement_count": field_agree[field],
                "token_agreement_rate": ratio(field_agree[field], token_count),
                "both_true_tokens": field_both_true[field], "llm_ok_token_rate": ratio(field_both_true[field], token_count),
                "both_true_sentences": sum(both_runs_ok(row, (field,)) for row in finished),
                "llm_ok_sentence_rate": ratio(sum(both_runs_ok(row, (field,)) for row in finished), n)}
                for field in FIELDS},
            "disagreements": disagreements,
            "disagreement_sentence_ids": [item["sentence_id"] for item in disagreements]}
