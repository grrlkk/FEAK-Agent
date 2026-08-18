#!/usr/bin/env python
"""Run two independent API-backed blind reviews and evaluate the proxy gate.

The reviewers only receive the public A/B form.  The hidden answer key is read
after both reviews (and optional blind re-review of disagreements) are done.
Completed rows are checkpointed so interrupted runs can resume without paying
for the same review twice.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.corruption.g2 import evaluate_two_human_reviews
from feak_tc.mvp.llm import LLMResponseError, LLMUnavailable, request_json


DEFAULT_MODEL_ONE = "gpt-5-mini-2025-08-07"
DEFAULT_MODEL_TWO = "gpt-4.1-2025-04-14"
ALLOWED_PREFERENCES = {"A", "B", "TIE"}
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "preference": {"type": "string", "enum": ["A", "B", "TIE"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "notes": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": ["preference", "confidence", "notes"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """당신은 한국어 논술 품질을 평가하는 독립 블라인드 검수자다.
두 글 중 주어진 질문에 더 잘 답하고, 더 자연스럽고, 논리적이며, 응집성 있고,
불필요한 반복이나 주제 이탈이 적은 글을 고른다.

규칙:
- A/B가 어떤 방식으로 만들어졌는지 추측하지 말고 보이는 글만 평가한다.
- 글의 길이만으로 판단하지 않는다.
- 질문 적합성, 문장 유창성, 논리적 연결, 구성, 중복/잡음 순으로 종합한다.
- A가 명확히 낫다면 A, B가 명확히 낫다면 B를 선택한다.
- 실질적인 품질 차이가 없을 때만 TIE를 사용한다.
- notes에는 결정적인 근거를 한국어 한두 문장으로 간결하게 적는다.
- 다른 평가자나 숨은 정답은 존재하지 않는다고 가정하고 독립적으로 판단한다.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rater-one", required=True)
    parser.add_argument("--rater-two", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--disagreements-out", required=True)
    parser.add_argument("--adjudication-out", required=True)
    parser.add_argument("--model-one", default=DEFAULT_MODEL_ONE)
    parser.add_argument("--model-two", default=DEFAULT_MODEL_TWO)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument(
        "--skip-adjudication",
        action="store_true",
        help="Leave disagreements unresolved instead of blind re-reviewing them.",
    )
    args = parser.parse_args()

    if args.model_one == args.model_two:
        raise SystemExit("the two reviewers must use different model identifiers")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    rater_one_path = Path(args.rater_one)
    rater_two_path = Path(args.rater_two)
    key_path = Path(args.key)
    report_path = Path(args.report_out)
    disagreements_path = Path(args.disagreements_out)
    adjudication_path = Path(args.adjudication_out)

    first_rows = _read_jsonl(rater_one_path)
    second_rows = _read_jsonl(rater_two_path)
    _validate_same_blind_forms(first_rows, second_rows)

    _review_form(
        rows=first_rows,
        path=rater_one_path,
        model=args.model_one,
        reviewer_id="LLM-R1",
        workers=args.workers,
        max_attempts=args.max_attempts,
        timeout=args.timeout,
    )
    _review_form(
        rows=second_rows,
        path=rater_two_path,
        model=args.model_two,
        reviewer_id="LLM-R2",
        workers=args.workers,
        max_attempts=args.max_attempts,
        timeout=args.timeout,
    )

    # Determine conflicts only from the two public forms.  The hidden key is
    # deliberately not loaded before all model-facing work is complete.
    conflicts = _find_disagreements(first_rows, second_rows)
    _write_jsonl_atomic(disagreements_path, conflicts)
    adjudication_rows: list[dict[str, Any]] = []
    if conflicts and not args.skip_adjudication:
        adjudication_rows = _blind_readjudicate(
            conflicts=conflicts,
            public_rows=first_rows,
            path=adjudication_path,
            models=(args.model_one, args.model_two),
            workers=args.workers,
            max_attempts=args.max_attempts,
            timeout=args.timeout,
        )
    elif adjudication_path.exists():
        adjudication_rows = _read_jsonl(adjudication_path)

    key_rows = _read_jsonl(key_path)
    base_report, unresolved = evaluate_two_human_reviews(
        first_rows,
        second_rows,
        key_rows,
        adjudication_rows=adjudication_rows,
        threshold=args.threshold,
    )
    both_models_pass = bool(
        base_report["rater_one"]["meets_threshold"]
        and base_report["rater_two"]["meets_threshold"]
    )
    proxy_status = (
        "passed"
        if base_report["status"] == "passed" and both_models_pass
        else "pending_adjudication"
        if base_report["status"] == "pending_adjudication"
        else "failed"
    )
    report = {
        "gate": "G2_TWO_LLM_API_BLIND_REVIEW_PROXY",
        "status": proxy_status,
        "human_gate_status": "not_performed",
        "is_human_review": False,
        "models": {
            "rater_one": args.model_one,
            "rater_two": args.model_two,
        },
        "policy": {
            "threshold": args.threshold,
            "require_both_individual_models": True,
            "adjudication": "fresh blind re-review alternating the two models",
            "key_excluded_from_all_model_inputs": True,
        },
        **base_report,
        "status": proxy_status,
    }
    _write_json_atomic(report_path, report)
    _write_jsonl_atomic(disagreements_path, unresolved)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if proxy_status == "passed" else 2


def _review_form(
    *,
    rows: list[dict[str, Any]],
    path: Path,
    model: str,
    reviewer_id: str,
    workers: int,
    max_attempts: int,
    timeout: float,
) -> None:
    pending = [row for row in rows if not _has_valid_decision(row, model)]
    print(
        f"{reviewer_id}: model={model} complete={len(rows) - len(pending)} "
        f"pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return

    lock = threading.Lock()
    row_by_id = {str(row["pair_id"]): row for row in rows}
    completed = len(rows) - len(pending)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _request_decision,
                row,
                model=model,
                max_attempts=max_attempts,
                timeout=timeout,
            ): str(row["pair_id"])
            for row in pending
        }
        for future in as_completed(futures):
            pair_id = futures[future]
            decision = future.result()
            with lock:
                target = row_by_id[pair_id]
                target.update(decision)
                target["rater_id"] = reviewer_id
                target["review_kind"] = "openai_api_blind"
                target["model"] = model
                completed += 1
                _write_jsonl_atomic(path, rows)
                print(
                    f"{reviewer_id}: {completed}/{len(rows)} {pair_id} "
                    f"preference={decision['preference']}",
                    flush=True,
                )


def _blind_readjudicate(
    *,
    conflicts: Sequence[Mapping[str, Any]],
    public_rows: Sequence[Mapping[str, Any]],
    path: Path,
    models: tuple[str, str],
    workers: int,
    max_attempts: int,
    timeout: float,
) -> list[dict[str, Any]]:
    existing = _read_jsonl(path) if path.exists() else []
    by_id = {str(row["pair_id"]): dict(row) for row in existing}
    public_by_id = {str(row["pair_id"]): row for row in public_rows}
    pending = [
        conflict
        for conflict in conflicts
        if str(conflict["pair_id"]) not in by_id
        or str(by_id[str(conflict["pair_id"])].get("adjudicated_preference", ""))
        not in ALLOWED_PREFERENCES
    ]
    print(
        f"adjudication: complete={len(conflicts) - len(pending)} "
        f"pending={len(pending)}",
        flush=True,
    )

    def submit_one(conflict: Mapping[str, Any]) -> dict[str, Any]:
        pair_id = str(conflict["pair_id"])
        # Alternate which of the same two reviewer models supplies the fresh
        # tiebreak sample.  It sees neither prior preferences nor notes.
        model = models[sum(ord(char) for char in pair_id) % 2]
        decision = _request_decision(
            public_by_id[pair_id],
            model=model,
            max_attempts=max_attempts,
            timeout=timeout,
        )
        return {
            "pair_id": pair_id,
            "adjudicated_preference": decision["preference"],
            "notes": decision["notes"],
            "confidence": decision["confidence"],
            "model": model,
            "review_kind": "openai_api_fresh_blind_readjudication",
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(submit_one, row): row for row in pending}
        for future in as_completed(futures):
            result = future.result()
            by_id[result["pair_id"]] = result
            ordered = [by_id[str(row["pair_id"])] for row in conflicts if str(row["pair_id"]) in by_id]
            _write_jsonl_atomic(path, ordered)
            print(
                f"adjudication: {len(ordered)}/{len(conflicts)} "
                f"{result['pair_id']} preference={result['adjudicated_preference']}",
                flush=True,
            )
    return [by_id[str(row["pair_id"])] for row in conflicts]


def _request_decision(
    row: Mapping[str, Any],
    *,
    model: str,
    max_attempts: int,
    timeout: float,
    requester: Callable[..., dict[str, Any]] = request_json,
) -> dict[str, Any]:
    user_prompt = (
        f"pair_id: {row['pair_id']}\n"
        f"질문: {row.get('question', '')}\n\n"
        f"[글 A]\n{row['text_a']}\n\n"
        f"[글 B]\n{row['text_b']}\n\n"
        "A, B, TIE 중 하나와 판단 근거를 JSON으로 답하라."
    )
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw = requester(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                model=model,
                temperature=None,
                reasoning_effort="low" if model.startswith("gpt-5") else None,
                verbosity="low" if model.startswith("gpt-5") else None,
                json_schema=DECISION_SCHEMA,
                response_format_name="blind_writing_preference",
                timeout=timeout,
            )
            return _validate_decision(raw)
        except (LLMUnavailable, LLMResponseError, ValueError) as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(
        f"review failed for {row.get('pair_id')} with {model} "
        f"after {max_attempts} attempts: {last_error}"
    )


def _validate_decision(raw: Mapping[str, Any]) -> dict[str, Any]:
    preference = str(raw.get("preference", "")).strip().upper()
    if preference not in ALLOWED_PREFERENCES:
        raise ValueError(f"invalid preference: {preference!r}")
    confidence = int(raw.get("confidence", -1))
    if not 0 <= confidence <= 100:
        raise ValueError(f"invalid confidence: {confidence}")
    notes = str(raw.get("notes", "")).strip()
    if not notes:
        raise ValueError("notes must not be empty")
    return {
        "preference": preference,
        "confidence": confidence,
        "notes": notes[:500],
    }


def _has_valid_decision(row: Mapping[str, Any], model: str) -> bool:
    return (
        str(row.get("preference", "")).strip().upper() in ALLOWED_PREFERENCES
        and row.get("model") == model
        and row.get("review_kind") == "openai_api_blind"
    )


def _find_disagreements(
    first_rows: Sequence[Mapping[str, Any]],
    second_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    second = {str(row["pair_id"]): row for row in second_rows}
    conflicts = []
    for first in first_rows:
        pair_id = str(first["pair_id"])
        second_row = second[pair_id]
        first_answer = str(first.get("preference", "")).strip().upper()
        second_answer = str(second_row.get("preference", "")).strip().upper()
        if first_answer != second_answer:
            conflicts.append(
                {
                    "pair_id": pair_id,
                    "rater_one_preference": first_answer,
                    "rater_two_preference": second_answer,
                    "adjudicated_preference": "",
                    "notes": "",
                }
            )
    return conflicts


def _validate_same_blind_forms(
    first_rows: Sequence[Mapping[str, Any]],
    second_rows: Sequence[Mapping[str, Any]],
) -> None:
    if len(first_rows) != len(second_rows) or not first_rows:
        raise ValueError("rater forms must contain the same non-zero number of rows")
    first = {str(row["pair_id"]): row for row in first_rows}
    second = {str(row["pair_id"]): row for row in second_rows}
    if set(first) != set(second):
        raise ValueError("rater forms contain different pair IDs")
    public_fields = ("essay_id", "question", "text_a", "text_b")
    for pair_id in first:
        for field in public_fields:
            if first[pair_id].get(field) != second[pair_id].get(field):
                raise ValueError(f"rater forms differ at {pair_id}.{field}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
