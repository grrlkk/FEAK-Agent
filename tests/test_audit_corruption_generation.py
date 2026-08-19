from typing import Optional

from scripts.audit_corruption_generation import audit_generated_chains


def test_generated_chain_audit_passes_clean_balanced_rows():
    chains = [
        _chain("r1", "DELETE_SPECIFICS"),
        _chain("r2", "SHUFFLE_FLOW"),
        _chain("r3", "INSERT_OFFTOPIC", "서로 다른 삽입 문장 하나입니다."),
        _chain("r4", "INJECT_LEX_REPEAT", "고유하게 반복되는 표현을 더한 문장입니다."),
    ]
    cfg = {
        "artifact_audit": {"min_distinct_essays": 3},
        "balance": {"max_operator_fraction": 0.4},
    }

    report = audit_generated_chains(chains, cfg)

    assert report["passed"] is True
    assert report["chains"] == 4
    assert report["transitions"] == 4
    assert report["preservation_failures"] == 0


def test_generated_chain_audit_rejects_failed_preservation():
    chain = _chain("r1", "DELETE_SPECIFICS")
    chain["steps"][0]["preservation_check"] = {"passed": False}

    report = audit_generated_chains(
        [chain],
        {"artifact_audit": {}, "balance": {"enabled": False}},
    )

    assert report["passed"] is False
    assert report["preservation_failures"] == 1


def test_generated_chain_audit_reports_nonblocking_balance():
    chains = [
        _chain("r1", "INSERT_OFFTOPIC", "첫 번째 고유 문장입니다."),
        _chain("r2", "INSERT_OFFTOPIC", "두 번째 고유 문장입니다."),
        _chain("r3", "DELETE_SPECIFICS"),
    ]
    report = audit_generated_chains(
        chains,
        {
            "artifact_audit": {"enabled": False},
            "balance": {"max_operator_fraction": 0.4, "fail_pipeline": False},
        },
    )
    assert report["generated_operator_balance"]["passed"] is False
    assert report["passed"] is True


def test_generated_chain_audit_allows_only_validated_partial_prefixes_when_enabled():
    partial = _chain("r1", "DELETE_SPECIFICS")
    partial["status"] = "partial"

    strict = audit_generated_chains(
        [partial],
        {"artifact_audit": {"enabled": False}, "balance": {"enabled": False}},
    )
    allowed = audit_generated_chains(
        [partial],
        {"artifact_audit": {"enabled": False}, "balance": {"enabled": False}},
        allow_partial=True,
    )

    assert strict["passed"] is False
    assert strict["malformed_chains"] == 1
    assert allowed["passed"] is True
    assert allowed["malformed_chains"] == 0
    assert allowed["chain_statuses"] == {"partial": 1}


def _chain(record_id: str, operator: str, inserted_text: Optional[str] = None) -> dict:
    edits = []
    if inserted_text is not None:
        edits.append({"operation": "insert_after", "text": inserted_text})
    return {
        "record_id": record_id,
        "status": "ok",
        "states": ["전", "후"],
        "steps": [
            {
                "operator": operator,
                "edits": edits,
                "preservation_check": {"passed": True},
                "fallback": False,
                "normalized": False,
            }
        ],
    }
