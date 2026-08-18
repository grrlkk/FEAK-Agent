from scripts.build_corruption_rulev5_dataset import build_rulev5_pool


def test_rulev5_pool_drops_current_and_downstream_old_offtopic():
    chains = [
        {
            "record_id": "r1",
            "steps": [
                {"operator": "DELETE_SPECIFICS"},
                {"operator": "INSERT_OFFTOPIC"},
                {"operator": "SHUFFLE_FLOW"},
            ],
        },
        {"record_id": "r2", "steps": [{"operator": "INJECT_LEX_REPEAT"}]},
    ]
    original = [
        _row("r1", 1, "DELETE_SPECIFICS"),
        _row("r1", 2, "INSERT_OFFTOPIC"),
        _row("r1", 3, "SHUFFLE_FLOW"),
        _row("r2", 1, "INJECT_LEX_REPEAT"),
    ]
    replacement = [_row("r1", 2, "INSERT_OFFTOPIC", drop=0.7)]
    replacement[0]["edits"] = [
        {
            "operation": "insert_after",
            "text": "화산재는 바람을 타고 넓은 지역까지 퍼질 수 있다.",
        }
    ]
    cfg = {
        "artifact_audit": {"enabled": False},
        "semantic_artifact_audit": {"enabled": False},
        "balance": {"enabled": True, "max_operator_fraction": 0.5},
    }

    final, report = build_rulev5_pool(chains, original, replacement, cfg)

    assert report["passed"] is True
    assert report["clean_prefix_retained"] == 2
    assert report["old_current_or_downstream_offtopic_dropped"] == 2
    assert [(row["essay_id"], row["stage_k"]) for row in final] == [
        ("r1", 1),
        ("r1", 2),
        ("r2", 1),
    ]


def test_rulev5_pool_rejects_context_dependent_retrieval_sentence():
    chains = [
        {"record_id": "r1", "steps": [{"operator": "INSERT_OFFTOPIC"}]},
        {"record_id": "r2", "steps": [{"operator": "INJECT_LEX_REPEAT"}]},
    ]
    original = [_row("r2", 1, "INJECT_LEX_REPEAT")]
    replacement = [_row("r1", 1, "INSERT_OFFTOPIC", drop=0.7)]
    replacement[0]["edits"] = [
        {
            "operation": "insert_after",
            "text": "그러면 전쟁을 하는 나라의 인구도 급격히 줄어든다.",
        }
    ]
    cfg = {
        "artifact_audit": {"enabled": False},
        "semantic_artifact_audit": {"enabled": False},
        "balance": {"enabled": False, "max_operator_fraction": 1.0},
    }

    final, report = build_rulev5_pool(chains, original, replacement, cfg)

    assert final == original
    assert report["replacement_offtopic_standalone_rejected"] == 1


def _row(essay_id: str, stage: int, operator: str, drop: float = 0.5) -> dict:
    return {
        "essay_id": essay_id,
        "chain_id": f"{essay_id}:g1",
        "stage_k": stage,
        "corruption_op": operator,
        "target_drop": drop,
        "accepted": True,
        "edits": [],
    }
