import json
from collections import Counter

from feak_tc.rv.judge import (
    LABEL_FIELDS,
    analyze_judgments,
    build_blind_packets,
    fisher_exact_two_sided,
    fleiss_kappa,
    request_judgment,
    review_packet_digest,
    wilson_interval,
)
from scripts.evaluate_rv_pilot_llm_judges import _review_model


WEAK_LABELS = {
    "wrong_target": ("fail", "pass", "partial", "fail"),
    "over_edit": ("pass", "partial", "fail", "partial"),
}


def test_blind_packets_are_balanced_and_hide_answer_fields():
    rows = _pilot_rows()

    public, key = build_blind_packets(rows, sample_states=8, seed=17)

    assert len(public) == len(key) == 8
    assert Counter((row["corruption_type"], row["stage_k"]) for row in public) == {
        (operator, stage): 1
        for operator in ("DELETE", "OFFTOPIC", "REPEAT", "SHUFFLE")
        for stage in (1, 2)
    }
    forbidden = {
        "candidate_type",
        "candidate_source",
        "expected_labels",
        *LABEL_FIELDS,
    }
    assert all(not forbidden.intersection(row) for row in public)
    assert all("candidate_type" in row["candidate_a"] for row in key)
    assert all(row["reference_repair"].endswith("correct") for row in public)


def test_request_judgment_never_sends_hidden_key():
    public, _ = build_blind_packets(_pilot_rows(), sample_states=1, seed=1)
    captured = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return _judgment("wrong_target", WEAK_LABELS["wrong_target"])

    result = request_judgment(
        public[0],
        model="test-model",
        max_attempts=1,
        timeout=1,
        requester=fake_request,
    )

    assert result["candidate_a"]["confidence"] == 80
    assert "expected_labels" not in captured["user"]
    assert "candidate_source" not in captured["user"]
    assert public[0]["candidate_a_text"] in captured["user"]
    assert review_packet_digest(public[0]) == review_packet_digest(dict(public[0]))
    changed = dict(public[0], candidate_a_text="changed")
    assert review_packet_digest(public[0]) != review_packet_digest(changed)


def test_model_cache_is_invalidated_when_packet_digest_changes(tmp_path, monkeypatch):
    public, _ = build_blind_packets(_pilot_rows(), sample_states=1, seed=2)
    judgment = _judgment("wrong_target", WEAK_LABELS["wrong_target"])
    path = tmp_path / "judge.jsonl"
    path.write_text(
        json.dumps(
            {
                "review_id": public[0]["review_id"],
                "model": "test-model",
                "review_kind": "openai_api_independent_blind",
                "public_packet_sha256": "stale",
                **judgment,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    calls = []

    def fake_request(*args, **kwargs):
        calls.append(kwargs)
        return judgment

    monkeypatch.setattr(
        "scripts.evaluate_rv_pilot_llm_judges.request_judgment", fake_request
    )

    rows = _review_model(
        public_rows=public,
        path=path,
        model_config={"name": "test", "id": "test-model"},
        workers=1,
        max_attempts=1,
        timeout=1,
    )
    assert len(calls) == 1
    assert rows[0]["public_packet_sha256"] == review_packet_digest(public[0])

    _review_model(
        public_rows=public,
        path=path,
        model_config={"name": "test", "id": "test-model"},
        workers=1,
        max_attempts=1,
        timeout=1,
    )
    assert len(calls) == 1


def test_analysis_scores_majority_and_agreement():
    public, key = build_blind_packets(_pilot_rows(), sample_states=1, seed=3)
    expected = key[0]
    results = {}
    for model_index, model in enumerate(("m1", "m2", "m3")):
        row = {"review_id": public[0]["review_id"]}
        for side in ("candidate_a", "candidate_b"):
            candidate_type = expected[side]["candidate_type"]
            labels = tuple(expected[side]["expected_labels"][field] for field in LABEL_FIELDS)
            row[side] = _candidate_judgment(candidate_type, labels)
        if model_index == 2:
            row["candidate_a"]["inferred_candidate_type"] = "other"
        results[model] = [row]

    report, disagreements = analyze_judgments(results, key)

    assert report["reviewed_candidates"] == 2
    assert report["majority_vote"]["intended_type"]["accuracy_on_decided"] == 1.0
    assert report["majority_vote"]["by_candidate_type"]["wrong_target"][
        "intended_type_exact_rate"
    ] == 1.0
    assert list(
        report["majority_vote"]["by_candidate_type"]["wrong_target"][
            "intended_type_by_stratum"
        ].values()
    )[0]["exact"] == 1
    assert report["per_model"]["m3"]["overall"]["intended_type_accuracy"] == 0.5
    assert report["per_model"]["m3"]["inferred_type_confusion"]["wrong_target"] == {
        "other": 1
    }
    assert report["per_model"]["m1"]["inferred_type_classification"]["wrong_target"][
        "precision"
    ] == 1.0
    assert report["inter_rater_agreement"]["three_way_unanimity"]["inferred_candidate_type"] == 0.5
    assert report["inter_rater_agreement"]["pairwise_chance_diagnostics"]["m1__m2"][
        "usable_for_weak_supervision"
    ]["observed_agreement"] == 1.0
    assert len(disagreements) == 1
    assert fleiss_kappa([["a", "a", "a"], ["b", "b", "b"]]) == 1.0


def test_binomial_interval_and_fisher_exact_match_reference_values():
    assert wilson_interval(14, 50) == {"low": 0.174742, "high": 0.416651}
    assert wilson_interval(39, 50) == {"low": 0.647585, "high": 0.872461}
    assert fisher_exact_two_sided(3, 10, 36, 1) == 2.8538e-07


def _pilot_rows():
    rows = []
    index = 0
    for operator in ("DELETE", "OFFTOPIC", "REPEAT", "SHUFFLE"):
        for stage in (1, 2):
            for duplicate in range(2):
                state_id = f"state-{index}"
                common = {
                    "dataset_version": "test",
                    "essay_id": f"essay-{index}",
                    "chain_id": f"chain-{index}",
                    "state_id": state_id,
                    "stage_k": stage,
                    "question": "질문",
                    "before_text": f"{state_id} current",
                    "target_rubric": "content_1",
                    "intended_action": "ADD_DETAIL",
                    "intent": "RESTORE_DETAIL",
                    "corruption_type": operator,
                    "changed_spans": [{"operation": "delete", "target_span": "근거"}],
                }
                correct = {
                    **common,
                    "sample_id": f"{state_id}:correct_repair",
                    "candidate_type": "correct_repair",
                    "after_text": f"{state_id} correct",
                    **_label_dict(("pass", "pass", "pass", "pass")),
                }
                rows.append(correct)
                for candidate_type in ("wrong_target", "over_edit"):
                    rows.append(
                        {
                            **common,
                            "sample_id": f"{state_id}:{candidate_type}",
                            "candidate_type": candidate_type,
                            "after_text": f"{state_id} {candidate_type}",
                            **_label_dict(WEAK_LABELS[candidate_type]),
                        }
                    )
                index += 1
    return rows


def _judgment(candidate_type, labels):
    candidate = _candidate_judgment(candidate_type, labels)
    return {"candidate_a": dict(candidate), "candidate_b": dict(candidate)}


def _candidate_judgment(candidate_type, labels):
    return {
        "inferred_candidate_type": candidate_type,
        **_label_dict(labels),
        "usable_for_weak_supervision": True,
        "confidence": 80,
        "notes": "판정 근거",
    }


def _label_dict(values):
    return dict(zip(LABEL_FIELDS, values))
