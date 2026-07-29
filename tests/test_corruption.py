import random

import pytest
import yaml

from feak_tc.corruption import OPERATOR_SPECS, generate_chain
from feak_tc.corruption.measure import evaluate_chain
from feak_tc.corruption.normalize import normalize_text
from feak_tc.corruption.operators import (
    _inject_grammar_error,
    build_payload_schema,
    build_prompt,
    generate_rule_payload,
    parse_and_apply,
    validate_normalized_corruption,
)


VALIDITY = {
    "min_ratio_vs_source": 0.6,
    "max_ratio_vs_source": 1.8,
    "insertion_min_chars": 10,
    "insertion_max_chars": 250,
    "grammar_min_token_retention": 0.6,
}

ESSAY = (
    "인권이란 모든 인간이 태어나면서부터 가지는 기본적인 권리이다. "
    "예를 들어 표현의 자유와 거주 이전의 자유가 이에 해당한다. "
    "또한 헌법 제10조는 인간의 존엄과 가치를 명시하고 있다. "
    "인권은 국가가 함부로 제한할 수 없다. "
    "장애인 이동권 시위는 인권 보장의 실제 사례로 자주 언급된다. "
    "우리는 서로의 인권을 존중하며 살아야 한다."
)


def _spec(operator: str, edits: int = 1) -> dict:
    return {**OPERATOR_SPECS[operator], "edits_per_step": edits}


def test_v2_operator_mapping_is_exact():
    assert {
        operator: spec["target_rubric"]
        for operator, spec in OPERATOR_SPECS.items()
    } == {
        "DELETE_SPECIFICS": "content_2",
        "SHUFFLE_FLOW": "organization_1",
        "INSERT_OFFTOPIC": "organization_2",
        "INJECT_LEX_REPEAT": "expression_1",
        "INJECT_GRAMMAR_ERR": "expression_2",
    }
    assert "INFLATE_REDUNDANCY" not in OPERATOR_SPECS
    assert all(spec["preserve_constraints"] for spec in OPERATOR_SPECS.values())


@pytest.mark.parametrize("operator", sorted(OPERATOR_SPECS))
def test_operator_prompts_include_preserve_but_not_rubric_or_feature_targets(operator):
    spec = _spec(operator)
    prompt = build_prompt(operator, ESSAY, "인권을 설명하세요.", spec, "conservative")
    assert "[보존 조건]" in prompt
    assert spec["target_rubric"] not in prompt
    assert "target_rubric" not in prompt
    assert "NN_repRatio" not in prompt
    assert "topicConsistency" not in prompt


def test_delete_specifics_only_deletes_exact_spans():
    spans = [
        "예를 들어 표현의 자유와 거주 이전의 자유가 이에 해당한다.",
        "장애인 이동권 시위는 인권 보장의 실제 사례로 자주 언급된다.",
    ]
    new_text, edits = parse_and_apply(
        "DELETE_SPECIFICS",
        {"target_spans": spans},
        ESSAY,
        ESSAY,
        VALIDITY,
        _spec("DELETE_SPECIFICS", edits=2),
    )
    assert all(span not in new_text for span in spans)
    assert [edit["operation"] for edit in edits] == ["delete", "delete"]
    assert "인권은 국가가 함부로 제한할 수 없다." in new_text


def test_shuffle_flow_preserves_sentence_text():
    moved = "인권이란 모든 인간이 태어나면서부터 가지는 기본적인 권리이다."
    anchor = "인권은 국가가 함부로 제한할 수 없다."
    new_text, edits = parse_and_apply(
        "SHUFFLE_FLOW",
        {"moves": [{"moved_span": moved, "anchor_span": anchor}]},
        ESSAY,
        ESSAY,
        VALIDITY,
        _spec("SHUFFLE_FLOW"),
    )
    assert new_text.index(anchor) < new_text.index(moved)
    assert new_text.count(moved) == 1
    assert edits[0]["operation"] == "move_after"


def test_insert_offtopic_preserves_existing_text():
    anchor = "인권은 국가가 함부로 제한할 수 없다."
    insertion = "어제는 날씨가 좋아서 공원에 다녀왔다."
    new_text, _ = parse_and_apply(
        "INSERT_OFFTOPIC",
        {"edits": [{"anchor_span": anchor, "insertion": insertion}]},
        ESSAY,
        ESSAY,
        VALIDITY,
        _spec("INSERT_OFFTOPIC"),
    )
    assert anchor + " " + insertion in new_text
    assert new_text.replace(" " + insertion, "") == ESSAY


def test_inject_lex_repeat_requires_repeated_word_and_keeps_anchor():
    anchor = "인권은 국가가 함부로 제한할 수 없다."
    repetition = "인권, 인권, 인권은 반드시 중요하게 생각해야 한다."
    new_text, _ = parse_and_apply(
        "INJECT_LEX_REPEAT",
        {"edits": [{"anchor_span": anchor, "repetition": repetition}]},
        ESSAY,
        ESSAY,
        VALIDITY,
        _spec("INJECT_LEX_REPEAT"),
    )
    assert anchor + " " + repetition in new_text
    with pytest.raises(ValueError):
        parse_and_apply(
            "INJECT_LEX_REPEAT",
            {"edits": [{"anchor_span": anchor, "repetition": "인권은 중요하게 생각해야 한다."}]},
            ESSAY,
            ESSAY,
            VALIDITY,
            _spec("INJECT_LEX_REPEAT"),
        )


def test_inject_grammar_error_preserves_numbers_and_content_tokens():
    span = "또한 헌법 제10조는 인간의 존엄과 가치를 명시하고 있다."
    replacement = "또한 헌법 제10조는는 인간의 존엄과 가치를 명시하고 있다."
    new_text, _ = parse_and_apply(
        "INJECT_GRAMMAR_ERR",
        {"edits": [{"target_span": span, "replacement": replacement}]},
        ESSAY,
        ESSAY,
        VALIDITY,
        _spec("INJECT_GRAMMAR_ERR"),
    )
    assert replacement in new_text
    with pytest.raises(ValueError):
        parse_and_apply(
            "INJECT_GRAMMAR_ERR",
            {"edits": [{"target_span": span, "replacement": "헌법은 전혀 다른 사실이다."}]},
            ESSAY,
            ESSAY,
            VALIDITY,
            _spec("INJECT_GRAMMAR_ERR"),
        )


def test_rule_grammar_injects_three_distinct_verifiable_error_types():
    spec = _spec("INJECT_GRAMMAR_ERR", edits=3)
    payload = generate_rule_payload(
        "INJECT_GRAMMAR_ERR",
        ESSAY,
        spec,
        random.Random(7),
    )
    assert [edit["error_type"] for edit in payload["edits"]] == [
        "particle_swap",
        "spacing",
        "spelling_typo",
    ]
    assert len({edit["target_span"] for edit in payload["edits"]}) == 3

    corrupted, edits = parse_and_apply(
        "INJECT_GRAMMAR_ERR",
        payload,
        ESSAY,
        ESSAY,
        VALIDITY,
        spec,
    )
    assert [edit["error_type"] for edit in edits] == [
        "particle_swap",
        "spacing",
        "spelling_typo",
    ]
    validate_normalized_corruption("INJECT_GRAMMAR_ERR", edits, corrupted)

    restored_first = corrupted.replace(
        edits[0]["text"],
        edits[0]["target_span"],
        1,
    )
    with pytest.raises(ValueError, match="removed or rewrote"):
        validate_normalized_corruption(
            "INJECT_GRAMMAR_ERR",
            edits,
            restored_first,
        )


def test_rule_grammar_does_not_mistake_adnominal_ending_for_particle():
    sentence = (
        "가장 기억에 남는 것은 작년 여름 할머니와 화상통화를 했을 때의 일이다."
    )
    corrupted = _inject_grammar_error(sentence, "particle_swap")
    assert corrupted is not None
    assert "남은 것은" not in corrupted
    assert "화상통화을" in corrupted
    assert _inject_grammar_error(
        "그래서 우리는 앞으로도 계속 노력해야 한다.",
        "spelling_typo",
    ).startswith("그랫서")


def test_payload_schema_is_strict_and_uses_exact_edit_count():
    schema = build_payload_schema(
        "INSERT_OFFTOPIC",
        _spec("INSERT_OFFTOPIC", edits=2),
    )
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["edits"]
    assert schema["properties"]["edits"]["minItems"] == 2
    assert schema["properties"]["edits"]["maxItems"] == 2
    assert schema["properties"]["edits"]["items"]["additionalProperties"] is False


@pytest.mark.parametrize("operator", sorted(OPERATOR_SPECS))
def test_every_operator_has_rule_generator(operator):
    payload = generate_rule_payload(operator, ESSAY, _spec(operator), random.Random(7))
    new_text, edits = parse_and_apply(
        operator,
        payload,
        ESSAY,
        ESSAY,
        VALIDITY,
        _spec(operator),
    )
    assert new_text != ESSAY
    assert edits


def test_uniform_normalizer_rejects_meaning_rewrite(monkeypatch):
    monkeypatch.setattr(
        "feak_tc.corruption.normalize.request_json",
        lambda **kwargs: {"normalized_text": "완전히 다른 짧은 글이다."},
    )
    with pytest.raises(RuntimeError):
        normalize_text(
            ESSAY,
            {
                "enabled": True,
                "max_attempts": 1,
                "min_similarity": 0.75,
                "min_length_ratio": 0.8,
                "max_length_ratio": 1.2,
            },
        )


def test_uniform_normalizer_retries_when_corruption_is_repaired(monkeypatch):
    spec = _spec("INJECT_GRAMMAR_ERR")
    payload = generate_rule_payload(
        "INJECT_GRAMMAR_ERR",
        ESSAY,
        spec,
        random.Random(7),
    )
    corrupted, edits = parse_and_apply(
        "INJECT_GRAMMAR_ERR",
        payload,
        ESSAY,
        ESSAY,
        VALIDITY,
        spec,
    )
    responses = iter(
        [
            {"normalized_text": ESSAY},
            {"normalized_text": corrupted},
        ]
    )
    monkeypatch.setattr(
        "feak_tc.corruption.normalize.request_json",
        lambda **kwargs: next(responses),
    )
    normalized, metadata = normalize_text(
        corrupted,
        {
            "enabled": True,
            "max_attempts": 2,
            "min_similarity": 0.75,
            "min_length_ratio": 0.8,
            "max_length_ratio": 1.2,
        },
        post_validate=lambda candidate: validate_normalized_corruption(
            "INJECT_GRAMMAR_ERR",
            edits,
            candidate,
        ),
    )
    assert normalized == corrupted
    assert metadata["attempts"] == 2
    assert metadata["postcondition_checked"] is True
    assert "removed or rewrote" in metadata["errors"][0]


def test_uniform_normalizer_falls_back_to_validated_input_if_every_output_repairs_error(
    monkeypatch,
):
    spec = _spec("INJECT_GRAMMAR_ERR")
    payload = generate_rule_payload(
        "INJECT_GRAMMAR_ERR",
        ESSAY,
        spec,
        random.Random(7),
    )
    corrupted, edits = parse_and_apply(
        "INJECT_GRAMMAR_ERR",
        payload,
        ESSAY,
        ESSAY,
        VALIDITY,
        spec,
    )
    monkeypatch.setattr(
        "feak_tc.corruption.normalize.request_json",
        lambda **kwargs: {"normalized_text": ESSAY},
    )
    normalized, metadata = normalize_text(
        corrupted,
        {
            "enabled": True,
            "max_attempts": 2,
            "fallback_to_input_on_postcondition_failure": True,
            "min_similarity": 0.75,
            "min_length_ratio": 0.8,
            "max_length_ratio": 1.2,
        },
        post_validate=lambda candidate: validate_normalized_corruption(
            "INJECT_GRAMMAR_ERR",
            edits,
            candidate,
        ),
    )
    assert normalized == corrupted
    assert metadata["normalized"] is False
    assert metadata["generator"] == "identity_postcondition_fallback"
    assert metadata["postcondition_checked"] is True
    assert len(metadata["errors"]) == 2


def test_generate_chain_records_v2_labels_preservation_and_normalization(monkeypatch):
    payloads = iter(
        [
            {"target_spans": [
                "예를 들어 표현의 자유와 거주 이전의 자유가 이에 해당한다.",
            ]},
            {"edits": [{
                "anchor_span": "인권은 국가가 함부로 제한할 수 없다.",
                "insertion": "요즘 유행하는 드라마는 정말 재미있다.",
            }]},
            {"moves": [{
                "moved_span": "우리는 서로의 인권을 존중하며 살아야 한다.",
                "anchor_span": "인권이란 모든 인간이 태어나면서부터 가지는 기본적인 권리이다.",
            }]},
        ]
    )
    monkeypatch.setattr(
        "feak_tc.corruption.chain.request_json",
        lambda **kwargs: next(payloads),
    )
    def fake_normalize(text, cfg, post_validate=None):
        if post_validate is not None:
            post_validate(text)
        return (
            text,
            {"normalized": True, "changed": False, "generator": "test", "attempts": 1},
        )

    monkeypatch.setattr("feak_tc.corruption.chain.normalize_text", fake_normalize)
    cfg = {
        "seed": 1,
        "depth": 3,
        "llm": {"max_attempts": 1},
        "normalization": {"enabled": True},
        "generation": {"modes": ["llm:conservative"]},
        "validity": VALIDITY,
        "operators": {
            name: {**spec, "edits_per_step": 1}
            for name, spec in OPERATOR_SPECS.items()
        },
    }
    rng = random.Random(0)
    monkeypatch.setattr(
        rng,
        "sample",
        lambda seq, k: ["DELETE_SPECIFICS", "INSERT_OFFTOPIC", "SHUFFLE_FLOW"],
    )
    chain = generate_chain(
        {"record_id": "r1", "question": "인권이란?", "text": ESSAY},
        cfg,
        rng=rng,
    )
    assert chain["status"] == "ok"
    assert len(chain["states"]) == 4
    assert len(chain["normalizations"]) == 4
    assert [step["target_rubric"] for step in chain["steps"]] == [
        "content_2",
        "organization_2",
        "organization_1",
    ]
    assert all(step["preserve_constraints"] for step in chain["steps"])
    assert all(step["target_features"] == [] for step in chain["steps"])


def test_grammar_operator_override_forces_rule_generator(monkeypatch):
    monkeypatch.setattr(
        "feak_tc.corruption.chain.request_json",
        lambda **kwargs: pytest.fail("grammar override must not call the LLM"),
    )
    cfg = {
        "seed": 1,
        "depth": 1,
        "llm": {
            "model": "gpt-5-mini-2025-08-07",
            "reasoning": {"effort": "minimal"},
            "text": {"verbosity": "low"},
        },
        "normalization": {"enabled": False},
        "generation": {"modes": ["llm:student_natural"]},
        "validity": VALIDITY,
        "operators": {
            "INJECT_GRAMMAR_ERR": {
                **OPERATOR_SPECS["INJECT_GRAMMAR_ERR"],
                "generation_modes": ["rule"],
                "edits_per_step": 3,
            }
        },
    }
    chain = generate_chain(
        {"record_id": "grammar-rule", "question": "인권이란?", "text": ESSAY},
        cfg,
    )
    assert chain["status"] == "ok"
    assert chain["steps"][0]["generator"] == "rule"
    assert chain["steps"][0]["model"] is None
    assert chain["steps"][0]["normalization"]["postcondition_checked"] is True


def test_invalid_llm_payload_falls_back_to_same_rule_operator(monkeypatch):
    monkeypatch.setattr(
        "feak_tc.corruption.chain.request_json",
        lambda **kwargs: {"target_spans": ["원문에 없는 문장이다."]},
    )
    cfg = {
        "seed": 1,
        "depth": 1,
        "llm": {"max_attempts": 2, "fallback_to_rule": True},
        "normalization": {"enabled": False},
        "generation": {"modes": ["llm:student_natural"]},
        "validity": VALIDITY,
        "operators": {
            "DELETE_SPECIFICS": {
                **OPERATOR_SPECS["DELETE_SPECIFICS"],
                "edits_per_step": 1,
            }
        },
    }
    chain = generate_chain(
        {"record_id": "fallback", "question": "인권이란?", "text": ESSAY},
        cfg,
    )
    assert chain["status"] == "ok"
    step = chain["steps"][0]
    assert step["generator"] == "rule_fallback"
    assert step["requested_generator"] == "llm:student_natural"
    assert step["fallback"] is True
    assert len(step["errors"]) == 2


def test_measurement_accepts_each_strict_target_drop_independently():
    with open("configs/schema.yaml", encoding="utf-8") as file:
        schema = yaml.safe_load(file)
    features = {key: 0.0 for key in schema["features"]["keys"]}
    base_scores = [5.0] * 8
    after_first = list(base_scores)
    after_first[2] = 4.8  # content_2 decreases
    after_second = list(after_first)
    after_second[5] = 5.1  # organization_2 improves, so reject
    measurements = {
        ("r1", 0): _measurement(schema, base_scores, features),
        ("r1", 1): _measurement(schema, after_first, features),
        ("r1", 2): _measurement(schema, after_second, {**features, "NN_repRatio": 999.0}),
        ("r1", 3): _measurement(schema, [1.0] * 8, features),
    }
    chain = {
        "record_id": "r1",
        "states": ["x0", "x1", "x2", "x3"],
        "steps": [
            _measured_step("DELETE_SPECIFICS", "content_2"),
            _measured_step("INSERT_OFFTOPIC", "organization_2"),
            _measured_step("SHUFFLE_FLOW", "organization_1"),
        ],
    }
    rows = evaluate_chain(
        chain,
        measurements,
        schema,
        {"score_basis": "rf_corrected", "target_drop_min": 0.0},
    )
    assert [row["accepted"] for row in rows] == [True, False, True]
    assert rows[0]["target_drop"] == pytest.approx(0.2)
    assert rows[1]["acceptance_reason"] == "target_rubric_not_decreased"
    assert rows[2]["acceptance_reason"] == "target_rubric_decreased"
    assert rows[1]["target_features"] == []


def _measurement(schema: dict, rf_scores: list[float], features: dict) -> dict:
    return {
        "rubrics": {
            key: round(value)
            for key, value in zip(schema["rubrics"]["keys"], rf_scores)
        },
        "rf_corrected": rf_scores,
        "features": features,
    }


def _measured_step(operator: str, target: str) -> dict:
    spec = OPERATOR_SPECS[operator]
    return {
        "corruption_op": operator,
        "target_rubric": target,
        "generator": "rule",
        "normalized": True,
        "normalization": {"normalized": True},
        "reverse_action": spec["reverse_action"],
        "intent": spec["intent"],
        "preserve_constraints": list(spec["preserve_constraints"]),
        "edits": [{"operation": "test"}],
    }
