import random

import pytest
import yaml

from feak_tc.corruption import (
    MAIN_CHAIN_OPERATORS,
    OPERATOR_SPECS,
    generate_chain,
    generate_surface_sample,
)
from feak_tc.corruption.measure import evaluate_chain
from feak_tc.corruption.normalize import normalize_text
from feak_tc.corruption.operators import (
    LLM_ONLY_OPERATORS,
    RULE_UNSUPPORTED_OPERATORS,
    _inject_grammar_error,
    build_payload_schema,
    build_prompt,
    generate_rule_payload,
    parse_and_apply,
    validate_normalized_corruption,
    validate_operator_preservation,
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
    assert set(MAIN_CHAIN_OPERATORS) == {
        "DELETE_SPECIFICS",
        "SHUFFLE_FLOW",
        "INSERT_OFFTOPIC",
        "INJECT_LEX_REPEAT",
    }
    assert "INJECT_GRAMMAR_ERR" not in MAIN_CHAIN_OPERATORS


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
    check = validate_operator_preservation(
        "DELETE_SPECIFICS",
        ESSAY,
        new_text,
        edits,
    )
    assert check["method"] == "exact_target_span_subtraction"

    with pytest.raises(ValueError, match="outside target spans"):
        validate_operator_preservation(
            "DELETE_SPECIFICS",
            ESSAY,
            new_text.replace("존중하며", "존중하지 않으며"),
            edits,
        )


def test_shuffle_flow_preserves_sentence_text():
    moved = "또한 헌법 제10조는 인간의 존엄과 가치를 명시하고 있다."
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
    check = validate_operator_preservation(
        "SHUFFLE_FLOW",
        ESSAY,
        new_text,
        edits,
    )
    assert check["method"] == "sentence_multiset_and_discourse_dependency"
    assert check["dependency_breaks"][0]["markers"] == ["또한"]


def test_shuffle_rule_builds_two_effective_moves():
    spec = _spec("SHUFFLE_FLOW", edits=2)
    payload = generate_rule_payload(
        "SHUFFLE_FLOW",
        ESSAY,
        spec,
        random.Random(1),
    )
    assert len(payload["moves"]) == 2
    new_text, edits = parse_and_apply(
        "SHUFFLE_FLOW",
        payload,
        ESSAY,
        ESSAY,
        VALIDITY,
        spec,
    )
    check = validate_operator_preservation(
        "SHUFFLE_FLOW",
        ESSAY,
        new_text,
        edits,
    )
    assert len(check["dependency_breaks"]) == 2


def test_structural_rules_only_target_source_sentences_after_insertion():
    inserted = "개성은 개성은 개성은 중요하다고 생각한다."
    current = ESSAY.replace(
        "인권은 국가가 함부로 제한할 수 없다.",
        "인권은 국가가 함부로 제한할 수 없다. " + inserted,
    )
    for operator in ("DELETE_SPECIFICS", "SHUFFLE_FLOW"):
        spec = _spec(operator, edits=2)
        payload = generate_rule_payload(
            operator,
            current,
            spec,
            random.Random(1),
            source_text=ESSAY,
        )
        _, edits = parse_and_apply(
            operator,
            payload,
            current,
            ESSAY,
            VALIDITY,
            spec,
        )
        assert all(edit["target_span"] in ESSAY for edit in edits)
        assert all(edit["target_span"] != inserted for edit in edits)


def test_shuffle_flow_rejects_ineffective_or_rewritten_moves():
    moved = "인권은 국가가 함부로 제한할 수 없다."
    anchor = "우리는 서로의 인권을 존중하며 살아야 한다."
    new_text, edits = parse_and_apply(
        "SHUFFLE_FLOW",
        {"moves": [{"moved_span": moved, "anchor_span": anchor}]},
        ESSAY,
        ESSAY,
        VALIDITY,
        _spec("SHUFFLE_FLOW"),
    )
    implicit = validate_operator_preservation(
        "SHUFFLE_FLOW",
        ESSAY,
        new_text,
        edits,
    )
    assert implicit["dependency_breaks"][0]["markers"] == ["implicit_adjacency"]

    dependent = "또한 헌법 제10조는 인간의 존엄과 가치를 명시하고 있다."
    moved_text, dependent_edits = parse_and_apply(
        "SHUFFLE_FLOW",
        {"moves": [{"moved_span": dependent, "anchor_span": anchor}]},
        ESSAY,
        ESSAY,
        VALIDITY,
        _spec("SHUFFLE_FLOW"),
    )
    with pytest.raises(ValueError, match="sentence text or membership changed"):
        validate_operator_preservation(
            "SHUFFLE_FLOW",
            ESSAY,
            moved_text.replace("살아야 한다.", "살아가야 한다."),
            dependent_edits,
        )


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


@pytest.mark.parametrize(
    ("anchor", "insertion", "message"),
    [
        (
            "핵심 키워드: 인권, 존엄",
            "우주에는 아직 밝혀지지 않은 현상이 많이 남아 있다.",
            "must not be metadata",
        ),
        (
            "인권은 국가가 함부로 제한할 수 없다.",
            "인권은 국가가 함부로 제한할 수 없다. 우주에는 별이 많다.",
            "exactly one sentence",
        ),
        (
            "인권은 국가가 함부로 제한할 수 없다.",
            "인권은 국가가 함부로 제한할 수 없다.",
            "already exists|contain its anchor",
        ),
    ],
)
def test_insert_offtopic_rejects_metadata_anchor_and_copied_context(
    anchor, insertion, message
):
    essay = ESSAY + " 핵심 키워드: 인권, 존엄"
    with pytest.raises(ValueError, match=message):
        parse_and_apply(
            "INSERT_OFFTOPIC",
            {"edits": [{"anchor_span": anchor, "insertion": insertion}]},
            essay,
            essay,
            VALIDITY,
            _spec("INSERT_OFFTOPIC"),
        )


def test_insert_offtopic_rejects_source_eight_gram_overlap():
    anchor = "인권은 국가가 함부로 제한할 수 없다."
    insertion = (
        "모든 인간이 태어나면서부터 가지는 기본적인 권리이다 예를 들어 표현의 자유는 중요하다."
    )
    with pytest.raises(ValueError, match="overlaps source text at 8-gram"):
        parse_and_apply(
            "INSERT_OFFTOPIC",
            {"edits": [{"anchor_span": anchor, "insertion": insertion}]},
            ESSAY,
            ESSAY,
            VALIDITY,
            _spec("INSERT_OFFTOPIC"),
        )


def test_lex_repeat_allows_only_numbers_already_present_in_source():
    essay = (
        "형광등은 실내를 밝히는 도구이다. "
        "가게에서는 20W 형광등을 판매한다. "
        "소비자는 방 크기에 맞는 제품을 선택한다. "
        "적절한 조명은 눈의 피로를 줄인다."
    )
    anchor = "가게에서는 20W 형광등을 판매한다."
    allowed = "20W 형광등은 20W 형광등은 20W 형광등은 중요하다."
    new_text, _ = parse_and_apply(
        "INJECT_LEX_REPEAT",
        {"edits": [{"anchor_span": anchor, "repetition": allowed}]},
        essay,
        essay,
        VALIDITY,
        {**_spec("INJECT_LEX_REPEAT"), "edits_per_step": 1},
    )
    assert allowed in new_text

    introduced = "21W 형광등은 21W 형광등은 21W 형광등은 중요하다."
    with pytest.raises(ValueError, match="must not introduce numeric facts"):
        parse_and_apply(
            "INJECT_LEX_REPEAT",
            {"edits": [{"anchor_span": anchor, "repetition": introduced}]},
            essay,
            essay,
            VALIDITY,
            {**_spec("INJECT_LEX_REPEAT"), "edits_per_step": 1},
        )


def test_lex_repeat_rejects_text_beyond_operator_length_limit():
    anchor = "인권은 국가가 함부로 제한할 수 없다."
    repetition = "권리는 " * 25 + "중요하다."
    with pytest.raises(ValueError, match="lexical repetition length"):
        parse_and_apply(
            "INJECT_LEX_REPEAT",
            {"edits": [{"anchor_span": anchor, "repetition": repetition}]},
            ESSAY,
            ESSAY,
            VALIDITY,
            {
                **_spec("INJECT_LEX_REPEAT"),
                "edits_per_step": 1,
                "repetition_max_chars": 60,
            },
        )


def test_lex_repeat_schema_requires_terminal_punctuation():
    schema = build_payload_schema(
        "INJECT_LEX_REPEAT",
        {
            **_spec("INJECT_LEX_REPEAT"),
            "repetition_min_chars": 15,
            "repetition_max_chars": 60,
        },
    )
    repetition = schema["properties"]["edits"]["items"]["properties"]["repetition"]
    assert repetition == {
        "type": "string",
        "minLength": 15,
        "maxLength": 60,
        "pattern": r"다[.]$",
    }


def test_shuffle_rule_avoids_sentence_fragment_from_double_punctuation():
    essay = (
        "첫 문장은 문제를 소개한다. "
        "두 번째 문장은 잘못된 이중 마침표를 가진다.. "
        "세 번째 문장은 원인을 설명한다. "
        "네 번째 문장은 해결 방법을 제안한다. "
        "다섯 번째 문장은 결론을 제시한다. "
        "여섯 번째 문장은 의미를 정리한다."
    )
    spec = {**_spec("SHUFFLE_FLOW"), "edits_per_step": 2}
    payload = generate_rule_payload(
        "SHUFFLE_FLOW",
        essay,
        spec,
        random.Random(7),
        source_text=essay,
    )
    moved = {item["moved_span"] for item in payload["moves"]}
    assert "두 번째 문장은 잘못된 이중 마침표를 가진다." not in moved
    new_text, edits = parse_and_apply(
        "SHUFFLE_FLOW",
        payload,
        essay,
        essay,
        VALIDITY,
        spec,
    )
    assert validate_operator_preservation(
        "SHUFFLE_FLOW", essay, new_text, edits
    )["passed"] is True


def test_llm_insertion_anchor_is_repaired_to_an_exact_source_sentence():
    insertion = "어제는 날씨가 좋아서 공원에 다녀왔다."
    new_text, edits = parse_and_apply(
        "INSERT_OFFTOPIC",
        {
            "edits": [
                {
                    "anchor_span": "인권은 국가가 함부로 제한할 수 있습니다.",
                    "insertion": insertion,
                }
            ]
        },
        ESSAY,
        ESSAY,
        VALIDITY,
        _spec("INSERT_OFFTOPIC"),
    )
    edit = edits[0]
    assert edit["anchor_repaired"] is True
    assert edit["requested_target_span"].endswith("있습니다.")
    assert edit["target_span"] in ESSAY
    assert f'{edit["target_span"]} {insertion}' in new_text


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


@pytest.mark.parametrize(
    "operator",
    sorted(set(OPERATOR_SPECS) - set(RULE_UNSUPPORTED_OPERATORS)),
)
def test_rule_capable_operator_has_rule_generator(operator):
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


@pytest.mark.parametrize("operator", RULE_UNSUPPORTED_OPERATORS)
def test_v5_non_rule_operators_reject_rule_generation(operator):
    with pytest.raises(ValueError, match="no rule generator"):
        generate_rule_payload(operator, ESSAY, _spec(operator), random.Random(7))


def test_delete_specifics_rejects_repeated_content_confound():
    repeated = (
        "예를 들어 세계은행은 개발도상국에 금융 지원을 한다 "
        "예를 들어 세계은행은 개발도상국에 금융 지원을 한다."
    )
    essay = f"도입 문장이다. {repeated} 결론 문장이다. 마지막 문장이다."
    with pytest.raises(ValueError, match="target contains repeated 5-gram"):
        parse_and_apply(
            "DELETE_SPECIFICS",
            {"target_spans": [repeated]},
            essay,
            essay,
            {**VALIDITY, "delete_repeated_ngram_size": 5},
            _spec("DELETE_SPECIFICS", edits=1),
        )


def test_delete_specifics_cannot_remove_prior_corruption_insertions():
    source = "첫 문장이다. 원래의 구체적인 사례 문장이다. 결론 문장이다. 마지막 문장이다."
    inserted = "오늘 아침에 본 드라마가 재미있었다."
    current = source.replace("첫 문장이다.", f"첫 문장이다. {inserted}")
    with pytest.raises(ValueError, match="must originate in the source text"):
        parse_and_apply(
            "DELETE_SPECIFICS",
            {"target_spans": [inserted]},
            current,
            source,
            VALIDITY,
            _spec("DELETE_SPECIFICS", edits=1),
        )


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
                "moved_span": "또한 헌법 제10조는 인간의 존엄과 가치를 명시하고 있다.",
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


def test_grammar_operator_is_chain_external_surface_sample(monkeypatch):
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
    assert chain["status"] == "failed"
    assert chain["steps"] == []
    assert "no main-chain" in chain["failure_errors"][0]

    cfg["surface_validation"] = {
        "operator": "INJECT_GRAMMAR_ERR",
        "generation_mode": "rule",
        "edits_per_sample": 3,
    }
    sample = generate_surface_sample(
        {"record_id": "grammar-rule", "question": "인권이란?", "text": ESSAY},
        cfg,
    )
    assert sample["usage"] == "surface_correction_validation_only"
    assert sample["surface_input_text"] != sample["clean_text"]
    assert len(sample["edits"]) == 3


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


def test_v4_llm_only_operator_never_falls_back_to_rule(monkeypatch):
    monkeypatch.setattr(
        "feak_tc.corruption.chain.request_json",
        lambda **kwargs: {"edits": []},
    )
    cfg = {
        "seed": 1,
        "depth": 1,
        "llm": {"max_attempts": 2, "fallback_to_rule": True},
        "normalization": {"enabled": False},
        "generation": {"modes": ["rule"]},
        "validity": VALIDITY,
        "operators": {
            "INJECT_LEX_REPEAT": {
                **OPERATOR_SPECS["INJECT_LEX_REPEAT"],
                "generation_modes": ["llm:student_natural"],
                "fallback_to_rule": False,
                "edits_per_step": 1,
            }
        },
    }
    chain = generate_chain(
        {"record_id": "no-template-fallback", "question": "인권이란?", "text": ESSAY},
        cfg,
    )
    assert chain["status"] == "failed"
    assert chain["steps"] == []
    assert len(chain["failure_errors"]) == 2
    assert all("rule_fallback" not in error for error in chain["failure_errors"])


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
    assert [row["transition_id"] for row in rows] == [
        "r1:g1:stage1",
        "r1:g1:stage2",
        "r1:g1:stage3",
    ]
    assert rows[1]["acceptance_reason"] == "target_rubric_not_decreased"
    assert rows[2]["acceptance_reason"] == "target_rubric_decreased"
    assert rows[1]["target_features"] == []


def test_measurement_supports_operator_specific_target_drop_minimums():
    with open("configs/schema.yaml", encoding="utf-8") as file:
        schema = yaml.safe_load(file)
    features = {key: 0.0 for key in schema["features"]["keys"]}
    before = [5.0] * 8
    after = list(before)
    after[6] = 4.6
    chain = {
        "record_id": "operator-threshold",
        "states": ["clean", "repeated"],
        "steps": [_measured_step("INJECT_LEX_REPEAT", "expression_1")],
    }
    rows = evaluate_chain(
        chain,
        {
            ("operator-threshold", 0): _measurement(schema, before, features),
            ("operator-threshold", 1): _measurement(schema, after, features),
        },
        schema,
        {
            "score_basis": "rf_corrected",
            "target_drop_min": 0.225,
            "target_drop_min_by_operator": {"INJECT_LEX_REPEAT": 0.5},
        },
    )
    assert rows[0]["target_drop"] == pytest.approx(0.4)
    assert rows[0]["target_drop_min"] == 0.5
    assert rows[0]["accepted"] is False


def test_measurement_rejects_delete_that_improves_repetition_metrics():
    with open("configs/schema.yaml", encoding="utf-8") as file:
        schema = yaml.safe_load(file)
    before_features = {key: 0.0 for key in schema["features"]["keys"]}
    before_features.update({"NN_repRatio": 0.40, "lemma_MATTR": 0.50})
    after_features = {
        **before_features,
        "NN_repRatio": 0.38,
        "lemma_MATTR": 0.53,
    }
    before_scores = [5.0] * 8
    after_scores = list(before_scores)
    after_scores[2] = 4.0
    chain = {
        "record_id": "delete-confound",
        "states": ["clean", "corrupted"],
        "steps": [_measured_step("DELETE_SPECIFICS", "content_2")],
    }
    rows = evaluate_chain(
        chain,
        {
            ("delete-confound", 0): _measurement(
                schema, before_scores, before_features
            ),
            ("delete-confound", 1): _measurement(
                schema, after_scores, after_features
            ),
        },
        schema,
        {
            "score_basis": "rf_corrected",
            "target_drop_min": 0.5,
            "delete_confound_guard": {
                "enabled": True,
                "nn_rep_ratio_reduction_min": 0.01,
                "lemma_mattr_gain_min": 0.02,
                "max_non_target_rubric_gain": 0.3,
            },
        },
    )
    assert rows[0]["accepted"] is False
    assert rows[0]["acceptance_reason"] == "delete_specifics_cross_axis_improvement"
    assert rows[0]["quality_checks"]["delete_confound"]["flagged"] is True


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
