"""Action-type-stratified candidate proposal for MVP."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from feak_tc.diagnose import Diagnosis
from feak_tc.diagnose.constants import ACTION_TYPES, RUBRIC_KEYS, RUBRIC_NAMES_KO

from .llm import LLMResponseError, LLMUnavailable, request_json
from .schemas import Candidate
from .targeting import rank_target_spans, select_target_span


_ACTION_INSTRUCTIONS = {
    "ADD_DETAIL": "핵심 주장과 직접 관련된 구체 사례나 설명을 한 문장 추가한다.",
    "DELETE_OR_FOCUS": "주제와 직접 관련이 약한 문장을 삭제하거나 초점을 맞춘 표현으로 바꾼다.",
    "COMPRESS": "반복되거나 장황한 표현을 의미 손실 없이 압축한다.",
    "RESTRUCTURE": "문장 순서나 연결 표현을 조정해 흐름을 분명하게 한다.",
    "STYLE_REFINE": "어색한 어휘, 맞춤법, 문장 표현을 자연스럽게 다듬는다.",
    "STOP": "현재 상태에서 추가 수정하지 않는다.",
}

PROPOSAL_ACTION_TYPES = tuple(action_type for action_type in ACTION_TYPES if action_type != "STOP")

_RUBRIC_TO_ACTION_HINT = {
    "task_1": "ADD_DETAIL",
    "content_1": "ADD_DETAIL",
    "content_2": "ADD_DETAIL",
    "content_3": "DELETE_OR_FOCUS",
    "organization_1": "RESTRUCTURE",
    "organization_2": "DELETE_OR_FOCUS",
    "expression_1": "STYLE_REFINE",
    "expression_2": "STYLE_REFINE",
}


def propose(
    diag: Diagnosis,
    n_per_action: int = 1,
    cfg: Optional[Mapping[str, Any]] = None,
) -> list[Candidate]:
    """Generate action-stratified, schema-valid candidates.

    `proposer.mode` controls the implementation:
    - deterministic: no external API, stable fallback.
    - llm: require a JSON LLM response.
    - auto: try LLM, fill/fallback deterministically on failure.
    """

    proposer_cfg = _section(cfg, "proposer")
    mode = str(proposer_cfg.get("mode", "deterministic"))
    if mode not in {"deterministic", "llm", "auto"}:
        raise ValueError(f"Unknown proposer mode: {mode}")

    if mode in {"llm", "auto"}:
        try:
            return _propose_with_llm(diag, n_per_action=n_per_action, cfg=proposer_cfg)
        except (LLMUnavailable, LLMResponseError, ValueError, RuntimeError) as exc:
            if mode == "llm":
                raise
            fallback = _propose_deterministic(diag, n_per_action=n_per_action)
            for cand in fallback:
                cand.metadata["llm_error"] = str(exc)
            return fallback

    return _propose_deterministic(diag, n_per_action=n_per_action)


def _propose_deterministic(diag: Diagnosis, n_per_action: int = 1) -> list[Candidate]:
    """Generate deterministic fallback candidates."""

    candidates: list[Candidate] = []
    target_order = list(diag.weak_rubrics) or list(RUBRIC_KEYS[:3])

    for action_type in PROPOSAL_ACTION_TYPES:
        preferred = [rubric for rubric in target_order if _RUBRIC_TO_ACTION_HINT.get(rubric) == action_type]
        target_rubric = (preferred or target_order)[0]
        target_hints = rank_target_spans(
            diag.text,
            target_rubric,
            action_type=action_type,
            limit=max(3, n_per_action),
            question=_diagnosis_question(diag),
        )
        for rank in range(max(1, n_per_action)):
            hint = target_hints[min(rank, len(target_hints) - 1)] if target_hints else {}
            target_span = str(hint.get("span", "")) if hint else ""
            candidates.append(
                Candidate(
                    action_type=action_type,
                    target_rubric=target_rubric,
                    target_span=target_span,
                    instruction=_build_instruction(action_type, target_rubric),
                    metadata={
                        "source": "deterministic_proposer",
                        "target_hints": target_hints,
                        "candidate_rank": rank,
                    },
                )
            )
    return candidates


def _propose_with_llm(
    diag: Diagnosis,
    n_per_action: int,
    cfg: Mapping[str, Any],
) -> list[Candidate]:
    payload = request_json(
        system=(
            "You are a Korean essay revision planner. Return only a JSON object. "
            "Do not rewrite the essay. Produce local, reversible revision candidates."
        ),
        user=_build_llm_prompt(diag, n_per_action=n_per_action),
        model=str(cfg.get("model", "gpt-4o-mini")),
        temperature=float(cfg.get("temperature", 0.2)),
        env_file=cfg.get("env_file"),
        timeout=float(cfg["timeout"]) if cfg.get("timeout") is not None else None,
    )
    candidates = _parse_candidate_payload(payload, diag, n_per_action=n_per_action)
    if not candidates:
        raise LLMResponseError("LLM produced no valid candidates.")

    # Keep non-STOP action coverage explicit. Missing actions are filled by
    # deterministic candidates so the counterfactual comparison has a complete
    # edit slate while stopping remains a controller decision.
    deterministic = _propose_deterministic(diag, n_per_action=n_per_action)
    by_action: dict[str, list[Candidate]] = {action: [] for action in PROPOSAL_ACTION_TYPES}
    for cand in candidates:
        if cand.action_type in by_action:
            by_action[cand.action_type].append(cand)
    for cand in deterministic:
        if len(by_action[cand.action_type]) < max(1, n_per_action):
            cand.metadata["source"] = "deterministic_fill"
            by_action[cand.action_type].append(cand)

    ordered: list[Candidate] = []
    for action_type in PROPOSAL_ACTION_TYPES:
        ordered.extend(by_action[action_type][: max(1, n_per_action)])
    return ordered


def _build_llm_prompt(diag: Diagnosis, n_per_action: int) -> str:
    weak_features = _diagnostic_feature_slice(diag)
    target_hints = _target_hint_payload(diag)
    schema = {
        "candidates": [
            {
                "action_type": "ADD_DETAIL",
                "target_rubric": "content_2",
                "target_span": "exact substring from essay",
                "instruction": "one local revision instruction in Korean",
            }
        ]
    }
    return "\n".join(
        [
            "Return JSON matching this schema:",
            json.dumps(schema, ensure_ascii=False),
            "",
            f"Create {n_per_action} candidate(s) for each non-STOP action_type:",
            ", ".join(PROPOSAL_ACTION_TYPES),
            "Rules:",
            "- Do not create STOP candidates; stopping is decided later by the controller.",
            "- target_span must be an exact contiguous substring from the essay.",
            "- instruction must be local and reversible; do not request a full rewrite.",
            "- Prefer weak rubrics and the related diagnostic features.",
            "- Follow the Korean action definition for each action_type.",
            "",
            "[Action definitions]",
            json.dumps(_proposal_action_definitions(), ensure_ascii=False),
            "",
            "[Rubric keys]",
            json.dumps(RUBRIC_NAMES_KO, ensure_ascii=False),
            "",
            "[Essay]",
            diag.text,
            "",
            "[Rubric scores]",
            json.dumps(diag.rubrics, ensure_ascii=False),
            "",
            "[Weak rubrics]",
            json.dumps(diag.weak_rubrics, ensure_ascii=False),
            "",
            "[Related features]",
            json.dumps(weak_features, ensure_ascii=False),
            "",
            "[Suggested target spans]",
            json.dumps(target_hints, ensure_ascii=False),
        ]
    )


def _parse_candidate_payload(
    payload: Mapping[str, Any],
    diag: Diagnosis,
    n_per_action: int,
) -> list[Candidate]:
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise LLMResponseError("LLM JSON must contain a `candidates` list.")

    candidates: list[Candidate] = []
    per_action_count = {action: 0 for action in PROPOSAL_ACTION_TYPES}
    default_target = (diag.weak_rubrics or list(RUBRIC_KEYS))[0]

    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            continue
        try:
            action_type = str(raw["action_type"])
            if action_type not in PROPOSAL_ACTION_TYPES:
                continue
            target_rubric = str(raw.get("target_rubric") or default_target)
            target_span = str(raw.get("target_span") or "")
            instruction = str(raw["instruction"]).strip()
            if not instruction:
                raise ValueError("empty instruction")
            if not target_span or target_span not in diag.text:
                target_span = select_target_span(
                    diag.text,
                    target_rubric,
                    action_type=action_type,
                    question=_diagnosis_question(diag),
                )
            cand = Candidate(
                action_type=action_type,
                target_rubric=target_rubric,
                target_span=target_span,
                instruction=instruction,
                metadata={"source": "llm_proposer"},
            )
        except (KeyError, ValueError):
            continue
        if per_action_count[cand.action_type] >= max(1, n_per_action):
            continue
        per_action_count[cand.action_type] += 1
        candidates.append(cand)
    return candidates


def _diagnostic_feature_slice(diag: Diagnosis) -> dict[str, float]:
    keys: list[str] = []
    for rubric in diag.weak_rubrics:
        if rubric in ("task_1", "content_1", "content_3", "organization_2"):
            keys.extend(["topicConsistency", "avgSentSimilarity", "C_Cnt"])
        elif rubric == "content_2":
            keys.extend(["word_Cnt", "char_Cnt", "2-gram_NDW"])
        elif rubric == "organization_1":
            keys.extend(["adjacent_sentence_overlap_function_lemmas", "avgSentSimilarity"])
        elif rubric == "expression_1":
            keys.extend(["lemma_MATTR", "morph_NDW", "N_MSTTR"])
        elif rubric == "expression_2":
            keys.extend(["E_NDW", "morph_LenAvg", "morph_LenStd"])
    return {key: float(diag.features.get(key, 0.0)) for key in dict.fromkeys(keys)}


def _section(cfg: Optional[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    if not cfg:
        return {}
    section = cfg.get(key, {})
    return section if isinstance(section, Mapping) else {}


def _target_hint_payload(diag: Diagnosis) -> dict[str, list[dict[str, Any]]]:
    target_order = list(diag.weak_rubrics) or list(RUBRIC_KEYS[:3])
    hints: dict[str, list[dict[str, Any]]] = {}
    for action_type in PROPOSAL_ACTION_TYPES:
        preferred = [rubric for rubric in target_order if _RUBRIC_TO_ACTION_HINT.get(rubric) == action_type]
        target_rubric = (preferred or target_order)[0]
        hints[action_type] = rank_target_spans(
            diag.text,
            target_rubric,
            action_type=action_type,
            limit=3,
            question=_diagnosis_question(diag),
        )
    return hints


def _diagnosis_question(diag: Diagnosis) -> Optional[str]:
    question = diag.metadata.get("question")
    return str(question) if question else None


def _proposal_action_definitions() -> dict[str, str]:
    return {action_type: _ACTION_INSTRUCTIONS[action_type] for action_type in PROPOSAL_ACTION_TYPES}


def _build_instruction(action_type: str, target_rubric: str) -> str:
    rubric_name = RUBRIC_NAMES_KO.get(target_rubric, target_rubric)
    return f"{rubric_name} 개선을 목표로 {_ACTION_INSTRUCTIONS[action_type]}"
