"""Uniform, meaning-preserving normalization for all corruption states.

The pass is deliberately operator-blind: it never receives a rubric, feature,
operator, or stage index. It only removes superficial generator fingerprints.
It must not repair the very degradation that a corruption operator introduced.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Callable, Mapping, Optional

from feak_tc.mvp.llm import LLMResponseError, LLMUnavailable, request_json


_SYSTEM = (
    "You normalize Korean essay text with the smallest possible surface edit. "
    "Return only a JSON object."
)

_PROMPT = """다음 글에 모든 상태에 공통으로 적용되는 경량 정규화를 1회 수행하라.

허용:
- 문장 사이 공백과 문장부호 표기의 일관성만 최소한으로 정리
- 생성기 특유의 머리말·목록 표지가 있으면 평문으로 정리

금지:
- 사실·주장·예시·수치의 추가, 삭제, 바꿔쓰기
- 문장 순서 변경
- 반복, 주제 이탈, 누락된 구체성의 보정
- 어법·조사·맞춤법 오류의 교정
- 문장을 더 잘 쓰거나 품질을 높이는 작업

원문:
{text}

반환 형식:
{{"normalized_text": "<정규화된 전체 글>"}}
"""

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")
_NORMALIZATION_SCHEMA = {
    "type": "object",
    "properties": {"normalized_text": {"type": "string"}},
    "required": ["normalized_text"],
    "additionalProperties": False,
}


def normalize_text(
    text: str,
    cfg: Mapping[str, Any],
    post_validate: Optional[Callable[[str], None]] = None,
) -> tuple[str, dict[str, Any]]:
    """Normalize one state with the same operator-blind pass.

    Raises RuntimeError after all attempts fail. A disabled pass still returns
    explicit metadata so every state records whether normalization occurred.
    """

    model = str(cfg.get("model", "gpt-4o-mini"))
    reasoning_effort = _nested_string(cfg, "reasoning", "effort")
    verbosity = _nested_string(cfg, "text", "verbosity")
    metadata = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "verbosity": verbosity,
        "postcondition_checked": post_validate is not None,
    }

    if not bool(cfg.get("enabled", True)):
        if post_validate is not None:
            post_validate(text)
        return text, {
            "normalized": False,
            "changed": False,
            "generator": "disabled",
            "attempts": 0,
            **metadata,
        }

    attempts = int(cfg.get("max_attempts", 3))
    errors: list[str] = []
    postcondition_rejections = 0
    for attempt in range(1, attempts + 1):
        try:
            payload = request_json(
                system=_SYSTEM,
                user=_PROMPT.format(text=text),
                model=model,
                temperature=(
                    float(cfg["temperature"])
                    if cfg.get("temperature") is not None
                    else None
                ),
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
                json_schema=_NORMALIZATION_SCHEMA,
                response_format_name="corruption_normalization",
                timeout=float(cfg["timeout"]) if cfg.get("timeout") is not None else None,
            )
            normalized = str(payload.get("normalized_text", "")).strip()
            _validate_normalized(text, normalized, cfg)
        except (LLMUnavailable, LLMResponseError, ValueError, RuntimeError) as exc:
            errors.append(f"attempt{attempt}: {exc}")
            continue

        if post_validate is not None:
            try:
                post_validate(normalized)
            except ValueError as exc:
                postcondition_rejections += 1
                errors.append(f"attempt{attempt}: {exc}")
                continue
        return normalized, {
            "normalized": True,
            "changed": normalized != text,
            "generator": "llm_uniform",
            "attempts": attempt,
            "errors": errors,
            **metadata,
        }

    if (
        post_validate is not None
        and postcondition_rejections == attempts
        and bool(cfg.get("fallback_to_input_on_postcondition_failure", False))
    ):
        _validate_normalized(text, text, cfg)
        post_validate(text)
        return text, {
            "normalized": False,
            "changed": False,
            "generator": "identity_postcondition_fallback",
            "attempts": attempts,
            "errors": errors,
            "fallback_reason": "all_normalized_outputs_removed_or_rewrote_corruption",
            **metadata,
        }

    raise RuntimeError("normalization failed: " + "; ".join(errors))


def _validate_normalized(before: str, after: str, cfg: Mapping[str, Any]) -> None:
    if not after:
        raise ValueError("normalizer returned empty text")
    ratio = len(after) / max(1, len(before))
    lo = float(cfg.get("min_length_ratio", 0.8))
    hi = float(cfg.get("max_length_ratio", 1.2))
    if not lo <= ratio <= hi:
        raise ValueError(f"normalization length ratio {ratio:.3f} outside [{lo}, {hi}]")
    if set(_NUMBER_RE.findall(before)) != set(_NUMBER_RE.findall(after)):
        raise ValueError("normalizer changed numeric expressions")
    similarity = SequenceMatcher(a=_compact(before), b=_compact(after)).ratio()
    minimum = float(cfg.get("min_similarity", 0.75))
    if similarity < minimum:
        raise ValueError(f"normalization similarity {similarity:.3f} below {minimum}")


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _nested_string(cfg: Mapping[str, Any], section: str, key: str) -> Optional[str]:
    nested = cfg.get(section)
    if not isinstance(nested, Mapping):
        return None
    value = nested.get(key)
    return str(value) if value is not None else None
