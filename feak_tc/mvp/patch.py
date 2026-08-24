"""Patch application for MVP candidate revisions."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from .llm import LLMResponseError, LLMUnavailable, request_json
from .schemas import Candidate, Patch


def apply_patch(
    text: str,
    cand: Candidate,
    cfg: Optional[Mapping[str, Any]] = None,
) -> Candidate:
    """Apply a reversible minimal patch to produce a candidate revision."""

    if cand.action_type == "STOP":
        cand.new_text = text
        cand.patch = Patch(
            operation="noop",
            target_span=cand.target_span,
            before="",
            after="",
            reason="STOP candidate leaves the essay unchanged.",
        )
        return cand

    patcher_cfg = _section(cfg, "patcher")
    mode = str(patcher_cfg.get("mode", "deterministic"))
    if mode not in {"deterministic", "llm", "auto"}:
        raise ValueError(f"Unknown patcher mode: {mode}")

    if mode in {"llm", "auto"}:
        try:
            return _apply_llm_patch(text, cand, patcher_cfg)
        except (LLMUnavailable, LLMResponseError, ValueError, RuntimeError) as exc:
            if mode == "llm":
                raise
            cand.metadata["patcher_error"] = str(exc)

    return _apply_deterministic_patch(text, cand)


def _apply_deterministic_patch(text: str, cand: Candidate) -> Candidate:
    target = cand.target_span if cand.target_span in text else _first_sentence(text)
    replacement = _rewrite_span(target, cand.action_type)
    if not target:
        new_text = replacement
    elif cand.action_type == "ADD_DETAIL":
        new_text = text.replace(target, f"{target} {replacement}", 1)
    elif cand.action_type == "DELETE_OR_FOCUS":
        new_text = text.replace(target, replacement, 1).strip()
    elif cand.action_type == "COMPRESS":
        new_text = text.replace(target, replacement, 1)
    elif cand.action_type == "RESTRUCTURE":
        new_text = text.replace(target, replacement, 1)
    elif cand.action_type == "STYLE_REFINE":
        new_text = text.replace(target, replacement, 1)
    else:
        raise ValueError(cand.action_type)

    cand.new_text = _clean_spacing(new_text)
    cand.patch = Patch(
        operation="replace" if cand.action_type != "ADD_DETAIL" else "insert_after",
        target_span=target,
        before=target,
        after=replacement,
        reason=cand.instruction,
    )
    return cand


# The LLM patcher is span-anchored: the edit location is fixed by the
# candidate's target_span and applied by offset, so the model can only
# produce local content, never choose where the edit lands.

_ACTION_TASKS = {
    "ADD_DETAIL": "대상 문장 바로 뒤에 이어질 뒷받침 문장 1개를 새로 작성한다.",
    "COMPRESS": "대상 문장을 의미 손실 없이 더 간결하게 압축한 문장으로 다시 쓴다.",
    "RESTRUCTURE": "대상 문장의 어순이나 연결 표현을 조정해 흐름이 분명한 문장으로 다시 쓴다.",
    "STYLE_REFINE": "대상 문장의 어색한 어휘, 맞춤법, 표현을 자연스럽게 다듬어 다시 쓴다.",
}

_DANGLING_OPENERS = (
    "그래서",
    "그러므로",
    "따라서",
    "이에",
    "이는",
    "이를",
    "이러한",
    "이처럼",
    "이와 같이",
    "그 중",
    "그중",
    "또한",
    "하지만",
    "그러나",
    "그리고",
    "그런데",
    "그렇기",
    "그로 인해",
    "예를 들",
    "즉",
    "결국",
    "왜냐하면",
    "때문에",
)

_CONTEXT_CHARS = 160


def _apply_llm_patch(text: str, cand: Candidate, cfg: Mapping[str, Any]) -> Candidate:
    start, end = _locate_span(text, cand.target_span)
    span = text[start:end]

    if cand.action_type == "DELETE_OR_FOCUS":
        return _apply_span_delete(text, cand, cfg, start, end)

    after = _request_generated_text(
        cfg,
        system=(
            "You are a Korean essay editor. Return only a JSON object with the "
            'single key "after". Write only the requested local text; never '
            "return the whole essay or surrounding sentences."
        ),
        user=_build_span_prompt(text, cand, start, end),
        max_len=_local_edit_limit(span),
    )

    if cand.action_type == "ADD_DETAIL":
        new_text = f"{text[:end]} {after}{text[end:]}"
        patch = Patch("insert_after", span, span, after, cand.instruction)
    else:
        new_text = text[:start] + after + text[end:]
        patch = Patch("replace", span, span, after, cand.instruction)

    cand.new_text = _clean_spacing(new_text)
    cand.patch = patch
    cand.metadata["patcher"] = "llm"
    return cand


def _apply_span_delete(
    text: str, cand: Candidate, cfg: Mapping[str, Any], start: int, end: int
) -> Candidate:
    span = text[start:end]
    if not text[:start].strip() and not text[end:].strip():
        raise ValueError("Deleting the whole essay is not allowed.")

    repaired = None
    bounds = _next_sentence_bounds(text, end)
    if bounds is not None:
        next_start, next_end = bounds
        next_sentence = text[next_start:next_end]
        if next_sentence.startswith(_DANGLING_OPENERS):
            try:
                repaired = _request_generated_text(
                    cfg,
                    system=(
                        "You are a Korean essay editor. Return only a JSON object "
                        'with the single key "after" containing exactly one sentence.'
                    ),
                    user=_build_repair_prompt(text, span, start, next_sentence),
                    max_len=_local_edit_limit(next_sentence),
                )
            except (LLMUnavailable, LLMResponseError) as exc:
                cand.metadata["connective_repair_error"] = str(exc)

    if repaired is not None:
        new_text = text[:start] + repaired + text[next_end:]
        patch = Patch("replace", span, text[start:next_end], repaired, cand.instruction)
        cand.metadata["connective_repair"] = True
    else:
        new_text = text[:start] + text[end:]
        patch = Patch("delete", span, span, "", cand.instruction)

    cand.new_text = _clean_spacing(new_text)
    cand.patch = patch
    cand.metadata["patcher"] = "llm"
    return cand


def _build_span_prompt(text: str, cand: Candidate, start: int, end: int) -> str:
    span = text[start:end]
    if cand.action_type == "ADD_DETAIL":
        output_desc = "대상 문장 바로 뒤에 삽입할 새 문장 1개"
    else:
        output_desc = "대상 문장을 대신할 수정된 문장"
    return "\n".join(
        [
            'Return JSON: {"after": "<' + output_desc + '>"}',
            "",
            f"Task: {_ACTION_TASKS[cand.action_type]}",
            f"Instruction: {cand.instruction}",
            "",
            "Rules:",
            f'- "after"에는 {output_desc}만 담는다. 글 전체나 앞뒤 문맥을 포함하지 않는다.',
            "- 원문에 없는 수치, 통계, 연도, 기관명, 조사 결과를 만들어 내지 않는다.",
            "- 글쓴이의 의도와 어조를 유지한다.",
            "",
            "[앞 문맥]",
            text[max(0, start - _CONTEXT_CHARS) : start].strip(),
            "",
            "[대상 문장]",
            span,
            "",
            "[뒤 문맥]",
            text[end : end + _CONTEXT_CHARS].strip(),
        ]
    )


def _build_repair_prompt(text: str, deleted_span: str, start: int, next_sentence: str) -> str:
    return "\n".join(
        [
            'Return JSON: {"after": "<수정된 문장>"}',
            "",
            "에세이에서 한 문장이 삭제되었다. 삭제 직후에 오는 문장이 삭제된 문장을",
            "가리키는 접속어나 지시어로 시작해 어색해졌다. 이 문장이 남은 앞 문맥에",
            "자연스럽게 이어지도록 최소한으로만 고쳐라.",
            "",
            "Rules:",
            "- 문장의 내용은 유지하고 접속어와 지시어만 조정한다.",
            "- 새로운 정보를 추가하지 않는다.",
            "- 문장 1개만 반환한다.",
            "",
            "[삭제된 문장]",
            deleted_span,
            "",
            "[삭제 후 남은 앞 문맥]",
            text[max(0, start - _CONTEXT_CHARS) : start].strip(),
            "",
            "[고칠 문장]",
            next_sentence,
        ]
    )


def _request_generated_text(
    cfg: Mapping[str, Any], *, system: str, user: str, max_len: int
) -> str:
    payload = request_json(
        system=system,
        user=user,
        model=str(cfg.get("model", "gpt-5-mini-2025-08-07")),
        temperature=float(cfg.get("temperature", 0.1)),
        env_file=cfg.get("env_file"),
        timeout=float(cfg["timeout"]) if cfg.get("timeout") is not None else None,
    )
    after = str(payload.get("after") or "").strip()
    if not after:
        raise LLMResponseError("Patch `after` cannot be empty.")
    if len(after) > max_len:
        raise LLMResponseError(
            f"Patch `after` is too long ({len(after)} chars); must stay a local edit."
        )
    return after


def _locate_span(text: str, target_span: Optional[str]) -> tuple[int, int]:
    span = (target_span or "").strip()
    if not span:
        raise ValueError("Candidate target_span is empty.")
    start = text.find(span)
    if start < 0:
        raise ValueError("Candidate target_span is not an exact substring of the essay.")
    return start, start + len(span)


def _next_sentence_bounds(text: str, pos: int) -> Optional[tuple[int, int]]:
    match = re.search(r"\S", text[pos:])
    if match is None:
        return None
    start = pos + match.start()
    end_match = re.search(r"[.!?。？！]", text[start:])
    end = start + end_match.end() if end_match else len(text)
    return start, end


def _local_edit_limit(span: str) -> int:
    return max(2 * len(span) + 120, 200)


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?。？！])\s+", text.strip(), maxsplit=1)
    return parts[0] if parts else text.strip()


def _rewrite_span(span: str, action_type: str) -> str:
    stripped = span.strip()
    if action_type == "ADD_DETAIL":
        return _add_detail_sentence(stripped)
    if action_type == "DELETE_OR_FOCUS":
        return _focus_sentence(stripped)
    if action_type == "COMPRESS":
        return _compress_sentence(stripped)
    if action_type == "RESTRUCTURE":
        return _add_connector(stripped)
    if action_type == "STYLE_REFINE":
        return _style_refine(stripped)
    return stripped


def _add_detail_sentence(sentence: str) -> str:
    if any(keyword in sentence for keyword in ("인권", "권리", "존중")):
        return "예를 들어, 표현의 자유와 안전하게 살 권리가 이에 해당한다."
    return "예를 들어, 생활 속 사례를 덧붙이면 주장이 더 분명해진다."


def _focus_sentence(sentence: str) -> str:
    if not sentence:
        return sentence
    if any(keyword in sentence for keyword in ("인권", "권리", "존중", "주제")):
        return sentence
    return "이 문장은 글의 핵심 주제와 직접 연결되도록 초점을 분명히 해야 한다."


def _compress_sentence(sentence: str) -> str:
    words = sentence.split()
    if len(words) <= 10:
        return sentence
    compressed = " ".join(words[:10])
    if sentence.endswith(".") and not compressed.endswith("."):
        compressed += "."
    return compressed


def _add_connector(sentence: str) -> str:
    if sentence.startswith(("따라서", "또한", "그러나", "그리고")):
        return sentence
    return f"따라서 {sentence}"


def _style_refine(sentence: str) -> str:
    replacements = {
        "맛있는 밥": "기본적인 생활 조건",
        "화장실을 자유롭게 갈 수 있다": "기본적인 생활의 자유를 누릴 수 있다",
        "목숨을 가져갔다": "생명을 빼앗았다",
        "들고 일어났다": "저항했다",
    }
    refined = sentence
    for before, after in replacements.items():
        refined = refined.replace(before, after)
    refined = refined.replace("  ", " ")
    return refined


def _clean_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _section(cfg: Optional[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    if not cfg:
        return {}
    section = cfg.get(key, {})
    return section if isinstance(section, Mapping) else {}
