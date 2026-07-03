"""Patch application for MVP candidate revisions."""

from __future__ import annotations

import json
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


def _apply_llm_patch(text: str, cand: Candidate, cfg: Mapping[str, Any]) -> Candidate:
    payload = request_json(
        system=(
            "You are a Korean essay patch simulator. Return only JSON. "
            "Apply one minimal, reversible local edit. Never rewrite the whole essay."
        ),
        user=_build_patch_prompt(text, cand),
        model=str(cfg.get("model", "gpt-4o-mini")),
        temperature=float(cfg.get("temperature", 0.1)),
        env_file=cfg.get("env_file"),
        timeout=float(cfg["timeout"]) if cfg.get("timeout") is not None else None,
    )
    patch = _parse_patch_payload(text, cand, payload)
    cand.new_text = _clean_spacing(_materialize_patch(text, patch))
    cand.patch = patch
    cand.metadata["patcher"] = "llm"
    return cand


def _build_patch_prompt(text: str, cand: Candidate) -> str:
    schema = {
        "operation": "replace | insert_after | delete | noop",
        "target_span": "exact substring from essay",
        "before": "exact text to replace/delete or anchor for insert_after",
        "after": "new local text, empty only for delete/noop",
        "reason": "short Korean reason",
    }
    return "\n".join(
        [
            "Return JSON matching this schema:",
            json.dumps(schema, ensure_ascii=False),
            "",
            "Rules:",
            "- Use one local edit only.",
            "- before and target_span must be exact substrings from the essay unless operation is noop.",
            "- Do not rewrite the whole essay.",
            "- Preserve the writer's original intent.",
            "",
            "[Essay]",
            text,
            "",
            "[Candidate]",
            json.dumps(cand.to_dict(), ensure_ascii=False),
        ]
    )


def _parse_patch_payload(text: str, cand: Candidate, payload: Mapping[str, Any]) -> Patch:
    operation = str(payload.get("operation", "")).strip()
    if operation not in {"replace", "insert_after", "delete", "noop"}:
        raise LLMResponseError(f"Invalid patch operation: {operation}")

    target_span = str(payload.get("target_span") or cand.target_span or "").strip()
    before = str(payload.get("before") or target_span).strip()
    after = str(payload.get("after") or "").strip()
    reason = str(payload.get("reason") or cand.instruction).strip()

    if operation == "noop":
        return Patch("noop", target_span, "", "", reason)

    if not before:
        raise LLMResponseError("Patch `before` cannot be empty.")
    if before not in text:
        if cand.target_span and cand.target_span in text:
            before = cand.target_span
            target_span = cand.target_span
        else:
            raise LLMResponseError("Patch `before` is not an exact substring.")
    if operation in {"replace", "insert_after"} and not after:
        raise LLMResponseError("Patch `after` cannot be empty for replace/insert_after.")
    if operation == "delete" and before.strip() == text.strip():
        raise LLMResponseError("Deleting the whole essay is not allowed.")

    return Patch(operation, target_span or before, before, after, reason)


def _materialize_patch(text: str, patch: Patch) -> str:
    if patch.operation == "noop":
        return text
    if patch.operation == "replace":
        return text.replace(patch.before, patch.after, 1)
    if patch.operation == "insert_after":
        return text.replace(patch.before, f"{patch.before} {patch.after}", 1)
    if patch.operation == "delete":
        return text.replace(patch.before, "", 1)
    raise ValueError(patch.operation)


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?。？！])\s+", text.strip(), maxsplit=1)
    return parts[0] if parts else text.strip()


def _rewrite_span(span: str, action_type: str) -> str:
    stripped = span.strip()
    if action_type == "ADD_DETAIL":
        return "예를 들어, 이 내용은 글의 핵심 주장과 직접 연결되는 구체적인 사례로 보완할 수 있다."
    if action_type == "DELETE_OR_FOCUS":
        return _focus_sentence(stripped)
    if action_type == "COMPRESS":
        return _compress_sentence(stripped)
    if action_type == "RESTRUCTURE":
        return _add_connector(stripped)
    if action_type == "STYLE_REFINE":
        return _style_refine(stripped)
    return stripped


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
