"""Surface-level spelling and spacing normalization for MVP inputs."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_BAREUN_API_URL = "https://api.bareun.ai/bareun.RevisionService/CorrectError"

DEFAULT_SURFACE_CFG = {
    "mode": "off",
    "timeout": 20,
    "api_key_env": "BAREUN_API_KEY",
    "api_url_env": "BAREUN_API_URL",
    "api_url": DEFAULT_BAREUN_API_URL,
    "allowed_categories": ["SPACING", "TYPO", "GRAMMER", "WORD"],
    "max_char_edit_ratio": 0.2,
    "min_length_ratio": 0.8,
    "max_length_ratio": 1.25,
    "max_sentence_count_delta": 1,
    "reject_new_numbers": True,
}


class SurfaceNormalizerUnavailable(RuntimeError):
    """Raised when a requested surface normalizer cannot be used."""


class SurfaceNormalizerError(RuntimeError):
    """Raised when a surface normalizer returns an invalid response."""


@dataclass
class SurfaceNormalization:
    provider: str
    original_text: str
    normalized_text: str
    applied: bool
    rejected: bool = False
    reject_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "applied": self.applied,
            "rejected": self.rejected,
            "reject_reasons": list(self.reject_reasons),
            "metadata": {k: v for k, v in self.metadata.items() if k != "suggested_text"},
        }


def normalize_surface_text(
    text: str,
    cfg: Optional[Mapping[str, Any]] = None,
) -> Optional[SurfaceNormalization]:
    """Normalize spelling/spacing before diagnosis if enabled in config."""

    merged = _surface_config(cfg)
    mode = str(merged.get("mode", "off")).strip().lower()
    if mode in {"", "off", "none", "false", "0"}:
        return None
    if mode == "bareun":
        return _normalize_with_bareun(text, merged)
    if mode == "hanspell":
        return _normalize_with_hanspell(text, merged)
    raise ValueError(f"Unknown surface_normalizer mode: {mode}")


def _normalize_with_bareun(text: str, cfg: Mapping[str, Any]) -> SurfaceNormalization:
    _load_env(cfg.get("env_file"))
    api_key_env = str(cfg.get("api_key_env", "BAREUN_API_KEY"))
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise SurfaceNormalizerUnavailable(f"{api_key_env} is not set.")

    api_url_env = str(cfg.get("api_url_env", "BAREUN_API_URL"))
    api_url = os.getenv(api_url_env) or str(cfg.get("api_url") or DEFAULT_BAREUN_API_URL)
    custom_dict_names = _string_list(cfg.get("custom_dict_names"))
    payload: dict[str, Any] = {
        "document": {"content": text, "language": "ko-KR"},
        "encoding_type": "UTF32",
        "config": {
            "enable_cleanup_whitespace": True,
            "enable_sentence_check": bool(cfg.get("enable_sentence_check", False)),
        },
    }
    if custom_dict_names:
        payload["custom_dict_names"] = custom_dict_names

    response = _post_json(
        api_url,
        payload=payload,
        headers={"api-key": api_key, "Content-Type": "application/json"},
        timeout=float(cfg.get("timeout", 20)),
    )
    revised = response.get("revised")
    if not isinstance(revised, str):
        raise SurfaceNormalizerError("Bareun CorrectError response has no string `revised` field.")

    categories = sorted(_bareun_categories(response))
    metadata = {
        "categories": categories,
        "tokens_count": response.get("tokens_count"),
        "revised_block_count": len(response.get("revised_blocks") or []),
        "language": response.get("language"),
    }
    return _guard_result(provider="bareun", original=text, suggested=revised, metadata=metadata, cfg=cfg)


def _normalize_with_hanspell(text: str, cfg: Mapping[str, Any]) -> SurfaceNormalization:
    try:
        from hanspell import spell_checker
    except ModuleNotFoundError as exc:
        raise SurfaceNormalizerUnavailable("py-hanspell is not installed.") from exc

    checked = spell_checker.check(text)
    revised = getattr(checked, "checked", None)
    if not isinstance(revised, str):
        raise SurfaceNormalizerError("py-hanspell response has no string `checked` field.")

    words = getattr(checked, "words", {}) or {}
    categories = sorted(
        category
        for category in {_hanspell_category(value) for value in words.values() if value is not None}
        if category != "PASSED"
    )
    metadata = {
        "categories": categories,
        "errors": getattr(checked, "errors", None),
        "time": getattr(checked, "time", None),
    }
    return _guard_result(provider="hanspell", original=text, suggested=revised, metadata=metadata, cfg=cfg)


def _guard_result(
    *,
    provider: str,
    original: str,
    suggested: str,
    metadata: dict[str, Any],
    cfg: Mapping[str, Any],
) -> SurfaceNormalization:
    reasons: list[str] = []
    if not suggested.strip():
        reasons.append("surface:empty_suggestion")

    categories = set(str(category) for category in metadata.get("categories", []))
    allowed = set(_string_list(cfg.get("allowed_categories")) or DEFAULT_SURFACE_CFG["allowed_categories"])
    disallowed = sorted(categories - allowed)
    if disallowed:
        reasons.append("surface:disallowed_category:" + ",".join(disallowed))

    edit = _char_edit_ratio(original, suggested)
    metadata["char_edit_ratio"] = edit
    if edit > float(cfg.get("max_char_edit_ratio", 0.2)):
        reasons.append("surface:edit_ratio")

    original_len = max(len(original), 1)
    length_ratio = len(suggested) / original_len
    metadata["length_ratio"] = length_ratio
    if length_ratio < float(cfg.get("min_length_ratio", 0.8)):
        reasons.append("surface:length_shrink")
    if length_ratio > float(cfg.get("max_length_ratio", 1.25)):
        reasons.append("surface:length_expand")

    sentence_delta = abs(_sentence_count(suggested) - _sentence_count(original))
    metadata["sentence_count_delta"] = sentence_delta
    if sentence_delta > int(cfg.get("max_sentence_count_delta", 1)):
        reasons.append("surface:sentence_count")

    if bool(cfg.get("reject_new_numbers", True)) and re.findall(r"\d+", original) != re.findall(r"\d+", suggested):
        reasons.append("surface:number_change")

    if reasons:
        metadata["suggested_text"] = suggested
        return SurfaceNormalization(
            provider=provider,
            original_text=original,
            normalized_text=original,
            applied=False,
            rejected=True,
            reject_reasons=reasons,
            metadata=metadata,
        )

    return SurfaceNormalization(
        provider=provider,
        original_text=original,
        normalized_text=suggested,
        applied=suggested != original,
        rejected=False,
        metadata=metadata,
    )


def _post_json(url: str, *, payload: Mapping[str, Any], headers: Mapping[str, str], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SurfaceNormalizerUnavailable(f"Bareun request failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SurfaceNormalizerUnavailable(f"Bareun request failed: {exc}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SurfaceNormalizerError(f"Bareun returned invalid JSON: {raw}") from exc
    if not isinstance(parsed, dict):
        raise SurfaceNormalizerError("Bareun JSON response must be an object.")
    return parsed


def _bareun_categories(response: Mapping[str, Any]) -> set[str]:
    categories: set[str] = set()

    def visit_block(block: Any) -> None:
        if not isinstance(block, Mapping):
            return
        for revision in block.get("revisions") or []:
            if isinstance(revision, Mapping) and revision.get("category"):
                categories.add(str(revision["category"]))
        for nested in block.get("nested") or []:
            visit_block(nested)

    for block in response.get("revised_blocks") or []:
        visit_block(block)
    return categories


def _hanspell_category(value: Any) -> str:
    # py-hanspell CheckResult: 0 passed, 1 spelling, 2 spacing,
    # 3 ambiguous, 4 statistical correction.
    mapping = {
        1: "TYPO",
        2: "SPACING",
        3: "CONFIRM",
        4: "CONFIRM",
    }
    try:
        return mapping.get(int(value), "PASSED")
    except (TypeError, ValueError):
        return "UNKNOWN"


def _char_edit_ratio(before: str, after: str) -> float:
    if before == after:
        return 0.0
    return 1.0 - SequenceMatcher(None, before, after).ratio()


def _sentence_count(text: str) -> int:
    count = len(re.findall(r"[.!?…。？！]", text))
    if count:
        return count
    return 1 if text.strip() else 0


def _surface_config(cfg: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    merged = dict(DEFAULT_SURFACE_CFG)
    if cfg:
        section = cfg.get("surface_normalizer", cfg)
        if isinstance(section, Mapping):
            merged.update(section)
    return merged


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _load_env(env_file: Optional[Any]) -> None:
    explicit = env_file or os.getenv("FEAK_ENV_FILE")
    if explicit:
        load_dotenv(dotenv_path=Path(str(explicit)).expanduser(), override=False)
        return
    load_dotenv(dotenv_path=DEFAULT_ENV_FILE, override=False)
    load_dotenv(override=False)
