"""Small OpenAI-compatible JSON helper for MVP proposal and patching."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


class LLMUnavailable(RuntimeError):
    """Raised when the configured LLM client cannot be used."""


class LLMResponseError(RuntimeError):
    """Raised when the LLM response is not valid JSON for the caller."""


def request_json(
    *,
    system: str,
    user: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    env_file: Optional[str] = None,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat model and parse a JSON object response."""

    _load_env(env_file)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMUnavailable("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise LLMUnavailable("The `openai` package is not installed.") from exc

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    if timeout is not None:
        client_kwargs["timeout"] = timeout

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise LLMResponseError("LLM returned an empty response.")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"LLM returned invalid JSON: {content}") from exc
    if not isinstance(parsed, dict):
        raise LLMResponseError("LLM JSON response must be an object.")
    return parsed


def _load_env(env_file: Optional[str]) -> None:
    explicit = env_file or os.getenv("FEAK_ENV_FILE")
    if explicit:
        load_dotenv(dotenv_path=Path(explicit).expanduser(), override=False)
        return
    load_dotenv(dotenv_path=DEFAULT_ENV_FILE, override=False)
    load_dotenv(override=False)
