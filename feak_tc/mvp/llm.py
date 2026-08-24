"""Small OpenAI-compatible JSON helper for MVP proposal and patching."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

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
    model: str = "gpt-5-mini-2025-08-07",
    temperature: Optional[float] = 0.2,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
    json_schema: Optional[Mapping[str, Any]] = None,
    response_format_name: str = "json_response",
    env_file: Optional[str] = None,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    """Call an OpenAI model and parse a JSON object response.

    GPT-5 callers use the Responses API so reasoning effort, verbosity, and
    strict JSON Schema can be configured consistently across MVP and corruption
    generation.
    """

    _load_env(env_file)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMUnavailable("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI, OpenAIError
    except ModuleNotFoundError as exc:
        raise LLMUnavailable("The `openai` package is not installed.") from exc

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    if timeout is not None:
        client_kwargs["timeout"] = timeout

    client = OpenAI(**client_kwargs)
    try:
        if _use_responses_api(model, reasoning_effort, verbosity, json_schema):
            text: dict[str, Any] = {
                "format": (
                    {
                        "type": "json_schema",
                        "name": response_format_name,
                        "schema": dict(json_schema),
                        "strict": True,
                    }
                    if json_schema is not None
                    else {"type": "json_object"}
                )
            }
            if verbosity is not None:
                text["verbosity"] = verbosity
            response_kwargs: dict[str, Any] = {
                "model": model,
                "instructions": system,
                "input": user,
                "text": text,
            }
            if reasoning_effort is not None:
                response_kwargs["reasoning"] = {"effort": reasoning_effort}
            response = client.responses.create(**response_kwargs)
            content = response.output_text
        else:
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
    except OpenAIError as exc:
        raise LLMUnavailable(f"OpenAI request failed: {exc}") from exc
    if not content:
        raise LLMResponseError("LLM returned an empty response.")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"LLM returned invalid JSON: {content}") from exc
    if not isinstance(parsed, dict):
        raise LLMResponseError("LLM JSON response must be an object.")
    return parsed


def _use_responses_api(
    model: str,
    reasoning_effort: Optional[str],
    verbosity: Optional[str],
    json_schema: Optional[Mapping[str, Any]],
) -> bool:
    return (
        model.startswith("gpt-5")
        or reasoning_effort is not None
        or verbosity is not None
        or json_schema is not None
    )


def _load_env(env_file: Optional[str]) -> None:
    explicit = env_file or os.getenv("FEAK_ENV_FILE")
    if explicit:
        load_dotenv(dotenv_path=Path(explicit).expanduser(), override=False)
        return
    load_dotenv(dotenv_path=DEFAULT_ENV_FILE, override=False)
    load_dotenv(override=False)
