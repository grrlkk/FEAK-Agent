"""Bounded, independent Responses requests, adapted from Writing-Agent-Web."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv


class CallBudgetExceeded(RuntimeError):
    pass


class ResponseError(ValueError):
    pass


class APIUnavailable(RuntimeError):
    pass


def load_api_environment():
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)
    if os.getenv("FEAK_ENV_FILE"):
        load_dotenv(Path(os.environ["FEAK_ENV_FILE"]).expanduser(), override=False)


class OpenAIJSONClient:
    def __init__(self, cfg, on_record=None):
        self.cfg = cfg
        self.calls = 0
        self.sample_calls = 0
        self.records = []
        self.sample_id = None
        self.on_record = on_record
        self._client = None

    def start_sample(self, sample_id):
        self.sample_id = sample_id
        self.sample_calls = 0

    def _load(self):
        if self._client is None:
            load_api_environment()
            if not os.getenv("OPENAI_API_KEY"):
                raise APIUnavailable("OPENAI_API_KEY is missing; configure .env or FEAK_ENV_FILE")
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                                  base_url="https://api.openai.com/v1",
                                  timeout=self.cfg.timeout_s, max_retries=0)

    def __call__(self, *, system, user, schema):
        if self.calls >= self.cfg.max_calls_total or self.sample_calls >= self.cfg.max_calls_per_sample:
            raise CallBudgetExceeded("Configured GPT call budget exhausted")
        if len(system) + len(user) > self.cfg.max_input_chars:
            raise ResponseError("Input exceeds max_input_chars; refusing to truncate")
        self._load()
        from openai import OpenAIError
        self.calls += 1
        self.sample_calls += 1
        record = {"call": self.calls, "sample_id": self.sample_id, "role": schema.__name__,
                  "provider": "openai", "model": self.cfg.model, "system": system,
                  "input": json.loads(user), "schema": schema.model_json_schema()}
        self.records.append(record)
        try:
            response = self._client.responses.create(
                model=self.cfg.model, instructions=system, input=user,
                reasoning={"effort": self.cfg.reasoning_effort},
                text={"format": {"type": "json_schema", "name": schema.__name__,
                                 "schema": schema.model_json_schema(), "strict": True}},
                max_output_tokens=self.cfg.max_output_tokens, truncation="disabled", store=False,
            )
            record.update(response_id=response.id, response_model=response.model,
                          status=response.status, raw=response.output_text,
                          usage=response.usage.model_dump() if response.usage else None)
            if response.status != "completed":
                raise ResponseError("OpenAI response incomplete")
            if any(part.type == "refusal" for item in response.output
                   for part in getattr(item, "content", [])):
                raise APIUnavailable("OpenAI declined this request")
            try:
                parsed = json.loads(response.output_text)
            except (ValueError, TypeError) as exc:
                raise ResponseError("Response is not a complete JSON object") from exc
            if not isinstance(parsed, dict):
                raise ResponseError("Response must be a JSON object")
            return parsed
        except OpenAIError as exc:
            # SDK exception bodies may contain request data or credentials.
            record.update(status="error", error_type=type(exc).__name__,
                          http_status=getattr(exc, "status_code", None))
            raise APIUnavailable(f"OpenAI request failed ({record['error_type']}, HTTP {record['http_status']})") from None
        finally:
            if self.on_record:
                self.on_record(record)

    def close(self):
        if self._client is not None:
            self._client.close()
