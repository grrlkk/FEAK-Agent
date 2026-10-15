"""Fresh bounded GPT requests via the existing API adapter, with P1 retry policy."""

import time

from feak_tc.agent.schemas import OpenAIConfig
from feak_tc.runtime.openai import (APIUnavailable, CallBudgetExceeded, OpenAIJSONClient,
                                   ResponseError)
from pydantic import ValidationError


class JSONFailure(RuntimeError):
    pass


class LLM:
    def __init__(self, model_config, request_config=None, on_record=None, client=None):
        if model_config.get("provider") != "openai":
            raise ValueError("This P1 deployment uses the user-selected current OpenAI API")
        self.on_record = on_record or (lambda row: None)
        self.context = {}
        cfg = OpenAIConfig(**{**(request_config or {}), "model": model_config["model"],
                              "reasoning_effort": model_config.get("reasoning_effort", "low")})
        self.client = client if client is not None else OpenAIJSONClient(cfg, self._record)

    def _record(self, record):
        record = dict(record)
        record.update(self.context)
        record["elapsed_s"] = time.monotonic() - self.started
        self.on_record(record)

    def start_sample(self, sample_id):
        self.client.start_sample(sample_id)

    def request(self, schema, prompt, payload, *, role, retries=0, condition=None, validate=None):
        import json
        for attempt in range(retries + 1):
            self.context = {"stage": role, "condition": condition, "attempt": attempt}
            self.started = time.monotonic()
            try:
                response = schema.model_validate(self.client(system=prompt,
                    user=json.dumps(payload, ensure_ascii=False), schema=schema))
                if validate is not None:
                    validate(response)
                return response
            except CallBudgetExceeded:
                raise
            except (APIUnavailable, ResponseError, ValidationError, ValueError) as exc:
                # Repeat the same request. No earlier answer or decision enters the prompt.
                self.on_record({"stage": role, "condition": condition, "attempt": attempt,
                                "event": "validation_or_request_error", "error_type": type(exc).__name__})
                if attempt == retries:
                    raise JSONFailure(f"{role} failed after {attempt + 1} attempts") from exc

    def close(self):
        self.client.close()
