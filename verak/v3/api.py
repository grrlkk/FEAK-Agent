"""Environment-only OpenAI adapter with a durable phase-wide request budget."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from threading import Lock
import time

from pydantic import BaseModel, ConfigDict

from feak_tc.agent.schemas import OpenAIConfig
from feak_tc.runtime.openai import APIUnavailable, CallBudgetExceeded, OpenAIJSONClient
from .common import read_json, write_json


class StrictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PhaseBudget:
    def __init__(self, path, max_api_calls, *, authorized_ceiling=700):
        if not isinstance(max_api_calls, int) or not 0 <= max_api_calls <= authorized_ceiling:
            raise ValueError("max-api-calls must be between zero and the authorized phase ceiling")
        self.path, self.limit, self.ceiling = Path(path), max_api_calls, authorized_ceiling
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(".lock")

    @property
    def used(self):
        return read_json(self.path)["reserved_calls"] if self.path.exists() else 0

    def reserve(self):
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = read_json(self.path) if self.path.exists() else {
                "phase": "v3_phase1", "authorized_ceiling": self.ceiling, "reserved_calls": 0}
            if data["phase"] != "v3_phase1" or data["authorized_ceiling"] != self.ceiling:
                raise ValueError("Budget ledger belongs to another phase or authorization")
            if data["reserved_calls"] >= min(self.limit, data["authorized_ceiling"]):
                raise CallBudgetExceeded("Phase 1 GPT request budget exhausted")
            data["reserved_calls"] += 1
            write_json(self.path, data)  # Reserve before any network request, including retries.
            return data["reserved_calls"]


class EnvironmentJSONClient(OpenAIJSONClient):
    def _load(self):
        if self._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise APIUnavailable("Set OPENAI_API_KEY in the process environment")
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                base_url="https://api.openai.com/v1", timeout=self.cfg.timeout_s, max_retries=0)


class PhaseTeacher:
    """One client per request; safe to share the phase budget across worker threads."""
    def __init__(self, config, max_api_calls, *, client_factory=EnvironmentJSONClient):
        self.output = config["paths"]["output"]
        self.output.mkdir(parents=True, exist_ok=True)
        self.budget = PhaseBudget(self.output / "api_budget.json", max_api_calls,
            authorized_ceiling=config["teacher"]["phase_api_ceiling"])
        self.config = OpenAIConfig(model=config["teacher"]["model"],
            reasoning_effort=config["teacher"]["reasoning_effort"],
            max_output_tokens=config["teacher"]["max_output_tokens"],
            max_input_chars=30000, max_calls_total=700, max_calls_per_sample=700,
            timeout_s=config["teacher"]["timeout_s"])
        self.client_factory = client_factory
        self.write_lock = Lock()

    def _log(self, record):
        # Across processes the same log file is locked; credentials are never recorded.
        with self.write_lock, (self.output / "calls.jsonl").open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()

    def request(self, schema, system, payload, *, stage, sample_id):
        user = json.dumps(payload, ensure_ascii=False)
        if len(system) + len(user) > self.config.max_input_chars:
            raise ValueError("API input too long; refusing to truncate")
        started = time.monotonic()
        context = {"stage": stage, "sample_id": sample_id}
        client = self.client_factory(self.config,
            lambda record: self._log({**record, **context, "elapsed_s": time.monotonic() - started}))
        try:
            client._load()  # A missing key must not consume an API reservation.
            context["phase_call"] = self.budget.reserve()
            self._log({**context, "event": "reserved_before_request"})
            client.start_sample(sample_id)
            return schema.model_validate(client(system=system, user=user, schema=schema))
        finally:
            client.close()
