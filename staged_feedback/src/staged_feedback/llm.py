"""Structured calls, explicit reasoning settings and one persistent shared ledger."""
import json
from pathlib import Path
import time

import yaml

from .adapters import bootstrap
from .events import jsonable


class LiveLLM:
    def __init__(self, cfg, directory, project_root, *, budget_ledger=None):
        root = bootstrap(cfg.legacy_root)
        from para_agent.config import ModelConfig, load_config
        from para_agent.llm.openai_transport import OpenAITransport
        from para_agent.eval.costs import ReservedFailureBudgetTransport
        legacy = load_config(root / cfg.legacy_config)
        self.rates = yaml.safe_load((root / "configs/api_pricing_2026-10-15.yaml").read_text())["models"][cfg.model]
        self.cfg, self.directory = cfg, Path(directory)
        self.call_count = 0
        self.prompts = Path(project_root) / cfg.prompts
        self.directory.mkdir(parents=True, exist_ok=True)
        self.transports = {}
        for effort in {cfg.worker_reasoning_effort, cfg.coordinator_reasoning_effort}:
            model = ModelConfig(provider="openai", model=cfg.model,
                temperature=cfg.temperature if effort == "none" else None,
                reasoning_effort=effort, max_output_tokens=cfg.max_output_tokens, timeout_s=cfg.timeout_s)
            transport = OpenAITransport(model, Path(legacy.llm.env_file), max_calls=cfg.max_api_calls)
            self.transports[effort] = ReservedFailureBudgetTransport(transport, self.rates, cfg.budget_usd,
                ledger_path=Path(budget_ledger) if budget_ledger else self.directory / "budget.json")

    def request(self, role, payload, schema, *, context):
        from para_agent.eval.costs import token_cost
        if self.call_count >= self.cfg.max_api_calls:
            raise RuntimeError("전체 API 호출 상한에 도달했습니다.")
        effort = self.cfg.coordinator_reasoning_effort if role.startswith("coordinator") else self.cfg.worker_reasoning_effort
        transport = self.transports[effort]
        prompt = (self.prompts / f"{role}.txt").read_text(encoding="utf-8")
        if len(prompt) > 1500:
            raise ValueError(f"Prompt exceeds 1500 characters: {role}")
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(jsonable(payload), ensure_ascii=False, allow_nan=False)}]
        if sum(len(m["content"]) for m in messages) > self.cfg.max_input_chars:
            raise ValueError("Input exceeds configured limit; document text was not truncated")
        params = transport.parameters(schema.model_json_schema(), role)
        started = time.monotonic()
        before_count = transport.call_count
        transport.transport.last_usage = {}
        transport.transport.last_metadata = {}
        record = {**context, "role": role, "messages": messages, "params": params,
                  "reasoning_effort": effort, "cache_hit": False}
        try:
            self.call_count += 1
            print(f"API {context['essay_id']} turn={context['turn']} {role}", flush=True)
            raw, usage = transport.complete(agent=role, messages=messages, params=params)
            record["raw_output"] = raw
            result = schema.model_validate_json(raw)
            record["status"] = "completed"
            return result
        except Exception as exc:
            record.update(status="error", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            usage = getattr(transport, "last_usage", {}) or {}
            record.update(usage=usage, provider_response=getattr(transport, "last_metadata", {}),
                dispatched=transport.call_count > before_count,
                cost_usd=token_cost(usage, self.rates) if usage else None,
                latency_s=time.monotonic() - started)
            with (self.directory / "calls.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()

    def close(self):
        for transport in self.transports.values():
            transport.close()
