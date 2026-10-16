import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from staged_feedback.config import Settings
from staged_feedback.llm import LiveLLM
from staged_feedback.models import Revision


def install_transports(monkeypatch, tmp_path):
    """Exercise our wiring without credentials, networking or a legacy checkout."""
    from staged_feedback import llm as module
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/api_pricing_2026-10-15.yaml").write_text("models:\n  gpt-6-luna: {input: 0.10, output: 0.50}\n")
    monkeypatch.setattr(module, "bootstrap", lambda _: tmp_path)
    models, budgets = [], []

    class Transport:
        def __init__(self, model, *args, **kwargs):
            self.model, self.call_count, self.last_usage, self.last_metadata = model, 0, {}, {}
            models.append(model)

        def parameters(self, schema, role):
            return {"model": self.model.model, "schema": schema, "role": role}

        def complete(self, **kwargs):
            self.call_count += 1
            self.last_usage = {"input_tokens": 100, "output_tokens": 10}
            self.last_metadata = {"response_id": "fake"}
            return '{"operations": [], "student_tasks": []}', self.last_usage

        def close(self):
            pass

    class Budget:
        def __init__(self, transport, rates, cap, *, ledger_path):
            self.transport, self.ledger_path = transport, ledger_path
            self.block = False
            budgets.append(self)

        def __getattr__(self, key):
            return getattr(self.transport, key)

        def complete(self, **kwargs):
            if self.block:
                raise RuntimeError("budget exhausted")
            return self.transport.complete(**kwargs)

    monkeypatch.setitem(sys.modules, "para_agent.config", SimpleNamespace(ModelConfig=lambda **kw: SimpleNamespace(**kw),
        load_config=lambda _: SimpleNamespace(llm=SimpleNamespace(env_file="not-read.env"))))
    monkeypatch.setitem(sys.modules, "para_agent.llm.openai_transport", SimpleNamespace(OpenAITransport=Transport))
    monkeypatch.setitem(sys.modules, "para_agent.eval.costs", SimpleNamespace(ReservedFailureBudgetTransport=Budget,
        token_cost=lambda usage, _: usage.get("input_tokens", 0) / 1000000))
    prompt_dir = tmp_path / "prompts/staged_feedback"
    prompt_dir.mkdir(parents=True)
    for role in ("reviser", "coordinator_review"):
        (prompt_dir / f"{role}.txt").write_text("목표: 합성 검증")
    return models, budgets


def test_roles_share_budget_and_reasoning_temperature(monkeypatch, tmp_path):
    models, budgets = install_transports(monkeypatch, tmp_path)
    client = LiveLLM(Settings(), tmp_path / "run", tmp_path)
    assert len(models) == 2
    assert {m.reasoning_effort: m.temperature for m in models} == {"none": 0, "low": None}
    assert len({b.ledger_path for b in budgets}) == 1
    context = dict(essay_id="synthetic", phase="planned", turn=1)
    client.request("reviser", {}, Revision, context=context)
    client.request("coordinator_review", {}, Revision, context=context)
    calls = [json.loads(line) for line in (tmp_path / "run/calls.jsonl").read_text().splitlines()]
    assert len(calls) == 2
    assert all(c["dispatched"] and c["cost_usd"] == 0.0001 for c in calls)
    assert all(c["raw_output"] for c in calls)


def test_blocked_request_does_not_repeat_previous_usage(monkeypatch, tmp_path):
    _, budgets = install_transports(monkeypatch, tmp_path)
    client = LiveLLM(Settings(), tmp_path / "run", tmp_path)
    context = dict(essay_id="synthetic", phase="planned", turn=1)
    client.request("reviser", {}, Revision, context=context)
    client.transports["none"].block = True
    with pytest.raises(RuntimeError, match="budget"):
        client.request("reviser", {}, Revision, context=context)
    calls = [json.loads(line) for line in (tmp_path / "run/calls.jsonl").read_text().splitlines()]
    assert not calls[-1]["dispatched"]
    assert calls[-1]["cost_usd"] is None
    assert not calls[-1]["usage"]


def test_max_calls_is_shared_across_roles(monkeypatch, tmp_path):
    install_transports(monkeypatch, tmp_path)
    client = LiveLLM(Settings(max_api_calls=1), tmp_path / "run", tmp_path)
    context = dict(essay_id="synthetic", phase="planned", turn=1)
    client.request("reviser", {}, Revision, context=context)
    with pytest.raises(RuntimeError, match="상한"):
        client.request("coordinator_review", {}, Revision, context=context)


def test_retry_attempt_can_share_original_budget_ledger(monkeypatch, tmp_path):
    _, budgets = install_transports(monkeypatch, tmp_path)
    shared = tmp_path / "original_run/budget.json"
    LiveLLM(Settings(), tmp_path / "retry_run", tmp_path, budget_ledger=shared)
    assert all(b.ledger_path == shared for b in budgets)


def test_prepare_from_run_preserves_selection_and_budget(monkeypatch, tmp_path):
    from staged_feedback import cli, metadata
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    selection = {"essays": [dict(essay_id="synthetic", prompt="일상", text="읽는다.",
                                stratum="middle", human_original_total=40)]}
    (old / "selection.json").write_text(json.dumps(selection))
    (old / "evaluation_exclusions.json").write_text('{"exclude_essay_ids":["synthetic"]}')
    called = []
    monkeypatch.setattr(metadata, "prepare_metadata", lambda cfg, directory: called.append(directory))
    assert cli.main(["prepare", "--from-run", str(old), "--run-dir", str(new)]) == 0
    assert json.loads((new / "selection.json").read_text()) == selection
    assert json.loads((new / "budget_reference.json").read_text())["ledger"] == str(old / "budget.json")
    assert (new / "prompts/feedback.txt").exists()
    assert called == [new]


def test_local_metadata_overrides_legacy_path_only_in_new_config(monkeypatch, tmp_path):
    from staged_feedback import adapters
    root = tmp_path / "legacy"
    run = tmp_path / "new_run"
    (run / "scorer_metadata").mkdir(parents=True)
    (run / "scorer_metadata/genres.json").write_text("{}")
    monkeypatch.setattr(adapters, "bootstrap", lambda _: root)
    model = SimpleNamespace(internal_kanana_n=None, scorer=SimpleNamespace(cache_dir="old", metadata_dir="old"))
    monkeypatch.setitem(sys.modules, "para_agent.config", SimpleNamespace(load_config=lambda _: model))
    value = adapters.legacy_config(Settings(), run)
    assert value.scorer.metadata_dir == str(run / "scorer_metadata")
    assert value.scorer.cache_dir == str(run / "cache/scorer")
