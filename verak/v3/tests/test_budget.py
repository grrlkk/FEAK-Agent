from concurrent.futures import ThreadPoolExecutor

import pytest

from feak_tc.runtime.openai import CallBudgetExceeded
from verak.v3.api import PhaseBudget, PhaseTeacher, StrictResponse


def test_budget_is_shared_across_threads_restarts_and_lower_cli_limits(tmp_path):
    path = tmp_path / "phase.json"
    def reserve(_):
        try:
            return PhaseBudget(path, 7).reserve()
        except CallBudgetExceeded:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reserve, range(20)))
    assert sorted(x for x in results if x is not None) == list(range(1, 8))
    assert PhaseBudget(path, 7).used == 7
    with pytest.raises(CallBudgetExceeded):
        PhaseBudget(path, 7).reserve()
    with pytest.raises(CallBudgetExceeded):
        PhaseBudget(path, 6).reserve()
    with pytest.raises(ValueError):
        PhaseBudget(path, 701)


def test_zero_budget_stops_before_any_api_request(tmp_path):
    budget = PhaseBudget(tmp_path / "phase.json", 0)
    with pytest.raises(CallBudgetExceeded):
        budget.reserve()
    assert budget.used == 0


def test_failed_requests_count_and_no_request_is_made_after_budget(tmp_path):
    class Response(StrictResponse):
        ok: bool

    class Client:
        requests = 0

        def __init__(self, config, on_record):
            pass

        def _load(self):
            pass

        def start_sample(self, sample_id):
            pass

        def __call__(self, **kwargs):
            Client.requests += 1
            if Client.requests == 1:
                raise RuntimeError("simulated transport failure")
            return {"ok": True}

        def close(self):
            pass

    config = {"paths": {"output": tmp_path}, "teacher": {"model": "gpt-5-mini",
        "reasoning_effort": "low", "phase_api_ceiling": 700, "max_output_tokens": 256, "timeout_s": 5}}
    teacher = PhaseTeacher(config, 2, client_factory=Client)
    with pytest.raises(RuntimeError):
        teacher.request(Response, "prompt", {}, stage="test", sample_id="a")
    assert teacher.request(Response, "prompt", {}, stage="test", sample_id="b").ok
    with pytest.raises(CallBudgetExceeded):
        teacher.request(Response, "prompt", {}, stage="test", sample_id="c")
    assert teacher.budget.used == Client.requests == 2
