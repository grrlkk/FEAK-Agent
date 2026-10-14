import json
from pathlib import Path
import signal
import subprocess
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from pydantic import ValidationError

from feak_tc.web.server import RunManager, RunRequest, make_server, read_events, recover_result


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("local_llm:\n  model: kakaocorp/kanana-1.5-8b-instruct-2505\n"
                    "controller:\n  max_steps: 3\n  candidates_per_step: 2\n")
    return path


@pytest.mark.parametrize("update", [
    {"text": " "}, {"question": "\n"}, {"text": "가" * 4001},
    {"question": "가" * 501}, {"max_steps": 0}, {"max_steps": 7},
    {"max_steps": True}, {"candidates": 4}, {"candidates": "2"},
    {"offline_smoke": "false"}, {"config": "/tmp/arbitrary.yaml"},
])
def test_request_limits(update):
    with pytest.raises(ValidationError):
        RunRequest.model_validate({"text": "나의 글", "question": "과제", **update})


def test_incremental_events(tmp_path):
    path = tmp_path / "events.jsonl"
    assert read_events(path) == []
    path.write_text('{"event":"start"}\n42\n{"event":')
    assert read_events(path) == [{"event": "start"}]
    path.write_text('{"event":"start"}\n{"event":"plan"}\n')
    assert len(read_events(path)) == 2


def accept(checkpoint, parent=0):
    return {"event": "accept", "checkpoint_id": checkpoint, "parent_id": parent,
            "text": f"수정 {checkpoint}", "diagnosis": {"rubrics": {}}}


def guard(reasons=None):
    return {"event": "guard", "verdict": {"preservation": {"label": "pass"}},
            "reasons": reasons or []}


@pytest.mark.parametrize("events,expected", [
    ([], 0), ([accept(1)], 0), ([accept(1), guard()], 1),
    ([accept(1), guard(["guard_preservation"])], 0),
    ([accept(1), {"event": "guard", "error": "unavailable"}], 0),
    ([accept(1), guard(), accept(2, 1)], 1),
    ([accept(1), guard(), accept(2, 1), guard(),
      {"event": "rollback", "to_checkpoint": 1}], 1),
])
def test_recovery_never_returns_unguarded_candidate(events, expected):
    result = recover_result({"text": "원문", "question": "과제"}, events, "cancelled", "중단")
    assert result["final_checkpoint_id"] == expected
    assert result["final_text"] == (f"수정 {expected}" if expected else "원문")
    assert result["events"] == events
    assert result["stop_reason"] == "user_cancelled"


class FakeProcess:
    pid = 987654321

    def __init__(self):
        self.done = threading.Event()
        self.returncode = None

    def wait(self, timeout=None):
        if not self.done.wait(timeout):
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode

    def poll(self):
        return self.returncode

    def finish(self, code=0):
        self.returncode = code
        self.done.set()


@pytest.fixture
def manager(tmp_path, config):
    process = FakeProcess()
    commands = []

    def spawn(command, **kwargs):
        commands.append((command, kwargs))
        return process

    manager = RunManager(tmp_path / "runs", config, popen=spawn)
    manager.test_process, manager.test_commands = process, commands
    yield manager
    process.finish(1)
    for run in manager.runs.values():
        if run.get("watcher"):
            run["watcher"].join(timeout=3)


def test_single_job_fixed_command_and_persistence(manager):
    request = RunRequest(text="나의 글", question="과제; $(실행하지 않음)", offline_smoke=True)
    run_id = manager.start(request)
    with pytest.raises(RuntimeError):
        manager.start(request)
    command, kwargs = manager.test_commands[0]
    assert "--offline-smoke" in command
    assert request.question in command
    assert kwargs["start_new_session"] is True
    assert "shell" not in kwargs
    directory = manager.runs[run_id]["directory"]
    assert (directory / "input.txt").read_text() == request.text
    (directory / "result.events.jsonl").write_text('{"event":"phase","stage":"diagnose"}\n')
    assert manager.get(run_id)["cursor"] == 1
    assert manager.get(run_id, 1)["events"] == []
    result = recover_result(request.model_dump(), [], "completed", "")
    (directory / "result.json").write_text(json.dumps(result))
    manager.test_process.finish()
    manager.runs[run_id]["watcher"].join(timeout=3)
    assert manager.active is None
    assert manager.get(run_id)["status"] == "completed"
    restored = RunManager(manager.runs_dir, manager.config)
    assert restored.get(run_id)["result"] == result


def test_cancel_own_group_saves_safe_text(manager, monkeypatch):
    run_id = manager.start(RunRequest(text="원문", question="과제"))
    directory = manager.runs[run_id]["directory"]
    events = [accept(1), guard(), accept(2, 1)]
    (directory / "result.events.jsonl").write_text("\n".join(map(json.dumps, events)))
    signals = []

    def killpg(pid, sig):
        signals.append((pid, sig))
        manager.test_process.finish(-sig)

    monkeypatch.setattr("feak_tc.web.server.os.killpg", killpg)
    manager.close()
    assert signals == [(manager.test_process.pid, signal.SIGTERM)]
    state = manager.get(run_id)
    assert state["status"] == "cancelled"
    assert state["result"]["final_text"] == "수정 1"
    assert json.loads((directory / "result.json").read_text())["status"] == "cancelled"


def test_crash_recovers_with_model_metadata(manager):
    run_id = manager.start(RunRequest(text="원문", question="과제", offline_smoke=True))
    manager.test_process.finish(1)
    manager.runs[run_id]["watcher"].join(timeout=3)
    state = manager.get(run_id)
    assert state["status"] == "error"
    assert state["result"]["final_text"] == "원문"
    assert state["result"]["runtime"]["offline_smoke"] is True


def test_abandoned_run_and_historical_model(tmp_path, config):
    directory = tmp_path / "runs" / ("a" * 32)
    directory.mkdir(parents=True)
    meta = {"created_at": "2026-09-11", "model": "kanana", "request": {"text": "원문", "question": "과제"}}
    (directory / "request.json").write_text(json.dumps(meta))
    (directory / "result.events.jsonl").write_text(json.dumps(accept(1)))
    history = tmp_path / "history"
    history.mkdir()
    (history / "local_agent_old.json").write_text(json.dumps({
        "schema_version": "training_free_agent_v1", "status": "completed",
        "runtime": {"config": {"local_llm": {"model": "Qwen/Qwen2.5-7B-Instruct"}}},
        "original_text": "이전 글", "question": "과제", "events": [],
    }))
    manager = RunManager(directory.parent, config, history)
    assert manager.get("a" * 32)["result"]["final_text"] == "원문"
    assert (directory / "result.json").exists()
    imported = next(row for row in manager.list() if row["imported"])
    assert imported["model"].startswith("Qwen/")
    assert "kanana" in manager.metadata()["model"]


@pytest.fixture
def http_server(manager):
    server = make_server(manager, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def fetch(base, path, body=None, headers=None):
    request = Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                      headers=headers or ({"Content-Type": "application/json"} if body is not None else {}))
    try:
        return urlopen(request, timeout=5)
    except HTTPError as response:
        return response


def test_http_static_and_invalid_requests(http_server):
    response = fetch(http_server, "/")
    assert response.status == 200
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert "글쓰기" in response.read().decode()
    assert fetch(http_server, "/app.js").headers["Content-Type"].startswith("text/javascript")
    assert fetch(http_server, "/api/meta").status == 200
    assert fetch(http_server, "/../../configs/agent_local.yaml").status == 404
    assert fetch(http_server, "/api/runs/unknown").status == 404
    assert fetch(http_server, "/api/runs/unknown?cursor=-1").status == 400
    assert fetch(http_server, "/api/runs", {}).status == 400
    assert fetch(http_server, "/api/runs", {}, {"Content-Type": "text/plain"}).status == 415
    assert fetch(http_server, "/api/runs", {}, {
        "Content-Type": "application/json", "Origin": "https://untrusted.example"}).status == 403


def test_http_start_and_busy(http_server, manager):
    payload = {"text": "원문", "question": "과제", "offline_smoke": True}
    response = fetch(http_server, "/api/runs", payload)
    assert response.status == 202
    run_id = json.load(response)["id"]
    assert json.load(fetch(http_server, "/api/runs"))["active_id"] == run_id
    assert fetch(http_server, "/api/runs", payload).status == 409
    assert fetch(http_server, f"/api/runs/{run_id}/result").status == 409
