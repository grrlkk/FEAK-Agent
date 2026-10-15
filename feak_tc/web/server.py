"""Local, single-job web console backed by the same CLI and durable event logs."""

from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urlparse
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).with_name("static")


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=4000)
    question: str = Field(min_length=1, max_length=500)
    max_steps: int = Field(default=3, ge=1, le=6)
    candidates: int = Field(default=2, ge=1, le=3)
    offline_smoke: bool = False

    @field_validator("text", "question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("글과 과제를 입력해 주세요.")
        return value


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_events(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
            if isinstance(row, dict) and isinstance(row.get("event"), str):
                rows.append(row)
        except ValueError:
            # A writer may still be completing the final line. Retry on next poll.
            continue
    return rows


def recover_result(request, events, status, message):
    """Never expose an accepted-but-not-guarded text after cancellation/crash."""
    checkpoints = {0: {"id": 0, "parent_id": None, "text": request["text"], "safe": True}}
    current = 0
    pending = None
    for row in events:
        if row["event"] == "start":
            checkpoints[0]["diagnosis"] = row.get("diagnosis")
        elif row["event"] == "accept":
            pending = row["checkpoint_id"]
            checkpoints[pending] = {"id": pending, "parent_id": row["parent_id"],
                                    "text": row["text"], "diagnosis": row.get("diagnosis"), "safe": False}
        elif row["event"] == "guard" and pending is not None:
            if row.get("verdict") and not row.get("reasons") and not row.get("error"):
                current = pending
                checkpoints[current]["safe"] = True
            pending = None
        elif row["event"] == "rollback":
            target = row["to_checkpoint"]
            if target in checkpoints and checkpoints[target]["safe"]:
                current = target
            pending = None
    return {"schema_version": "training_free_agent_v1", "status": status,
            "stop_reason": "user_cancelled" if status == "cancelled" else "process_error",
            "message": message, "original_text": request["text"], "question": request["question"],
            "final_text": checkpoints[current]["text"], "final_checkpoint_id": current,
            "checkpoints": list(checkpoints.values()), "events": events,
            "accepted": sum(row["event"] == "accept" for row in events),
            "rollbacks": sum(row["event"] == "rollback" for row in events)}


class RunManager:
    def __init__(self, runs_dir, config, history_dir=None, popen=subprocess.Popen):
        self.runs_dir = Path(runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.config = Path(config).resolve()
        self.cfg = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        self.popen = popen
        self.lock = threading.RLock()
        self.runs = {}
        self.active = None
        self._load_runs(history_dir)

    def _load_runs(self, history_dir):
        for directory in self.runs_dir.iterdir():
            if not directory.is_dir() or not re.fullmatch(r"[0-9a-f]{32}", directory.name):
                continue
            meta = read_json(directory / "request.json")
            if not isinstance(meta, dict) or "request" not in meta:
                continue
            result = read_json(directory / "result.json")
            if not result:
                events = read_events(directory / "result.events.jsonl")
                result = recover_result(meta["request"], events, "error", "이전 서버 실행이 중단되었습니다.")
                result["runtime"] = {"offline_smoke": meta["request"].get("offline_smoke", False),
                                     "config": {"local_llm": {"model": meta["model"]}}}
                (directory / "result.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            self.runs[directory.name] = {**meta, "id": directory.name, "directory": directory,
                                          "status": result.get("status", "error"), "result": result}
        if history_dir:
            for path in sorted(Path(history_dir).glob("local_agent*.json")):
                result = read_json(path)
                if not isinstance(result, dict) or result.get("schema_version") != "training_free_agent_v1":
                    continue
                run_id = "history-" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
                runtime = result.get("runtime", {})
                self.runs[run_id] = {
                    "id": run_id, "status": result.get("status", "completed"), "result": result,
                    "created_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                    "model": runtime.get("config", {}).get("local_llm", {}).get("model", "unknown"),
                    "request": {"question": result.get("question", path.stem),
                                "text": result.get("original_text", ""),
                                "offline_smoke": runtime.get("offline_smoke", False)},
                    "imported": True,
                }

    def metadata(self):
        return {"model": self.cfg["local_llm"]["model"], "diagnoser": "Kanana + 학습된 LoRA",
                "max_steps": min(6, self.cfg["controller"]["max_steps"]),
                "candidates": min(3, self.cfg["controller"]["candidates_per_step"]),
                "example": (PROJECT_ROOT / "examples/local_agent_essay.txt").read_text(encoding="utf-8"),
                "example_question": "학교에서 휴대전화 사용을 어떻게 정하면 좋을지 자신의 주장과 이유를 쓰시오."}

    def start(self, request):
        with self.lock:
            if self.active is not None:
                raise RuntimeError("이미 실행 중인 글이 있습니다. 완료를 기다리거나 실행을 중단해 주세요.")
            run_id = uuid.uuid4().hex
            directory = self.runs_dir / run_id
            directory.mkdir()
            meta = {"id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
                    "request": request.model_dump(), "model": self.cfg["local_llm"]["model"]}
            (directory / "request.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            (directory / "input.txt").write_text(request.text, encoding="utf-8")
            command = [sys.executable, "-u", str(PROJECT_ROOT / "scripts/run_agent.py"),
                       "--text-file", str(directory / "input.txt"), "--question", request.question,
                       "--config", str(self.config), "--output", str(directory / "result.json"),
                       "--max-steps", str(request.max_steps), "--candidates", str(request.candidates)]
            if request.offline_smoke:
                command.append("--offline-smoke")
            log = (directory / "process.log").open("w", encoding="utf-8")
            try:
                process = self.popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            except Exception:
                log.close()
                raise
            self.runs[run_id] = {**meta, "status": "running", "directory": directory,
                                 "process": process, "log": log, "cancel_requested": False}
            self.active = run_id
            watcher = threading.Thread(target=self._watch, args=(run_id,), daemon=True)
            self.runs[run_id]["watcher"] = watcher
            watcher.start()
            return run_id

    def _watch(self, run_id):
        run = self.runs[run_id]
        returncode = run["process"].wait()
        with self.lock:
            run["log"].close()
            events = read_events(run["directory"] / "result.events.jsonl")
            result = read_json(run["directory"] / "result.json")
            if run["cancel_requested"]:
                status, message = "cancelled", "사용자가 실행을 중단했습니다. 마지막으로 검증된 글을 보관합니다."
            elif returncode != 0 or not result or result.get("status") == "error":
                status, message = "error", "실행을 마치지 못했습니다. 오류 내용을 확인해 주세요."
            else:
                status, message = "completed", ""
            if not result or status == "cancelled":
                result = recover_result(run["request"], events, status, message)
                result["runtime"] = {"offline_smoke": run["request"]["offline_smoke"],
                                     "config": {"local_llm": {"model": run["model"]}}}
                (run["directory"] / "result.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            run.update(status=status, result=result, returncode=returncode)
            self.active = None

    def list(self):
        with self.lock:
            return [{key: run.get(key) for key in ("id", "created_at", "status", "model", "request", "imported")}
                    for run in sorted(self.runs.values(), key=lambda row: row["created_at"], reverse=True)]

    def get(self, run_id, cursor=0):
        with self.lock:
            run = self.runs[run_id]
            result = run.get("result")
            events = result.get("events", []) if result else read_events(run["directory"] / "result.events.jsonl")
            error = None
            if run["status"] == "error":
                errors = [row.get("message", row.get("error")) for row in events if row["event"] == "error"]
                error = next((value for value in reversed(errors) if value), None)
                error = error or (result or {}).get("message", "로컬 모델 실행을 확인해 주세요.")
            return {"id": run_id, "status": run["status"], "model": run["model"],
                    "request": run["request"], "created_at": run["created_at"],
                    "events": events[cursor:], "cursor": len(events), "result": result, "error": error}

    def cancel(self, run_id):
        with self.lock:
            run = self.runs[run_id]
            process = run.get("process")
            if run["status"] != "running" or process is None or process.poll() is not None:
                return
            run["cancel_requested"] = True
            # Only this manager's own process group, including Kanana/feature workers.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    def close(self):
        with self.lock:
            active = self.active
            watcher = self.runs[active].get("watcher") if active else None
        if active:
            self.cancel(active)
            if watcher:
                watcher.join(timeout=10)


def make_server(manager, host="127.0.0.1", port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, body, content_type="application/json; charset=utf-8"):
            data = json.dumps(body, ensure_ascii=False).encode() if content_type.startswith("application/json") else body
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if path in ("/", "/app.js", "/styles.css"):
                name, mime = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                              "/styles.css": ("styles.css", "text/css")}[path]
                return self.send(200, (STATIC_DIR / name).read_bytes(), mime + "; charset=utf-8")
            if path == "/api/meta":
                return self.send(200, manager.metadata())
            if path == "/api/runs":
                return self.send(200, {"runs": manager.list(), "active_id": manager.active})
            match = re.fullmatch(r"/api/runs/([\w-]+)(/result)?", path)
            if match:
                try:
                    cursor = int(parse_qs(parsed.query).get("cursor", ["0"])[0])
                    if cursor < 0:
                        raise ValueError()
                    state = manager.get(match[1], cursor)
                    if match[2]:
                        if state["result"] is None:
                            return self.send(409, {"error": "실행이 아직 끝나지 않았습니다."})
                        return self.send(200, state["result"])
                    return self.send(200, state)
                except KeyError:
                    return self.send(404, {"error": "실행 기록을 찾을 수 없습니다."})
                except ValueError:
                    return self.send(400, {"error": "잘못된 cursor입니다."})
            self.send(404, {"error": "찾을 수 없는 주소입니다."})

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.send(403, {"error": "같은 웹 화면에서 요청해 주세요."})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send(415, {"error": "JSON 요청이 필요합니다."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 64000:
                    raise ValueError("요청 크기가 올바르지 않습니다.")
                body = json.loads(self.rfile.read(length))
                if self.path == "/api/runs":
                    request = RunRequest.model_validate(body)
                    return self.send(202, {"id": manager.start(request)})
                match = re.fullmatch(r"/api/runs/([\w-]+)/cancel", self.path)
                if match:
                    manager.cancel(match[1])
                    return self.send(200, {"ok": True})
                return self.send(404, {"error": "찾을 수 없는 주소입니다."})
            except KeyError:
                return self.send(404, {"error": "실행 기록을 찾을 수 없습니다."})
            except ValueError:
                return self.send(400, {"error": "글·과제와 실행 설정을 확인해 주세요. 글은 최대 4,000자입니다."})
            except RuntimeError as exc:
                return self.send(409, {"error": str(exc)})
            except OSError:
                return self.send(500, {"error": "실행 파일을 준비하지 못했습니다. 서버 로그를 확인해 주세요."})

    return ThreadingHTTPServer((host, port), Handler)
