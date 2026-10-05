"""Approved Phase 3b continuation: reuse the pilot, one request per new candidate."""

from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import json
import os
import random
from threading import Lock
import time

from feak_tc.runtime.openai import CallBudgetExceeded
from ..api import EnvironmentJSONClient
from ..common import file_sha, read_json, sha_text, write_json
from ..phase2 import read_jsonl
from .api import CorruptionAPI, QC_PROMPT, QCResponse
from .instance_pilot import SPLITS, filter_row, usage_cost
from .qc import qc_payload, validate_judgments

NANOS = 1_000_000_000


class CostUnavailable(RuntimeError):
    """Wait for outstanding calls to settle, or stop without exceeding the cap."""


def usage_nanos(usage):
    return round(usage_cost(usage)["cost_usd_usage_estimate"] * NANOS)


def request_bound(payload):
    # Byte fallback bounds visible tokens above by UTF-8 bytes. Reserve 4096
    # additional tokens for framing/schema overhead, plus the full 8192 output
    # cap, at the most expensive standard input rate (cache writes).
    visible = QC_PROMPT + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    visible += json.dumps(QCResponse.model_json_schema(), ensure_ascii=False)
    byte_count = len(visible.encode("utf-8"))
    return (byte_count+4096)*2500 + 8192*10000, byte_count


class CostLedger:
    """One cumulative Phase 3b ledger, including the immutable pilot's cost/calls."""
    def __init__(self, path, pilot, *, max_api_calls, max_cost_usd=45):
        if not isinstance(max_api_calls, int) or not 100 <= max_api_calls <= 3033:
            raise ValueError("Cumulative --max-api-calls must be 100..3033")
        if not 0 < max_cost_usd <= 45:
            raise ValueError("Phase 3b cumulative cost cap cannot exceed $45")
        self.path, self.limit = path, max_api_calls
        self.cap = round(max_cost_usd*NANOS)
        self.lock_path = path.with_suffix(".lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pilot = pilot
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not path.exists():
                write_json(path, {"phase": "v3_phase3b_cumulative", "pilot": pilot,
                    "cost_cap_nanos": self.cap, "authorized_calls": 3033, "entries": {}})
            else:
                data = read_json(path)
                if data["pilot"] != pilot or data["cost_cap_nanos"] != self.cap:
                    raise ValueError("Pilot provenance or cost authorization changed")

    def snapshot(self):
        data = read_json(self.path)
        settled = sum(e["cost_nanos"] for e in data["entries"].values() if "cost_nanos" in e)
        outstanding = sum(e["bound_nanos"] for e in data["entries"].values() if "cost_nanos" not in e)
        unknown = sum(e["cost_nanos"] for e in data["entries"].values()
                      if "cost_nanos" in e and not e.get("cost_known"))
        return {"calls": self.pilot["calls"]+len(data["entries"]),
            "cost_nanos": self.pilot["cost_nanos"]+settled, "outstanding_nanos": outstanding,
            "cost_usd": (self.pilot["cost_nanos"]+settled)/NANOS,
            "confirmed_cost_usd": (self.pilot["cost_nanos"]+settled-unknown)/NANOS,
            "unknown_cost_upper_usd": unknown/NANOS, "entries": data["entries"]}

    def reserve(self, sid, bound):
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data, state = read_json(self.path), self.snapshot()
            if sid in data["entries"] or sid in self.pilot["ids"]:
                raise ValueError("This candidate has already been sent; no re-judging")
            if state["calls"] >= self.limit:
                raise CallBudgetExceeded("Cumulative Phase 3b request ceiling reached")
            if state["cost_nanos"]+state["outstanding_nanos"]+bound > self.cap:
                raise CostUnavailable("Insufficient remaining cost reservation")
            number = state["calls"]+1
            data["entries"][sid] = {"call": number, "bound_nanos": bound, "state": "reserved"}
            write_json(self.path, data)
            return number

    def settle(self, sid, record):
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = read_json(self.path)
            entry = data["entries"][sid]
            known = bool(record.get("usage"))
            cost = usage_nanos(record["usage"]) if known else entry["bound_nanos"]
            if "cost_nanos" in entry:
                if entry["cost_nanos"] != cost:
                    raise ValueError("Conflicting settlement for the same request")
                return
            entry.update(cost_nanos=cost, cost_known=known, state=record.get("status", "error"))
            if record.get("error_type"):
                entry.update(error_type=record["error_type"], http_status=record.get("http_status"))
            write_json(self.path, data)
            if cost > entry["bound_nanos"]:
                raise ValueError("Observed usage exceeded the conservative reservation; stop")


class RateGate:
    """Use returned account rate-limit headers; do not make discovery requests."""
    def __init__(self):
        self.lock, self.history = Lock(), deque()
        self.token_limit, self.request_limit = 400_000, 60
        self.pause_until = 0.

    def backoff(self, retry_after=None):
        try:
            delay=max(30.,float(retry_after or 30))
        except ValueError:
            delay=30.
        with self.lock:
            self.pause_until=max(self.pause_until,time.monotonic()+delay)

    def observe(self, headers):
        with self.lock:
            try:
                self.token_limit = max(1, int(int(headers["x-ratelimit-limit-tokens"])*.8))
                self.request_limit = min(120, max(1, int(int(headers["x-ratelimit-limit-requests"])*.8)))
            except (KeyError, ValueError):
                pass

    def acquire(self, byte_count):
        # Korean UTF-8 bytes substantially exceed model tokens. Half the visible
        # bytes plus full output allowance is a conservative scheduling estimate.
        estimate = byte_count//2+8192+1024
        while True:
            with self.lock:
                now = time.monotonic()
                while self.history and now-self.history[0][0] >= 61:
                    self.history.popleft()
                if (now >= self.pause_until and (not self.history or (len(self.history) < self.request_limit and
                        sum(x[1] for x in self.history)+estimate <= self.token_limit))):
                    self.history.append((now, estimate))
                    return
                delay = 1.
            time.sleep(delay)


class HeaderClient(EnvironmentJSONClient):
    def __init__(self, cfg, on_record, rate_gate):
        self.rate_gate, self.headers, self.error_metadata = rate_gate, {}, {}
        super().__init__(cfg, lambda r: on_record({**r, "rate_limits": self.headers,
                                                  "error_metadata": self.error_metadata}))

    def _load(self):
        if self._client is None:
            from feak_tc.runtime.openai import APIUnavailable
            if not os.environ.get("OPENAI_API_KEY"):
                raise APIUnavailable("Set OPENAI_API_KEY in the process environment")
            import httpx
            from openai import OpenAI
            def response_hook(response):
                self.headers = {k: v for k, v in response.headers.items() if k.startswith("x-ratelimit-")}
                self.rate_gate.observe(self.headers)
                if response.status_code >= 400:
                    self.error_metadata = {k:response.headers[k] for k in
                        ("retry-after","x-request-id","server","content-type") if k in response.headers}
                    try:
                        response.read()
                        error=response.json().get("error",{})
                        if isinstance(error,dict):
                            self.error_metadata.update({k:error.get(k) for k in ("code","type")})
                    except (ValueError,AttributeError):
                        pass
                    if response.status_code in (429,500,502,503,504):
                        self.rate_gate.backoff(response.headers.get("retry-after"))
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1",
                timeout=600, max_retries=0,
                http_client=httpx.Client(timeout=600, event_hooks={"response": [response_hook]}))


class FullAPI(CorruptionAPI):
    def __init__(self, config, ledger, row, rate_gate):
        self.output = config["paths"]["phase3b_full_output"]
        self.config = config["corruption_finalize"]
        if (self.config["judge_model"], self.config["judge_reasoning_effort"]) != ("gpt-6.1-sol", "high"):
            raise ValueError("Judge model or reasoning effort changed")
        self.ledger, self.sid, self.rate_gate = ledger, row["episode_id"], rate_gate
        self.payload = qc_payload(row)
        bound, self.byte_count = request_bound(self.payload)
        class Reservation:
            def reserve(inner):
                return ledger.reserve(row["episode_id"], bound)
        self.budget = Reservation()

    def log(self, record):
        super().log(record)  # Persist raw usage before settling; replay is safe on resume.
        if not record.get("event"):
            self.ledger.settle(self.sid, record)
            if record.get("status") == "completed" and record.get("response_model") != "gpt-6.1-sol":
                raise ValueError("Unexpected response model")

    def judge(self):
        self.rate_gate.acquire(self.byte_count)
        return super().request("corruption_qc", self.sid, self.payload,
            client_factory=lambda cfg, cb: HeaderClient(cfg, cb, self.rate_gate))


def load_population(config):
    source = config["paths"]["phase3b_output"]
    for name, digest in read_json(source/"frozen_pilot_hashes.json").items():
        if file_sha(source/name) != digest:
            raise ValueError(f"Frozen pilot artifact changed: {name}")
    rows = [r for split in SPLITS for r in read_jsonl(source/f"candidates_{split}.jsonl")]
    pilot_rows = read_jsonl(source/"pilot_sample.jsonl")
    lookup = {r["episode_id"]: r for r in rows}
    for row in pilot_rows:
        if qc_payload(lookup[row["episode_id"]]) != qc_payload(row):
            raise ValueError("Pilot no longer matches candidate population")
        filter_row(row)
    pilot_calls = [r for r in read_jsonl(source/"calls.jsonl") if r.get("usage")]
    if len(pilot_calls) != 100 or len(rows) != 3033 or len(lookup) != len(rows):
        raise ValueError("Population or pilot count differs from authorization")
    pilot = {"calls": 100, "cost_nanos": sum(usage_nanos(r["usage"]) for r in pilot_calls),
        "ids": sorted(r["episode_id"] for r in pilot_rows),
        "calls_sha256": file_sha(source/"calls.jsonl"), "judgments_sha256": file_sha(source/"pilot_sample.jsonl")}
    return rows, pilot_rows, pilot


def recovered_response(call):
    if call.get("status") != "completed" or call.get("response_model") != "gpt-6.1-sol":
        raise ValueError("Failed or unexpected-model response; no automatic retry")
    parsed = QCResponse.model_validate_json(call["raw"]).model_dump()
    return {**parsed, "model": call["model"], "reasoning_effort": "high", "phase_call": call["phase_call"],
        "prompt_sha256": call["prompt_sha256"], "usage": call["usage"], "response_id": call.get("response_id")}


def run_full(config, *, max_api_calls, max_cost_usd=45, workers=12):
    output = config["paths"]["phase3b_full_output"]
    output.mkdir(parents=True, exist_ok=True)
    with (output/"run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_full(config, max_api_calls, max_cost_usd, workers)


def _run_full(config, max_api_calls, max_cost_usd, workers):
    output = config["paths"]["phase3b_full_output"]
    rows, pilot_rows, pilot = load_population(config)
    ledger = CostLedger(output/"cost_ledger.json", pilot, max_api_calls=max_api_calls, max_cost_usd=max_cost_usd)
    lookup = {r["episode_id"]: r for r in rows}
    # Per-essay results plus the durable raw log allow recovery without repeating
    # a request if a process stops between receiving a response and saving a row.
    completed = {r["episode_id"]: r["llm_judgment"] for r in pilot_rows}
    calls = read_jsonl(output/"calls.jsonl") if (output/"calls.jsonl").exists() else []
    actual = [r for r in calls if not r.get("event")]
    counts = Counter(r["sample_id"] for r in actual)
    if any(v != 1 for v in counts.values()) or set(counts) & set(pilot["ids"]):
        raise ValueError("Duplicate calls or pilot re-judgment")
    quarantined = []
    for call in actual:
        sid = call["sample_id"]
        ledger.settle(sid, call)
        if (call.get("error_type") in {"APITimeoutError", "APIConnectionError", "InternalServerError", "RateLimitError"}
                and not call.get("usage")):
            quarantined.append({"episode_id": sid, "error": call["error_type"],
                                "retry_requires_user_decision": True})
            continue
        response = recovered_response(call)
        validate_judgments(lookup[sid], response)
        write_json(output/"judgments"/(sid+".json"), response)
        completed[sid] = response
    unfinished = set(ledger.snapshot()["entries"])-set(counts)
    if unfinished:
        raise ValueError("Unsettled prior requests need inspection; refusing to send them again")
    quarantined_ids = {r["episode_id"] for r in quarantined}
    pending = [r for r in rows if r["episode_id"] not in completed and r["episode_id"] not in quarantined_ids]
    pending.sort(key=lambda r: r["episode_id"])
    random.Random(41).shuffle(pending)
    manifest = {"population": len(rows), "pilot_reused": 100, "pending_ids": [r["episode_id"] for r in pending],
        "prompt_sha256": sha_text(QC_PROMPT), "model": "gpt-6.1-sol", "reasoning_effort": "high",
        "cost_cap_usd": max_cost_usd, "cumulative_call_cap": max_api_calls}
    if not (output/"preregistration.json").exists():
        write_json(output/"preregistration.json", manifest)
    gate, errors, stop_reason = RateGate(), [], None
    queue = deque(pending)
    def save_status():
        state = ledger.snapshot()
        status = {k: state[k] for k in ("calls", "cost_nanos", "cost_usd", "outstanding_nanos",
                                       "confirmed_cost_usd", "unknown_cost_upper_usd")}
        status.update(judged=len(completed), total=len(rows), remaining=len(rows)-len(completed),
                      errors=errors, quarantined=quarantined, stop_reason=stop_reason,
                      complete=len(completed)==len(rows), phase4_started=False)
        write_json(output/"judging_status.json", status)
        return status
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        while queue or futures:
            # First request discovers this account's rate limits without a paid
            # extra query; later requests use the same prompt and token cap.
            concurrency = 1 if len(completed)==100 and not actual else workers
            while queue and len(futures)<concurrency and not stop_reason:
                row = queue.popleft()
                futures[executor.submit(FullAPI(config, ledger, row, gate).judge)] = row
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                row = futures.pop(future)
                try:
                    response = future.result()
                    validate_judgments(row, response)
                    write_json(output/"judgments"/(row["episode_id"]+".json"), response)
                    completed[row["episode_id"]] = response
                except CostUnavailable:
                    queue.appendleft(row)
                    fresh = ledger.snapshot()
                    bound, _ = request_bound(qc_payload(row))
                    if not futures and fresh["cost_nanos"]+fresh["outstanding_nanos"]+bound > ledger.cap:
                        stop_reason = "cost_cap_insufficient_for_next_request_reservation"
                except CallBudgetExceeded:
                    queue.appendleft(row)
                    stop_reason = "request_cap"
                except Exception as error:
                    entry = ledger.snapshot()["entries"].get(row["episode_id"], {})
                    if entry.get("error_type") in {"APITimeoutError", "APIConnectionError", "InternalServerError", "RateLimitError"}:
                        quarantined.append({"episode_id": row["episode_id"], "error": entry["error_type"],
                                            "retry_requires_user_decision": True})
                    else:
                        errors.append({"episode_id": row["episode_id"], "error": type(error).__name__})
                        stop_reason = "request_error_no_automatic_retry"
                state = save_status()
                if state["cost_usd"] > max_cost_usd:
                    stop_reason = "cost_cap_exceeded"
                print(f"Judged {len(completed)}/{len(rows)}; calls={state['calls']}; "
                      f"usage_cost=${state['confirmed_cost_usd']:.6f}; "
                      f"unknown_cost_upper=${state['unknown_cost_upper_usd']:.6f}; stop={stop_reason}", flush=True)
    return save_status()
