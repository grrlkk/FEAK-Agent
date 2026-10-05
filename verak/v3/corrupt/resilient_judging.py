"""User-approved Phase 3b retry accounting; one valid judgment per candidate."""

from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import json
import random
from threading import Lock
import time

from feak_tc.runtime.openai import CallBudgetExceeded
from ..common import file_sha, read_json, write_json
from ..phase2 import read_jsonl
from .api import CorruptionAPI
from .full_judging import (CostUnavailable, FullAPI, NANOS, RateGate, load_population,
                          recovered_response, request_bound, usage_nanos)
from .qc import qc_payload, validate_judgments

MAX_CALLS = 7797  # 1445 complete + 155*(1+3) + 1433*(1+3), including the pilot.
RETRY_DELAYS = (10, 40, 120)


def is_5xx(record):
    status = record.get("http_status")
    return isinstance(status, int) and 500 <= status < 600


def retryable(record):
    return is_5xx(record) or record.get("error_type") == "APITimeoutError"


def accounting(record, bound):
    if record.get("usage"):
        return usage_nanos(record["usage"]), "confirmed_usage"
    if is_5xx(record):
        return 0, "unreported_5xx_zero_by_user_decision"
    if record.get("error_type") == "APITimeoutError":
        return bound, "timeout_reservation"
    # Only timeouts retain a reservation after settlement. Other failures with
    # no confirmed usage are uncharged and non-retryable, so the runner stops.
    return 0, "no_confirmed_usage_non_retryable"


class RetryLedger:
    """Every HTTP attempt has a separate durable reservation and settlement."""

    def __init__(self, path, pilot, *, max_api_calls, max_cost_usd=50, prior_calls=None):
        if not isinstance(max_api_calls, int) or not 100 <= max_api_calls <= MAX_CALLS:
            raise ValueError(f"Cumulative call limit must be 100..{MAX_CALLS}")
        if not 0 < max_cost_usd <= 50:
            raise ValueError("Authorized confirmed-usage plus timeout cap is at most $50")
        self.path, self.pilot, self.limit = path, pilot, max_api_calls
        self.cap = round(max_cost_usd*NANOS)
        self.lock_path = path.with_suffix(".lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not path.exists():
                write_json(path, self._empty())
            data = read_json(path)
            if data["pilot"] != pilot:
                raise ValueError("Pilot provenance changed")
            if data.get("schema_version", 1) == 1:
                self._migrate(data, prior_calls)
                data = read_json(path)
            if data["schema_version"] != 2 or data["cost_cap_nanos"] != self.cap:
                raise ValueError("Ledger policy/cap differs from this authorization")

    def _empty(self):
        return {"phase":"v3_phase3b_cumulative", "schema_version":2, "pilot":self.pilot,
            "cost_policy":"confirmed_usage_plus_timeout_reservations_5xx_without_usage_zero",
            "cost_cap_nanos":self.cap, "authorized_calls":MAX_CALLS, "max_attempts_per_candidate":4,
            "entries":{}}

    def _migrate(self, data, calls):
        if calls is None:
            raise ValueError("Raw prior calls required for authorized cost reconciliation")
        lookup = {r["sample_id"]:r for r in calls}
        if len(lookup) != len(calls) or set(lookup) != set(data["entries"]):
            raise ValueError("Legacy calls/ledger mismatch or unsettled request")
        backup = self.path.with_name("cost_ledger_before_retry_approval.json")
        if not backup.exists():
            # Store the original bytes, not a reconstruction with changed formatting.
            backup.write_bytes(self.path.read_bytes())
        elif file_sha(backup) != file_sha(self.path):
            raise ValueError("Legacy ledger differs from the preserved migration source")
        migrated = self._empty()
        for sid, old in data["entries"].items():
            record = lookup[sid]
            cost, category = accounting(record, old["bound_nanos"])
            migrated["entries"][sid] = {**old, "episode_id":sid, "attempt":1,
                "legacy_cost_nanos":old.get("cost_nanos"), "cost_nanos":cost, "accounting":category,
                "state":record.get("status", "error"), "error_type":record.get("error_type"),
                "http_status":record.get("http_status"), "cost_known":bool(record.get("usage"))}
        migrated["migration"] = {"source_path":str(backup), "source_sha256":file_sha(backup),
            "time":time.time(), "user_approved":True, "source_calls":len(calls)+self.pilot["calls"]}
        write_json(self.path, migrated)

    def snapshot(self):
        data = read_json(self.path)
        entries = data["entries"]
        confirmed = self.pilot["cost_nanos"] + sum(e.get("cost_nanos",0) for e in entries.values()
            if e.get("accounting") == "confirmed_usage")
        timeout = sum(e.get("cost_nanos",0) for e in entries.values() if e.get("accounting") == "timeout_reservation")
        unresolved = sum(e.get("cost_nanos",0) for e in entries.values() if e.get("accounting") == "unresolved_reservation")
        outstanding = sum(e["bound_nanos"] for e in entries.values() if "cost_nanos" not in e)
        return {"calls":self.pilot["calls"]+len(entries), "cost_nanos":confirmed+timeout+unresolved,
            "cost_usd":(confirmed+timeout+unresolved)/NANOS, "confirmed_cost_usd":confirmed/NANOS,
            "timeout_reserved_usd":timeout/NANOS, "unknown_cost_upper_usd":(timeout+unresolved)/NANOS,
            "unresolved_reserved_usd":unresolved/NANOS, "outstanding_nanos":outstanding,
            "zero_usage_5xx":sum(e.get("accounting") == "unreported_5xx_zero_by_user_decision" for e in entries.values()),
            "entries":entries}

    def reserve(self, sid, bound):
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data, state = read_json(self.path), self.snapshot()
            previous = sorted((e for e in data["entries"].values() if e["episode_id"] == sid), key=lambda e:e["attempt"])
            if sid in self.pilot["ids"] or any(e["state"] == "completed" for e in previous):
                raise ValueError("Completed candidate must not be re-judged")
            if previous and ("cost_nanos" not in previous[-1] or not retryable(previous[-1])):
                raise ValueError("Only settled 5xx/timeouts can be retried")
            attempt = len(previous)+1
            if attempt > 4:
                raise ValueError("Three retries exhausted for this candidate")
            if state["calls"] >= self.limit:
                raise CallBudgetExceeded("Cumulative attempt limit reached")
            if state["cost_nanos"]+state["outstanding_nanos"]+bound > self.cap:
                raise CostUnavailable("Insufficient confirmed-usage/timeout/inflight reservation")
            attempt_id = f"{sid}:attempt:{attempt}"
            if attempt_id in data["entries"]:
                raise ValueError("Duplicate attempt identifier")
            entry = {"episode_id":sid, "attempt":attempt, "call":state["calls"]+1,
                     "bound_nanos":bound, "state":"reserved", "started_at":time.time()}
            data["entries"][attempt_id] = entry
            write_json(self.path, data)
            return {"attempt_id":attempt_id, **entry}

    def settle(self, attempt_id, record):
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = read_json(self.path)
            entry = data["entries"][attempt_id]
            if record["sample_id"] != entry["episode_id"]:
                raise ValueError("Attempt belongs to another candidate")
            cost, category = accounting(record, entry["bound_nanos"])
            if "cost_nanos" in entry:
                if (entry["cost_nanos"], entry["accounting"]) != (cost, category):
                    raise ValueError("Conflicting attempt settlement")
                return
            entry.update(cost_nanos=cost, accounting=category, cost_known=bool(record.get("usage")),
                state=record.get("status", "error"), error_type=record.get("error_type"),
                http_status=record.get("http_status"), finished_at=time.time())
            write_json(self.path, data)
            if cost > entry["bound_nanos"]:
                raise ValueError("Usage exceeded reserved bound; stop")


class ReliabilityGate(RateGate):
    """Global 50-response circuit breaker, persisted across process restarts."""

    def __init__(self, path, previous_calls=(), *, now=time.time):
        super().__init__()
        self.path, self.now = path, now
        if path.exists():
            self.state = read_json(path)
        else:
            self.state = {"recent":[is_5xx(r) for r in list(previous_calls)[-50:]],
                          "pause_until_epoch":0., "events":[]}
            self._trip_if_needed()
            write_json(path, self.state)

    def _trip_if_needed(self):
        recent = self.state["recent"]
        if len(recent) == 50 and sum(recent) > 10:
            deadline = self.now()+600
            self.state["pause_until_epoch"] = max(self.state["pause_until_epoch"], deadline)
            self.state["events"].append({"time":self.now(), "5xx":sum(recent), "window":50,
                                         "pause_seconds":600, "resume_after":deadline})
            # Require a fresh response window after cooling down; don't trigger
            # another ten minutes repeatedly from the same old failure cluster.
            self.state["recent"] = []

    def record(self, record):
        with self.lock:
            self.state["recent"] = (self.state["recent"]+[is_5xx(record)])[-50:]
            self._trip_if_needed()
            write_json(self.path, self.state)

    def backoff(self, retry_after=None):
        # Candidate retries use 10/40/120 seconds. Respect an additional server
        # delay if supplied; no extra request is sent to discover limits.
        if retry_after is None:
            return
        try:
            delay = max(0.,float(retry_after))
        except (ValueError,TypeError):
            return
        with self.lock:
            self.state["pause_until_epoch"] = max(self.state["pause_until_epoch"],self.now()+delay)
            write_json(self.path,self.state)

    def seconds_remaining(self):
        with self.lock:
            return max(0.,self.state["pause_until_epoch"]-self.now())

    def acquire(self, byte_count):
        while self.seconds_remaining() > 0:
            time.sleep(min(1.,self.seconds_remaining()))
        with self.lock:
            self.pause_until = time.monotonic()+max(0.,self.state["pause_until_epoch"]-self.now())
        super().acquire(byte_count)
        # The breaker may have tripped while rate-limit pacing was in progress.
        while self.seconds_remaining() > 0:
            time.sleep(min(1.,self.seconds_remaining()))


class AttemptAPI(FullAPI):
    def __init__(self, config, ledger, row, rate_gate):
        super().__init__(config, ledger, row, rate_gate)
        self.attempt = None
        bound, _ = request_bound(self.payload)
        owner = self
        class Reservation:
            def reserve(inner):
                owner.attempt = ledger.reserve(owner.sid, bound)
                return owner.attempt["call"]
        self.budget = Reservation()

    def log(self, record):
        enriched = {**record, "attempt_id":self.attempt["attempt_id"], "attempt":self.attempt["attempt"],
                    "requested_reasoning_effort":"high", "transport_timeout_s":600}
        CorruptionAPI.log(self, enriched)
        if not record.get("event"):
            self.ledger.settle(self.attempt["attempt_id"], enriched)
            self.rate_gate.record(enriched)
            if record.get("status") == "completed" and record.get("response_model") != "gpt-6.1-sol":
                raise ValueError("Unexpected response model")


def collect_responses(rows, pilot_rows, calls, ledger):
    """Validate the attempt chain and return exactly one successful response per ID."""
    lookup = {r["episode_id"]:r for r in rows}
    responses = {r["episode_id"]:r["llm_judgment"] for r in pilot_rows}
    entries = ledger.snapshot()["entries"]
    seen_attempts, numbers = set(), set()
    history = {}
    for call in sorted(calls, key=lambda r:r["phase_call"]):
        sid = call["sample_id"]
        aid = call.get("attempt_id",sid)  # Immutable pre-approval calls have no attempt ID.
        if sid not in lookup or sid in responses or aid in seen_attempts or call["phase_call"] in numbers:
            raise ValueError("Unknown candidate, duplicate attempt, or completed candidate re-judged")
        entry = entries[aid]
        previous = history.setdefault(sid,[])
        if entry["episode_id"] != sid or entry["attempt"] != len(previous)+1 or entry["call"] != call["phase_call"]:
            raise ValueError("Attempt sequence does not match the ledger")
        if previous and not retryable(previous[-1]):
            raise ValueError("Non-retryable error was retried")
        if "cost_nanos" not in entry:
            ledger.settle(aid,call)
        elif (entry["cost_nanos"],entry["accounting"]) != accounting(call,entry["bound_nanos"]):
            raise ValueError("Raw usage and settled attempt cost disagree")
        seen_attempts.add(aid); numbers.add(call["phase_call"]); previous.append(call)
        if call.get("status") == "completed":
            response = recovered_response(call)
            validate_judgments(lookup[sid], response)
            responses[sid] = response
    if seen_attempts != set(entries):
        raise ValueError("Unsettled attempt needs inspection; do not repeat an uncertain request")
    return responses, history


def open_progress(config, *, max_api_calls=MAX_CALLS, max_cost_usd=50):
    output = config["paths"]["phase3b_full_output"]
    rows, pilot_rows, pilot = load_population(config)
    calls = [r for r in read_jsonl(output/"calls.jsonl") if not r.get("event")]
    ledger = RetryLedger(output/"cost_ledger.json", pilot, max_api_calls=max_api_calls,
                         max_cost_usd=max_cost_usd, prior_calls=calls)
    responses, history = collect_responses(rows,pilot_rows,calls,ledger)
    return rows,pilot_rows,pilot,calls,ledger,responses,history


def run_resilient(config, *, max_api_calls, max_cost_usd=50, workers=4):
    if not 1 <= workers <= 4:
        raise ValueError("At most four concurrent requests are authorized")
    output = config["paths"]["phase3b_full_output"]
    output.mkdir(parents=True,exist_ok=True)
    with (output/"run.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run(config,max_api_calls,max_cost_usd,workers)


def _run(config,max_api_calls,max_cost_usd,workers):
    output = config["paths"]["phase3b_full_output"]
    rows,pilot_rows,pilot,calls,ledger,responses,history = open_progress(config,
        max_api_calls=max_api_calls,max_cost_usd=max_cost_usd)
    gate = ReliabilityGate(output/"reliability_state.json",calls)
    lookup = {r["episode_id"]:r for r in rows}
    completed_at_start = set(responses)
    manifest_path = output/"retry_preregistration.json"
    if not manifest_path.exists():
        write_json(manifest_path,{"completed_ids":sorted(completed_at_start),
            "pending_ids":sorted(set(lookup)-completed_at_start),"pilot_reused":len(pilot_rows),
            "model":"gpt-6.1-sol","reasoning_effort":"high","max_concurrency":4,
            "retry_delays":list(RETRY_DELAYS),"max_attempts_per_candidate":4,"transport_timeout_s":600,
            "cost_cap_usd":max_cost_usd,"cumulative_call_cap":max_api_calls,
            "cost_policy":"confirmed_usage_plus_timeout_reservations_5xx_without_usage_zero"})
    else:
        original = read_json(manifest_path)
        if not set(original["completed_ids"]) <= completed_at_start:
            raise ValueError("Previously completed judgments were lost")
    errors, stop_reason = [], None
    pending = sorted(set(lookup)-completed_at_start)
    random.Random(41).shuffle(pending)
    # Put previously failed candidates first; retry delays are enforced even on
    # restart. Completed candidates are never submitted again.
    pending.sort(key=lambda sid:0 if history.get(sid) else 1)
    queue = deque(pending)
    def status():
        state = ledger.snapshot()
        value = {k:v for k,v in state.items() if k != "entries"}
        value.update(judged=len(responses),total=len(rows),remaining=len(rows)-len(responses),
            complete=len(responses)==len(rows),errors=errors,stop_reason=stop_reason,
            pause_seconds_remaining=gate.seconds_remaining(),workers=workers,
            max_cost_usd=max_cost_usd,phase4_started=False)
        write_json(output/"judging_status.json",value)
        return value
    def judge(sid):
        prior = history.get(sid,[])
        n = len(prior)
        if n:
            if n >= 4 or not retryable(prior[-1]):
                raise ValueError("Retry limit exhausted or prior error is not retryable")
            # Each sleep is short; the CLI remains observable during pauses.
            deadline = time.monotonic()+RETRY_DELAYS[n-1]
            while time.monotonic() < deadline:
                time.sleep(max(0.,min(1.,deadline-time.monotonic())))
        api = AttemptAPI(config,ledger,lookup[sid],gate)
        try:
            result = api.judge()
        except Exception:
            # Raw callback has already saved this attempt and its budget charge.
            if api.attempt is not None:
                entry = ledger.snapshot()["entries"][api.attempt["attempt_id"]]
                return None, entry
            raise
        return result,ledger.snapshot()["entries"][api.attempt["attempt_id"]]
    status()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        active = {}
        while queue or active:
            while queue and len(active)<workers and not stop_reason:
                sid = queue.popleft()
                active[executor.submit(judge,sid)] = sid
            if not active:
                break
            done,_ = wait(active,timeout=15,return_when=FIRST_COMPLETED)
            if not done:
                state=status()
                print(f"Waiting; judged={len(responses)}/{len(rows)}; pause={state['pause_seconds_remaining']:.0f}s",flush=True)
                continue
            for future in done:
                sid = active.pop(future)
                try:
                    response,entry = future.result()
                    if response is not None:
                        validate_judgments(lookup[sid],response)
                        write_json(output/"judgments"/(sid+".json"),response)
                        responses[sid]=response
                    else:
                        history.setdefault(sid,[]).append(entry)
                        if retryable(entry) and entry["attempt"]<4:
                            queue.appendleft(sid)
                        else:
                            errors.append({"episode_id":sid,"attempt":entry["attempt"],
                                           "error_type":entry.get("error_type"),"http_status":entry.get("http_status")})
                            if not retryable(entry):
                                stop_reason="non_retryable_request_error"
                except CostUnavailable:
                    queue.appendleft(sid)
                    state=ledger.snapshot()
                    bound,_=request_bound(qc_payload(lookup[sid]))
                    if not active and state["cost_nanos"]+state["outstanding_nanos"]+bound>ledger.cap:
                        stop_reason="cost_cap_insufficient_for_next_request_reservation"
                except CallBudgetExceeded:
                    queue.appendleft(sid); stop_reason="request_cap"
                except Exception as error:
                    errors.append({"episode_id":sid,"error_type":type(error).__name__})
                    stop_reason="non_retryable_error"
                state=status()
                if state["cost_usd"]>max_cost_usd:
                    stop_reason="cost_cap_exceeded"
                print(f"Judged {len(responses)}/{len(rows)}; calls={state['calls']}; "
                      f"usage=${state['confirmed_cost_usd']:.6f}; timeout=${state['timeout_reserved_usd']:.6f}; "
                      f"pause={state['pause_seconds_remaining']:.0f}s; stop={stop_reason}",flush=True)
    if errors and not stop_reason:
        stop_reason="unresolved_candidates_after_retry_limit"
    return status()
