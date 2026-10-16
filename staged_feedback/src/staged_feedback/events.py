"""Append-only events and deterministic, role-specific memory projections."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


KINDS = {
    "feedback_presented", "operation_applied", "operation_rejected", "test_result",
    "issue_resolved", "issue_added", "plan_changed", "commitment_added", "turn_ended",
}


def jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Journal:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.events = []
        if self.path and self.path.exists():
            self.events = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for i, event in enumerate(self.events, 1):
                if event["seq"] != i or event["kind"] not in KINDS:
                    raise ValueError("Invalid event journal")

    def append(self, kind, *, phase, turn, **payload):
        if kind not in KINDS:
            raise ValueError(f"Unknown event kind: {kind}")
        event = {"seq": len(self.events) + 1, "time": datetime.now(timezone.utc).isoformat(),
                 "kind": kind, "phase": phase, "turn": turn, "payload": jsonable(payload)}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
        self.events.append(event)
        return event

    def summary(self, role, *, phase, remaining_turns, enabled=True):
        if not enabled:
            return ""
        issues, resolved, tests, commitments, feedback, applied, flow = {}, set(), {}, [], [], [], []
        strategy, turn_plan, dispositions = None, [], {}
        for event in self.events:
            data, kind = event["payload"], event["kind"]
            active = event["phase"] in {"initial", phase}
            if kind == "turn_ended":
                flow.append({"phase": event["phase"], "turn": event["turn"],
                             "summary": data.get("summary", ""), "tests": data.get("tests")})
            if not active:
                continue
            if kind == "issue_added":
                issue = data["issue"]
                issues[issue["id"]] = issue
            elif kind == "issue_resolved":
                resolved.add(data["issue_id"])
            elif kind == "test_result":
                tests[data["result"]["issue_id"]] = data["result"]
            elif kind == "commitment_added":
                commitments.append(data["commitment"])
            elif kind == "feedback_presented":
                feedback.extend(data["items"])
                for item in data["items"]:
                    if item["issue_id"] in issues:
                        issues[item["issue_id"]] = {**issues[item["issue_id"]], "test": item["test"]}
            elif kind == "operation_applied":
                applied.append({"turn": event["turn"], "description": data["description"],
                                "preserve_quotes": data.get("preserve_quotes", [])})
            elif kind == "plan_changed":
                if "strategy" in data:
                    strategy = data["strategy"]["after"]
                if "turn_plan" in data:
                    turn_plan = data["turn_plan"]["after"]
                resolved.difference_update(data.get("reopened_issue_ids", []))
                for decision in data.get("dispositions", []):
                    dispositions[decision["issue_id"]] = decision
                for update in data.get("priorities", []):
                    if update["issue_id"] in issues:
                        issues[update["issue_id"]] = {**issues[update["issue_id"]], "priority": update["priority"]}
        remaining = sorted((value for key, value in issues.items() if key not in resolved), key=lambda i: (i["priority"], i["id"]))
        if role == "coordinator":
            return {"flow": flow[-8:], "remaining_issues": remaining, "test_results": list(tests.values()),
                    "remaining_turns": remaining_turns, "strategy": strategy, "turn_plan": turn_plan,
                    "dispositions": list(dispositions.values())}
        if role == "feedback":
            return {"candidates": remaining, "handled_issue_ids": sorted({f["issue_id"] for f in feedback}),
                    "resolved_issue_ids": sorted(resolved), "commitments": commitments,
                    "strategy": strategy, "turn_plan": turn_plan}
        if role == "reviser":
            return {"feedback": feedback[-2:], "commitments": commitments,
                    "earlier_edits": [a for a in applied if a["turn"] < max((e["turn"] for e in self.events if e["phase"] == phase), default=0)]}
        raise ValueError(f"Unknown memory role: {role}")

    def digest(self):
        return hashlib.sha256(json.dumps(self.events, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
