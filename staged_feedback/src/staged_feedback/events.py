"""Append-only events and deterministic, role-specific memory projections."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


KINDS = {
    "feedback_presented", "operation_applied", "operation_rejected", "test_result",
    "issue_resolved", "issue_added", "plan_changed", "commitment_added", "turn_ended",
    "issue_excluded", "student_task_added", "student_task_deduplicated", "gate_blocked",
    "initial_plan_rejected", "revision_recorded", "review_recorded",
    "optional_suggestion_added", "scope_revision_recorded", "restoration_recorded",
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
        issues, resolved, student_ids = {}, set(), set()
        tasks, suggestions, feedback, revisions, reviews, flow = [], [], [], [], [], []
        strategy, turn_plan = None, []
        for event in self.events:
            data, kind = event["payload"], event["kind"]
            if kind == "issue_added":
                issues[data["issue"]["id"]] = dict(data["issue"])
            elif kind == "issue_resolved":
                resolved.add(data["issue_id"])
            elif kind in {"student_task_added", "student_task_deduplicated"}:
                if data["task"].get("issue_id"):
                    student_ids.add(data["task"]["issue_id"])
                if kind == "student_task_added":
                    tasks.append(data["task"])
            elif kind == "feedback_presented":
                feedback = data["items"]
            elif kind == "optional_suggestion_added":
                suggestions.append(data["suggestion"])
            elif kind == "revision_recorded":
                revisions.append({"turn": event["turn"], "changed_sentences": data["changed_sentences"],
                                  "observational_flags": data["flags"]})
            elif kind == "review_recorded":
                reviews.append({"turn": event["turn"], **data["review"]})
            elif kind == "restoration_recorded":
                if revisions and revisions[-1]["turn"] == event["turn"]:
                    revisions[-1].update(changed_sentences=data["changed_sentences"],
                        observational_flags=data["flags"], coordinator_restore=data["requests"])
            elif kind == "turn_ended":
                flow.append({"turn": event["turn"], **data})
            elif kind == "plan_changed":
                if "strategy" in data:
                    strategy = data["strategy"]["after"]
                if "turn_plan" in data:
                    turn_plan = data["turn_plan"]["after"]
                resolved.difference_update(data.get("reopened_issue_ids", []))
                for update in data.get("priorities", []):
                    if update["issue_id"] in issues:
                        issues[update["issue_id"]]["priority"] = update["priority"]
        remaining = sorted((v for k, v in issues.items() if k not in resolved | student_ids),
                           key=lambda i: (i["priority"], i["id"]))
        if role == "coordinator":
            return {"flow": flow, "remaining_issues": remaining, "reviews": reviews,
                    "remaining_turns": remaining_turns, "strategy": strategy,
                    "turn_plan": turn_plan, "student_tasks": tasks, "optional_suggestions": suggestions}
        if role == "feedback":
            return {"candidates": remaining, "resolved_issue_ids": sorted(resolved),
                    "strategy": strategy, "turn_plan": turn_plan, "earlier_revisions": revisions}
        if role == "reviser":
            return {"feedback": [i for i in feedback if i["issue_id"] not in student_ids],
                    "earlier_revisions": revisions, "student_tasks": tasks}
        raise ValueError(f"Unknown memory role: {role}")

    def digest(self):
        return hashlib.sha256(json.dumps(self.events, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
