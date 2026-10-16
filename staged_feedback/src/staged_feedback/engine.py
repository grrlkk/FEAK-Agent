"""One adaptive loop: grounded difficulty, strategy, turn plan and review."""
from difflib import SequenceMatcher
import math
from pathlib import Path

from .checks import test_summary, run_tests
from .edits import changed_char_ratio, quote_spans, unique_span, Rejected
from .events import Journal, jsonable, save_json
from .models import (
    Commitment, Feedback, FeedbackItem, InitialPlan, Revision, Review, SingleCallStages,
)


class Engine:
    def __init__(self, cfg, *, llm, diagnostic, executor, directory):
        self.cfg, self.llm, self.diagnostic, self.executor = cfg, llm, diagnostic, executor
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.directory / "events.jsonl")
        if self.journal.events:
            raise ValueError("Use a fresh essay directory; do not append a new run to old events")
        self.issues, self.origins, self.commitments, self.resolved, self.deferred = {}, {}, [], set(), set()
        self.tasks, self.turns, self.memory_views = [], [], []
        self.memory_enabled = cfg.condition not in {"routed_no_memory", "planned_no_memory"}
        self.budget = cfg.max_turns
        self.strategy, self.turn_plan, self.initial = None, [], None

    def _evidence(self, evidence, diagnostic):
        for item in evidence:
            values = diagnostic.get("scores" if item.source == "kanana" else "features", {})
            if item.key not in values:
                raise ValueError(f"Unknown diagnostic evidence key: {item.key}")
            value = values[item.key]
            if (value is None) != (item.value is None) or (value is not None and
                    not math.isclose(value, item.value, rel_tol=1e-6, abs_tol=1e-6)):
                raise ValueError(f"Diagnostic evidence differs from the measured value: {item.key}")

    def _validate_plan(self, plan, budget):
        numbers = [item.turn for item in plan]
        if len(numbers) != len(set(numbers)) or set(numbers) != set(range(1, budget + 1)):
            raise ValueError("턴 계획에는 예산 안의 매 턴을 한 번씩 지정해야 합니다.")
        for item in plan:
            if len(item.issue_ids) != len(set(item.issue_ids)) or not set(item.issue_ids) <= set(self.issues):
                raise ValueError("턴 계획에 중복 또는 알 수 없는 이슈 ID가 있습니다.")

    def _summary(self, role, phase, turn):
        view = self.journal.summary(role, phase=phase, remaining_turns=max(0, self.budget - turn), enabled=self.memory_enabled)
        self.memory_views.append({"role": role, "phase": phase, "turn": turn, "summary": view})
        return view

    def _request(self, role, payload, schema, phase, turn):
        if role != "macro_tests":
            memory_role = "coordinator" if role.startswith("coordinator") else role
            payload = {**payload, "memory_summary": self._summary(memory_role, phase, turn)}
        return self.llm.request(role, payload, schema, context={"essay_id": self.essay_id, "phase": phase, "turn": turn})

    def _add_issue(self, issue, text, phase, turn):
        if issue.id in self.issues:
            raise ValueError(f"Duplicate issue ID: {issue.id}")
        for quote in [*issue.evidence_quotes, *issue.test.target_quotes]:
            unique_span(text, quote)
        if issue.test.source_quote:
            unique_span(text, issue.test.source_quote)
        self.issues[issue.id] = issue
        self.origins[issue.id] = text
        self.journal.append("issue_added", phase=phase, turn=turn, issue=issue)

    def _commit(self, commitment, text, phase, turn):
        if commitment.quote and not quote_spans(text, commitment.quote):
            raise ValueError("확정 사항의 인용이 현재 글에 없습니다.")
        if commitment.model_dump() not in [c.model_dump() for c in self.commitments]:
            self.commitments.append(commitment)
            self.journal.append("commitment_added", phase=phase, turn=turn, commitment=commitment)

    def _keep(self):
        return list(dict.fromkeys(c.quote for c in self.commitments if c.quote))

    def _feedback_payload(self, feedback, text):
        return {"writing_prompt": self.prompt, "current_text": text,
                "feedback": jsonable(feedback), "preserve": jsonable(self.commitments),
                "allowed_operations": ["replace", "move", "merge", "split", "connect"]}

    def _apply(self, text, feedback, phase, turn):
        revision = self._request("reviser", self._feedback_payload(feedback, text), Revision, phase, turn)
        self.tasks.extend(revision.student_tasks)
        allowed = {item.issue_id for item in feedback.items}
        records, failed = [], []

        def attempt(operation, retry):
            nonlocal text
            try:
                text, record = self.executor.apply(text, operation, keep=self._keep(), allowed_issue_ids=allowed)
                record.update(applied=True, retry=retry)
                self.journal.append("operation_applied", phase=phase, turn=turn, **record)
                records.append(record)
                return None
            except (Rejected, ValueError) as exc:
                record = {"operation": operation.model_dump(), "applied": False, "retry": retry,
                          "reason_code": getattr(exc, "code", "invalid_operation"), "reason": str(exc),
                          "preserve_violation_attempt": getattr(exc, "code", "").startswith("preserve_")}
                self.journal.append("operation_rejected", phase=phase, turn=turn, **record)
                records.append(record)
                return record

        seen = set()
        for operation in revision.operations:
            if operation.id in seen:
                record = {"operation": operation.model_dump(), "applied": False, "retry": False,
                    "reason_code": "duplicate_operation_id", "reason": "중복된 연산 ID는 적용하지 않았습니다.",
                    "preserve_violation_attempt": False}
                self.journal.append("operation_rejected", phase=phase, turn=turn, **record)
                records.append(record)
                continue
            seen.add(operation.id)
            failure = attempt(operation, False)
            if failure:
                failed.append(failure)
        if failed and self.cfg.operation_retries:
            retry = self._request("reviser", {
                **self._feedback_payload(feedback, text),
                "retry": {"remaining_attempts": 1, "failed_operations": failed,
                          "instruction": "이미 적용된 연산을 반복하지 말고 실패한 연산 ID만 한 번 고치세요. 불가능하면 생략하세요."},
            }, Revision, phase, turn)
            self.tasks.extend(retry.student_tasks)
            wanted = {item["operation"]["id"] for item in failed}
            observed = set()
            for operation in retry.operations:
                if operation.id not in wanted or operation.id in observed:
                    record = dict(operation=operation.model_dump(), applied=False, retry=True,
                        reason_code="invalid_retry_id", reason="실패한 연산마다 같은 ID로 한 번만 재시도할 수 있습니다.",
                        preserve_violation_attempt=False)
                    self.journal.append("operation_rejected", phase=phase, turn=turn, **record)
                    records.append(record)
                    continue
                observed.add(operation.id)
                attempt(operation, True)
            for operation_id in wanted - observed:
                record = dict(operation_id=operation_id, applied=False, retry=True, reason_code="retry_omitted",
                    reason="재시도 응답에서 생략되어 연산을 버렸습니다.", preserve_violation_attempt=False)
                self.journal.append("operation_rejected", phase=phase, turn=turn, **record)
                records.append(record)
        return text, records

    def _tests(self, text, phase, turn):
        results = run_tests(list(self.issues.values()), text=text, original=self.original,
            sentence_spans=self.executor.sentence_spans, llm=self.llm, origins=self.origins,
            context={"essay_id": self.essay_id, "phase": phase, "turn": turn})
        for result in results:
            self.journal.append("test_result", phase=phase, turn=turn, result=result)
        return results

    def _freeze_edits(self, before, after, phase, turn):
        changed = [(c, d) for kind, a, b, c, d in SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
                   if kind != "equal" and c != d]
        for sentence in self.executor.sentence_spans(after):
            if any(a < sentence["end"] and sentence["start"] < b for a, b in changed):
                self._commit(Commitment(statement="앞 턴에서 고친 문장의 내용을 유지합니다.", quote=sentence["text"]), after, phase, turn)

    def _turn_record(self, phase, turn, feedback, operations, text, results, **extra):
        record = {"phase": phase, "turn": turn, "feedback": jsonable(feedback),
                  "checklist": [r.checklist for r in results], "operations": operations,
                  "revised_text": text, "tests": jsonable(results), **jsonable(extra)}
        self.turns.append(record)
        save_json(self.directory / f"{phase}_turn_{turn}.json", record)
        return record

    def _review(self, text, results, operations, turn):
        review = self._request("coordinator_review", {
            "writing_prompt": self.prompt, "current_text": text,
            "issues": jsonable(list(self.issues.values())), "test_results": jsonable(results),
            "operations": [{k: v for k, v in r.items() if k not in {"text_before", "text_after"}} for r in operations],
            "current_turn": turn, "current_turn_budget": self.budget, "max_turns": self.cfg.max_turns,
            "commitments": jsonable(self.commitments), "strategy": jsonable(self.strategy),
            "turn_plan": jsonable(self.turn_plan),
        }, Review, "planned", turn)
        passed = {r.issue_id for r in results if r.passed}
        changes = {}
        for issue_id in set(review.resolved_issue_ids) - passed:
            changes.setdefault("rejected_resolutions", []).append(issue_id)
        reopened = self.resolved & {r.issue_id for r in results if not r.passed}
        if reopened:
            changes["reopened_issue_ids"] = sorted(reopened)
            self.resolved.difference_update(reopened)
        for issue_id in sorted(passed - self.resolved):
            self.resolved.add(issue_id)
            self.deferred.discard(issue_id)
            self.journal.append("issue_resolved", phase="planned", turn=turn, issue_id=issue_id)
        for issue in review.new_issues:
            try:
                self._add_issue(issue, text, "planned", turn)
                changes.setdefault("new_issue_ids", []).append(issue.id)
            except ValueError as exc:
                changes.setdefault("rejected_new_issues", []).append({"id": issue.id, "reason": str(exc)})
        priorities = []
        for update in review.priorities:
            if update.issue_id in self.issues:
                issue = self.issues[update.issue_id]
                if issue.priority != update.priority:
                    self.issues[issue.id] = issue.model_copy(update={"priority": update.priority})
                    priorities.append(update.model_dump())
        if priorities:
            changes["priorities"] = priorities
        for decision in review.dispositions:
            if decision.issue_id not in self.issues or decision.issue_id in self.resolved:
                continue
            if decision.action == "retry":
                self.deferred.discard(decision.issue_id)
            else:
                self.deferred.add(decision.issue_id)
            changes.setdefault("dispositions", []).append(decision.model_dump())
            if decision.action == "student_task":
                from .models import StudentTask
                self.tasks.append(StudentTask(issue_id=decision.issue_id,
                    instruction=self.issues[decision.issue_id].description, reason=decision.reason))
        new_budget = max(turn, min(review.turn_budget, self.cfg.max_turns))
        if self.cfg.condition == "always_direct":
            new_budget = 1
        new_plan = review.turn_plan if review.turn_plan is not None else self.turn_plan
        self._validate_plan(new_plan, new_budget)
        if jsonable(new_plan) != jsonable(self.turn_plan):
            changes["turn_plan"] = {"before": jsonable(self.turn_plan), "after": jsonable(new_plan)}
            self.turn_plan = new_plan
        if review.strategy is not None and review.strategy != self.strategy:
            changes["strategy"] = {"before": jsonable(self.strategy), "after": jsonable(review.strategy)}
            self.strategy = review.strategy
        if new_budget != self.budget:
            changes["turn_budget"] = {"before": self.budget, "after": new_budget}
            self.budget = new_budget
        for commitment in review.new_commitments:
            try:
                self._commit(commitment, text, "planned", turn)
            except ValueError as exc:
                changes.setdefault("rejected_commitments", []).append(str(exc))
        self.tasks.extend(review.student_tasks)
        if changes:
            self.journal.append("plan_changed", phase="planned", turn=turn, **changes, reason=review.reason)
        return review

    def run(self, essay):
        self.essay_id, self.prompt = essay["essay_id"], essay["prompt"]
        self.original = essay["text"]
        save_json(self.directory / "input.json", essay)
        before = self.diagnostic(self.original, self.prompt, self.essay_id)
        save_json(self.directory / "diagnosis_initial.json", before)
        if self.cfg.condition == "single_call_staged":
            return self._single_call(before)
        effective_max = 1 if self.cfg.condition == "always_direct" else self.cfg.max_turns
        initial = self._request("coordinator_initial", {
            "writing_prompt": self.prompt, "current_text": self.original, "diagnosis": before,
            "max_turns": effective_max, "condition": self.cfg.condition,
            "forced_initial_budget": effective_max if self.cfg.condition == "always_staged" else None,
        }, InitialPlan, "initial", 0)
        self.budget = min(initial.turn_budget, effective_max)
        for issue in initial.issues:
            self._add_issue(issue, self.original, "initial", 0)
        self._evidence(initial.difficulty.diagnostic_evidence, before)
        for quote in initial.difficulty.evidence_quotes:
            unique_span(self.original, quote)
        self._validate_plan(initial.turn_plan, self.budget)
        if initial.strategy.name == "single_pass" and self.budget != 1:
            raise ValueError("single_pass starts with a one-turn plan")
        if self.cfg.condition == "always_staged" and self.budget != self.cfg.max_turns:
            raise ValueError("always_staged uses the configured initial turn budget")
        self.initial, self.strategy, self.turn_plan = initial, initial.strategy, initial.turn_plan
        self.journal.append("plan_changed", phase="initial", turn=0, initial=True,
            difficulty=initial.difficulty, strategy={"before": None, "after": initial.strategy},
            turn_plan={"before": [], "after": initial.turn_plan},
            turn_budget={"before": None, "after": self.budget}, reason=initial.reason)
        for commitment in initial.commitments:
            self._commit(commitment, self.original, "initial", 0)
        self.tasks.extend(initial.student_tasks)
        save_json(self.directory / "initial_plan.json", initial)
        if self.cfg.condition == "always_direct":
            return self._always_direct(before)
        current, stop_reason = self.original, "계획된 턴 예산에 도달했습니다."
        for turn in range(1, self.cfg.max_turns + 1):
            if turn > self.budget:
                break
            direction = next(item for item in self.turn_plan if item.turn == turn)
            strategy_used = self.strategy
            candidates = sorted((self.issues[i] for i in direction.issue_ids
                if i not in self.resolved and i not in self.deferred), key=lambda i: (i.priority, i.id))
            current_features = self.diagnostic.features(current)
            save_json(self.directory / f"features_turn_{turn}.json", current_features)
            feedback = self._request("feedback", {
                "writing_prompt": self.prompt, "current_text": current,
                "strategy": jsonable(self.strategy), "turn_direction": jsonable(direction),
                "candidates": jsonable(candidates), "max_feedback": self.cfg.feedback_per_turn,
                "current_features": current_features, "commitments": jsonable(self.commitments),
            }, Feedback, "planned", turn)
            ids = [item.issue_id for item in feedback.items]
            if len(ids) != len(set(ids)) or not set(ids) <= {i.id for i in candidates} or len(ids) > self.cfg.feedback_per_turn:
                raise ValueError("Feedback must stay within this turn's planned issues and configured limit")
            for item in feedback.items:
                unique_span(current, item.target_quote)
                self._evidence(item.feature_evidence, {"features": current_features["features"]})
                issue = self.issues[item.issue_id]
                if (item.test.kind == "macro") != (issue.level == "macro"):
                    raise ValueError("Feedback cannot switch an issue's test level")
                self.issues[item.issue_id] = issue.model_copy(update={"test": item.test})
                self.origins[item.issue_id] = current
            self.tasks.extend(feedback.student_tasks)
            self.journal.append("feedback_presented", phase="planned", turn=turn,
                items=feedback.items, strategy=self.strategy, direction=direction, features=current_features)
            previous = current
            if feedback.items:
                current, operations = self._apply(current, feedback, "planned", turn)
            else:
                operations = []
            results = self._tests(current, "planned", turn)
            self._freeze_edits(previous, current, "planned", turn)
            review = self._review(current, results, operations, turn)
            summary = test_summary(results)
            all_passed = summary["all_passed"] and {r.issue_id for r in results} == set(self.issues)
            self._turn_record("planned", turn, feedback, operations, current, results,
                strategy=strategy_used, strategy_after_review=self.strategy,
                turn_direction=direction, current_features=current_features,
                review=review, test_summary=summary,
                untested_issue_ids=sorted(set(self.issues) - {r.issue_id for r in results}))
            self.journal.append("turn_ended", phase="planned", turn=turn, summary=review.reason,
                remaining_issue_ids=sorted(set(self.issues) - self.resolved),
                remaining_turns=max(0, self.budget - turn), tests=summary)
            if all_passed:
                stop_reason = "현재 이슈의 모든 테스트를 통과했습니다."
                break
            if not review.continue_work:
                stop_reason = review.reason
                break
            if not (set(self.issues) - self.resolved - self.deferred):
                stop_reason = "남은 이슈를 보류하거나 학생 과제로 넘겼습니다."
                break
        route = "single_turn" if len(self.turns) == 1 else "multi_turn"
        return self._finish(current, before, route, stop_reason)

    def _always_direct(self, before):
        """Comparison only: every initial issue goes to one revision, without routing."""
        feedback = Feedback(items=[FeedbackItem(issue_id=issue.id,
            target_quote=issue.evidence_quotes[0], what=issue.description,
            how=issue.test.checklist, student_explanation=issue.description, test=issue.test)
            for issue in self.issues.values()])
        self.journal.append("feedback_presented", phase="comparison_direct", turn=1,
            items=feedback.items, strategy=self.strategy, direction=self.turn_plan[0], features={})
        current, operations = self._apply(self.original, feedback, "comparison_direct", 1) if feedback.items else (self.original, [])
        results = self._tests(current, "comparison_direct", 1)
        for result in results:
            if result.passed:
                self.resolved.add(result.issue_id)
                self.journal.append("issue_resolved", phase="comparison_direct", turn=1, issue_id=result.issue_id)
        summary = test_summary(results)
        self._turn_record("comparison_direct", 1, feedback, operations, current, results,
            test_summary=summary, comparison_all_issues=True)
        self.journal.append("turn_ended", phase="comparison_direct", turn=1,
            summary="비교 조건: 초기 이슈 전체를 한 번 수정하고 종료했습니다.", tests=summary)
        return self._finish(current, before, "single_turn", "비교 조건: 초기 이슈 전체를 한 번 수정했습니다. 재시작이나 경로 분기는 없습니다.")

    def _single_call(self, before):
        stages = self._request("reviser", {"mode": "single_call_staged", "writing_prompt": self.prompt,
            "current_text": self.original, "diagnosis": before,
            "instruction": "우선순위를 정해 2~3단계로 나눠 피드백하고 단계별 수정본도 보여주세요."},
            SingleCallStages, "single_call_staged", 0)
        current = self.original
        for number, stage in enumerate(stages.stages, 1):
            self.executor._markers(current, stage.revised_text)
            current = stage.revised_text
            self.turns.append({"phase": "single_call_staged", "turn": number, **stage.model_dump(),
                "tests": [], "operations": [], "comparison_baseline_full_text": True})
        self.tasks.extend(stages.student_tasks)
        return self._finish(current, before, "single_call_staged", "한 번의 LLM 호출로 단계별 피드백·수정본을 생성하는 비교 조건")

    def _finish(self, text, before, route, reason):
        after = self.diagnostic(text, self.prompt, self.essay_id)
        save_json(self.directory / "diagnosis_final.json", after)
        initial_scores, final_scores = before.get("scores", {}), after.get("scores", {})
        tasks, seen = [], set()
        for task in self.tasks:
            data = task.model_dump()
            key = (data["issue_id"], data["instruction"], data["reason"])
            if key not in seen:
                tasks.append(data)
                seen.add(key)
        strategy_history, plan_history = [], []
        for event in self.journal.events:
            data = event["payload"]
            if event["kind"] == "plan_changed":
                for field, history in (("strategy", strategy_history), ("turn_plan", plan_history)):
                    if field in data:
                        history.append({"turn": event["turn"], "phase": event["phase"],
                            **data[field], "reason": data.get("reason", "")})
        result = {"essay_id": self.essay_id, "condition": self.cfg.condition,
            "mock": bool(self.cfg.mock or before.get("mock")),
            "route": route, "route_reason": reason,
            "difficulty": jsonable(self.initial.difficulty) if self.initial else None,
            "initial_strategy": jsonable(self.initial.strategy) if self.initial else None,
            "strategy": jsonable(self.strategy), "strategy_history": strategy_history,
            "initial_turn_plan": jsonable(self.initial.turn_plan) if self.initial else [],
            "turn_plan": jsonable(self.turn_plan), "turn_plan_history": plan_history,
            "final_text": text, "initial_text": self.original, "turns": self.turns,
            "initial_scores": initial_scores, "final_scores": final_scores,
            "score_delta": {key: final_scores[key] - value for key, value in initial_scores.items() if key in final_scores},
            "changed_char_ratio": changed_char_ratio(self.original, text),
            "score_n": 32, "diagnostic_calls": 2, "student_tasks": tasks,
            "remaining_issue_ids": sorted(set(self.issues) - self.resolved),
            "memory_summaries": self.memory_views, "event_count": len(self.journal.events),
            "journal_sha256": self.journal.digest()}
        save_json(self.directory / "result.json", result)
        (self.directory / "final.txt").write_text(text, encoding="utf-8")
        return result
