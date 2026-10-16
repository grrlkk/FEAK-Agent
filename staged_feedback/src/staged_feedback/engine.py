"""Full-text revision, passive observations and direct coordinator review."""
import json
from pathlib import Path
import re
import unicodedata

from .events import Journal, jsonable, save_json
from .models import Feedback, InitialPlan, Revision, Review, SingleCallStages, StudentTask, TurnDirection
from .observations import changed_char_ratio


class Engine:
    def __init__(self, cfg, *, llm, diagnostic, observer, anonymizer_factory, directory):
        self.cfg, self.llm, self.diagnostic, self.observer = cfg, llm, diagnostic, observer
        self.anonymizer_factory = anonymizer_factory
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.directory / "events.jsonl")
        if self.journal.events:
            raise ValueError("Use a fresh essay directory; do not overwrite an experiment")
        self.issues, self.resolved, self.deferred, self.student_ids = {}, set(), set(), set()
        self.tasks, self.turns, self.memory_views = [], [], []
        self.memory_enabled = cfg.condition not in {"routed_no_memory", "planned_no_memory"}
        self.budget, self.effective_max = cfg.max_turns, cfg.max_turns
        self.strategy, self.turn_plan, self.initial = None, [], None
        self.deferred_micro_feedback, self.gate_blockers, self.changed_ranges = [], [], []

    def _editable(self):
        return {key: issue for key, issue in self.issues.items() if key not in self.student_ids}

    def _pending(self):
        return set(self._editable()) - self.resolved - self.deferred

    @staticmethod
    def _task_text(text):
        return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text)).casefold()

    def _queue_tasks(self, tasks, phase, turn):
        for task in tasks:
            if task.issue_id in self.issues:
                self.student_ids.add(task.issue_id)
                self.resolved.discard(task.issue_id)
                self.deferred.add(task.issue_id)
                self.issues[task.issue_id] = self.issues[task.issue_id].model_copy(update={"needs_student_content": True})
            duplicate = next((kept for kept in self.tasks if
                (task.issue_id is not None and task.issue_id == kept.issue_id) or
                self._task_text(task.instruction) == self._task_text(kept.instruction)), None)
            if duplicate:
                self.journal.append("student_task_deduplicated", phase=phase, turn=turn,
                    task=task, kept=duplicate, reason="같은 이슈 또는 같은 안내의 학생 과제를 하나로 유지합니다.")
            else:
                self.tasks.append(task)
                self.journal.append("student_task_added", phase=phase, turn=turn, task=task)

    def _classify_students(self, phase, turn):
        for issue in list(self.issues.values()):
            if issue.needs_student_content and issue.id not in self.student_ids:
                self._queue_tasks([StudentTask(issue_id=issue.id, instruction=issue.description,
                    reason="새 내용·실제 경험·학생의 의도 확인이 필요합니다.")], phase, turn)

    def _filter_plan(self, plan, turn):
        return [item.model_copy(update={"issue_ids": [key for key in item.issue_ids if key in self._pending()]})
                if item.turn > turn else item for item in plan]

    def _student_gate(self, turn):
        blockers = sorted(key for key in self.student_ids if self.issues[key].level == "macro")
        if not blockers:
            return False
        self.gate_blockers = blockers
        self.deferred_micro_feedback = [{"issue_id": i.id, "when": "큰 틀을 고친 뒤 다룰 것", "feedback": i.description}
            for i in sorted(self.issues.values(), key=lambda i: (i.priority, i.id))
            if i.level == "micro" and i.id not in self.resolved]
        phase = "planned" if turn else "baseline"
        self.journal.append("gate_blocked", phase=phase, turn=turn, student_issue_ids=blockers,
            deferred_micro_feedback=self.deferred_micro_feedback,
            reason="학생의 내용·경험·의도 확인이 필요한 거시 과제를 제시하고 종료합니다.")
        old, old_budget = self.turn_plan, self.budget
        self.turn_plan, self.budget = [p for p in self.turn_plan if p.turn <= turn], turn
        if old != self.turn_plan or old_budget != turn:
            self.journal.append("plan_changed", phase=phase, turn=turn,
                turn_plan={"before": old, "after": self.turn_plan}, turn_budget={"before": old_budget, "after": turn},
                reason_categories=["student_task_transfer"], reason="거시 학생 과제를 먼저 제시하고 남은 턴을 취소했습니다.")
        return True

    def _request(self, role, payload, schema, phase, turn):
        memory_role = "coordinator" if role.startswith("coordinator") else role
        view = self.journal.summary(memory_role, phase=phase,
            remaining_turns=max(0, self.budget - turn), enabled=self.memory_enabled)
        self.memory_views.append({"role": memory_role, "phase": phase, "turn": turn, "summary": view})
        payload = {**payload, "memory_summary": view}
        anonymizer = self.anonymizer_factory(self.essay_id, self.cfg.condition + ":" + role, turn)
        # Mask every text-bearing field, including memory. Restore known tokens only.
        # Never invoke the legacy validate_revision/fallback_to_original mechanism.
        masked = anonymizer.mask_data(payload)
        with (self.directory / "anonymization.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"role": role, "phase": phase, "turn": turn,
                "mapping": anonymizer.mapping}, ensure_ascii=False) + "\n")
        response = self.llm.request(role, masked, schema,
            context={"essay_id": self.essay_id, "phase": phase, "turn": turn})
        return schema.model_validate(anonymizer.restore_data(response))

    def _add_issue(self, issue, phase, turn):
        replaced = issue.id in self.issues
        self.issues[issue.id] = issue
        self.journal.append("issue_added", phase=phase, turn=turn, issue=issue, updated_existing_id=replaced)

    def _direction(self, turn):
        pending = self._pending()
        old = next((item for item in self.turn_plan if item.turn == turn), None)
        if self.strategy.name == "single_pass" or self.cfg.condition == "always_direct":
            ids = sorted(pending, key=lambda key: (self.issues[key].priority, key))
            direction = TurnDirection(turn=turn, direction=old.direction if old else "수정 가능한 모든 이슈를 이번 한 턴에 다룹니다.",
                issue_ids=ids, feedback_count=len(ids))
        elif old and set(old.issue_ids) & pending:
            direction = old.model_copy(update={"issue_ids": [key for key in old.issue_ids if key in pending]})
        else:
            future = next((p for p in self.turn_plan if p.turn > turn and set(p.issue_ids) & pending), None)
            ids = [key for key in future.issue_ids if key in pending] if future else sorted(pending)
            direction = TurnDirection(turn=turn, direction=future.direction if future else "남은 수정 이슈를 다룹니다.",
                issue_ids=ids, feedback_count=future.feedback_count if future else len(ids))
        if direction != old:
            before = self.turn_plan
            self.turn_plan = sorted([p for p in before if p.turn != turn] + [direction], key=lambda p: p.turn)
            assigned = set(old.issue_ids) if old else set()
            categories = [name for name, values in (("student_task_transfer", self.student_ids),
                          ("resolved_issue", self.resolved)) if assigned & values]
            self.journal.append("plan_changed", phase="planned", turn=turn,
                turn_plan={"before": before, "after": self.turn_plan}, reason_categories=categories or ["plan_adjustment"],
                reason="단일 턴의 전체 이슈 배정 또는 현재 비어 있는 배정을 남은 이슈에 맞췄습니다.")
        return direction

    def _review(self, current, feedback, observed, turn, students_before_turn):
        review = self._request("coordinator_review", {
            "writing_prompt": self.prompt, "current_text": current, "previous_text": self.previous,
            "feedback": jsonable(feedback), "changed_sentences": observed["changed_sentences"],
            "layout_changes": observed["layout_changes"], "observational_flags": observed["flags"],
            "feedback_targets": observed["feedback_targets"],
            "unlocated_feedback_issue_ids": observed["unlocated_feedback_issue_ids"],
            "issues": jsonable(list(self.issues.values())), "resolved_issue_ids": sorted(self.resolved),
            "student_tasks": jsonable(self.tasks), "current_turn": turn, "current_turn_budget": self.budget,
            "max_turns": self.effective_max, "strategy": jsonable(self.strategy), "turn_plan": jsonable(self.turn_plan),
        }, Review, "planned", turn)
        changes, newly_resolved, reopened = {}, [], []
        # These are the coordinator's decisions, not computed correctness verdicts.
        for assessment in review.assessments:
            if assessment.issue_id not in self._editable():
                continue
            if assessment.status == "applied":
                if assessment.issue_id not in self.resolved:
                    newly_resolved.append(assessment.issue_id)
                    self.journal.append("issue_resolved", phase="planned", turn=turn,
                        issue_id=assessment.issue_id, reason=assessment.reason, source="coordinator_review")
                self.resolved.add(assessment.issue_id)
                self.deferred.discard(assessment.issue_id)
            elif assessment.issue_id in self.resolved:
                reopened.append(assessment.issue_id)
                self.resolved.discard(assessment.issue_id)
        for key in review.reopened_issue_ids:
            if key in self.resolved:
                reopened.append(key)
                self.resolved.discard(key)
                self.deferred.discard(key)
        if reopened:
            changes["reopened_issue_ids"] = reopened
        for issue in review.new_issues:
            self._add_issue(issue, "planned", turn)
            self.resolved.discard(issue.id)
            changes.setdefault("new_issue_ids", []).append(issue.id)
        for update in review.priorities:
            if update.issue_id in self.issues and self.issues[update.issue_id].priority != update.priority:
                self.issues[update.issue_id] = self.issues[update.issue_id].model_copy(update={"priority": update.priority})
                changes.setdefault("priorities", []).append(jsonable(update))
        old_students = students_before_turn
        self._queue_tasks(review.student_tasks, "planned", turn)
        for decision in review.dispositions:
            if decision.issue_id not in self.issues:
                continue
            if decision.action == "retry":
                self.deferred.discard(decision.issue_id)
            else:
                self.deferred.add(decision.issue_id)
            changes.setdefault("dispositions", []).append(jsonable(decision))
            if decision.action == "student_task":
                self._queue_tasks([StudentTask(issue_id=decision.issue_id,
                    instruction=self.issues[decision.issue_id].description, reason=decision.reason)], "planned", turn)
        self._classify_students("planned", turn)
        if self.student_ids != old_students:
            changes["student_issue_ids"] = sorted(self.student_ids - old_students)
        if review.strategy is not None and review.strategy != self.strategy:
            changes["strategy"] = {"before": self.strategy, "after": review.strategy}
            self.strategy = review.strategy
        new_budget = max(turn, min(review.turn_budget, self.effective_max))
        if self.strategy.name == "single_pass":
            new_budget = turn
        if new_budget != self.budget:
            changes["turn_budget"] = {"before": self.budget, "after": new_budget}
            self.budget = new_budget
        plan = self._filter_plan(review.turn_plan if review.turn_plan is not None else self.turn_plan, turn)
        if plan != self.turn_plan:
            changes["turn_plan"] = {"before": self.turn_plan, "after": plan}
            self.turn_plan = plan
        if changes:
            categories = [name for name, values in (("student_task_transfer", changes.get("student_issue_ids")),
                ("new_issue", changes.get("new_issue_ids")), ("resolved_issue", newly_resolved), ("reopened_issue", reopened)) if values]
            self.journal.append("plan_changed", phase="planned", turn=turn, **changes,
                reason_categories=categories or ["plan_adjustment"], reason=review.reason)
        self.journal.append("review_recorded", phase="planned", turn=turn, review=review,
            missing_assessment_ids=sorted({i.issue_id for i in feedback.items} - {a.issue_id for a in review.assessments}))
        return review

    def run(self, essay):
        self.essay_id, self.prompt, self.original = essay["essay_id"], essay["prompt"], essay["text"]
        save_json(self.directory / "input.json", essay)
        before = self.diagnostic(self.original, self.prompt, self.essay_id)
        save_json(self.directory / "diagnosis_initial.json", before)
        if self.cfg.condition == "single_call_staged":
            return self._single_call(before)
        self.effective_max = 1 if self.cfg.condition == "always_direct" else self.cfg.max_turns
        initial = self._request("coordinator_initial", {
            "writing_prompt": self.prompt, "current_text": self.original, "diagnosis": before,
            "max_turns": self.effective_max, "condition": self.cfg.condition,
            "forced_initial_budget": self.effective_max if self.cfg.condition == "always_staged" else None,
        }, InitialPlan, "initial", 0)
        self.initial, self.strategy, self.turn_plan = initial, initial.strategy, initial.turn_plan
        if initial.strategy.name == "single_pass":
            self.effective_max = 1
        self.budget = min(initial.turn_budget, self.effective_max)
        self.journal.append("plan_changed", phase="initial", turn=0, initial=True,
            difficulty=initial.difficulty, strategy={"before": None, "after": self.strategy},
            turn_plan={"before": [], "after": self.turn_plan}, turn_budget={"before": None, "after": self.budget}, reason=initial.reason)
        save_json(self.directory / "initial_plan.json", initial)
        for issue in initial.issues:
            self._add_issue(issue, "initial", 0)
        self._queue_tasks(initial.student_tasks, "baseline", 0)
        self._classify_students("baseline", 0)
        if self.student_ids:
            old = self.turn_plan
            self.turn_plan = self._filter_plan(old, 0)
            self.journal.append("plan_changed", phase="baseline", turn=0,
                turn_plan={"before": old, "after": self.turn_plan}, student_issue_ids=sorted(self.student_ids),
                reason_categories=["student_task_transfer"], reason="학생이 해야 할 과제를 수정 계획에서 분리했습니다.")
        if self._student_gate(0):
            return self._finish(self.original, before, "student_tasks", "거시 학생 과제를 제시하고 종료했습니다.")
        if not self._pending():
            return self._finish(self.original, before, "no_revision", "총괄 초기 판단에서 자동 수정할 이슈가 없습니다.")
        current, stop_reason = self.original, "계획된 턴 예산에 도달했습니다."
        for turn in range(1, self.effective_max + 1):
            if turn > self.budget or not self._pending():
                break
            direction, strategy_used, budget_before = self._direction(turn), self.strategy, self.budget
            students_before_turn = self.student_ids.copy()
            features = self.diagnostic.features(current)
            save_json(self.directory / f"features_turn_{turn}.json", features)
            feedback = self._request("feedback", {
                "writing_prompt": self.prompt, "current_text": current, "strategy": jsonable(self.strategy),
                "turn_direction": jsonable(direction), "feedback_count": direction.feedback_count,
                "candidates": jsonable([self.issues[key] for key in direction.issue_ids]), "current_features": features,
            }, Feedback, "planned", turn)
            self._queue_tasks(feedback.student_tasks, "planned", turn)
            self.journal.append("feedback_presented", phase="planned", turn=turn,
                items=feedback.items, strategy=self.strategy, direction=direction, features=features)
            if self._student_gate(turn):
                self.turns.append({"phase": "planned", "turn": turn, "feedback": jsonable(feedback.items),
                    "revised_text": current, "changed_sentences": [], "observational_flags": [], "review": None,
                    "turn_direction": jsonable(direction), "budget_before": budget_before, "budget_after": self.budget,
                    "stop_reason": "피드백 단계에서 거시 학생 과제를 확인하여 수정 호출 전에 종료했습니다."})
                stop_reason = self.turns[-1]["stop_reason"]
                self.journal.append("turn_ended", phase="planned", turn=turn, summary=stop_reason)
                break
            editable = feedback.model_copy(update={"items": [i for i in feedback.items if i.issue_id not in self.student_ids]})
            self.previous = current
            if editable.items:
                revision = self._request("reviser", {"writing_prompt": self.prompt, "current_text": current,
                    "feedback": jsonable(editable)}, Revision, "planned", turn)
                # Always adopt the complete restored text. No executor, retry, tests or rollback.
                current = revision.revised_text
                self._queue_tasks(revision.student_tasks, "planned", turn)
            observed = self.observer.compare(self.previous, current, feedback=jsonable(editable.items),
                previous_changed_ranges=self.changed_ranges)
            self.changed_ranges = observed.pop("next_changed_ranges")
            self.journal.append("revision_recorded", phase="planned", turn=turn,
                text_before=self.previous, text_after=current, **observed)
            # Persist before review too: a later API/budget error cannot erase a revision.
            save_json(self.directory / f"revision_turn_{turn}.json", {"previous_text": self.previous,
                "revised_text": current, **observed})
            review = self._review(current, editable, observed, turn, students_before_turn)
            gate = self._student_gate(turn)
            continue_work = bool(review.continue_work and self._pending() and not gate and turn < self.budget)
            stop_reason = ("거시 학생 과제를 제시하고 종료했습니다." if gate else
                "총괄이 수정 이슈의 반영을 확인했습니다." if not self._pending() and not self.deferred else
                review.reason if not review.continue_work else
                "계획된 턴 예산에 도달했습니다." if turn >= self.budget else review.reason)
            self.turns.append({"phase": "planned", "turn": turn, "feedback": jsonable(feedback.items),
                "feedback_count_planned": direction.feedback_count, "feedback_count_actual": len(feedback.items),
                "previous_text": self.previous, "revised_text": current, "changed_sentences": observed["changed_sentences"],
                "layout_changes": observed["layout_changes"], "observational_flags": observed["flags"],
                "feedback_targets": observed["feedback_targets"], "unlocated_feedback_issue_ids": observed["unlocated_feedback_issue_ids"],
                "strategy": jsonable(strategy_used), "strategy_after_review": jsonable(self.strategy),
                "turn_direction": jsonable(direction), "current_features": features, "review": jsonable(review),
                "budget_before": budget_before, "budget_after": self.budget, "continue_work": continue_work, "stop_reason": stop_reason})
            self.journal.append("turn_ended", phase="planned", turn=turn, summary=stop_reason,
                remaining_issue_ids=sorted(set(self._editable()) - self.resolved),
                remaining_turns=max(0, self.budget - turn), continue_work=continue_work)
            if not continue_work:
                break
        return self._finish(current, before, "single_turn" if len(self.turns) == 1 else "multi_turn", stop_reason)

    def _single_call(self, before):
        response = self._request("reviser", {"mode": "single_call_staged", "writing_prompt": self.prompt,
            "current_text": self.original, "diagnosis": before,
            "instruction": "피드백을 2~3단계로 나누고 단계별 전체 수정본을 보여 주세요."}, SingleCallStages, "single_call_staged", 0)
        current = self.original
        for number, stage in enumerate(response.stages, 1):
            observed = self.observer.compare(current, stage.revised_text, previous_changed_ranges=self.changed_ranges)
            self.changed_ranges = observed.pop("next_changed_ranges")
            self.journal.append("revision_recorded", phase="single_call_staged", turn=number,
                text_before=current, text_after=stage.revised_text, **observed)
            self.turns.append({"phase": "single_call_staged", "turn": number, **stage.model_dump(),
                "changed_sentences": observed["changed_sentences"], "observational_flags": observed["flags"],
                "comparison_baseline_full_text": True})
            current = stage.revised_text
        self._queue_tasks(response.student_tasks, "single_call_staged", 0)
        return self._finish(current, before, "single_call_staged", "한 호출의 단계별 수정 비교 조건입니다. 별도 총괄 검토는 없습니다.")

    def _finish(self, text, before, route, reason):
        try:
            after = self.diagnostic(text, self.prompt, self.essay_id)
            save_json(self.directory / "diagnosis_final.json", after)
        except Exception as exc:
            after = {}
            save_json(self.directory / "diagnosis_final.error.json", {"error_type": type(exc).__name__, "error": str(exc),
                "reason": "최종 점수는 미측정입니다. 수정본은 그대로 보존했습니다."})
        initial_scores, final_scores = before.get("scores", {}), after.get("scores", {})
        deferred_ids = {i["issue_id"] for i in self.deferred_micro_feedback}
        tasks = [jsonable(task) for task in self.tasks if task.issue_id not in deferred_ids]
        histories = {"strategy": [], "turn_plan": []}
        for event in self.journal.events:
            if event["kind"] == "plan_changed":
                for key in histories:
                    if key in event["payload"]:
                        histories[key].append({"turn": event["turn"], "phase": event["phase"],
                            **event["payload"][key], "reason": event["payload"].get("reason", "")})
        result = {"essay_id": self.essay_id, "condition": self.cfg.condition, "mock": bool(self.cfg.mock or before.get("mock")),
            "revision_mode": "full_text", "review_method": "coordinator", "route": route, "route_reason": reason,
            "difficulty": jsonable(self.initial.difficulty) if self.initial else None,
            "initial_strategy": jsonable(self.initial.strategy) if self.initial else None, "strategy": jsonable(self.strategy),
            "strategy_history": histories["strategy"], "initial_turn_plan": jsonable(self.initial.turn_plan) if self.initial else [],
            "turn_plan": jsonable(self.turn_plan), "turn_plan_history": histories["turn_plan"],
            "initial_text": self.original, "final_text": text, "turns": self.turns,
            "initial_scores": initial_scores, "final_scores": final_scores,
            "score_delta": {key: final_scores[key] - value for key, value in initial_scores.items() if key in final_scores},
            "changed_char_ratio": changed_char_ratio(self.original, text), "score_n": 32, "diagnostic_calls": 2,
            "student_tasks": tasks, "issues": jsonable(list(self.issues.values())), "resolved_issue_ids": sorted(self.resolved),
            "remaining_issue_ids": sorted(set(self._editable()) - self.resolved - deferred_ids),
            "student_task_issue_ids": sorted(self.student_ids), "macro_gate_blockers": self.gate_blockers,
            "deferred_micro_feedback": self.deferred_micro_feedback, "memory_summaries": self.memory_views,
            "event_count": len(self.journal.events), "journal_sha256": self.journal.digest()}
        save_json(self.directory / "result.json", result)
        (self.directory / "final.txt").write_text(text, encoding="utf-8")
        return result
