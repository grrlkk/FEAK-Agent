"""Scoped revision, coordinator-owned student tasks/restoration and remaining turns."""
import json
from pathlib import Path
import re
import unicodedata

from .events import Journal, jsonable, save_json
from .models import Feedback, InitialPlan, Revision, Review, SingleCallStages, StudentTask, TurnDirection
from .observations import changed_char_ratio
from .scopes import assemble, document_units, public_units, restore_sentences, revision_scopes


class Engine:
    def __init__(self, cfg, *, llm, diagnostic, observer, anonymizer_factory, directory):
        self.cfg, self.llm, self.diagnostic, self.observer = cfg, llm, diagnostic, observer
        self.anonymizer_factory = anonymizer_factory
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.directory / "events.jsonl")
        if self.journal.events:
            raise ValueError("Use a fresh essay directory; do not overwrite an experiment")
        self.issues, self.resolved, self.student_ids = {}, set(), set()
        self.tasks, self.optional_suggestions, self.turns, self.memory_views = [], [], [], []
        self.memory_enabled = cfg.condition not in {"routed_no_memory", "planned_no_memory"}
        self.remaining, self.effective_max = cfg.max_turns, cfg.max_turns
        self.strategy, self.turn_plan, self.initial = None, [], None
        self.changed_ranges = []

    def _editable(self):
        return {key: issue for key, issue in self.issues.items() if key not in self.student_ids}

    def _pending(self):
        return set(self._editable()) - self.resolved

    def _waiting_micro(self):
        restructure = {key for key in self._pending() if self.issues[key].level == "macro"
                       and self.issues[key].macro_type == "restructure"}
        return {key: sorted(dependencies) for key in sorted(self._pending())
                if self.issues[key].level == "micro"
                and (dependencies := set(self.issues[key].restructure_issue_ids) & restructure)}

    def _available(self):
        return self._pending() - self._waiting_micro().keys()

    def _deferred_feedback(self):
        return [dict(issue_id=key, restructure_issue_ids=dependencies,
                     when="해당 문장의 재배열을 마친 뒤 다룰 것", feedback=self.issues[key].description)
                for key, dependencies in self._waiting_micro().items()]

    @staticmethod
    def _task_text(text):
        return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text)).casefold()

    def _queue_tasks(self, tasks, phase, turn):
        # Called only with a coordinator's response/issue classifications.
        for task in tasks:
            if task.issue_id in self.issues:
                self.student_ids.add(task.issue_id)
                self.resolved.discard(task.issue_id)
                issue = self.issues[task.issue_id]
                self.issues[task.issue_id] = issue.model_copy(update={"needs_student_content": True,
                    "macro_type": "student_input" if issue.level == "macro" else issue.macro_type})
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
            if (issue.needs_student_content or issue.macro_type == "student_input") and issue.id not in self.student_ids:
                self._queue_tasks([StudentTask(issue_id=issue.id, instruction=issue.description,
                    reason="총괄이 학생의 내용·경험·의도 확인이 꼭 필요하다고 판단했습니다.")], phase, turn)

    def _queue_suggestions(self, suggestions, phase, turn):
        for suggestion in suggestions:
            if self._task_text(suggestion) not in {self._task_text(s) for s in self.optional_suggestions}:
                self.optional_suggestions.append(suggestion)
                self.journal.append("optional_suggestion_added", phase=phase, turn=turn, suggestion=suggestion)

    def _filter_plan(self, plan, turn):
        return [item.model_copy(update={"issue_ids": [key for key in item.issue_ids if key in self._pending()]})
                if item.turn > turn else item for item in plan]

    def _request(self, role, payload, schema, phase, turn, *, scope_id=None):
        memory_role = "coordinator" if role.startswith("coordinator") else role
        # Reviser gets only its scope, neighboring context and that scope's feedback.
        # Do not leak a full previous essay through the old journal projection.
        if role not in {"reviser", "single_call_staged"}:
            view = self.journal.summary(memory_role, phase=phase,
                remaining_turns=self.remaining, enabled=self.memory_enabled)
            self.memory_views.append({"role": memory_role, "phase": phase, "turn": turn, "summary": view})
            payload = {**payload, "memory_summary": view}
        anonymizer = self.anonymizer_factory(self.essay_id, self.cfg.condition + ":" + role, turn)
        masked = anonymizer.mask_data(payload)
        with (self.directory / "anonymization.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"role": role, "phase": phase, "turn": turn, "scope_id": scope_id,
                "mapping": anonymizer.mapping}, ensure_ascii=False) + "\n")
        response = self.llm.request(role, masked, schema,
            context={"essay_id": self.essay_id, "phase": phase, "turn": turn, "scope_id": scope_id})
        return schema.model_validate(anonymizer.restore_data(response))

    def _add_issue(self, issue, phase, turn):
        replaced = issue.id in self.issues
        self.issues[issue.id] = issue
        self.journal.append("issue_added", phase=phase, turn=turn, issue=issue, updated_existing_id=replaced)

    def _direction(self, turn):
        available = self._available()
        old = next((item for item in self.turn_plan if item.turn == turn), None)
        if self.strategy.name == "single_pass" or self.cfg.condition == "always_direct":
            ids = sorted(available, key=lambda key: (self.issues[key].priority, key))
            direction = TurnDirection(turn=turn, direction=old.direction if old else "현재 수정 가능한 이슈를 이번 한 턴에 다룹니다.",
                issue_ids=ids, feedback_count=len(ids))
        elif old and set(old.issue_ids) & available:
            ids = [key for key in old.issue_ids if key in available]
            direction = old.model_copy(update={"issue_ids": ids, "feedback_count": len(ids)})
        else:
            future = next((p for p in self.turn_plan if p.turn > turn and set(p.issue_ids) & available), None)
            ids = [key for key in future.issue_ids if key in available] if future else sorted(available)
            direction = TurnDirection(turn=turn, direction=future.direction if future else "현재 수정 가능한 이슈를 다룹니다.",
                issue_ids=ids, feedback_count=len(ids))
        if direction != old:
            before = self.turn_plan
            self.turn_plan = sorted([p for p in before if p.turn != turn] + [direction], key=lambda p: p.turn)
            assigned = set(old.issue_ids) if old else set()
            categories = [name for name, values in (("student_task_transfer", self.student_ids),
                ("resolved_issue", self.resolved), ("restructure_dependency", set(self._waiting_micro()))) if assigned & values]
            self.journal.append("plan_changed", phase="planned", turn=turn,
                turn_plan={"before": before, "after": self.turn_plan}, reason_categories=categories or ["plan_adjustment"],
                reason="학생 과제는 분리하고 재배열을 기다리는 문장만 미룬 뒤 현재 수정 가능한 이슈를 배정했습니다.")
        return direction

    def _review(self, current, feedback, observed, turn, scope_records, scope_unavailable):
        splitter = self.observer.sentence_spans
        review = self._request("coordinator_review", {
            "writing_prompt": self.prompt, "current_text": current, "previous_text": self.previous,
            "previous_document_units": public_units(document_units(self.previous, splitter)),
            "current_document_units": public_units(document_units(current, splitter)),
            "feedback": jsonable(feedback), "changed_sentences": observed["changed_sentences"],
            "layout_changes": observed["layout_changes"], "observational_flags": observed["flags"],
            "feedback_targets": observed["feedback_targets"], "scope_revisions": scope_records,
            "scope_unavailable": scope_unavailable, "unlocated_feedback_issue_ids": observed["unlocated_feedback_issue_ids"],
            "issues": jsonable(list(self.issues.values())), "resolved_issue_ids": sorted(self.resolved),
            "student_tasks": jsonable(self.tasks), "optional_suggestions": self.optional_suggestions,
            "deferred_micro_feedback": self._deferred_feedback(), "current_turn": turn,
            "remaining_turns": self.remaining, "max_remaining_turns": self.effective_max - turn,
            "strategy": jsonable(self.strategy), "turn_plan": jsonable(self.turn_plan),
        }, Review, "planned", turn)
        changes, newly_resolved, reopened = {}, [], []
        old_students = self.student_ids.copy()
        for assessment in review.assessments:
            if assessment.issue_id not in self._editable():
                continue
            if assessment.status == "applied":
                if assessment.issue_id not in self.resolved:
                    newly_resolved.append(assessment.issue_id)
                    self.journal.append("issue_resolved", phase="planned", turn=turn,
                        issue_id=assessment.issue_id, reason=assessment.reason, source="coordinator_review")
                self.resolved.add(assessment.issue_id)
            elif assessment.issue_id in self.resolved:
                reopened.append(assessment.issue_id)
                self.resolved.discard(assessment.issue_id)
        for key in review.reopened:
            if key in self.resolved:
                reopened.append(key)
                self.resolved.discard(key)
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
        self._queue_tasks(review.student_tasks, "planned", turn)
        self._classify_students("planned", turn)
        self._queue_suggestions(review.optional_suggestions, "planned", turn)
        if self.student_ids != old_students:
            changes["student_issue_ids"] = sorted(self.student_ids - old_students)
        strategy_used = self.strategy
        if review.strategy is not None and review.strategy != self.strategy:
            changes["strategy"] = {"before": self.strategy, "after": review.strategy}
            self.strategy = review.strategy
        new_remaining = min(review.remaining_turns, self.effective_max - turn)
        if strategy_used.name == "single_pass":
            new_remaining = 0
        elif self.strategy.name == "single_pass":
            new_remaining = min(new_remaining, 1)
        if not review.continue_work:
            new_remaining = 0
        if new_remaining != self.remaining:
            changes["remaining_turns"] = {"before": self.remaining, "after": new_remaining}
        self.remaining = new_remaining
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
            "forced_remaining_turns": self.effective_max if self.cfg.condition == "always_staged" else None,
        }, InitialPlan, "initial", 0)
        self.initial, self.strategy, self.turn_plan = initial, initial.strategy, initial.turn_plan
        if initial.strategy.name == "single_pass":
            self.effective_max = 1
        self.remaining = min(initial.remaining_turns, self.effective_max)
        self.journal.append("plan_changed", phase="initial", turn=0, initial=True,
            difficulty=initial.difficulty, strategy={"before": None, "after": self.strategy},
            turn_plan={"before": [], "after": self.turn_plan}, remaining_turns={"before": None, "after": self.remaining}, reason=initial.reason)
        save_json(self.directory / "initial_plan.json", initial)
        for issue in initial.issues:
            self._add_issue(issue, "initial", 0)
        self._queue_tasks(initial.student_tasks, "baseline", 0)
        self._classify_students("baseline", 0)
        self._queue_suggestions(initial.optional_suggestions, "baseline", 0)
        if self.student_ids:
            old = self.turn_plan
            self.turn_plan = self._filter_plan(old, 0)
            self.journal.append("plan_changed", phase="baseline", turn=0,
                turn_plan={"before": old, "after": self.turn_plan}, student_issue_ids=sorted(self.student_ids),
                reason_categories=["student_task_transfer"], reason="학생 과제를 먼저 안내하고 나머지 수정 계획을 계속합니다.")
        if not self._pending():
            return self._finish(self.original, before, "student_tasks" if self.tasks else "no_revision",
                                "총괄 초기 판단에서 자동 수정할 이슈가 없습니다.")
        current, stop_reason = self.original, "계획된 남은 턴 수에 도달했습니다."
        splitter = self.observer.sentence_spans
        for turn in range(1, self.effective_max + 1):
            if not self.remaining or not self._available():
                break
            direction, strategy_used, remaining_before = self._direction(turn), self.strategy, self.remaining
            self.remaining -= 1
            features = self.diagnostic.features(current)
            save_json(self.directory / f"features_turn_{turn}.json", features)
            feedback = self._request("feedback", {
                "writing_prompt": self.prompt, "current_text": current, "strategy": jsonable(self.strategy),
                "document_units": public_units(document_units(current, splitter)),
                "turn_direction": jsonable(direction), "feedback_count": direction.feedback_count,
                "candidates": jsonable([self.issues[key] for key in direction.issue_ids]), "current_features": features,
            }, Feedback, "planned", turn)
            self.journal.append("feedback_presented", phase="planned", turn=turn,
                items=feedback.items, strategy=self.strategy, direction=direction, features=features)
            self.previous = current
            editable = [i for i in feedback.items if i.issue_id not in self.student_ids
                        and i.issue_id not in self._waiting_micro()]
            scopes, unavailable = revision_scopes(current, editable, self.issues, splitter)
            replacements, scope_records = [], []
            for scope in scopes:
                revision = self._request("reviser", {"source_text": scope["source_text"],
                    "context_before": scope["context_before"], "context_after": scope["context_after"],
                    "feedback": jsonable(scope["feedback"])}, Revision, "planned", turn, scope_id=scope["scope_id"])
                replacement = scope["source_text"] if revision.cannot_apply else revision.revised_text
                replacements.append(dict(start=scope["start"], end=scope["end"], replacement=replacement))
                record = {"scope_id": scope["scope_id"], "target_ids": scope["target_ids"],
                    "issue_ids": [i.issue_id for i in scope["feedback"]], "before": scope["source_text"],
                    "after": replacement, "submitted_text": revision.revised_text, "cannot_apply": revision.cannot_apply,
                    "source_span": [scope["start"], scope["end"]]}
                scope_records.append(record)
                self.journal.append("scope_revision_recorded", phase="planned", turn=turn, **record)
            candidate = assemble(current, replacements)
            target_ranges = [dict(start=s["start"], end=s["end"], issue_ids=[i.issue_id for i in s["feedback"]]) for s in scopes]
            observed = self.observer.compare(current, candidate, feedback=jsonable(feedback.items),
                previous_changed_ranges=self.changed_ranges, target_ranges=target_ranges)
            observed.pop("next_changed_ranges")
            self.journal.append("revision_recorded", phase="planned", turn=turn,
                text_before=current, text_after=candidate, **observed)
            save_json(self.directory / f"revision_turn_{turn}.json", {"previous_text": current,
                "revised_text": candidate, "scope_revisions": scope_records, "scope_unavailable": unavailable, **observed})
            review = self._review(candidate, feedback, observed, turn, scope_records, unavailable)
            current, restored = restore_sentences(self.previous, candidate, review.restore, splitter)
            final_observed = self.observer.compare(self.previous, current, feedback=jsonable(feedback.items),
                previous_changed_ranges=self.changed_ranges, target_ranges=target_ranges)
            self.changed_ranges = final_observed.pop("next_changed_ranges")
            if restored:
                self.journal.append("restoration_recorded", phase="planned", turn=turn,
                    requests=restored, text_before=candidate, text_after=current,
                    changed_sentences=final_observed["changed_sentences"], flags=final_observed["flags"])
            save_json(self.directory / f"reviewed_turn_{turn}.json", {"revised_text": current,
                "candidate_text": candidate, "restore_results": restored, "review": review, **final_observed})
            continue_work = bool(review.continue_work and self._available() and self.remaining)
            stop_reason = ("총괄이 수정 이슈의 반영을 확인했습니다." if not self._pending() else
                review.reason if not review.continue_work else
                "계획된 남은 턴 수에 도달했습니다." if not self.remaining else review.reason)
            self.turns.append({"phase": "planned", "turn": turn, "feedback": jsonable(feedback.items),
                "feedback_count_planned": direction.feedback_count, "feedback_count_actual": len(feedback.items),
                "previous_text": self.previous, "candidate_text": candidate, "revised_text": current,
                "scope_revisions": scope_records, "scope_unavailable": unavailable, "restore_results": restored,
                "changed_sentences": observed["changed_sentences"], "final_changed_sentences": final_observed["changed_sentences"],
                "layout_changes": observed["layout_changes"], "observational_flags": observed["flags"],
                "feedback_targets": observed["feedback_targets"], "unlocated_feedback_issue_ids": observed["unlocated_feedback_issue_ids"],
                "strategy": jsonable(strategy_used), "strategy_after_review": jsonable(self.strategy),
                "turn_direction": jsonable(direction), "current_features": features, "review": jsonable(review),
                "remaining_before": remaining_before, "remaining_after_consumption": remaining_before - 1,
                "remaining_after_review": self.remaining, "deferred_micro_feedback": self._deferred_feedback(),
                "continue_work": continue_work, "stop_reason": stop_reason})
            self.journal.append("turn_ended", phase="planned", turn=turn, summary=stop_reason,
                remaining_issue_ids=sorted(self._pending()), remaining_turns=self.remaining, continue_work=continue_work)
            if not continue_work:
                break
        return self._finish(current, before, "single_turn" if len(self.turns) == 1 else "multi_turn", stop_reason)

    def _single_call(self, before):
        response = self._request("single_call_staged", {"writing_prompt": self.prompt,
            "current_text": self.original, "diagnosis": before}, SingleCallStages, "single_call_staged", 0)
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
        # The comparison's single call is the combined coordinator, not the scoped worker.
        self._queue_tasks(response.student_tasks, "single_call_staged", 0)
        self._queue_suggestions(response.optional_suggestions, "single_call_staged", 0)
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
        histories = {"strategy": [], "turn_plan": []}
        for event in self.journal.events:
            if event["kind"] == "plan_changed":
                for key in histories:
                    if key in event["payload"]:
                        histories[key].append({"turn": event["turn"], "phase": event["phase"],
                            **event["payload"][key], "reason": event["payload"].get("reason", "")})
        result = {"essay_id": self.essay_id, "student_tasks": jsonable(self.tasks),
            "condition": self.cfg.condition, "mock": bool(self.cfg.mock or before.get("mock")),
            "revision_mode": "scoped", "review_method": "coordinator", "route": route, "route_reason": reason,
            "difficulty": jsonable(self.initial.difficulty) if self.initial else None,
            "initial_strategy": jsonable(self.initial.strategy) if self.initial else None, "strategy": jsonable(self.strategy),
            "strategy_history": histories["strategy"], "initial_turn_plan": jsonable(self.initial.turn_plan) if self.initial else [],
            "turn_plan": jsonable(self.turn_plan), "turn_plan_history": histories["turn_plan"],
            "initial_text": self.original, "final_text": text, "turns": self.turns,
            "initial_scores": initial_scores, "final_scores": final_scores,
            "score_delta": {key: final_scores[key] - value for key, value in initial_scores.items() if key in final_scores},
            "changed_char_ratio": changed_char_ratio(self.original, text), "score_n": 32, "diagnostic_calls": 2,
            "issues": jsonable(list(self.issues.values())), "resolved_issue_ids": sorted(self.resolved),
            "remaining_issue_ids": sorted(self._pending()), "student_task_issue_ids": sorted(self.student_ids),
            "deferred_micro_feedback": self._deferred_feedback(), "remaining_turns": self.remaining,
            "memory_summaries": self.memory_views, "event_count": len(self.journal.events),
            "journal_sha256": self.journal.digest(), "optional_suggestions": self.optional_suggestions}
        save_json(self.directory / "result.json", result)
        (self.directory / "final.txt").write_text(text, encoding="utf-8")
        return result
