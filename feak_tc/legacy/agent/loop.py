"""Bounded revision trajectory with independent local and global decisions."""

from dataclasses import asdict, replace
import math
from typing import Any

from feak_tc.mvp.heuristic import build_result, heuristic_score
from feak_tc.mvp.llm import LLMResponseError
from feak_tc.mvp.targeting import is_protected_deletion_span
from feak_tc.mvp.transition import _continuous_rubrics, compute_transition
from feak_tc.mvp.validity import patch_validity_violations

from .local_llm import LocalCallBudgetExceeded
from .retrieval import ExemplarStore
from .schemas import ControllerConfig


def _quality(diagnosis):
    scores = list(_continuous_rubrics(diagnosis).values())
    if not all(math.isfinite(x) for x in scores):
        raise ValueError("Diagnosis contains non-finite rubric scores")
    return sum(scores) / len(scores)


def run_agent(text, *, question, diagnoser, roles, cfg, exemplars=None,
              essay_id=None, source_group=None, on_event=None, similarity_fn=None) -> dict[str, Any]:
    """Return original/final snapshots and all decisions; never change the input.

    `diagnoser` and `roles` are injected so a future learned verifier can replace
    only roles.verify. No training or dataset generation occurs in this path.
    """
    if not text.strip() or not question.strip():
        raise ValueError("Essay and question must not be empty")
    control = ControllerConfig.model_validate(cfg["controller"])
    exemplars = exemplars or ExemplarStore()
    if exemplars.rows and not source_group:
        raise ValueError("source_group is required when using exemplars")
    events = []
    checkpoints = []
    history = []
    current_text = text
    current_id = best_id = 0
    no_progress = rollbacks = accepted = 0
    seen = {text}
    failed_plans = set()
    cache = {}

    def emit(event, **data):
        row = {"event": event, **data}
        events.append(row)
        if on_event:
            on_event(row)

    def diagnose(value):
        if value not in cache:
            cache[value] = diagnoser.diagnose(value)
            _quality(cache[value])
        return cache[value]

    stop_reason = "max_steps"
    status = "completed"
    try:
        emit("phase", step=0, stage="diagnose")
        current = diagnose(text)
        checkpoints.append({"id": 0, "parent_id": None, "text": text,
                            "diagnosis": asdict(current), "quality": _quality(current), "safe": True})
        emit("start", original_text=text, question=question, diagnosis=asdict(current))
        for step in range(1, control.max_steps + 1):
            emit("phase", step=step, stage="plan")
            memory = history[-control.memory_steps:]
            examples = exemplars.retrieve(text, current.weak_rubrics,
                                          essay_id=essay_id, source_group=source_group)
            response = roles.plan(question, current, memory, examples)
            request = response.plan
            emit("plan", step=step, checkpoint_id=current_id, response=response.model_dump(),
                 exemplar_ids=[row["essay_id"] for row in examples])
            if request is None:
                stop_reason = "planner_no_request"
                break
            invalid_plan = None
            plan_key = (current_text, request.action_type, request.target_span)
            if plan_key in failed_plans:
                invalid_plan = "repeated_failed_plan"
            elif request.target_span not in current_text:
                invalid_plan = "target_span_not_found"
            elif current_text.count(request.target_span) != 1:
                invalid_plan = "ambiguous_target_span"
            elif request.action_type == "DELETE_OR_FOCUS" and is_protected_deletion_span(
                current_text, request.target_span, question=question
            ):
                invalid_plan = "protected_definition"

            viable = []
            rejected = []
            candidate_texts = set()
            for index in range(0 if invalid_plan else control.candidates_per_step):
                row = {"step": step, "index": index, "request": request.model_dump()}
                try:
                    emit("phase", step=step, index=index, stage="generate")
                    candidate = roles.patch(current_text, request, index)
                    row["candidate"] = candidate.to_dict()
                    emit("patch", **row)
                    after_text = candidate.new_text
                    reasons = patch_validity_violations(current_text, candidate, cfg=cfg)
                    if not after_text or after_text == current_text:
                        reasons.append("no_effect")
                    if after_text in seen:
                        reasons.append("visited_state")
                    if after_text in candidate_texts:
                        reasons.append("duplicate_candidate")
                    candidate_texts.add(after_text)
                    if not reasons:
                        emit("phase", step=step, index=index, stage="rediagnose")
                        after = diagnose(after_text)
                        row["after"] = asdict(after)
                        transition = compute_transition(current, after, candidate, similarity_fn=similarity_fn)
                        result = build_result(candidate, transition, cfg)
                        row.update(result.to_dict())
                        reasons.extend(result.reject_reasons)
                        # Give the RV only operational text/request inputs, regardless of FEAK gain.
                        emit("phase", step=step, index=index, stage="verify")
                        rv = roles.verify(question, current_text, request, after_text)
                        row["rv"] = rv.model_dump()
                        for axis, allowed in (
                            ("target_fulfillment", control.rv_target_labels),
                            ("preservation", control.rv_preservation_labels),
                        ):
                            if getattr(rv, axis).label not in allowed:
                                reasons.append("rv_" + axis)
                        noop = replace(transition, target_gain=0.0, target_gap_reduction=0.0,
                                       non_target_drop=0.0, edit_ratio=0.0,
                                       goal_preservation=1.0, emb_sim=1.0)
                        noop_score = heuristic_score(noop, cfg)
                        advantage = result.heuristic_score - noop_score
                        row.update(noop_score=noop_score, advantage=advantage,
                                   quality_delta=_quality(after) - _quality(current))
                        if not math.isfinite(advantage) or advantage <= control.noop_margin:
                            reasons.append("no_improvement_over_noop")
                        if _quality(after) < _quality(current) - control.quality_drop_max:
                            reasons.append("quality_drop")
                        if not reasons:
                            viable.append((advantage, index, after))
                    row["reject_reasons"] = list(dict.fromkeys(reasons))
                    row["rejected"] = bool(reasons)
                except LocalCallBudgetExceeded:
                    raise
                except (LLMResponseError, ValueError) as exc:
                    row.update(rejected=True, reject_reasons=["candidate_error"], error=str(exc))
                emit("candidate", **row)
                if row["rejected"]:
                    rejected.append({"index": index, "reasons": row["reject_reasons"]})

            if not viable:
                failed_plans.add(plan_key)
                no_progress += 1
                record = {"step": step, "decision": "reject", "request": request.model_dump(),
                          "reason": invalid_plan or "no_viable_candidate", "rejections": rejected}
                history.append(record)
                emit("reject", **{key: value for key, value in record.items() if key != "decision"})
                if no_progress >= control.no_progress_patience:
                    stop_reason = "no_progress"
                    break
                emit("replan", step=step, reason=record["reason"])
                continue

            _, chosen_index, proposed = max(viable, key=lambda item: (item[0], -item[1]))
            previous_id = current_id
            current_id = len(checkpoints)
            current, current_text = proposed, proposed.text
            seen.add(current_text)
            accepted += 1
            checkpoints.append({"id": current_id, "parent_id": previous_id, "text": current_text,
                                "diagnosis": asdict(current), "quality": _quality(current), "safe": False})
            record = {"step": step, "decision": "accept", "request": request.model_dump(),
                      "checkpoint_id": current_id, "chosen_index": chosen_index}
            history.append(record)
            emit("accept", step=step, checkpoint_id=current_id, parent_id=previous_id,
                 chosen_index=chosen_index, text=current_text, diagnosis=asdict(current))

            # Global checks happen after adoption. On failure, restore a stored full snapshot.
            best = checkpoints[best_id]
            try:
                emit("phase", step=step, stage="guard")
                guard = roles.guard(question, text, best["text"], current_text,
                                    history[-control.memory_steps:])
                guard_reasons = []
                for axis in ("preservation", "coherence"):
                    label = getattr(guard, axis).label
                    if label in (None, "fail") or (control.guard_require_pass and label != "pass"):
                        guard_reasons.append("guard_" + axis)
                if _quality(current) < best["quality"] - control.guard_quality_drop_max:
                    guard_reasons.append("checkpoint_quality_drop")
                emit("guard", step=step, checkpoint_id=current_id,
                     verdict=guard.model_dump(), reasons=guard_reasons)
            except Exception as exc:
                # A verifier failure of any kind must not expose an unguarded final state.
                emit("guard", step=step, checkpoint_id=current_id, error=str(exc),
                     reasons=["guard_unavailable"])
                emit("rollback", step=step, from_checkpoint=current_id, to_checkpoint=best_id,
                     text=best["text"], reasons=["guard_unavailable"])
                current_id, current_text = best_id, best["text"]
                current = diagnose(current_text)
                rollbacks += 1
                raise
            if guard_reasons:
                failed_plans.add(plan_key)
                emit("rollback", step=step, from_checkpoint=current_id, to_checkpoint=best_id,
                     text=best["text"], reasons=guard_reasons)
                history.append({"step": step, "decision": "rollback", "request": request.model_dump(),
                                "reasons": guard_reasons, "to_checkpoint": best_id})
                current_id, current_text = best_id, best["text"]
                current = diagnose(current_text)
                rollbacks += 1
                no_progress += 1
                if rollbacks >= control.max_rollbacks:
                    stop_reason = "rollback_limit"
                    break
                if no_progress >= control.no_progress_patience:
                    stop_reason = "no_progress"
                    break
                emit("replan", step=step, reason="trajectory_damage")
                continue
            checkpoints[current_id]["safe"] = True
            if _quality(current) >= best["quality"]:
                best_id = current_id
            no_progress = 0
    except LocalCallBudgetExceeded:
        stop_reason = "llm_call_budget"
    except Exception as exc:
        status, stop_reason = "error", "runtime_error"
        emit("error", error_type=type(exc).__name__, message=str(exc))

    emit("stop", reason=stop_reason, status=status, checkpoint_id=current_id)
    return {"schema_version": "training_free_agent_v1", "status": status,
            "original_text": text, "final_text": current_text, "question": question,
            "final_checkpoint_id": current_id, "best_checkpoint_id": best_id,
            "stop_reason": stop_reason, "accepted": accepted, "rollbacks": rollbacks,
            "diagnosis_calls": len(cache), "checkpoints": checkpoints, "events": events}
