"""Single-issue revision with RV-only adoption and bounded fresh verification."""

from dataclasses import asdict

from feak_tc.runtime.openai import CallBudgetExceeded

from .roles import PROMPT_VERSION, changed_passages
from .schemas import CRITERIA, ControllerConfig


def rejection_reasons(verdicts):
    verdict = verdicts[-1]
    return [f"{name}: {getattr(verdict, name).label}: {getattr(verdict, name).reason}"
            for name in CRITERIA if getattr(verdict, name).label != "PASS"]


def score_baseline(before, candidate, target):
    target_delta = candidate[target] - before[target]
    aggregate_delta = sum(candidate.values()) - sum(before.values())
    return {"target_delta": target_delta, "aggregate_delta": aggregate_delta,
            "target_accept": target_delta > 0, "aggregate_accept": aggregate_delta > 0,
            "accept_either": target_delta > 0 or aggregate_delta > 0,
            "used_for_control": False}


def run_pilot(sample, *, scorer, roles, config=None, on_attempt=None, on_event=None):
    control = config or ControllerConfig()
    current = sample.text
    state = None
    original_state = None
    accepted = 0
    attempts = []
    events = []
    status, stop_reason = "completed", "max_iterations"
    pending = None

    def emit(event, **data):
        row = {"sample_id": sample.sample_id, "event": event, **data}
        events.append(row)
        if on_event:
            on_event(row)

    def save(row):
        attempts.append(row)
        if on_attempt:
            on_attempt(row)

    try:
        emit("phase", stage="score_initial")
        state = scorer.score(current)
        original_state = state
        emit("state", iteration=0, text=current, **asdict(state))
        for iteration in range(1, control.max_iterations + 1):
            emit("phase", iteration=iteration, stage="plan")
            response = roles.plan(sample.writing_prompt, current, state.scores)
            emit("plan", iteration=iteration, **response.model_dump())
            if response.plan is None:
                stop_reason = response.outcome
                break
            plan = response.plan
            if plan.target_span not in current:
                raise ValueError("Planner target_span is not in current draft")
            feedback = None
            issue_accepted = False
            for attempt in range(control.max_retry_per_issue + 1):
                before, before_state = current, state
                pending = {"sample_id": sample.sample_id, "iteration": iteration, "attempt": attempt,
                           "text_before": before, "scores_before": before_state.scores,
                           "score_metadata_before": before_state.metadata, "plan": plan.model_dump(),
                           "candidate": None, "rv": None, "rv_checks": [],
                           "planner_review": {"priority_checks": [c.model_dump() for c in response.priority_checks],
                                              "reason": response.reason,
                                              "source_review": (response.source_review.model_dump()
                                                                if response.source_review else None)},
                           "acceptance_decision": None, "controller_decision": None,
                           "text_after": before, "scores_after": before_state.scores,
                           "scores_candidate": None, "score_baseline": None}
                emit("phase", iteration=iteration, attempt=attempt, stage="revise")
                revision = roles.revise(sample.writing_prompt, before, plan, feedback)
                if revision.outcome == "cannot_revise":
                    pending.update(acceptance_decision="NOT_GENERATED", controller_decision="STOP",
                                   revision_outcome=revision.outcome,
                                   summary_of_change=revision.summary_of_change)
                    save(pending)
                    pending = None
                    emit("revision_unavailable", iteration=iteration, attempt=attempt,
                         reason=revision.summary_of_change)
                    break
                candidate = revision.revised_text
                pending.update(candidate=candidate, summary_of_change=revision.summary_of_change,
                               revision_outcome=revision.outcome,
                               edit_patch=revision.patch.model_dump() if revision.patch else None,
                               text_diff=changed_passages(before, candidate))
                # Persist the candidate before any verifier/measurement can fail.
                emit("candidate", iteration=iteration, attempt=attempt, text_before=before,
                     candidate=candidate, plan=plan.model_dump())
                verdicts = []
                reasons = []
                if candidate == before:
                    decision, reasons = "REJECT", ["no_change: candidate equals the current draft"]
                else:
                    for check in range(control.max_reverify + 1):
                        emit("phase", iteration=iteration, attempt=attempt, stage="verify", check=check)
                        # All checks get identical allowed text inputs, with a fresh API context.
                        rv = roles.verify(sample.writing_prompt, before, candidate, plan)
                        verdicts.append(rv)
                        pending["rv"] = rv.model_dump()
                        pending["rv_checks"].append(rv.model_dump())
                        emit("rv", iteration=iteration, attempt=attempt, check=check, verdict=rv.model_dump())
                        decision = rv.rule_decision()
                        if decision != "REVERIFY":
                            break
                    if decision == "REVERIFY":
                        decision = "REJECT"
                    if decision == "REJECT":
                        reasons = rejection_reasons(verdicts)

                # Adoption is finalized before candidate scores are requested.
                is_accepted = decision == "ACCEPT"
                controller_decision = "ACCEPT" if is_accepted else (
                    "RETRY" if attempt < control.max_retry_per_issue else "STOP")
                if is_accepted:
                    current, state = candidate, None
                    accepted += 1
                    issue_accepted = True
                pending.update(acceptance_decision=decision, controller_decision=controller_decision,
                               rejection_reasons=reasons, text_after=current,
                               scores_after=None if is_accepted else before_state.scores)
                emit("decision", iteration=iteration, attempt=attempt,
                     acceptance_decision=decision, controller_decision=controller_decision,
                     text_after=current, rejection_reasons=reasons)

                if is_accepted or control.log_score_baseline:
                    emit("phase", iteration=iteration, attempt=attempt, stage="score_candidate")
                    measured = scorer.score(candidate)
                    pending.update(scores_candidate=measured.scores, score_metadata_candidate=measured.metadata)
                    if control.log_score_baseline:
                        pending["score_baseline"] = score_baseline(before_state.scores, measured.scores,
                                                                  plan.target_rubric)
                    if is_accepted:
                        state = measured
                        pending["scores_after"] = state.scores
                        emit("state", iteration=iteration, text=current, **asdict(state))
                save(pending)
                pending = None
                if is_accepted:
                    break
                feedback = {"rejected_candidate": candidate, "reasons": reasons}
            if not issue_accepted:
                stop_reason = "revision_not_feasible" if revision.outcome == "cannot_revise" else "repeated_rejection"
                break
    except Exception as exc:
        status = "error"
        stop_reason = "llm_call_budget" if isinstance(exc, CallBudgetExceeded) else "runtime_error"
        # API errors have already been sanitized by the shared client.
        error = {"type": type(exc).__name__, "message": str(exc)}
        if pending is not None:
            pending["error"] = error
            pending["text_after"] = current
            if pending["acceptance_decision"] is None:
                pending.update(acceptance_decision="ERROR", controller_decision="STOP")
            save(pending)
        emit("error", stop_reason=stop_reason, error=error)
    emit("stop", reason=stop_reason, status=status, accepted=accepted, final_text=current)
    return {"sample_id": sample.sample_id, "writing_prompt": sample.writing_prompt,
            "prompt_version": PROMPT_VERSION, "status": status, "stop_reason": stop_reason,
            "original_text": sample.text, "final_text": current, "accepted": accepted,
            "initial_scores": original_state.scores if original_state else None,
            "final_scores": state.scores if state else None, "attempts": attempts, "events": events}
