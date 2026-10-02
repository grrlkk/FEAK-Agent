"""Scoped revision loop, with the original P1 comparison available via --mode single."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from dotenv import load_dotenv
import yaml

from .analyzer import Analyzer, BareunBackend
from .change_info import extract, surface_diff, changed_korean_info
from .diagnoser import Diagnoser, LocalKanana, set_goal, plan_scoped
from .judge import judge_pair, judge_scoped, LOOP_REQUIREMENTS
from .llm import LLM
from .reviser import revise, revise_scoped, hard_checks
from .schemas import CONDITIONS, Goal, Unit
from .spellcheck import SpellChecker, attach, compare
from feak_tc.runtime.openai import CallBudgetExceeded

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def read_sample(row, source_line):
    if all(key in row for key in ("question", "draft")):
        question, draft = row["question"], row["draft"]
    elif isinstance(row.get("user"), str):
        user = row["user"]
        if not user.startswith("질문:") or "\n에세이:" not in user:
            raise ValueError(f"Input line {source_line}: expected 질문:/에세이: fields")
        question, draft = user[len("질문:"):].split("\n에세이:", 1)
        question, draft = question.removeprefix(" "), draft.removeprefix(" ")
        if "\n핵심 키워드:" in draft:
            draft, _ = draft.rsplit("\n핵심 키워드:", 1)
    else:
        raise ValueError(f"Input line {source_line}: expected question/draft or training-format user")
    if not all(isinstance(text, str) and text.strip() for text in (question, draft)):
        raise ValueError("Question and draft must be nonblank strings")
    digest = hashlib.sha256(json.dumps([question, draft], ensure_ascii=False).encode()).hexdigest()
    # Gold answers, human scores and keyword annotations never enter the pipeline.
    return {"id": str(row.get("id", digest[:16])), "question": question, "draft": draft,
            "source_line": source_line, "original_group_sha256": digest,
            "constraints": row.get("constraints", [])}


def load_samples(path, limit):
    samples, seen = [], set()
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            sample = read_sample(json.loads(line), line_number)
            if sample["id"] in seen:
                raise ValueError("Duplicate input id")
            seen.add(sample["id"])
            samples.append(sample)
            if len(samples) == limit:
                break
    if not samples:
        raise ValueError("No input samples")
    return samples


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if config.get("mode", "loop") not in {"loop", "single"}:
        raise ValueError("mode must be loop or single")
    if tuple(config["judge_info"]) != CONDITIONS:
        raise ValueError("P1 runs exactly criteria_only, surface_diff, korean")
    if config["analyzer"]["name"] != "bareun" or config.get("keywords") is not False:
        raise ValueError("User selected Bareun and no keyword input")
    if config["context_window"] != 2:
        raise ValueError("P1 context_window must be two sentences")
    same = config["models"]["reviser"]["provider"] == config["models"]["judge"]["provider"]
    if same and not config["models"].get("same_family_exception"):
        raise ValueError("Same-family models require a recorded user exception")
    re.compile(config["anonymization_pattern"])
    for env in config.get("env_files", []):
        load_dotenv(path.parent / env, override=False)
    cache = Path(config["spellcheck"]["cache"])
    config["spellcheck"]["cache"] = str((path.parent / cache).resolve())
    loop = config.setdefault("loop", {"max_steps": 5, "allow_document_rewrite": False})
    if type(loop.get("max_steps")) is not int or not 1 <= loop["max_steps"] <= 20:
        raise ValueError("loop.max_steps must be an integer from 1 to 20")
    if type(loop.get("allow_document_rewrite")) is not bool:
        raise ValueError("loop.allow_document_rewrite must be a boolean")
    return config


def process_sample(sample, *, analyzer, diagnoser, generator, judge_llm, spellchecker, config,
                   prompts, on_pair, on_event=None):
    event = on_event or (lambda row: None)
    started = time.monotonic()
    question, before = sample["question"], sample["draft"]
    event({"stage": "analyze_before"})
    before_profile = analyzer.profile(before)
    event({"stage": "diagnose"})
    diagnosis = diagnoser.diagnose(question, before)
    event({"stage": "diagnosis", "diagnosis": diagnosis})
    goal, reason = set_goal(question, before, diagnosis, before_profile, llm=generator, prompt=prompts["goal"])
    base = {"id": sample["id"], "input": sample, "diagnosis": diagnosis, "goal_reason": reason,
            "original_text": before, "returned_text": before}
    if goal is None:
        return {**base, "status": "no_edit", "goal": None, "judgments": {},
                "elapsed_s": time.monotonic() - started}
    event({"stage": "goal", "goal": goal.model_dump()})
    after = revise(question, before, goal, before_profile, llm=generator, prompt=prompts["revise"],
                   anonymization_pattern=config["anonymization_pattern"])
    event({"stage": "candidate", "before": before, "after": after, "goal": goal.model_dump()})
    after_profile = analyzer.profile(after)
    units = extract(before, after, before_profile, after_profile, analyzer.lexicons["focus"])
    before_spell, after_spell = spellchecker.report(before), spellchecker.report(after)
    spelling = compare(before, after, before_spell, after_spell)
    attach(units, spelling)
    pair = {"id": sample["id"], "question": question, "before": before, "after": after,
            "goal": goal.model_dump(), "units": [u.to_dict() for u in units], "spelling": spelling,
            "spellcheck_before": before_spell, "spellcheck_after": after_spell,
            "before_profile": before_profile.to_dict(), "after_profile": after_profile.to_dict(),
            "constraints": sample.get("constraints", []),
            "original_group_sha256": sample.get("original_group_sha256")}
    on_pair(pair)  # The pair is durable before the first judgment and never regenerated per condition.
    event({"stage": "judge"})
    judgments = judge_saved_pair(pair, llm=judge_llm, config=config, prompts=prompts)
    return {**base, "status": "completed", "goal": goal.model_dump(), "candidate": after,
            "units": pair["units"], "spelling": spelling, "judgments": judgments,
            "elapsed_s": time.monotonic() - started}


def judge_saved_pair(pair, *, llm, config, prompts):
    return judge_pair(pair["id"], pair["question"], pair["before"], pair["after"],
        Goal.model_validate(pair["goal"]), [Unit(**unit) for unit in pair["units"]], pair["spelling"],
        llm=llm, unit_prompt=prompts["j_units"], global_prompt=prompts["j_global"],
        seed=config["seed"], conditions=config["judge_info"],
        anonymization_pattern=config["anonymization_pattern"], constraints=pair.get("constraints", []))


def process_loop(sample, *, analyzer, diagnoser, generator, judge_llm, config, prompts,
                 on_step=None, on_event=None):
    """Commit only RV-approved edits; all failures retain the current accepted text."""
    event, save = on_event or (lambda row: None), on_step or (lambda row: None)
    original, question = sample["draft"], sample["question"]
    current, diagnosis, profile = original, None, None
    trajectory, history, seen_plans = [], [], set()
    seen_texts, score_cache = {original}, {}
    stop_reason, status = "max_steps", "completed"
    settings = config["loop"]
    allow_rewrite = settings["allow_document_rewrite"]

    def finish_step(row):
        row["current_text_after"] = current
        row["kanana_after"] = diagnosis
        row["scores_after"] = diagnosis["scores"] if diagnosis else None
        trajectory.append(row)
        save({"sample_id": sample["id"], **row})
        history.append({key: row[key] for key in
                        ("planner", "candidate_text", "rv", "decision", "errors")})

    try:
        event({"stage": "analyze_current", "step": 0, "current_text": current})
        profile = analyzer.profile(current)
        diagnosis = diagnoser.diagnose(question, current)
        score_cache[current] = diagnosis
        event({"stage": "initial_state", "step": 0, "current_text": current, "kanana": diagnosis})
    except Exception as exc:
        event({"stage": "initial_error", "step": 0, "error_type": type(exc).__name__})
        return {"id": sample["id"], "input": sample, "mode": "loop", "status": "error",
                "original_text": original, "returned_text": current, "trajectory": [],
                "stop_reason": "initial_analysis_error", "error_type": type(exc).__name__}

    for number in range(1, settings["max_steps"] + 1):
        row = {"step": number, "question": question, "current_text": current, "planner": None, "candidate_text": None,
               "edit": None, "diff": [], "korean_changes": None, "rv": None,
               "kanana_before": diagnosis, "kanana_candidate": None,
               "scores_before": diagnosis["scores"], "scores_candidate": None,
               "decision": "REJECT", "errors": []}
        event({"stage": "step_start", "step": number, "current_text": current})
        active_stage = "planner"
        try:
            plan, reason = plan_scoped(question, current, diagnosis, profile, history,
                llm=generator, prompt=prompts["plan_loop"], allow_document_rewrite=allow_rewrite)
            row["planner"] = {"plan": plan.model_dump() if plan else None, "reason": reason}
            event({"stage": "planner", "step": number, **row["planner"]})
            if plan is None:
                row["decision"], stop_reason = "STOP", "no_clear_revision"
                finish_step(row)
                break
            # A changed Preserve/evidence contract can make a retry a different plan.
            # Rephrasing only the smallest-scope explanation is not a new attempt.
            signature = (current, plan.model_dump_json(exclude={"minimal_scope_reason"}))
            if signature in seen_plans:
                row["decision"], stop_reason = "STOP", "repeated_plan"
                finish_step(row)
                break
            seen_plans.add(signature)
            active_stage = "reviser"
            candidate, edit = revise_scoped(question, current, plan, llm=generator,
                prompt=prompts["revise_loop"], allow_document_rewrite=allow_rewrite,
                anonymization_pattern=config["anonymization_pattern"])
            diff = surface_diff(current, candidate)
            row.update(candidate_text=candidate, edit=edit, diff=diff)
            event({"stage": "candidate", "step": number, "current_text": current,
                   "planner": row["planner"], "candidate_text": candidate, "edit": edit, "diff": diff})
            hard_ok, reasons = hard_checks(current, candidate, plan,
                config["anonymization_pattern"], sample.get("constraints", []))
            if candidate != current and candidate in seen_texts:
                hard_ok, reasons = False, [*reasons, "previously_accepted_state"]
            candidate_profile, korean_changes = None, {}
            if hard_ok:
                active_stage = "barun_candidate"
                candidate_profile = analyzer.profile(candidate)
                korean_changes = changed_korean_info(profile, candidate_profile, diff)
            row["korean_changes"] = korean_changes
            active_stage = "rv"
            rv = judge_scoped(question, current, candidate, plan, diff, korean_changes,
                llm=judge_llm, prompt=prompts["rv_loop"], hard_ok=hard_ok, hard_reasons=reasons)
            row["rv"] = rv
            # Adoption is determined BEFORE post-revision scoring. Scores cannot override the RV.
            if rv["accept"]:
                current, profile, diagnosis = candidate, candidate_profile, None
                seen_texts.add(current)
                row["decision"] = "ACCEPT"
            event({"stage": "decision", "step": number, "decision": row["decision"],
                   "rv": rv, "current_text_after": current})
            # Score rejected candidates too, for trajectory analysis, but never replace state scores.
            active_stage = "kanana_candidate"
            if candidate.strip():
                row["candidate_score_cached"] = candidate in score_cache
                if candidate not in score_cache:
                    score_cache[candidate] = diagnoser.diagnose(question, candidate)
                row["kanana_candidate"] = score_cache[candidate]
                row["scores_candidate"] = score_cache[candidate]["scores"]
            if row["decision"] == "ACCEPT":
                diagnosis = row["kanana_candidate"]
            finish_step(row)
        except CallBudgetExceeded:
            row["errors"].append({"stage": "budget", "error_type": "CallBudgetExceeded"})
            if row["decision"] != "ACCEPT":
                row["decision"] = "STOP"
            stop_reason = "call_budget"
            finish_step(row)
            break
        except Exception as exc:
            # Keep successful earlier adoptions, including one whose follow-up scoring failed.
            row["errors"].append({"stage": active_stage, "error_type": type(exc).__name__, "message": str(exc)})
            event({"stage": "step_error", "failed_stage": active_stage, "step": number,
                   "error_type": type(exc).__name__})
            finish_step(row)
            if row["decision"] == "ACCEPT":
                stop_reason, status = "accepted_state_scoring_error", "error"
                break
            # A bad plan, invalid patch or failed analysis becomes evidence for a different plan.
            continue
    return {"id": sample["id"], "input": sample, "mode": "loop", "status": status,
            "original_text": original, "returned_text": current, "trajectory": trajectory,
            "initial_diagnosis": score_cache[original], "final_diagnosis": diagnosis,
            "accepted_steps": sum(step["decision"] == "ACCEPT" for step in trajectory),
            "rejected_steps": sum(step["decision"] == "REJECT" for step in trajectory),
            "stop_reason": stop_reason}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Defaults to the confirmed validation file")
    source.add_argument("--pairs", type=Path, help="Judge saved fixed pairs without generating new candidates")
    parser.add_argument("--limit", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnose-only", action="store_true", help="T6 raw scorer/feedback integration check")
    parser.add_argument("--mode", choices=("loop", "single"), help="Default: scoped loop; single replays P1")
    parser.add_argument("--max-steps", type=int, choices=range(1, 21), help="Maximum loop attempts per essay")
    args = parser.parse_args(argv)
    if args.pairs and args.diagnose_only:
        parser.error("--pairs and --diagnose-only cannot be combined")
    config = load_config(args.config)
    if args.pairs and args.mode == "loop":
        parser.error("--pairs replays P1 fixed pairs; use --mode single or omit --mode")
    mode = "single" if args.pairs else args.mode or config.get("mode", "loop")
    if args.max_steps is not None:
        if mode != "loop" or args.diagnose_only:
            parser.error("--max-steps requires loop mode")
        config["loop"]["max_steps"] = args.max_steps
    if args.output_dir.exists():
        parser.error("Output directory already exists; use a new directory")
    if args.pairs:
        with args.pairs.open() as handle:
            samples = [json.loads(line) for line in handle if line.strip()][:args.limit]
        if not samples or len({s["id"] for s in samples}) != len(samples):
            parser.error("Fixed-pair input must be nonempty with unique ids")
    else:
        samples = load_samples(args.input or config["paths"]["val"], args.limit)
    names = ("plan_loop", "revise_loop", "rv_loop") if mode == "loop" else ("goal", "revise", "j_units", "j_global")
    prompts = {name: (ROOT / "prompts" / f"{name}.txt").read_text() for name in names}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pair_path = ROOT / "data" / "pairs" / f"{args.output_dir.name}-{time.time_ns()}.jsonl"
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    source_files = [*ROOT.glob("src/*.py"), *ROOT.glob("prompts/*.txt"), *ROOT.glob("lexicons/*.yaml")]
    manifest = {"config": config, "created_at": datetime.now(timezone.utc).isoformat(),
                "mode": "judge_fixed_pairs" if args.pairs else "diagnose_only" if args.diagnose_only else mode,
                "input": str(args.pairs or args.input or config["paths"]["val"]), "limit": args.limit,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                "pair_path": str(pair_path), "prompts": prompts}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    records, results, current_id, current_step = [], [], None, None
    with (args.output_dir / "calls.jsonl").open("x") as calls, (args.output_dir / "results.jsonl").open("x") as output, \
         (args.output_dir / "events.jsonl").open("x") as events, pair_path.open("x") as pairs, \
         (args.output_dir / "trajectory.jsonl").open("x") as trajectories:
        def record(row):
            row = {**row, "sample_id": current_id, "step": current_step}
            records.append(row)
            write_jsonl(calls, row)
        def event(row):
            nonlocal current_step
            current_step = row.get("step", current_step)
            write_jsonl(events, {"sample_id": current_id, **row})
            print(f"[{current_id}] {row['stage']}", flush=True)
        generator = LLM(config["models"]["reviser"], config["request"], record)
        judge_llm = (generator if config["models"]["reviser"] == config["models"]["judge"]
                     else LLM(config["models"]["judge"], config["request"], record))
        backend_args = {key: value for key, value in config["analyzer"].items() if key != "name"}
        analyzer = None if args.pairs or args.diagnose_only else Analyzer(BareunBackend(**backend_args))
        diagnoser = Diagnoser(LocalKanana(config["models"]["scorer"]), record)
        spellchecker = SpellChecker(config["spellcheck"], on_record=record)
        try:
            for sample in samples:
                current_id = sample["id"]
                current_step = None
                generator.start_sample(current_id)
                if judge_llm is not generator: judge_llm.start_sample(current_id)
                count_before, start = len(records), time.monotonic()
                try:
                    if args.pairs:
                        write_jsonl(pairs, sample)
                        result = {"id": current_id, "status": "completed", "original_text": sample["before"],
                                  "returned_text": sample["before"], "candidate": sample["after"],
                                  "judgments": judge_saved_pair(sample, llm=judge_llm, config=config, prompts=prompts)}
                    elif args.diagnose_only:
                        event({"stage": "diagnose"})
                        result = {"id": current_id, "status": "completed", "input": sample,
                                  "diagnosis": diagnoser.diagnose(sample["question"], sample["draft"])}
                    elif mode == "loop":
                        result = process_loop(sample, analyzer=analyzer, diagnoser=diagnoser,
                            generator=generator, judge_llm=judge_llm, config=config, prompts=prompts,
                            on_step=lambda row: write_jsonl(trajectories, row), on_event=event)
                    else:
                        result = process_sample(sample, analyzer=analyzer, diagnoser=diagnoser,
                            generator=generator, judge_llm=judge_llm, spellchecker=spellchecker,
                            config=config, prompts=prompts, on_pair=lambda pair: write_jsonl(pairs, pair), on_event=event)
                except Exception as exc:
                    result = {"id": current_id, "status": "error", "error_type": type(exc).__name__,
                              "input": sample, "returned_text": sample.get("draft", sample.get("before"))}
                    event({"stage": "error", "error_type": type(exc).__name__})
                sample_records = records[count_before:]
                result["usage"] = summarize_calls(sample_records)
                result["elapsed_s"] = time.monotonic() - start
                write_jsonl(output, result)
                results.append(result)
                print(f"[{current_id}] {result['status']}", flush=True)
                if result["status"] == "error":
                    break
        finally:
            generator.close()
            if judge_llm is not generator: judge_llm.close()
    report = {"samples": len(results), "completed": sum(r["status"] == "completed" for r in results),
              "no_edit": sum(r["status"] == "no_edit" for r in results),
              "errors": sum(r["status"] == "error" for r in results), "usage": summarize_calls(records),
              "accept_by_condition": {c: sum(r.get("judgments", {}).get(c, {}).get("accept", False) for r in results)
                                      for c in CONDITIONS}, "pair_path": str(pair_path),
              "note": "Execution and model judgments only; no human gold or demonstrated effect."}
    if mode == "loop" and not args.diagnose_only:
        report.pop("accept_by_condition")
        report.update(mode="loop", accepted_steps=sum(r.get("accepted_steps", 0) for r in results),
                      rejected_steps=sum(r.get("rejected_steps", 0) for r in results),
                      step_errors=sum(len(s["errors"]) for r in results for s in r.get("trajectory", [])),
                      rv_errors=sum(len((s.get("rv") or {}).get("errors", []))
                                    for r in results for s in r.get("trajectory", [])),
                      stop_reasons={r["id"]: r.get("stop_reason") for r in results},
                      trajectory_path=str(args.output_dir / "trajectory.jsonl"))
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.output_dir / "summary.md").write_text(render_summary(results, report))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 1 if report["errors"] else 0


def render_summary(results, report):
    if report.get("mode") == "loop":
        return render_loop_summary(results, report)
    lines = ["# VERAK P1 실행 기록", "", "후보는 고정되어 있고 원문에 채택하지 않았다. 판정은 모델의 출력이며 사람 정답이나 품질 향상의 증거가 아니다.",
             "", f"고정 쌍: {report['pair_path']}", ""]
    for row in results:
        lines += [f"## {row['id']} — {row['status']}", ""]
        if row.get("goal"):
            lines += [f"목표: {row['goal']['rubric']} / {row['goal']['intent']}", ""]
        for title, key in (("원문", "original_text"), ("후보", "candidate")):
            if key in row:
                lines += [f"### {title}", "", *("    " + s for s in row[key].splitlines()), ""]
        judgments = row.get("judgments", {})
        if judgments:
            lines += ["| 조건 | goal_valid | goal_improved | selective | meaning | global | 채택 계산 |",
                      "|---|---|---|---|---|---|---|"]
            for condition in CONDITIONS:
                j = judgments[condition]
                values = [condition, *(j[k] for k in ("goal_valid", "goal_improved", "selective", "meaning", "global")),
                          str(j["accept"])]
                lines.append("| " + " | ".join(values) + " |")
            lines.append("")
    return "\n".join(lines)


def render_loop_summary(results, report):
    lines = ["# VERAK 범위 제한 수정 루프", "", "모델 판정·실행 기록이며 사람 평가나 성능 입증이 아니다.",
             "", f"단계별 기록: {report['trajectory_path']}", ""]
    for result in results:
        lines += [f"## {result['id']} — {result['status']}", "",
                  f"문항: {result['input']['question']}", "", f"종료: {result.get('stop_reason')}", ""]
        for title, key in (("최초 원문", "original_text"), ("최종 채택 글", "returned_text")):
            lines += [f"### {title}", "", *("    " + s for s in result[key].splitlines()), ""]
        for step in result.get("trajectory", []):
            lines += [f"### Step {step['step']} — {step['decision']}", ""]
            planner = step.get("planner") or {}
            plan = planner.get("plan")
            if plan:
                lines += [f"목표: {plan['rubric']} / {plan['goal']}",
                          f"범위: {plan['scope']} / 행동: {plan['action']}",
                          f"최소 범위 근거: {plan['minimal_scope_reason']}", ""]
            else:
                lines += [planner.get("reason", "계획 생성 실패"), ""]
            for title, key in (("수정 전", "current_text"), ("수정 후보", "candidate_text")):
                if step.get(key) is not None:
                    lines += [f"{title}:", "", *("    " + s for s in step[key].splitlines()), ""]
            for edit in step["diff"]:
                lines += [f"변경 {edit['operation']} {edit['before_span']} → {edit['after_span']}",
                          f"    전: {edit['before_text']}", f"    후: {edit['after_text']}", ""]
            rv = step.get("rv")
            if rv:
                lines += ["판정: " + ", ".join(f"{key}={rv[key]}" for key in LOOP_REQUIREMENTS), ""]
                lines += [f"- {issue['requirement']}: {issue['reason']}" for issue in rv["issues"]]
                lines += [f"- 기계 제약: {reason}" for reason in rv["hard_reasons"]]
            lines += ["", f"수정 전 점수: {step['scores_before']}",
                      f"후보 점수: {step['scores_candidate']}", f"채택 상태 점수: {step['scores_after']}", ""]
            if step["errors"]:
                lines += [f"오류: {step['errors']}", ""]
    return "\n".join(lines)


def summarize_calls(records):
    actual = [r for r in records if not r.get("cached") and not r.get("event")]
    return {"llm_api_calls": sum(r.get("provider") == "openai" for r in actual),
            "kanana_generations": sum(r.get("role") == "diagnose" for r in actual),
            "spellcheck_attempts": sum(r.get("stage") == "spellcheck" for r in actual),
            "llm_tokens": {key: sum((r.get("usage") or {}).get(key, 0) for r in actual if r.get("provider") == "openai")
                           for key in ("input_tokens", "output_tokens", "total_tokens")},
            "kanana_tokens": {key: sum((r.get("usage") or {}).get(key, 0) for r in actual if r.get("role") == "diagnose")
                              for key in ("input_tokens", "output_tokens", "total_tokens")}}


if __name__ == "__main__":
    raise SystemExit(main())
