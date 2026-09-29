"""CLI for P1: freeze one candidate, compare three judgments, retain the original."""

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
from .change_info import extract
from .diagnoser import Diagnoser, LocalKanana, set_goal
from .judge import judge_pair
from .llm import LLM
from .reviser import revise
from .schemas import CONDITIONS, Goal, Unit
from .spellcheck import SpellChecker, attach, compare

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Defaults to the confirmed validation file")
    source.add_argument("--pairs", type=Path, help="Judge saved fixed pairs without generating new candidates")
    parser.add_argument("--limit", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnose-only", action="store_true", help="T6 raw scorer/feedback integration check")
    args = parser.parse_args(argv)
    if args.pairs and args.diagnose_only:
        parser.error("--pairs and --diagnose-only cannot be combined")
    config = load_config(args.config)
    if args.output_dir.exists():
        parser.error("Output directory already exists; use a new directory")
    if args.pairs:
        with args.pairs.open() as handle:
            samples = [json.loads(line) for line in handle if line.strip()][:args.limit]
        if not samples or len({s["id"] for s in samples}) != len(samples):
            parser.error("Fixed-pair input must be nonempty with unique ids")
    else:
        samples = load_samples(args.input or config["paths"]["val"], args.limit)
    prompts = {name: (ROOT / "prompts" / f"{name}.txt").read_text() for name in ("goal", "revise", "j_units", "j_global")}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pair_path = ROOT / "data" / "pairs" / f"{args.output_dir.name}-{time.time_ns()}.jsonl"
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    source_files = [*ROOT.glob("src/*.py"), *ROOT.glob("prompts/*.txt"), *ROOT.glob("lexicons/*.yaml")]
    manifest = {"config": config, "created_at": datetime.now(timezone.utc).isoformat(),
                "mode": "judge_fixed_pairs" if args.pairs else "diagnose_only" if args.diagnose_only else "single",
                "input": str(args.pairs or args.input or config["paths"]["val"]), "limit": args.limit,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                "pair_path": str(pair_path), "prompts": prompts}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    records, results, current_id = [], [], None
    with (args.output_dir / "calls.jsonl").open("x") as calls, (args.output_dir / "results.jsonl").open("x") as output, \
         (args.output_dir / "events.jsonl").open("x") as events, pair_path.open("x") as pairs:
        def record(row):
            row = {**row, "sample_id": current_id}
            records.append(row)
            write_jsonl(calls, row)
        def event(row):
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
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.output_dir / "summary.md").write_text(render_summary(results, report))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 1 if report["errors"] else 0


def render_summary(results, report):
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
