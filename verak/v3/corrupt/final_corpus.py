"""Whole-essay balance selection, frozen-scoring execution, and final data audits."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import math
import multiprocessing
from pathlib import Path
import random
import sqlite3
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ..common import file_sha, pair_key, read_json, sha_text, write_json
from ..phase2 import read_jsonl, write_jsonl
from .full_judging import CostLedger, load_population, recovered_response
from .instance_pilot import LOCAL_LEVELS, SPLITS, filter_row, usage_cost
from .qc import validate_judgments


def distribution(rows):
    levels = Counter(r["level"] for row in rows for r in row["records"])
    local_n = sum(levels[l] for l in LOCAL_LEVELS)
    return {"essays": len(rows), "source_essays": len({r["source_id"] for r in rows}),
        "records": sum(levels.values()),
        "operators": dict(Counter(r["op"] for row in rows for r in row["records"])),
        "record_levels": dict(levels), "curriculum_levels": dict(Counter(r["level"] for r in rows)),
        "genres": dict(Counter(r["genre"] for r in rows)), "local_records": local_n,
        "local_shares": {level: levels[level]/local_n if local_n else None for level in LOCAL_LEVELS}}


def balanced_subset(rows, *, seed=41):
    """Sample without replacement within local-count patterns using integer counts.

    Lexicographic objective: keep as many non-TEXT-heavy essays as possible,
    then as many total essays as possible. TEXT-heavy means T/(W+S+T) > 1/3.
    Within a pattern, sample using the fixed seed. No text/record edits or copies.
    """
    if not rows or len({r["episode_id"] for r in rows}) != len(rows):
        raise ValueError("Need nonempty, uniquely identified kept essays")
    unique, duplicate_ids = {}, []
    for row in sorted(rows, key=lambda r: r["episode_id"]):
        key = pair_key(row["question"], row["corrupted_text"])
        if key in unique:
            duplicate_ids.append(row["episode_id"])
        else:
            unique[key] = row
    groups = defaultdict(list)
    for row in unique.values():
        counts = Counter(r["level"] for r in row["records"])
        groups[tuple(counts[l] for l in LOCAL_LEVELS)].append(row)
    patterns = sorted(groups)
    features = np.asarray(patterns, dtype=float).T
    totals = features.sum(axis=0)
    capacities = np.array([len(groups[p]) for p in patterns])
    non_text_heavy = (3*features[2] <= totals).astype(float)
    # Integer arithmetic expresses inclusive shares in [28%, 38%].
    coefficients = np.vstack([100*features-28*totals, 38*totals-100*features, totals])
    constraint = LinearConstraint(coefficients, np.array([0]*6+[1]), np.inf)
    bounds = Bounds(np.zeros(len(patterns)), capacities)
    integrality = np.ones(len(patterns))
    first = milp(-non_text_heavy, integrality=integrality, bounds=bounds,
                 constraints=constraint, options={"mip_rel_gap": 0., "time_limit": 60})
    if not first.success:
        raise ValueError("No feasible whole-essay balance; do not duplicate or relabel")
    non_text_optimum = int(np.rint(first.x).dot(non_text_heavy))
    second = milp(-np.ones(len(patterns)), integrality=integrality, bounds=bounds,
        constraints=[constraint, LinearConstraint(non_text_heavy, non_text_optimum, non_text_optimum)],
        options={"mip_rel_gap": 0., "time_limit": 60})
    if not second.success:
        raise ValueError("Could not finish the balance optimization")
    chosen, kept, removed = np.rint(second.x).astype(int), [], []
    rng = random.Random(seed)
    allocation = []
    for pattern, n in zip(patterns, chosen):
        pool = sorted(groups[pattern], key=lambda r: r["episode_id"])
        rng.shuffle(pool)
        kept.extend(pool[:n])
        removed.extend(pool[n:])
        allocation.append({"local_counts": dict(zip(LOCAL_LEVELS, pattern)), "available": len(pool),
                           "kept": int(n), "text_heavy": 3*pattern[2] > sum(pattern)})
    counts = Counter(r["level"] for row in kept for r in row["records"])
    total = sum(counts[l] for l in LOCAL_LEVELS)
    if not total or any(not 28*total <= 100*counts[l] <= 38*total for l in LOCAL_LEVELS):
        raise ValueError("Integer selection violated a share bound")
    kept.sort(key=lambda r: r["episode_id"])
    return kept, {"seed": seed, "share_bounds": [.28, .38], "selection_unit": "whole_essay",
        "method": "integer_pattern_allocation_then_seeded_sampling_without_replacement",
        "objective": ["minimize_non_TEXT_heavy_removals", "minimize_total_removals"],
        "duplicate_ids_removed": duplicate_ids,
        "balance_removed_ids": sorted(r["episode_id"] for r in removed),
        "non_text_heavy_removed": int(sum(len(groups[p]) for p in patterns if 3*p[2] <= sum(p))-non_text_optimum),
        "pattern_allocation": allocation, "before": distribution(rows), "after": distribution(kept)}


def completed_judgments(config):
    output = config["paths"]["phase3b_full_output"]
    rows, pilot_rows, pilot = load_population(config)
    status = read_json(output/"judging_status.json")
    retry_policy = (output/"cost_ledger.json").exists() and read_json(output/"cost_ledger.json").get("schema_version") == 2
    cap = 50 if retry_policy else 45
    if not status["complete"] or status["cost_usd"] > cap or status["errors"]:
        raise ValueError("Full judging must finish within budget before finalizing")
    if retry_policy:
        from .resilient_judging import open_progress
        rows,pilot_rows,pilot,calls,ledger,responses,_ = open_progress(config)
        state = ledger.snapshot()
        if state["cost_usd"] > cap or state["outstanding_nanos"]:
            raise ValueError("Unsettled or over-budget attempts prevent finalization")
    else:
        responses = {r["episode_id"]: r["llm_judgment"] for r in pilot_rows}
        calls = [r for r in read_jsonl(output/"calls.jsonl") if not r.get("event")]
        if len(calls) != 2933 or len({r["sample_id"] for r in calls}) != 2933:
            raise ValueError("Expected exactly one call per remaining candidate")
        for call in calls:
            sid = call["sample_id"]
            if sid in responses:
                raise ValueError("Pilot re-judged or duplicate request")
            responses[sid] = recovered_response(call)
    if set(responses) != {r["episode_id"] for r in rows}:
        raise ValueError("Not every candidate has one judgment")
    judged = []
    for row in rows:
        full = {**row, "llm_judgment": responses[row["episode_id"]], "human_ok": None,
                "judgment_origin": "pilot" if row["episode_id"] in pilot["ids"] else "continuation"}
        judged.append(filter_row(full))
    return judged


def judgment_metrics(rows, calls):
    by = {"operator": defaultdict(list), "record_level": defaultdict(list), "curriculum": defaultdict(list)}
    for row in rows:
        result = {j["record_id"]: j for j in row["llm_judgment"]["judgments"]}
        for rec in row["records"]:
            value = result[rec["record_id"]]
            for key, group in (("operator",rec["op"]),("record_level",rec["level"]),("curriculum",row["level"])):
                by[key][group].append(value)
    yields = {key: {group: {"judged": len(items),
        "kept": sum(v["damage_real"] and v["original_is_fix"] for v in items),
        "yield": sum(v["damage_real"] and v["original_is_fix"] for v in items)/len(items),
        "damage_real": sum(v["damage_real"] for v in items)/len(items),
        "original_is_fix": sum(v["original_is_fix"] for v in items)/len(items)}
        for group, items in sorted(groups.items())} for key, groups in by.items()}
    usage = [usage_cost(c["usage"]) for c in calls]
    total = {key: sum(v[key] for v in usage)
             for key in usage_cost({"input_tokens":0, "output_tokens":0})}
    essays = {}
    for group, members in {"overall": rows, **{s:[r for r in rows if r["split"]==s] for s in SPLITS},
            **{l:[r for r in rows if r["level"]==l] for l in ("L1","L2","L3","L4")}}.items():
        essays[group] = {"judged":len(members), "kept":sum(r["essay_kept"] for r in members),
                         "yield":sum(r["essay_kept"] for r in members)/len(members) if members else None}
    return {"label":"LLM-verified instance filtering", "record_yield":yields,"essay_yield":essays,
            "usage":total,"operator_gating":False,"human_accuracy":None}


def audit_judgments(config):
    """Export a recoverable, explicitly partial snapshot without balancing/scoring.

    Transport failures have no QC verdict. They are neither rejected records nor
    accepted essays, and the audit never promotes an incomplete run to a corpus.
    """
    output = config["paths"]["phase3b_full_output"]
    rows, pilot_rows, pilot = load_population(config)
    lookup = {r["episode_id"]: r for r in rows}
    calls = [r for r in read_jsonl(output/"calls.jsonl") if not r.get("event")]
    ids = [r["sample_id"] for r in calls]
    retry_policy = read_json(output/"cost_ledger.json").get("schema_version") == 2
    if retry_policy:
        from .resilient_judging import open_progress
        _,_,_,_,ledger,responses,history = open_progress(config)
        unresolved_calls = [items[-1] for sid,items in history.items() if sid not in responses]
    else:
        if len(ids) != len(set(ids)) or set(ids) & set(pilot["ids"]) or set(ids)-set(lookup):
            raise ValueError("Duplicate, pilot, or unknown candidate request in continuation")
        ledger = CostLedger(output/"cost_ledger.json", pilot, max_api_calls=3033)
        if set(ledger.snapshot()["entries"]) != set(ids):
            raise ValueError("Unsettled requests need inspection before exporting an audit")
        responses = {r["episode_id"]: r["llm_judgment"] for r in pilot_rows}
        unresolved_calls = []
        for call in calls:
            ledger.settle(call["sample_id"], call)
            if call.get("status") != "completed":
                unresolved_calls.append(call)
            else:
                responses[call["sample_id"]] = recovered_response(call)
    failures = [{"episode_id":call["sample_id"], "split":lookup[call["sample_id"]]["split"],
        "error_type":call.get("error_type"), "http_status":call.get("http_status"),
        "usage_reported":bool(call.get("usage")), "phase_call":call.get("phase_call")}
        for call in unresolved_calls]
    judged = []
    for row in rows:
        if row["episode_id"] in responses:
            validate_judgments(row, responses[row["episode_id"]])
            judged.append(filter_row({**row, "llm_judgment":responses[row["episode_id"]],
                "human_ok":None, "judgment_origin":"pilot" if row["episode_id"] in pilot["ids"] else "continuation"}))
    unattempted = sorted(set(lookup)-set(ids)-set(pilot["ids"]))
    usage_calls = [r for directory in (config["paths"]["phase3b_output"], output)
                   for r in read_jsonl(directory/"calls.jsonl") if r.get("usage")]
    metrics = judgment_metrics(judged, usage_calls)
    metrics.update(scope="completed_judgments_only", population=len(rows), judged=len(judged),
                   failed_requests=len(failures), unattempted=len(unattempted), final=False)
    preserved = read_json(output/"preserved_hashes.json")
    changed = [p for p,h in preserved.items() if file_sha(Path(p)) != h]
    if changed:
        raise ValueError(f"Frozen pilot/analyzer/scorer changed: {changed}")
    snapshot = output/("partial_after_retry" if retry_policy else "partial")
    snapshot.mkdir(exist_ok=True)
    write_jsonl(snapshot/"judged.jsonl", judged)
    split_counts = {}
    for split in SPLITS:
        completed = [r for r in judged if r["split"] == split]
        kept = [r for r in completed if r["essay_kept"]]
        write_jsonl(snapshot/f"kept_{split}.jsonl", kept)
        split_counts[split] = {"population":sum(r["split"] == split for r in rows),
            "judged":len(completed), "failed_requests":sum(r["split"] == split for r in failures),
            "unattempted":sum(lookup[sid]["split"] == split for sid in unattempted),
            "kept_before_balance":distribution(kept), "after_balance":None}
    write_json(snapshot/"metrics.json", metrics)
    write_json(snapshot/"unresolved.json", {"failed":failures, "unattempted_ids":unattempted})
    state = ledger.snapshot()
    result = {"scope":"completed_judgments_only", "final":False, "population":len(rows),
        "judged":len(judged), "pilot_reused":100, "failed_requests":len(failures),
        "unattempted":len(unattempted), "calls":state["calls"],
        "usage_cost_usd":state["confirmed_cost_usd"],
        "unknown_cost_upper_usd":state["unknown_cost_upper_usd"],
        "budget_upper_usd":state["cost_usd"], "outstanding_nanos":state["outstanding_nanos"],
        "budget_cap_usd":50 if retry_policy else 45, "splits":split_counts, "balanced":False, "scored":False,
        "preserved_files":len(preserved), "preserved_changes":[], "phase4_started":False,
        "cost_ledger_sha256":file_sha(output/"cost_ledger.json"),
        "files":{p.name:{"path":str(p), "sha256":file_sha(p)} for p in sorted(snapshot.iterdir())
                 if p.name != "manifest.json"}}
    write_json(snapshot/"manifest.json", result)
    return result


def balance_corpus(config):
    output = config["paths"]["phase3b_full_output"]
    if (output/"balance_manifest.json").exists():
        raise ValueError("Balanced selection is already frozen; do not overwrite after scoring")
    rows = completed_judgments(config)
    write_jsonl(output/"all_judged.jsonl",rows)
    calls = [r for directory in (config["paths"]["phase3b_output"],output)
             for r in read_jsonl(directory/"calls.jsonl") if r.get("usage")]
    write_json(output/"judgment_metrics.json",judgment_metrics(rows,calls))
    decisions = {}
    for split in SPLITS:
        kept = [r for r in rows if r["split"]==split and r["essay_kept"]]
        write_jsonl(output/f"kept_{split}.jsonl",kept)
        chosen, decision = balanced_subset(kept,seed=config["corruption_finalize"]["balance_seed"])
        for row in chosen:
            row["balance_selected"] = True
        path = output/f"balanced_{split}.jsonl"
        write_jsonl(path,chosen)
        decisions[split] = {**decision,"kept_path":str(output/f"kept_{split}.jsonl"),
            "kept_sha256":file_sha(output/f"kept_{split}.jsonl"),
            "balanced_path":str(path),"balanced_sha256":file_sha(path)}
    result = {"splits":decisions,"pilot_reused":100,"judged":3033,"scored":False,
              "selection_uses_q_corrupted":False,"phase4_started":False}
    write_json(output/"balance_manifest.json",result)
    return {s:d["after"] for s,d in decisions.items()}


def exclude_unscorable(config):
    """Retain QC-kept data, but exclude reproducibly malformed scores from final selection.

    Only parseability is considered, never the numerical Q. Preserve the original
    selection and refuse the refinement if either split would lose its balance.
    """
    from ..score.kanana import ScoreParseError, parse_first_line
    output = config["paths"]["phase3b_full_output"]
    final = config["paths"]["metadata"] / "corrupt"
    if (output/"scored_manifest.json").exists() or any((final/f"{s}.jsonl").exists() for s in SPLITS):
        raise ValueError("Cannot refine an already published corpus")
    manifest_path = output/"balance_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("scorer_exclusions"):
        raise ValueError("Scorer eligibility exclusions already applied")
    status = read_json(output/"scoring_status.json")
    errors = {e["episode_id"]:e for e in status["errors"]}
    diagnostics = read_json(output/"score_parse_diagnostics.json")
    if (not errors or status["scored"]+len(errors) != status["total"] or
            {d["episode_id"] for d in diagnostics} != set(errors) or len(diagnostics) != len(errors)):
        raise ValueError("Complete scoring and a repeated diagnosis of every failure are required")
    fingerprints = {read_json(p)["fingerprint"] for p in (output/"scores").glob("*.json")}
    if len(fingerprints) != 1:
        raise ValueError("Need successful scores from one frozen scorer")
    for item in diagnostics:
        sid = item["episode_id"]
        if (errors[sid]["error"] != "ScoreParseError" or item["status"] != "error" or
                item["error_type"] != "ScoreParseError" or item["fingerprint"] not in fingerprints or
                not item["first_lines"] or (output/"scores"/(sid+".json")).exists()):
            raise ValueError("Only repeatedly unparseable, unscored candidates may be excluded")
        for line in item["first_lines"]:
            try:
                parse_first_line(line["text"])
            except ScoreParseError:
                continue
            raise ValueError("A valid score line cannot justify exclusion")
    revised, plans, found = copy.deepcopy(manifest), {}, set()
    for split, decision in revised["splits"].items():
        path = Path(decision["balanced_path"])
        if file_sha(path) != decision["balanced_sha256"]:
            raise ValueError("Original balance selection changed")
        rows = read_jsonl(path)
        excluded = [r["episode_id"] for r in rows if r["episode_id"] in errors]
        kept = [r for r in rows if r["episode_id"] not in errors]
        after = distribution(kept)
        if not kept or any(v is None or not .28 <= v <= .38 for v in after["local_shares"].values()):
            raise ValueError("Scorer exclusions would violate the split balance; stop")
        found.update(excluded)
        new_path = output/f"balanced_scorable_{split}.jsonl"
        if new_path.exists() and read_jsonl(new_path) != kept:
            raise ValueError("Conflicting prior eligible selection")
        plans[split] = (new_path, kept)
        decision.update(pre_score_after=decision["after"], after=after,
            scorer_excluded_ids=excluded, pre_score_balanced_path=str(path),
            pre_score_balanced_sha256=decision["balanced_sha256"],
            pre_score_pattern_allocation=decision.pop("pattern_allocation", []))
    if found != set(errors):
        raise ValueError("A diagnosed failure is not in the original selection")
    backup = output/"balance_manifest_before_scorer_exclusions.json"
    if backup.exists() and file_sha(backup) != file_sha(manifest_path):
        raise ValueError("Original selection backup conflicts")
    if not backup.exists():
        backup.write_bytes(manifest_path.read_bytes())
    for split, (path, rows) in plans.items():
        write_jsonl(path, rows)
        revised["splits"][split].update(balanced_path=str(path), balanced_sha256=file_sha(path))
    revised.update(scorer_exclusions=diagnostics, selection_uses_q_corrupted=False,
        selection_requires_parseable_scores=True, pre_score_manifest=str(backup),
        pre_score_manifest_sha256=file_sha(backup))
    write_json(manifest_path, revised)
    return {"excluded_ids":sorted(found), "final_counts":{s:len(rows) for s,(_,rows) in plans.items()}}


_SCORER = None
_SCORE_DIR = None


def init_scorer(config):
    global _SCORER, _SCORE_DIR
    import torch
    torch.set_num_threads(1)
    from ..score.kanana import KananaScorer
    current = copy.deepcopy(config)
    current["paths"]["output"] = config["paths"]["phase3b_full_output"] / "scorer"
    if current["scorer"]["average_k"] != 1 or current["scorer"]["mode"] != "expected":
        raise ValueError("Phase 1 expected scorer with k=1 is required")
    _SCORER = KananaScorer(current)
    _SCORE_DIR = config["paths"]["phase3b_full_output"] / "scores"
    _SCORE_DIR.mkdir(exist_ok=True)


def score_one(row):
    started=time.monotonic()
    path=_SCORE_DIR/(row["episode_id"]+".json")
    # Persist every successful result before telling the parent. A resumed run
    # reuses exact question/text cache keys, never a score from a different text.
    value=_SCORER.score(row["question"],row["corrupted_text"]).to_dict()
    result={"episode_id":row["episode_id"],"question_hash":row["question_hash"],
            "corrupted_hash":row["corrupted_hash"],"score":value,"fingerprint":_SCORER.fingerprint,
            "seconds":time.monotonic()-started,"average_k":1}
    write_json(path,result)
    return {"episode_id":row["episode_id"],"mean":value["mean"],"seconds":result["seconds"]}


def validate_saved_score(row, result):
    score = result["score"]
    expected = score["expected"]
    if (result["episode_id"] != row["episode_id"] or
            result["question_hash"] != row["question_hash"] or
            result["corrupted_hash"] != row["corrupted_hash"] or
            score["cache_key"] != pair_key(row["question"], row["corrupted_text"]) or
            score["genre"] != row["genre"] or result["average_k"] != 1 or
            not 0 < score["input_tokens"] <= 3072 or len(expected) != 8 or
            not all(math.isfinite(v) and 1 <= v <= 9 for v in expected) or
            not math.isfinite(score["mean"]) or abs(sum(expected)/8-score["mean"]) > 1e-12):
        raise ValueError("Saved score has wrong input or invalid Phase 1 expected-score provenance")


def publish_scored(config, rows, *, workers):
    """Publish only a complete score set; permit an identical interrupted replay."""
    output = config["paths"]["phase3b_full_output"]
    final = config["paths"]["metadata"] / "corrupt"
    selected, fingerprints = {}, set()
    for split in SPLITS:
        selected[split] = []
        for source in rows:
            if source["split"] != split:
                continue
            result = read_json(output/"scores"/(source["episode_id"]+".json"))
            validate_saved_score(source, result)
            fingerprints.add(result["fingerprint"])
            selected[split].append({**source, "q_corrupted":result["score"]["mean"],
                "corrupted_score":result["score"], "scorer_fingerprint":result["fingerprint"],
                "scorer_average_k":1})
        path = final/f"{split}.jsonl"
        if path.exists() and read_jsonl(path) != selected[split]:
            raise ValueError("Refusing to overwrite a different prior final corpus")
    if len(fingerprints) != 1 or any(not selected[s] for s in SPLITS):
        raise ValueError("Both splits and one scorer fingerprint are required")
    final.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = final/f"{split}.jsonl"
        if not path.exists():
            write_jsonl(path, selected[split])
    write_json(output/"scored_manifest.json",{"paths":{s:str(final/f"{s}.jsonl") for s in SPLITS},
        "sha256":{s:file_sha(final/f"{s}.jsonl") for s in SPLITS},"scored":len(rows),
        "mode":"phase1_score_only_expected","average_k":1,"gpu":config["scorer"]["gpu"],
        "scorer_processes":workers,"phase4_started":False})
    return {"scored":len(rows),"final_directory":str(final)}


def score_corpus(config, *, workers=1):
    if workers not in (1,2):
        raise ValueError("Only one or two independent scorer processes on the configured single GPU")
    if config["scorer"]["average_k"] != 1 or config["scorer"]["mode"] != "expected":
        raise ValueError("Phase 1 expected scorer with k=1 is required")
    output=config["paths"]["phase3b_full_output"]
    manifest=read_json(output/"balance_manifest.json")
    rows=[]
    for split,d in manifest["splits"].items():
        path=Path(d["balanced_path"])
        if file_sha(path)!=d["balanced_sha256"]:
            raise ValueError("Balance selection changed before scoring")
        rows.extend(read_jsonl(path))
    cache=output/"scorer"/"score_cache.sqlite"
    cache.parent.mkdir(exist_ok=True)
    old=config["paths"]["phase3_output"] / "score_cache.sqlite"
    if not cache.exists() and old.exists():
        with sqlite3.connect(f"file:{old}?mode=ro",uri=True) as source,sqlite3.connect(cache) as target:
            source.backup(target)
    pending=[]
    for row in rows:
        path=output/"scores"/(row["episode_id"]+".json")
        if path.exists():
            result=read_json(path)
            validate_saved_score(row, result)
        else:
            pending.append(row)
    done=len(rows)-len(pending)
    started=time.monotonic()
    errors=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_scorer,initargs=(config,)) as pool:
        futures={pool.submit(score_one,row):row["episode_id"] for row in pending}
        for future in as_completed(futures):
            try:
                future.result()
                done+=1
            except Exception as error:
                errors.append({"episode_id":futures[future],"error":type(error).__name__,"message":str(error)[:240]})
            if done%20==0 or errors or done==len(rows):
                progress={"scored":done,"total":len(rows),"errors":errors,"seconds":time.monotonic()-started,
                          "workers":workers,"gpu":config["scorer"]["gpu"],"average_k":1}
                write_json(output/"scoring_status.json",progress)
                print(f"Scored {done}/{len(rows)}; errors={len(errors)}; elapsed={progress['seconds']:.1f}s",flush=True)
    if errors:
        raise ValueError("Scoring errors preserved; final corpus not published")
    # A resume may find all scores already saved. Refresh the status even when
    # no worker had to run; otherwise the previous failed-run status stays stale.
    write_json(output/"scoring_status.json", {"scored":done,"total":len(rows),"errors":[],
        "seconds":time.monotonic()-started,"workers":workers,"gpu":config["scorer"]["gpu"],
        "average_k":1,"reused_saved_scores":len(rows)-len(pending),"newly_scored":len(pending)})
    return publish_scored(config, rows, workers=workers)


def verify_corpus(config):
    output=config["paths"]["phase3b_full_output"]
    judged=completed_judgments(config)
    accepted={r["episode_id"]:r for r in judged if r["essay_kept"]}
    balance=read_json(output/"balance_manifest.json")
    scored=read_json(output/"scored_manifest.json")
    deny=set(read_json(config["paths"]["metadata"] / "audit_index.json")["train"]["essay_hashes"])
    questions,distributions,fingerprints={}, {},set()
    for split in SPLITS:
        path=Path(scored["paths"][split])
        if file_sha(path)!=scored["sha256"][split]: raise ValueError("Final corpus hash changed")
        rows=read_jsonl(path)
        selected=read_jsonl(Path(balance["splits"][split]["balanced_path"]))
        if [r["episode_id"] for r in rows]!=[r["episode_id"] for r in selected]:
            raise ValueError("Final data differ from the pre-score balance selection")
        seen=set()
        for row in rows:
            source=accepted[row["episode_id"]]
            key=pair_key(row["question"],row["corrupted_text"])
            if key in seen: raise ValueError("Duplicate final essay")
            seen.add(key)
            for field in ("question","source_text","corrupted_text","records","llm_judgment"):
                if row[field]!=source[field]: raise ValueError("Kept essay changed during balancing/scoring")
            if any(sha_text(row[field]) in deny for field in ("source_text","corrupted_text")):
                raise ValueError("Exact training essay in final data")
            score=row["corrupted_score"]
            if (score["cache_key"]!=key or len(score["expected"])!=8 or
                    abs(sum(score["expected"])/8-row["q_corrupted"])>1e-12 or
                    score["genre"]!=row["genre"] or score["input_tokens"]>3072 or row["scorer_average_k"]!=1):
                raise ValueError("Invalid Phase 1 score provenance")
            if not all(1<=v<=9 for v in score["expected"]): raise ValueError("Expected score outside rubric")
            fingerprints.add(row["scorer_fingerprint"])
        d=distribution(rows)
        if any(not .28<=v<=.38 for v in d["local_shares"].values()): raise ValueError("Final share bounds violated")
        distributions[split]=d
        questions[split]={r["question_hash"] for r in rows}
    if questions[SPLITS[0]] & questions[SPLITS[1]]: raise ValueError("Question split leakage")
    if len(fingerprints)!=1: raise ValueError("Scorer fingerprints disagree")
    preserved=read_json(output/"preserved_hashes.json")
    changed=[p for p,h in preserved.items() if file_sha(Path(p))!=h]
    if changed: raise ValueError(f"Frozen pilot/analyzer/scorer changed: {changed}")
    result={"final":distributions,"preserved_files":len(preserved),"preserved_changes":[],
            "question_overlap":0,"train_exact_matches":0,"pilot_reused":100,"total_judged":3033,
            "scorer_fingerprint":next(iter(fingerprints)),"phase4_started":False}
    write_json(output/"final_validation.json",result)
    return result
