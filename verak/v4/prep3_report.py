"""Read-only PREP3 summary. Does not dispatch API calls or start training."""
import json
from pathlib import Path

from .prep3_common import ROOT, REPO, PREP1, PREP2, GENRES, read_json, write_json, accounting, freeze_contract


def pct(value):
    return '—' if value is None else f'{100*value:.2f}%'


def load(path):
    return read_json(path) if path.exists() else None


def report():
    freeze_contract()
    lines=(ROOT/'freeze_report.md').read_text().split('## Remaining work')[0].rstrip().splitlines()
    lines+=['','New v4 teacher dispatches additionally reject noncanonical editor prompts; exact completed historical caches remain readable. '
        'The v1 RFT runtime and data are unchanged. No training or inference model was loaded for PREP3.','',
        '## 1. Edge-level map validation and fixed-cohort extraction','']
    gate=load(ROOT/'maps/gate.json')
    if gate:
        lines += [f'Saved-map gate: **{gate["valid_maps"]}/{gate["extractions"]} ({pct(gate["map_validity"])})** structurally valid maps; '
            f'**{gate["dropped_edges"]}/{gate["raw_edges"]} ({pct(gate["dropped_fraction"])})** edges dropped. '
            f'The ≥95% / ≤15% gate {"passed" if gate["passed"] else "failed"}. No new API calls were used for this decision.','',
            'An edge is checked before consuming its outgoing slot. A sentence can retain one support and one example independently; '
            'first valid edge in provider order wins within each type. Other relation types have no added outgoing cap. '
            'Above six paragraphs, cross-paragraph sentence edges need main/key endpoints. Invalid endpoints, directions, duplicates and excess edges '
            'are dropped individually. Metadata/JSON errors remain map errors. A structurally valid map is not a judgment of semantic correctness.','',
            '|Saved cohort|Valid maps|Valid pairs|Raw edges|Dropped|Empty maps|',
            '|---|---:|---:|---:|---:|---:|']
        reasons={}
        for name,m in gate['cohorts'].items():
            lines.append(f'|{name}|{m["valid_maps"]}/{m["extractions"]}|{m["valid_pairs"]}/{m["pairs"]}|{m["raw_edges"]}|{m["dropped_edges"]}|{m["empty_maps"]}|')
            for reason,n in m['dropped_by_reason'].items(): reasons[reason]=reasons.get(reason,0)+n
        lines+=['','|Edge drop reason|Count|','|---|---:|']+[f'|{k}|{v}|' for k,v in sorted(reasons.items())]
        lines+=['','All 300 pilot and 86 retry extractions are retained as separate saved observations. '
            'The 43 retry essays also belong to the pilot cohort; the gate counts extraction outputs, not 193 distinct essays. '
            'Both-run intersections and off_topic protection are recomputed after filtering. off_topic remains advisory and cannot justify deletion alone.','']
    scale=load(ROOT/'maps/scale2000/metrics.json')
    if scale:
        lines += [f'Frozen scale cohort: 2,000 source essays (667 argumentative, 667 explanatory, 666 emotional), '
            f'two requests each. Valid maps: {scale["valid_maps"]}/{scale["extractions"]}; usable two-run intersections: '
            f'{scale["valid_pairs"]}/{scale["pairs"]}. Edge drops: {scale["dropped_edges"]}/{scale["raw_edges"]} '
            f'({pct(scale["dropped_fraction"])}). Final intersected edges: {scale["final_edges"]}.','',
            '|Genre|Valid extractions|Usable pairs|','|---|---:|---:|']
        for g,m in scale['by_genre'].items(): lines.append(f'|{g}|{m.get("valid_maps",0)}/{m["extractions"]}|{m.get("valid_pairs",0)}/{m["sources"]}|')
        lines+=['','|Scale edge drop reason|Count|','|---|---:|']+[f'|{k}|{v}|' for k,v in sorted(scale['dropped_by_reason'].items())]
        lines+=['',f'Post-intersection hierarchy drops: `{json.dumps(scale["post_intersection_dropped_by_reason"],ensure_ascii=False)}`. '
            f'Advisory off_topic flags: {scale["off_topic_advisory"]}; relevance-protection overrides: {scale["protection_overrides"]}.','',
            '|Relation|Intersection / union|Jaccard|','|---|---:|---:|']
        for k,m in scale['agreement'].items(): lines.append(f'|{k}|{m["intersection"]}/{m["union"]}|{pct(m["jaccard"])}|')
        lines+=['','Invalid or unavailable maps and their reasons:','',
            '```json',json.dumps(scale['invalid_map_reasons'],ensure_ascii=False,indent=2),'```','']
    else:
        status=load(ROOT/'maps/status.json') or load(ROOT/'maps/materialization.json')
        lines += ['Scale collection is still pending/running. Latest saved status:','',
            '```json',json.dumps(status,ensure_ascii=False,indent=2),'```','']
    lines+=['The source manifest is the unchanged PREP2 `sample.json` (`maps2000`); no question overlaps the frozen test cohort. '
        'Source profiles use local Bareun with the existing low-priority service queue. Original pilot/retry responses are unchanged.','',
        '## 2. D v2 rejudgment without new edits','']
    rejudge=load(ROOT/'rejudge/metrics.json')
    if rejudge:
        lines += [f'All {rejudge["judged"]}/200 saved attempts received the new introduced-awkwardness judgment. '
            f'Kept: **{rejudge["old_kept"]}/200 ({pct(rejudge["old_rate"])}) → {rejudge["new_kept"]}/200 ({pct(rejudge["new_rate"])})**. '
            f'Newly kept {rejudge["newly_kept"]}; no longer kept {rejudge["no_longer_kept"]}; '
            f'new kept sources {rejudge["new_kept_sources"]}.','',
            'The new Sol call sees question, original and the two final texts, without prior verdicts. '
            'It judges only whether the edits introduced awkward or unnatural wording. All original invention, meaning, improvement, '
            'item-addressing and repetition verdicts remain verbatim. This isolates the gate change; it is not a fresh improvement claim about the editor.','',
            '|Genre|Attempts|Old kept|New kept|New kept rate|','|---|---:|---:|---:|---:|']
        for g,m in rejudge['by_genre'].items(): lines.append(f'|{g}|{m["attempts"]}|{m["old_kept"]}|{m["new_kept"]}|{pct(m["new_rate"])}|')
        lines += ['',f'New awkwardness verdicts: `{json.dumps(rejudge["awkwardness_verdicts"])}`. '
            f'Execution status among kept attempts: `{json.dumps(rejudge["kept_execution_status"])}`. '
            'As in D v2, STOP/completion is reported separately and is not an extra quality gate. '
            'Partial attempts are not described as complete episodes. These historical traces retain their historical prompts and '
            'are not exported as frozen-v4 demonstrations. New editing calls: **0**.','']
    else: lines+=['Rejudgment is pending.','']
    lines+=['## 3. D v3 with frozen prompts and grounded INSERT tasks','',
        '100 new train sources, stratified 34 argumentative / 33 explanatory / 33 emotional; two independent Luna-low attempts per source. '
        'Sources exclude all PREP1/PREP2 cohorts and the frozen 2,000-map cohort, normalized source duplicates and test-cohort questions. '
        'Low/middle uses the bottom two thirds of raw grader_1/2 mean scores within each remaining genre; no scorer inference.','',
        'Sol plans at most four located, grounded items. A warranted explanatory/linking INSERT built only from essay content is retained; '
        'only requests for new facts/examples/experiences go to the writer. There is at most one INSERT task per essay. '
        'The common frozen environment retains the two-successful-INSERT hard cap. Revision receives one or two tasks per six-step delegation; '
        'Korean gets its items and a 14-step whole-essay form pass. Each teacher output is action JSON, with the 8,192 context / 1,024 output limits.','',
        'Quality gate: no invented specifics/experiences, meaning preserved, better than original, at least half of assigned items fully addressed, '
        'no introduced repetition and no introduced awkwardness. Unknown fails the corresponding criterion. STOP/step limits are separate metrics.','']
    content=load(ROOT/'content/metrics.json')
    if content:
        m=content
        lines += [f'Completed plans {m["plans_completed"]}/100; saved attempts {m["attempts"]}/200; judged attempts {m["judged_attempts"]}/200. '
            f'Kept **{m["kept_attempts"]}/200 ({pct(m["kept_rate_all_planned"])})**, across {m["kept_sources"]} sources.','',
            f'INSERT tasks: {m["INSERT_tasks"]}/{m["assigned_items"]} total tasks ({pct(m["INSERT_task_share_all_items"])}), '
            f'{m["INSERT_tasks"]}/{m["revision_items"]} Revision tasks ({pct(m["INSERT_task_share_revision_items"])}); '
            f'{m["essays_with_INSERT_task"]}/{m["plans_completed"]} planned essays ({pct(m["essays_with_INSERT_task_share"])}). '
            f'Actual successful INSERT in {m["attempts_with_successful_INSERT"]}/{m["attempts"]} attempts; '
            f'surviving INSERT in {m["attempts_with_surviving_INSERT"]}, including {m["kept_with_surviving_INSERT"]} kept attempts.','',
            '|Genre|Attempts planned|Judged|Kept|Kept / planned|Essays with INSERT task|','|---|---:|---:|---:|---:|---:|']
        for g,v in m['by_genre'].items():
            lines.append(f'|{g}|{v["planned_attempts"]}|{v["judged"]}|{v["kept"]}|{pct(v["kept"]/v["planned_attempts"])}|{v["INSERT_tasks"]}|')
        lines+=['',f'Revision delegation STOP: {pct(m["delegation_STOP_rate"])} over {m["revision_delegations"]} delegations. '
            f'Endings `{json.dumps(m["delegation_endings"])}`; STOP statuses `{json.dumps(m["delegation_STOP_status"])}`. '
            f'Korean endings `{json.dumps(m["korean_endings"])}`. Valid action rate {pct(m["valid_action_rate"])}. '
            f'Two-steps-left notices in returned action records {m["notices_with_two_steps"]}. Maximum successful INSERTs in one attempt {m["max_successful_INSERTs"]}.','',
            '|Role|Valid action counts|','|---|---|']
        lines += [f'|{r}|`{json.dumps(v)}`|' for r,v in m['actions_by_role'].items()]
        lines+=['',f'Execution status `{json.dumps(m["execution_status"])}`; kept execution status `{json.dumps(m["kept_execution_status"])}`. '
            f'Gate failures (overlapping): `{json.dumps(m["gate_failures"])}`. '
            f'Planner drops {m["planner_dropped"]}; mechanical drops `{json.dumps(m["mechanical_drops"])}`; writer notes {m["writer_notes"]}.','',
            f'Manual review: {m["manual_review"]["cases"]}/15 uniformly sampled kept attempts with a **surviving** INSERT, '
            f'{m["manual_review"]["distinct_sources"]} distinct sources, seed 263; shortfall {m["manual_review"]["shortfall"]}. '
            '[Original | items | revised | verdicts](V4_CONTENT_V3_INSERT_REVIEW_15.md). '
            'UNDO-only insertion attempts are excluded from this review sample.','']
        audit=load(ROOT/'content/action_audit.json')
        if audit:
            lines += [f'Protocol completion (no API error and every required stage ends in STOP, including blocked STOP): '
                f'**{audit["protocol_complete_attempts"]}/200**. Among the {m["kept_attempts"]} quality-kept attempts, '
                f'{audit["protocol_complete_kept_attempts"]} meet that completion definition. '
                'Quality retention does not add a STOP gate; an execution status of completed only means the collection loop returned, '
                'and can include a stage ending at its step limit.','',
                f'Successful INSERT counts across all attempts: `{json.dumps(audit["INSERT_counts_all_attempts"])}`; '
                f'among kept attempts: `{json.dumps(audit["INSERT_counts_kept_attempts"])}`. '
                'The planner permits one INSERT task; the shared environment permits two successful INSERTs. '
                f'Of {audit["INSERT_counts_all_attempts"].get("2",0)} attempts with two successful INSERTs, '
                f'{audit["INSERT_counts_kept_attempts"].get("2",0)} were kept; '
                f'{audit["INSERT_counts_kept_attempts"].get("1",0)} kept attempts used one successful INSERT.','',
                f'Valid actions: {audit["valid_action_responses"]}/{audit["returned_action_responses"]} returned action responses. '
                'Mechanical rejection categories are separate from the Sol quality judgments:','',
                '|Role|Rejected action categories|','|---|---|']
            lines += [f'|{r}|`{json.dumps(v)}`|' for r,v in audit['invalid_categories_by_role'].items()]
            lines += ['','The sentence-boundary check also applies to Revision EDIT; a rejection reports the mechanical check, '
                'not the model’s intent. D v2 rejudging and D v3 use different source cohorts, so their kept rates are not a paired editor comparison.','']
        if 'export' in m:
            lines+=['|Export role|Valid action targets|Source essays|Maximum tokens|','|---|---:|---:|---:|']
            for role,v in m['export'].items(): lines.append(f'|{role}|{v["action_targets"]}|{v["source_essays"]}|{v["max_tokens"]}|')
            lines+=['','Exports assert the exact frozen system message and inference-prefix token IDs. '
                'Only the current valid assistant action and end-turn token receive loss; observations, notices and tool outputs are masked. '
                'Raw rubric feedback, scores and private item IDs are removed. Public Orchestrator assignments remain. **No training was run.**','']
    else: lines+=['D v3 collection is still running; final selection/export metrics are pending.','']
    lines+=['## Cost, verification and artifacts','',
        '|Component|Calls|Confirmed USD|Reserved USD|Cap USD|','|---|---:|---:|---:|---:|']
    total=0
    for name,cap in [('maps',15),('rejudge',3),('content',10)]:
        a=accounting(name); total+=a['confirmed_usd']
        lines.append(f'|{name}|{a["calls"]}|{a["confirmed_usd"]:.6f}|{a["reserved_usd"]:.6f}|{cap:.2f}|')
    lines+=['',f'Total confirmed cost: **${total:.6f}**. Independent durable ledgers reserve the maximum request cost before dispatch; '
        'blocked requests are not sent. Saved errors are reported without silent regeneration. Provider sampling is unseeded.','',
        f'Local artifact root: `{ROOT}`. Source manifests, prompt/design hashes, raw API requests, edge-drop records, attempts, judgments, '
        'selections and action-only exports are retained locally and excluded from Git.','']
    verify=load(ROOT/'verification.json')
    if verify:
        lines+=['Final verification:','', '```json',json.dumps(verify,ensure_ascii=False,indent=2),'```','']
    statuses={name:(ROOT/name/'complete.json').exists() for name in ('maps','rejudge','content')}
    lines+=['PREP3 collection status: `'+json.dumps(statuses)+'`. '
        +('All requested collections stopped; RFT1 and A were left untouched.' if all(statuses.values()) else 'This is an intermediate report; unfinished collections continue.'),'']
    path=REPO/'imple/reports/V4_PREP3.md'; path.write_text('\n'.join(lines)+'\n')
    write_json(ROOT/'report_status.json',{'complete':statuses,'report':str(path),'confirmed_usd':total})
    return path
