"""Local PREP2 report assembled only from saved artifacts; no inference or training."""
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter

from .prep2_common import REPO, ROOT, PREP1, GENRES, read_json, write_json, file_sha, accounting, contract


def pct(value):
    return 'unavailable' if value is None else f'{100*value:.2f}%'


def report():
    frozen=contract()
    c=read_json(ROOT/'C/retry43/metrics.json') if (ROOT/'C/retry43/metrics.json').exists() else None
    gate=read_json(ROOT/'C/gate.json') if (ROOT/'C/gate.json').exists() else None
    dpath=ROOT/'D/metrics.json'
    d=read_json(dpath) if dpath.exists() else None
    labels=read_json(ROOT/'scorer_labels.json') if (ROOT/'scorer_labels.json').exists() else None
    d_done=(ROOT/'D/complete.json').exists()
    c_done=(ROOT/'C/complete.json').exists()
    ca,da=accounting('C'),accounting('D')
    now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    lines=['# V4 PREP2','',f'Updated {now}. C collection stopped={c_done}; D collection stopped={d_done}.','',
        '## Provenance and execution contract','',
        '`assistant`의 `### Feedback`과 1–9 점수의 작성 출처는 사용자 확인에 따라 strong LLM으로 기록한다. '
        '사람 채점은 `grader_1_scores`/`grader_2_scores`의 각 1–5 값이다. '
        '피드백·작업 항목·Sol 판정은 LLM supervision 또는 LLM-flagged problems이며 human gold가 아니다.', '',
        'PREP2는 별도 worktree/config/output/ledger로 API와 낮은 우선순위 CPU만 사용한다. '
        'GPU/채점기 추론/학습 호출은 0이다. RFT1 및 extra GLOBAL A의 실행 코드·프로세스·스케줄을 바꾸지 않았다. '
        '`FEAK_AGENT_METHOD.md`도 수정하지 않았다.', '',
        f'- Method SHA-256: `{frozen["method_sha256"]}` (at-start/current comparison enforced).',
        f'- Previous source design SHA-256: `{frozen["prior_design_sha256"]}`.',
        '- PREP2 artifacts: `verak/v4/outputs/prep2/`; old PREP1 C/D artifacts retained.',
        '- Calls are independently unseeded; source order and review sampling use saved deterministic seeds.', '',
        '## C. Map repair and scale gate','',
        'Cross-paragraph main/key endpoints are now required only when the source has more than six paragraphs. '
        'The prompt provides a valid-ID list and each sentence’s paragraph; main must point to Q, other sentence edges to sentence IDs, '
        'and paragraph keys must belong to that paragraph. Earlier default behavior is retained for old runs.', '',
        'The gate was fixed before calls: both fresh extractions must be valid for at least 39/43 originally invalid-pair essays. '
        'No post-result replacement, third extraction, or silent relation repair is used. '
        'off_topic is advisory only and never sufficient for deletion. Union-edge two-hop protection and exact typed intersection remain in place.', '']
    if c and gate:
        valid=c['single_run_statuses'].get('valid',0)
        lines += [f'**Both-run validity: {gate["valid"]}/43 = {pct(gate["validity"])}. Gate passed: {gate["passed"]}.**',
            f'Single-run validity: {valid}/86 = {pct(valid/86)}. API errors: {c["single_run_statuses"].get("api_error",0)}.', '',
            '|Genre|Retried essays|Both valid|Rate|','|---|---:|---:|---:|']
        for g,v in c['by_genre'].items():
            n=v.get('valid_pairs',0)
            lines.append(f'|{g}|{v["sources"]}|{n}|{pct(n/v["sources"])}|')
        lines += ['','First failing validator reason per invalid extraction:', '', '|Reason|Extractions|','|---|---:|']
        lines += [f'|{reason}|{n}|' for reason,n in c['invalid_reasons'].items()]
        lines += ['', 'The four generic ID/type failures in this fixed retry are relation/endpoint mismatches '
            '(support→Q or main→sentence); they are not invented sentence IDs. '
            'The five main/key failures occur only in essays with more than six paragraphs. '
            'The remaining four failures violate the combined outgoing support/example cardinality.', '',
            'These are mechanical validity results, not a new estimate of semantic map accuracy. '
            'The earlier 60-map Sol review is retained as PREP1 evidence; no new Sol map judgments were requested.', '']
        if not gate['passed']:
            lines += ['**The 2,000-essay extraction was not started because the 90% gate failed.** '
                'A balanced 667/667/666-source manifest was frozen without API calls, but has no scaled map results. '
                'Its source pool excludes the prior B train300, duplicate sources and every frozen test-cohort question.', '']
        elif (ROOT/'C/scale2000/metrics.json').exists():
            scale=read_json(ROOT/'C/scale2000/metrics.json')
            lines += [f'Scale: {scale["valid_pairs"]}/{scale["sources"]} valid pairs; full per-genre/relation metrics in `C/scale2000/metrics.json`.', '']
        lines += ['Intersection agreement on the valid retry pairs only:', '',
            '|Relation|Intersection|Union|Jaccard|','|---|---:|---:|---:|']
        for key,v in c['agreement'].items():
            if key.startswith('all:sentence_relations:'):
                lines.append(f'|{key.split(":")[-1]}|{v["intersection"]}|{v["union"]}|{pct(v["jaccard"])}|')
        lines += ['',f'Final advisory off_topic sentences: {c["off_topic"].get("final_advisory_sentences",0)}; '
            f'protection overrides: {c["off_topic"].get("protected_overrides",0)}.', '']
    else:
        lines += ['Retry still in progress; no scale decision yet.', '']
    lines += [f'C confirmed ${ca["confirmed_usd"]:.6f}; reserved ${ca["reserved_usd"]:.6f}; pending {ca["pending"]}; cap $15.','',
        'The initial sandbox DNS/localhost denial dispatched no model requests and incurred $0. '
        'Its artifacts were preserved separately under `C/not_dispatched_dns_*`; they are excluded from model validity. '
        'Actual retry requests ran with approved network access.', '',
        '## D v2. Bounded Revision delegations','',
        '100 newly sampled train sources (34 argumentative, 33 explanatory, 33 emotional), excluding all prior B train300 '
        'and normalized duplicate sources. No selected question overlaps the frozen test cohort. '
        'Sampling is from the bottom two human-score thirds within each genre (ties retained); two independent attempts per source.', '',
        'Sol converts the stored LLM rubric feedback into at most four located concrete tasks per essay; '
        'generic advice is dropped, exact evidence must occur at the assigned S/P location, and new-fact/experience requests are left for the writer. '
        'Each Revision delegation gets one or two items and six actions, including STOP. '
        'The controller rejects edits/moves/inserts outside the assigned scope and caps successful INSERT calls at two per essay attempt; '
        'UNDO does not replenish the cap and cannot cross delegation boundaries. '
        'A notice is sent with two actions left. Step-limit endings are counted as failures to STOP, never converted into model STOP.', '',
        'Korean receives its own located tasks, actual Revision edits/marker notices and a whole-essay typo/spacing pass, '
        'with the previous 14-action stage budget. Revision content repetition is prohibited by prompt and independently judged by Sol. '
        'There is no scorer reward in this content task.', '',
        'Selection requires all six conditions: no invented specifics/experiences, preserved meaning, better than original, '
        'at least half of the assigned Revision/Korean items fully addressed, no repetition introduced, and natural reading. '
        'Partly addressed is not fully addressed. Writer-only notes are outside the item denominator. '
        'STOP is reported separately and is not silently added to the requested content-quality gate.', '']
    if d:
        lines += [f'Plans {d["plans_completed"]}/100; saved attempts {d["attempts"]}/200; judged pairs {d["judged_pairs"]}/100.',
            f'**Revision STOP per delegation: {d["delegation_endings"].get("STOP",0)}/{d["revision_delegations"]} = {pct(d["delegation_STOP_rate"])}.**',
            f'**Kept: {d["kept_attempts"]}/200 planned attempts = {pct(d["kept_rate_all_planned"])}; '
            f'{d["kept_sources"]} distinct sources. Judged-attempt kept rate: {pct(d["kept_rate_judged"])}.**','',
            f'- Revision delegation endings: `{d["delegation_endings"]}`.',
            f'- Actual STOP statuses: `{d["delegation_STOP_status"]}`.',
            f'- No Revision items, hence no delegation: {d["revision_no_items_attempts"]} attempts.',
            f'- Korean endings: `{d["korean_endings"]}`.',
            f'- Max successful INSERTs observed per attempt: {d["max_inserts_observed"]}.',
            f'- Assigned item owners/actions: `{d["item_owners"]}` / `{d["item_actions"]}`.',
            f'- Generic/other planner drops: {d["generic_or_other_drops"]}; exact-location evidence drops: {d["ungrounded_item_drops"]}; writer notes: {d["writer_notes"]}.',
            f'- Gate failures (overlap allowed): `{d["gate_failures"]}`.',
            f'- Execution/planning/judging errors: {len(d["errors"])} (details in `D/metrics.json`).','',
            '|Genre|Attempts|Kept|','|---|---:|---:|']
        for g,v in d['by_genre'].items():
            lines.append(f'|{g}|{v["attempts"]}|{v["kept"]}|')
        lines += ['',f'Manual file: [V4_CONTENT_V2_REVIEW_30.md](V4_CONTENT_V2_REVIEW_30.md), '
            f'{d["manual_cases"]} randomly sampled kept attempts from {d["manual_distinct_sources"]} sources (seed 233); '
            f'shortfall {d["manual_shortfall"]}. Each case includes original, located tasks, revised text and all verdicts.', '',
            'Exports retain only inference-visible Orchestrator assignments, state/tool observations and action JSON targets. '
            'Raw LLM feedback, rubric scores, private item IDs and judge verdicts are removed from policy observations. '
            'Only the current assistant action and end-of-turn receive loss; all observations are masked. '
            'No adapter was trained. Export checks/hashes are saved under `D/export/contract.json`.', '']
    else:
        lines += ['D collection is still in progress. No final STOP/kept rate is claimed.', '']
    lines += [f'D confirmed ${da["confirmed_usd"]:.6f}; reserved ${da["reserved_usd"]:.6f}; pending {da["pending"]}; cap $10.', '',
        '## Scorer labels: file-only audit','']
    if labels:
        lines += [f'Frozen adapter: `{labels["frozen_adapter"]}`. The saved adapter config specifies CAUSAL_LM, r=16, alpha=32 '
            'and Kanana base, but contains no training dataset path, formatting function or label-field selection. '
            'Its model-card Training Data/Procedure fields are unfilled. The local inference package and available git history '
            'do not include the original LoRA training recipe.', '',
            '`V3_PHASE_0.md` records train.jsonl as the training file based on prior user confirmation. '
            '**The user subsequently confirmed that training used aggregated teacher/grader scores.** '
            'The numerical target is therefore the combined human score on 1–9, independently checked below. '
            'Whether the original loader read the assistant header or recomputed it from the grader fields remains undocumented; '
            'these produce identical score vectors in the saved data. This is a statement about score targets, not human authorship of feedback.', '',
            'The score values themselves are unambiguous in the stored files:', '',
            '|Split|Rows audited|Rows with all 8 assistant scores = grader_1 + grader_2 − 1|Matching rubric values|','|---|---:|---:|---:|']
        for split,v in labels['data'].items():
            lines.append(f'|{split}|{v["valid_rows"]}|{v["all_eight_match_rater_sum_minus_one"]}|{v["matching_digits"]}/{v["total_digits"]}|')
        lines += ['', '**Scale mapping:** `s = r1 + r2 − 1 = 2 × mean(r1,r2) − 1` on 1–9; '
            '`mean(r1,r2) = (s+1)/2` on 1–5. For example, raters (3,4) map to 6, and system 6 maps to mean 3.5. '
            '`essay_scoring_llm/scaling.py::label_from_raters` implements this mapping for evaluation.', '',
            'Thus the literal assistant score vector and the recomputed human-score vector are numerically identical in all audited rows; '
            'the data alone cannot distinguish which construction the original trainer used. '
            'This equality does not establish who authored the surrounding feedback or that LLM feedback is a human standard. '
            'Feedback provenance remains recorded as supplied by the user.', '',
            'The running v3 scorer requests eight 1–9 digits, computes expectations from the raw nine-digit softmax, '
            'and averages those rubric expectations. It does not apply the legacy FEAK random-forest correction. '
            'No scale, scorer implementation, reward or running training was modified.', '',
            'The pinned base config has `architectures=[LlamaForCausalLM]`, `model_type=llama`. '
            'The loader uses AutoModelForCausalLM + frozen PEFT LoRA, without a separate regression head. '
            '**Whether this adapter is CE Only, CE+NTL, or Proposed CE+NTL+SAL is still unverified.** '
            'NTL/SAL are training losses, not additional inference operations. The loaded adapter would have to be linked '
            'to the original loss configuration or final Proposed checkpoint to establish that claim. '
            'Available adapter/config/inference files contain no such loss recipe; no retraining or replacement was performed.', '',
            'Evidence hashes and per-split/per-rubric counts are in `verak/v4/outputs/prep2/scorer_labels.json`.', '']
    else:
        lines += ['Scorer-label audit pending.', '']
    lines += ['## Cost and stopping status','',
        '|Component|Confirmed USD|Reserved USD|Cap USD|','|---|---:|---:|---:|',
        f'|Map repair / conditional scale|{ca["confirmed_usd"]:.6f}|{ca["reserved_usd"]:.6f}|15|',
        f'|D v2 planning / teachers / judges|{da["confirmed_usd"]:.6f}|{da["reserved_usd"]:.6f}|10|',
        '|Scorer-label file audit|0|0|0|','',
        'Separate PREP2 ledgers are not charged to A or RFT1. Paid outcomes are cached and reused on resume; '
        'invalid model outputs are preserved without outcome-driven retries. '
        'RFT1 and A continue under their existing authorization; PREP2 does not start any training or alter their schedule.', '']
    text='\n'.join(lines)
    target=REPO/'imple/reports/V4_PREP2.md'
    target.write_text(text)
    write_json(ROOT/'report_status.json',{'at':now,'C_stopped':c_done,'D_stopped':d_done,
        'method_unchanged':file_sha(REPO/'imple/FEAK_AGENT_METHOD.md')==frozen['method_sha256'],
        'scorer_training_recipe_confirmed':bool(labels and labels.get('training_recipe_locally_found')),
        'path':str(target),'sha256':file_sha(target)})
    return str(target)
