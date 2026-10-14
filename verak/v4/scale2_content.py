"""B2: new-source Dv3 content teachers, gated by the frozen v4.2 smoke.

No work runs at import. The runner uses the versioned teacher/export functions;
it neither changes the policy prefix nor adds a corruption reward/STOP gate.
"""
from collections import Counter
from pathlib import Path
import random
import time

from .common import (GENRES, REPO, atomic_new, collection_lock, file_sha, load_config,
                     read_json, safe_id, sha_text, write_json)
from .scale2_common import (ROOT, ATTEMPTS, api_for, component_accounting,
                           component_contract, freeze, saved_cost_estimate,
                           bounded_episodes, dispatch_limit)

QUOTAS = dict(zip(GENRES, (334, 333, 333)))
SEEDS = (421, 422, 423, 429)


def prior_cohorts(root=ROOT, *, allow_pending_smoke=False):
    from .data import normalized_source
    from verak.v3.phase2 import read_jsonl
    base = REPO/'verak/v4/outputs'
    paths = [base/'prep/design.json', base/'prep2/sample.json',
             base/'prep3/content/sample.json', base/'scale/B1/sample.json',
             Path(root)/'B1/sample.json']
    if allow_pending_smoke and not paths[-1].exists():
        # Read-only supply estimates may precede the same-20-source smoke copy.
        # Actual freezing never uses this branch and requires the new B1 file.
        paths = paths[:-1]
    metadata, questions = {}, set()
    for path in paths:
        value = read_json(path)
        metadata.update(value['source_metadata'])
        for key in ('selected_test_questions', 'test_question_hashes', 'eligible_test_questions'):
            questions.update(value.get(key, []))
    hashes = {str(p): file_sha(p) for p in paths}
    def add_source(row, cohort):
        sid = row['source_id']
        norm = normalized_source(row['source_text'])
        if sid in metadata and metadata[sid]['normalized_source_hash'] != norm:
            raise ValueError('Earlier cohorts disagree on the original source identity: '+sid)
        metadata.setdefault(sid, {'source_id': sid, 'normalized_source_hash': norm,
                                 'question_hash': sha_text(row['question']), 'cohort': cohort})
    # New content sources are disjoint from earlier corruption/extra-teacher
    # sources as well, even though those mostly use the valid:N namespace.
    active = load_config()['paths']['active_corrupt']
    for split in ('agent_train', 'agent_dev'):
        path = active/(split+'.jsonl')
        hashes[str(path)] = file_sha(path)
        for row in read_jsonl(path):
            add_source(row, 'v3_active_'+split)
            if split == 'agent_dev':
                questions.add(sha_text(row['question']))
    boost = REPO/'verak/v3/outputs/data_boost'
    global_manifest = boost/'global/gpu_rescore_manifest.json'
    insertion_design = boost/'insertion/teacher_design.json'
    candidate_paths = set()
    if global_manifest.exists():
        hashes[str(global_manifest)] = file_sha(global_manifest)
        for episode in read_json(global_manifest)['episodes']:
            candidate_paths.add(episode['candidate_path'])
    if insertion_design.exists():
        hashes[str(insertion_design)] = file_sha(insertion_design)
        candidate_paths.update(v['path'] for v in read_json(insertion_design)['corpus'].values())
    for name in sorted(candidate_paths):
        row = read_json(name)
        hashes[name] = file_sha(name)
        add_source(row, 'v3_extra_teacher')
    return metadata, questions, hashes


def select_sources(raw_rows, prior, questions, *, quotas=QUOTAS, seeds=SEEDS):
    """Pure selection: two graders' 16 values, genre lower two-thirds, no replacement."""
    from verak.v3.common import extract_question_essay
    from .data import genre_of, normalized_source
    forbidden = {r['normalized_source_hash'] for r in prior.values()}
    pool = {g: [] for g in GENRES}
    seen, excluded = set(), Counter()
    for number, raw in raw_rows:
        source = f'train:{number}'
        question, text = extract_question_essay(raw)
        normalized = normalized_source(text)
        if source in prior or normalized in forbidden:
            excluded['prior_source_or_normalized_text'] += 1
            continue
        if sha_text(question) in questions:
            excluded['test_question'] += 1
            continue
        if normalized in seen:
            excluded['duplicate_source'] += 1
            continue
        genre = genre_of(raw)
        if genre not in GENRES:
            excluded['unknown_genre'] += 1
            continue
        scores = raw.get('grader_1_scores', []) + raw.get('grader_2_scores', [])
        if len(scores) != 16 or any(type(x) not in (int, float) or not 1 <= x <= 5 for x in scores):
            excluded['invalid_human_scores'] += 1
            continue
        seen.add(normalized)
        pool[genre].append({'source_id': source, 'genre': genre, 'question_hash': sha_text(question),
            'essay_hash': sha_text(text), 'normalized_source_hash': normalized,
            'human_mean': sum(scores)/16, 'characters': len(text)})
    chosen, eligibility = [], {}
    for offset, genre in enumerate(GENRES):
        values = sorted(pool[genre], key=lambda r: (r['human_mean'], r['source_id']))
        if not values:
            raise ValueError('No eligible sources for genre: ' + genre)
        cutoff = values[(2*len(values)-1)//3]['human_mean']
        eligible = [r for r in values if r['human_mean'] <= cutoff]
        random.Random(seeds[offset]).shuffle(eligible)
        if len(eligible) < quotas[genre]:
            raise ValueError('Insufficient new low/middle sources for genre: ' + genre)
        chosen.extend(eligible[:quotas[genre]])
        eligibility[genre] = {'eligible': len(values), 'low_middle_cutoff': cutoff,
            'low_middle_count': len(eligible), 'chosen': quotas[genre]}
    random.Random(seeds[-1]).shuffle(chosen)
    return chosen, eligibility, dict(excluded)


def freeze_sample(root=ROOT):
    from .data import read_rows
    root = Path(root)
    contract = component_contract('B2', root)
    path = root/'B2/sample.json'
    if path.exists():
        saved = read_json(path)
        for source, digest in saved['prior_cohort_sha256'].items():
            if file_sha(source) != digest:
                raise ValueError('Prior cohort changed after B2 freeze')
        if file_sha(saved['input_path']) != saved['input_sha256']:
            raise ValueError('B2 raw source file changed')
        return saved
    prior, questions, hashes = prior_cohorts(root)
    canonical_test = read_json(REPO/'verak/v4/outputs/prep/design.json')
    test_split = set(canonical_test['eligible_test_questions'])
    test_cohort = set(canonical_test['selected_test_questions'])
    if not test_split <= questions or not test_cohort <= questions:
        raise ValueError('Frozen test-split or 100-test-cohort questions were omitted')
    rawpath = REPO/'data/data_jsonl/train.jsonl'
    chosen, eligibility, exclusions = select_sources(read_rows(rawpath), prior, questions)
    value = {'schema_version': 1, 'prompt_version': contract['prompt_version'],
        'input_path': str(rawpath), 'input_sha256': file_sha(rawpath),
        'source_ids': [r['source_id'] for r in chosen],
        'source_metadata': {r['source_id']: r for r in chosen},
        'prior_cohort_sha256': hashes, 'prior_distinct_sources': len(prior),
        'test_question_hashes': sorted(questions), 'exclusions': exclusions,
        'test_question_inventory': {'hash_rule': 'existing exact sha_text(question)',
            'canonical_test_split_questions': len(test_split),
            'frozen_100_test_cohort_questions': len(test_cohort),
            'additional_v3_dev_questions': len(questions-test_split),
            'canonical_split_missing': 0, 'frozen_test_cohort_missing': 0},
        'eligibility': eligibility, 'sampling_seeds': list(SEEDS),
        'provider_sampling_seed': None, 'quotas': QUOTAS,
        'human_sampling': 'mean of 16 stored grader values (1–5); lower two-thirds cutoff per genre; ties retained',
        'source_replacement_after_teacher_failure': False,
        'no_quality_or_length_cherry_picking': True, 'cost_estimate': saved_cost_estimate()}
    return freeze(path, value)


def materialize(root=ROOT, *, limit=None):
    from verak.v3.common import extract_question_essay
    from verak.v3.corrupt.document import Document
    from verak.v3.insertion_boost.resources import BoostParagraphs
    from verak.v3.v2_ops.local import load_environment
    from .data import read_rows, feedback_parts
    from .environment_v42 import normalize_row
    root = Path(root)
    sample = freeze_sample(root)
    config = load_config()
    load_environment(config)
    analysis = BoostParagraphs(config, cache_dir=root/'B2/bareun_sources')
    ordered = sample['source_ids'] if limit is None else sample['source_ids'][:limit]
    requested = set(ordered)
    raw_by_source = {}
    for number, raw in read_rows(sample['input_path']):
        source = f'train:{number}'
        if source in requested:
            raw_by_source[source] = raw
    if set(raw_by_source) != requested:
        raise ValueError('Frozen B2 source disappeared from its raw input')
    for source in ordered:
        raw = raw_by_source[source]
        path = root/'B2/essays'/(safe_id(source)+'.json')
        if path.exists():
            yield read_json(path)
            continue
        while dispatch_limit('B2', root, 0) == 0:
            time.sleep(5)
        question, text = extract_question_essay(raw)
        profile = analysis.profile(text)
        doc = Document.from_profile(text, profile)
        row = {**sample['source_metadata'][source], 'split': 'train',
            'question': question, 'text': text, 'profile': profile.to_dict(), 'layout': doc.snapshot(),
            'paragraphs': [{'id': p.pid, 'sentences': [{'id': u.sid, 'text': u.text} for u in p.units]}
                           for p in doc.paragraphs],
            'human_scores': [raw['grader_1_scores'], raw['grader_2_scores']],
            'feedback': feedback_parts(raw['assistant']), 'feedback_origin': 'strong_LLM', 'scorer_seen': True}
        if sha_text(text) != row['essay_hash'] or len(row['feedback']) != 8:
            raise ValueError('B2 materialized source does not match the frozen sample')
        # The planner, teacher and judge all use this same normalized baseline.
        # Its original layout/text and character ranges remain in provenance.
        row = normalize_row(row, analysis, document=doc)
        row['normalized_essay_hash'] = sha_text(row['text'])
        atomic_new(path, row)
        write_json(root/'B2/materialization.json', {'saved': len(list((root/'B2/essays').glob('*.json'))),
            'planned': 1000, 'at': time.time()})
        yield row
    files = {source: {'path': str(root/'B2/essays'/(safe_id(source)+'.json')),
                     'sha256': file_sha(root/'B2/essays'/(safe_id(source)+'.json'))}
             for source in ordered}
    if len(ordered) == len(sample['source_ids']):
        freeze(root/'B2/source_files.json', files)


def report(root=ROOT, *, tokenizer=None):
    from .prep3_content import selection
    from .runtime_v42 import export
    root = Path(root)
    base = root/'B2'
    sample = read_json(base/'sample.json')
    plans, attempts, judgments, cases, errors = [], [], [], [], []
    for source in sample['source_ids']:
        rowpath = base/'essays'/(safe_id(source)+'.json')
        planpath = base/'content/items'/(safe_id(source)+'.json')
        if not planpath.exists():
            continue
        plan = read_json(planpath)
        plans.append(plan)
        if plan['status'] != 'completed':
            errors.append(plan)
            continue
        row = read_json(rowpath)
        jp = base/'content/judgments'/(safe_id(source)+'.json')
        judgment = read_json(jp) if jp.exists() else None
        if judgment and judgment['status'] == 'completed':
            judgments.append(judgment)
        elif judgment:
            errors.append(judgment)
        for index in ATTEMPTS:
            ap = base/'content/attempts'/f'{safe_id(source)}_a{index}.json'
            if not ap.exists():
                continue
            attempt = read_json(ap)
            attempts.append(attempt)
            if attempt['status'] != 'completed':
                errors.append({'source_id': source, 'attempt': index, 'status': attempt['status'],
                               'error': attempt.get('error')})
            if judgment and judgment['status'] == 'completed':
                verdict = next(v for v in judgment['attempts'] if v['attempt'] == index)
                cases.append({'source_id': source, 'genre': row['genre'], 'items': plan['items'],
                    'attempt': attempt, 'verdict': verdict, 'selection': selection(plan['items'], attempt, verdict)})
    kept = [c for c in cases if c['selection']['export_keep']]
    calls = [c for a in attempts for c in a['calls']]
    account = component_accounting('B2', root)
    stats = {'component': 'B2', 'prompt_version': read_json(base/'contract.json')['prompt_version'],
        'planned_sources': 1000, 'planned_attempts': 2000, 'source_essays': len(sample['source_ids']),
        'plans': len(plans), 'plans_completed': sum(p['status'] == 'completed' for p in plans),
        'attempts_saved': len(attempts), 'attempts_started': sum(bool(a['calls']) for a in attempts),
        'execution_status': dict(Counter(a['status'] for a in attempts)), 'judged_pairs': len(judgments),
        'judged_attempts': len(cases), 'kept_attempts': len(kept),
        'kept_sources': len({c['source_id'] for c in kept}),
        'quality_gate_failures': dict(Counter(k for c in cases for k, v in c['selection']['criteria'].items() if not v)),
        'valid_actions': sum(c['action']['valid'] for c in calls), 'returned_actions': len(calls),
        'actions_by_role': {r: dict(Counter(c['action']['action'] for c in calls if c['role'] == r and c['action']['valid']))
                            for r in ('revision', 'korean')},
        'by_genre': {g: {'planned_sources': QUOTAS[g], 'attempts': sum(a['genre'] == g for a in attempts),
            'kept_attempts': sum(c['genre'] == g for c in kept)} for g in GENRES},
        'remaining_uncollected_attempts': 2000-len(attempts), 'errors': errors,
        'selection_uses_Dv3_quality_gates_only': True, 'extra_terminal_STOP_gate': False,
        'test_question_inventory': sample['test_question_inventory'],
        'test_questions_excluded': len(sample['test_question_hashes']),
        'prior_distinct_sources': sample['prior_distinct_sources'],
        'human_sampling_eligibility': sample['eligibility'],
        'api': account, 'GPU_used': False, 'scorer_calls': 0, 'training': False}
    write_json(base/'selections.json', [{'source_id': c['source_id'], 'attempt': c['attempt']['attempt'],
        **c['selection']} for c in cases])
    if tokenizer is not None:
        stats['export'] = export(kept, tokenizer, output_root=base)
    write_json(base/'metrics.json', stats)
    lines = ['# B2 v4.2 content teachers', '',
        f"Sources {len(sample['source_ids'])}/1000; saved attempts {len(attempts)}/2000; judged pairs {len(judgments)}.",
        f"Quality-kept {len(kept)} attempts from {stats['kept_sources']} source essays.",
        'Selection uses the six Dv3 quality conditions, including no introduced repetition or awkwardness. '
        'Both passing attempts are retained. Only valid action targets are exported with the exact frozen v4.2 prefix; '
        'observations, tool results, notices, raw feedback and scores are masked or absent.',
        'Low/middle sampling uses the mean of the two human graders’ 16 stored 1–5 values, lower two-thirds per genre; '
        'prior cohorts and test questions are excluded. Genre quotas are 334/333/333. No failed source is replaced.', '',
        f"Excluded questions: {stats['test_question_inventory']['canonical_test_split_questions']} canonical test-split questions "
        f"(including all {stats['test_question_inventory']['frozen_100_test_cohort_questions']} questions in the frozen 100-test cohort), "
        f"plus {stats['test_question_inventory']['additional_v3_dev_questions']} additional v3-dev questions; "
        f"{stats['test_questions_excluded']} unique exclusions. Missing frozen questions: "
        f"{stats['test_question_inventory']['canonical_split_missing']} split / "
        f"{stats['test_question_inventory']['frozen_test_cohort_missing']} cohort. "
        f"Earlier-cohort exclusions cover {stats['prior_distinct_sources']} distinct source IDs and their normalized-text duplicates.", '',
        f"Confirmed B2 cost ${account['confirmed_usd']:.8f}; reserved ${account['reserved_usd']:.8f}; "
        f"two-smoke-plus-B2 total ${account['cumulative_confirmed_and_reserved_usd']:.8f}/$50.",
        f"Uncollected attempts {stats['remaining_uncollected_attempts']}; saved errors {len(errors)}.",
        'No scorer, GPU, or training was run by B2.', '']
    (base/'report.md').write_text('\n'.join(lines))
    return stats


def run(root=ROOT, *, limit=None):
    """Explicit invocation only after B1 PASS; bounded workers never use GPUs."""
    from .runtime_v42 import make_plan, teach, bind
    from .prep3_content import judge
    from transformers import AutoTokenizer
    root = Path(root)
    with collection_lock(root/'B2'):
        component_contract('B2', root)
        rows = materialize(root, limit=limit)
        tokenizer = AutoTokenizer.from_pretrained(str(load_config()['paths']['policy_base']), local_files_only=True)
        luna, sol = api_for('B2', root=root), api_for('B2', 'sol', root=root)
        try:
            luna.settle_interrupted()
            observed = sum(bool(read_json(path).get('calls')) for path in (root/'B2/content/attempts').glob('*.json'))
            def one(row):
                account = component_accounting('B2', root)
                if account['confirmed_usd']+account['reserved_usd'] >= account['component_cap_usd']:
                    return {'budget_stop': True, 'source_id': row['source_id']}
                existing = {index: (root/'B2/content/attempts'/f'{safe_id(row["source_id"])}_a{index}.json').exists()
                            for index in ATTEMPTS}
                plan = make_plan(row, sol, output_root=root/'B2')
                attempts = []
                judged = None
                if plan['status'] == 'completed':
                    attempts = [teach(row, plan, index, luna, tokenizer, output_root=root/'B2') for index in ATTEMPTS]
                    judged = bind(judge, ROOT=root/'B2')(row, plan, attempts, sol)
                return {'source_id': row['source_id'],
                    'new_started_attempts': sum(bool(a['calls']) and not existing[a['attempt']] for a in attempts),
                    'budget_stop': plan['status'] == 'budget_stop' or any(a['status'] == 'budget_stop' for a in attempts)
                                   or bool(judged and judged['status'] == 'budget_stop')}
            def progress(number, value, started):
                write_json(root/'B2/status.json', {'processed_this_run': number, **value,
                    'observed_started_attempts': started, 'current_worker_limit': dispatch_limit('B2', root, started),
                    'api': component_accounting('B2', root), 'at': time.time()})
            bounded_episodes('B2', rows, one, root=root, observed_attempts=observed, progress=progress)
        finally:
            luna.close()
            sol.close()
        metrics = report(root, tokenizer=tokenizer)
        if metrics['api']['pending']:
            raise RuntimeError('Do not complete B2 while a paid call is live')
        write_json(root/'B2/collection_finished.json', {'status': 'complete' if metrics['attempts_saved'] == 2000 else 'partial',
            'metrics_path': str(root/'B2/metrics.json'), 'metrics_sha256': file_sha(root/'B2/metrics.json'),
            'no_live_paid_calls': True, 'training': False})
        return metrics
