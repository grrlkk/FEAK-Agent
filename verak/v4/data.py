"""Question-disjoint B/C sources; teacher feedback never enters a v1 loader."""
from collections import Counter
from pathlib import Path
import json
import random
import re
import time

from verak.v3.common import extract_question_essay
from verak.v3.data_policy import HEADING_SETS, feedback_headings
from verak.v3.corrupt.document import Document
from verak.v3.phase2 import restore_profile
from verak.v3.v2_ops.local import load_environment
from verak.v3.insertion_boost.resources import BoostParagraphs
from .common import (REPO, ROOT, GENRES, GENRE_KO, load_config, read_json, write_json,
                     file_sha, sha_text, safe_id, atomic_new, collection_lock)


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for index, line in enumerate(stream, 1):
            if line.strip():
                yield index, json.loads(line)


def genre_of(row):
    found = set(feedback_headings(row.get('assistant', '')))
    return next((en for en in GENRES if found == HEADING_SETS[GENRE_KO[en]]), None)


def feedback_parts(text):
    # Keep native rubric wording; normalization is a separate explicit mapping.
    matches = list(re.finditer(r'(?m)^\s*-\s*([^:\n]+):\s*', text))
    return [{'rubric_native': m.group(1).strip(), 'text': text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()}
            for i, m in enumerate(matches)]


def normalized_source(text):
    return sha_text(' '.join(text.split()))


def freeze_sample():
    target = ROOT / 'sample.json'
    if target.exists():
        result = read_json(target)
        for value in result['inputs'].values():
            if file_sha(value['path']) != value['sha256']:
                raise ValueError('Raw source file changed')
        return result
    paths = {s: REPO / 'data/data_jsonl' / (s+'.jsonl') for s in ('train', 'valid', 'test')}
    metadata = REPO / 'verak/v3/data'
    audit = read_json(metadata / 'audit_index.json')
    split = read_json(metadata / 'splits.json')
    training_questions = set(audit['train']['question_counts']) | set(split['agent_train'])
    source_deny = set()
    train_pool = {g: [] for g in GENRES}
    exclusions = Counter()
    seen_train = set()
    for name in ('train', 'valid'):
        for number, row in read_rows(paths[name]):
            question, essay = extract_question_essay(row)
            norm = normalized_source(essay)
            source_deny.add(norm)
            if name != 'train':
                continue
            genre = genre_of(row)
            if not genre:
                exclusions['train_unknown_feedback_genre'] += 1
                continue
            if norm in seen_train:
                exclusions['train_duplicate_source'] += 1
                continue
            seen_train.add(norm)
            scores = row.get('grader_1_scores', []) + row.get('grader_2_scores', [])
            if len(scores) != 16 or not all(isinstance(v, (int, float)) for v in scores):
                exclusions['train_invalid_human_scores'] += 1
                continue
            train_pool[genre].append({'source_id': f'train:{number}', 'genre': genre,
                'question_hash': sha_text(question), 'essay_hash': sha_text(essay),
                'normalized_source_hash': norm, 'human_mean': sum(scores)/len(scores), 'characters': len(essay)})
    test_pool = {g: [] for g in GENRES}
    seen_test, unseen_questions = set(), set()
    for number, row in read_rows(paths['test']):
        question, essay = extract_question_essay(row)
        if sha_text(question) in training_questions:
            exclusions['test_training_question'] += 1
            continue
        unseen_questions.add(sha_text(question))
        norm = normalized_source(essay)
        if norm in source_deny or norm in seen_test:
            exclusions['test_duplicate_source'] += 1
            continue
        seen_test.add(norm)
        genre = genre_of(row)
        if not genre:
            exclusions['test_unknown_feedback_genre'] += 1
            continue
        test_pool[genre].append({'source_id': f'test:{number}', 'genre': genre,
            'question_hash': sha_text(question), 'essay_hash': sha_text(essay), 'normalized_source_hash': norm,
            'characters': len(essay)})
    if len(unseen_questions) != 62:
        raise ValueError(f'Expected the approved 62 unseen questions, found {len(unseen_questions)}')
    chosen_train, chosen_test, maps = [], [], []
    for i, genre in enumerate(GENRES):
        values = sorted(train_pool[genre], key=lambda x: x['source_id'])
        random.Random(101+i).shuffle(values)
        if len(values) < 100:
            raise ValueError('Insufficient distinct training sources')
        chosen_train.extend(values[:100])
        map_values = list(values[:100])
        random.Random(103+i).shuffle(map_values)
        maps.extend(map_values[:50])
        values = sorted(test_pool[genre], key=lambda x: x['source_id'])
        random.Random(107+i).shuffle(values)
        count = 34 if i == 0 else 33
        if len(values) < count:
            raise ValueError('Insufficient question-disjoint evaluation sources')
        chosen_test.extend(values[:count])
    if {r['question_hash'] for r in chosen_train} & {r['question_hash'] for r in chosen_test}:
        raise ValueError('Feedback train/test question overlap')
    result = {'version': 'v4', 'seed_train': 101, 'seed_map': 103, 'seed_test': 107,
              'provider_sampling_seed': None, 'inputs': {k: {'path': str(p), 'sha256': file_sha(p)} for k,p in paths.items()},
              'train300': [r['source_id'] for r in chosen_train], 'test100': [r['source_id'] for r in chosen_test],
              'maps150': [r['source_id'] for r in maps],
              'source_metadata': {r['source_id']: r for r in chosen_train+chosen_test},
              'training_question_hashes': sorted(training_questions), 'eligible_test_questions': sorted(unseen_questions),
              'selected_test_questions': sorted({r['question_hash'] for r in chosen_test}),
              'exclusions': dict(exclusions),
              'eligible_train_per_genre': {g:len(v) for g,v in train_pool.items()},
              'eligible_test_per_genre': {g:len(v) for g,v in test_pool.items()},
              'human_score_source': 'mean of both stored grader_1_scores/grader_2_scores (16 raw rubric values); not scorer output',
              'feedback_source': 'stored rubric-wise assistant field; no separate per-rater feedback fields available',
              'sampling': '100 unique train sources per genre; evaluation 34/33/33 after all-training-question and normalized-source exclusions; maps are a 50/genre subset of B',
              'gpu_used': False, 'paid_calls': 0}
    atomic_new(target, result)
    return result


def materialize():
    config = load_config()
    load_environment(config)
    sample = freeze_sample()
    paths = {k: Path(v['path']) for k,v in sample['inputs'].items()}
    wanted = set(sample['train300']+sample['test100'])
    analyzer = BoostParagraphs(config, cache_dir=ROOT / 'bareun_paragraphs')
    files, errors = {}, []
    for split in ('train', 'test'):
        for index, raw in read_rows(paths[split]):
            source_id = f'{split}:{index}'
            if source_id not in wanted:
                continue
            path = ROOT / 'essays' / (safe_id(source_id)+'.json')
            if not path.exists():
                question, essay = extract_question_essay(raw)
                try:
                    profile = analyzer.profile(essay)
                    doc = Document.from_profile(essay, profile)
                    # A full profile is cached locally; only selected public fields leave this server.
                    row = {**sample['source_metadata'][source_id], 'split': split,
                        'question': question, 'text': essay, 'layout': doc.snapshot(), 'profile': profile.to_dict(),
                        'paragraphs': [{'id': p.pid, 'sentences': [{'id':u.sid, 'text':u.text} for u in p.units]} for p in doc.paragraphs],
                        'human_scores': [raw['grader_1_scores'], raw['grader_2_scores']],
                        'feedback': feedback_parts(raw['assistant']), 'scorer_seen': split == 'train'}
                    if len(row['feedback']) != 8:
                        raise ValueError('Expected exactly eight rubric feedback entries')
                    atomic_new(path, row)
                except Exception as exc:
                    errors.append({'source_id':source_id,'error':type(exc).__name__+': '+str(exc)})
                    write_json(ROOT / 'source_status.json', {'completed':len(files),'total':400,'errors':errors,'at':time.time()})
                    raise
            files[source_id] = {'path': str(path), 'sha256':file_sha(path)}
            write_json(ROOT / 'source_status.json', {'completed':len(files),'total':400,'errors':errors,'at':time.time()})
    if set(files) != wanted:
        raise ValueError('Incomplete frozen source materialization')
    design = {**sample, 'sample_sha256': file_sha(ROOT/'sample.json'), 'essay_files':files}
    target = ROOT/'design.json'
    if target.exists() and read_json(target) != design:
        raise ValueError('Frozen v4 design changed')
    if not target.exists():
        atomic_new(target, design)
    return design


def main():
    from .common import constrain_cpu
    constrain_cpu()
    with collection_lock(ROOT / 'source_collection'):
        design = materialize()
    print(json.dumps({'train':len(design['train300']),'test':len(design['test100']),'maps':len(design['maps150'])}))


if __name__ == '__main__':
    main()
