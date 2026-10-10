"""Freeze new source cohorts; all source/LLM feedback artifacts remain local."""
from collections import Counter
import random
import time

from verak.v3.common import extract_question_essay
from verak.v3.corrupt.document import Document
from verak.v3.insertion_boost.resources import BoostParagraphs
from verak.v3.v2_ops.local import load_environment
from .data import read_rows, genre_of, normalized_source, feedback_parts
from .prep2_common import (REPO, ROOT, PREP1, GENRES, read_json, write_json, file_sha,
    sha_text, safe_id, atomic_new, load_config, contract)


def freeze():
    contract()
    path = ROOT/'sample.json'
    if path.exists():
        saved = read_json(path)
        if file_sha(saved['input_path']) != saved['input_sha256']:
            raise ValueError('Raw train file changed')
        return saved
    old = read_json(PREP1/'design.json')
    forbidden_q = set(old['selected_test_questions'])
    old_sources = set(old['train300'])
    old_norms = {r['normalized_source_hash'] for r in old['source_metadata'].values()}
    pool = {g: [] for g in GENRES}
    seen = set()
    excluded = Counter()
    rawpath = REPO/'data/data_jsonl/train.jsonl'
    for number, raw in read_rows(rawpath):
        source = f'train:{number}'
        question, essay = extract_question_essay(raw)
        norm = normalized_source(essay)
        genre = genre_of(raw)
        if source in old_sources or norm in old_norms:
            excluded['prior_B_or_test_source'] += 1
            continue
        if sha_text(question) in forbidden_q:
            excluded['test_question'] += 1
            continue
        if norm in seen:
            excluded['duplicate_source'] += 1
            continue
        if genre not in GENRES:
            excluded['unknown_genre'] += 1
            continue
        scores = raw.get('grader_1_scores', [])+raw.get('grader_2_scores', [])
        if len(scores) != 16 or any(not isinstance(s, (int, float)) or not 1 <= s <= 5 for s in scores):
            excluded['invalid_human_scores'] += 1
            continue
        seen.add(norm)
        pool[genre].append({'source_id': source, 'genre': genre, 'question_hash': sha_text(question),
            'essay_hash': sha_text(essay), 'normalized_source_hash': norm,
            'human_mean': sum(scores)/16, 'characters': len(essay)})
    maps, content, eligibility = [], [], {}
    for offset, (genre, count) in enumerate(zip(GENRES, (667,667,666))):
        values = sorted(pool[genre], key=lambda r: r['source_id'])
        random.Random(211+offset).shuffle(values)
        if len(values) < count:
            raise ValueError('Not enough new map sources')
        maps += values[:count]
        by_score = sorted(values, key=lambda r: (r['human_mean'], r['source_id']))
        cutoff = by_score[(2*len(by_score)-1)//3]['human_mean']
        lowmid = [r for r in values if r['human_mean'] <= cutoff]
        random.Random(223+offset).shuffle(lowmid)
        n = 34 if offset == 0 else 33
        content += lowmid[:n]
        eligibility[genre] = {'eligible': len(values), 'low_middle_cutoff': cutoff,
            'low_middle_count': len(lowmid), 'D_chosen': n, 'C_chosen': count}
    random.Random(227).shuffle(content)
    random.Random(229).shuffle(maps)
    result = {'version': 'v4_prep2', 'input_path': str(rawpath), 'input_sha256': file_sha(rawpath),
        'prior_design_sha256': file_sha(PREP1/'design.json'), 'test_question_hashes': sorted(forbidden_q),
        'maps2000': [r['source_id'] for r in maps], 'content100': [r['source_id'] for r in content],
        'source_metadata': {r['source_id']: r for r in maps+content}, 'eligibility': eligibility,
        'exclusions': dict(excluded), 'human_scores': 'mean of 16 raw grader_1/2 values on 1–5; no scorer inference',
        'C_D_overlap': len({r['source_id'] for r in maps} & {r['source_id'] for r in content}),
        'feedback_is_LLM_authored': True}
    atomic_new(path, result)
    return result


def materialize(group):
    sample = freeze()
    wanted = set(sample[group])
    missing = {s for s in wanted if not (ROOT/'essays'/(safe_id(s)+'.json')).exists()}
    if missing:
        config = load_config()
        load_environment(config)
        analyzer = BoostParagraphs(config, cache_dir=ROOT/'bareun_paragraphs')
        for index, raw in read_rows(sample['input_path']):
            source = f'train:{index}'
            if source not in missing:
                continue
            question, text = extract_question_essay(raw)
            profile = analyzer.profile(text)
            document = Document.from_profile(text, profile)
            row = {**sample['source_metadata'][source], 'split': 'train', 'question': question, 'text': text,
                'profile': profile.to_dict(), 'layout': document.snapshot(),
                'paragraphs': [{'id': p.pid, 'sentences': [{'id': u.sid, 'text': u.text} for u in p.units]}
                               for p in document.paragraphs],
                'human_scores': [raw['grader_1_scores'], raw['grader_2_scores']],
                'feedback': feedback_parts(raw['assistant']), 'feedback_origin': 'strong_LLM', 'scorer_seen': True}
            if len(row['feedback']) != 8:
                raise ValueError('Expected eight stored LLM rubric feedback sections')
            atomic_new(ROOT/'essays'/(safe_id(source)+'.json'), row)
            missing.remove(source)
            write_json(ROOT/(group+'_materialization.json'), {'remaining':len(missing),'planned':len(wanted),'at':time.time()})
    if missing:
        raise ValueError('Missing selected raw sources')
    manifest = {s: {'path': str(ROOT/'essays'/(safe_id(s)+'.json')),
        'sha256': file_sha(ROOT/'essays'/(safe_id(s)+'.json'))} for s in sample[group]}
    path = ROOT/(group+'_files.json')
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Frozen essay materialization changed')
    if not path.exists():
        atomic_new(path, manifest)
    return [read_json(manifest[s]['path']) for s in sample[group]]
