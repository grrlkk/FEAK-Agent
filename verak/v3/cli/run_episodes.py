"""Phase 5 dev smoke only. No training and no scorer train/test inputs."""
import argparse
from collections import defaultdict, Counter
from pathlib import Path
import random

from ..common import load_config, read_json, write_json, file_sha
from ..phase2 import read_jsonl, write_jsonl
from ..view_data import load_episode_examples
from ..corrupt.document import BareunBank, Document, source_document
from ..env import RevisionEnv
from ..env.analysis import ParagraphAnalyzer
from ..agent.backends import TeacherBackend, PolicyBackend
from ..agent.runner import run_episode, PROMPTS
from ..agent.metrics import summarize


def sample_rows(rows, n, seed):
    groups = defaultdict(list)
    rng = random.Random(seed)
    for row in sorted(rows, key=lambda r: r['episode_id']):
        groups[row['level']].append(row)
    if set(groups) != {'L1', 'L2', 'L3', 'L4'} or n > len(rows):
        raise ValueError('Smoke requires all four levels and enough dev episodes')
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < n:
        for level in sorted(groups):
            if groups[level] and len(selected) < n:
                selected.append(groups[level].pop())
    return selected


class LazyScorer:
    def __init__(self, config):
        self.config, self.instance = config, None

    def score(self, question, text):
        if self.instance is None:
            from ..score.kanana import KananaScorer
            self.instance = KananaScorer(self.config)
        return self.instance.score(question, text)


class LazySimilarity:
    def __init__(self, config):
        self.config, self.instance = config, None

    def __call__(self, left, right):
        if self.instance is None:
            from ..reward.similarity import SentenceSimilarity
            self.instance = SentenceSimilarity(self.config['similarity']['model'], device='cuda:1')
        return self.instance(left, right)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True, choices=['teacher', 'policy'])
    parser.add_argument('--mode', default='two_stage', choices=['two_stage', 'single'])
    parser.add_argument('--input', type=Path)
    parser.add_argument('--limit', required=True, type=int)
    parser.add_argument('--seed', type=int, default=47)
    parser.add_argument('--max-api-calls', type=int, required=True)
    parser.add_argument('--max-cost-usd', type=float, default=10)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = load_config()
    approved = config['paths']['metadata']/'corrupt/agent_dev.jsonl'
    if args.input and args.input.resolve() != approved.resolve():
        raise ValueError('Phase 5 smoke accepts only the approved agent_dev corpus')
    limit = 5 if args.backend == 'policy' else 20 if args.mode == 'two_stage' else 10
    if not 1 <= args.limit <= limit:
        raise ValueError('Requested smoke size exceeds Phase 5 authorization')
    if args.backend == 'policy' and args.max_api_calls != 0:
        raise ValueError('Local policy smoke requires --max-api-calls 0')
    args.out.mkdir(parents=True, exist_ok=True)
    rows = sample_rows(read_jsonl(approved), args.limit, args.seed)
    examples = {e.id: e for e in load_episode_examples(config, 'agent_dev')}
    if any(r['source_id'] not in examples for r in rows):
        raise ValueError('Selected source is not view-eligible agent_dev data')
    manifest = {'backend': args.backend, 'mode': args.mode, 'seed': args.seed,
        'requested': args.limit, 'corpus_sha256': file_sha(approved),
        'selected_ids': [r['episode_id'] for r in rows], 'levels': dict(Counter(r['level'] for r in rows)),
        'prompt_sha256': {p.name: file_sha(p) for p in PROMPTS.glob('*.txt')},
        'config_sha256': file_sha(Path(__file__).parents[1]/'config.yaml')}
    manifest_path = args.out/'manifest.json'
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError('Smoke manifest changed; preserve the old run and inspect')
    write_json(manifest_path, manifest)
    bank = BareunBank(config, cache_dir=config['paths']['phase5_output']/'bareun_units',
        read_cache_dirs=(config['paths']['phase3b_output']/'bareun_units', config['paths']['phase3_output']/'bareun_units'))
    sources = {r['source_id']: source_document(config, examples[r['source_id']], bank) for r in rows}
    for row in rows:
        for record in row['records']:
            if record['op'] == 'G_OFFTOPIC':
                donor = record['params']['donor']['source_id']
                source_document(config, examples[donor], bank)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    backend = (TeacherBackend(config, config['paths']['phase5_output']/'teacher_api',
        max_api_calls=args.max_api_calls, max_cost_usd=args.max_cost_usd) if args.backend == 'teacher'
        else PolicyBackend(config, args.out))
    analysis = ParagraphAnalyzer(config)
    scorer, similarity = LazyScorer(config), LazySimilarity(config)
    completed = []
    try:
        for row in rows:
            path = args.out/'episodes'/(row['episode_id'].replace(':', '_')+'.json')
            if path.exists() and read_json(path)['completed']:
                result = read_json(path)
            else:
                episode = {key: row[key] for key in ('episode_id', 'source_id', 'genre', 'level',
                    'question', 'records', 'corrupted_score', 'preexisting_spell_spans')}
                episode.update(source=sources[row['source_id']], document=Document.restore(row['corrupted_layout'], bank))
                env = RevisionEnv(config, mode=args.mode, analysis=analysis, scorer=scorer,
                                  similarity=similarity, tokenizer=tokenizer)
                result = run_episode(env, episode, backend, event_path=args.out/'events.jsonl')
                write_json(path, result)
            completed.append(result)
            write_jsonl(args.out/'episodes.jsonl', completed)
            write_json(args.out/'metrics.json', summarize(completed, args.limit))
            if args.backend == 'teacher':
                write_json(config['paths']['phase5_output']/'teacher_api/accounting.json', backend.accounting())
            print({'mode': args.mode, 'finished': len(completed), 'requested': args.limit,
                'episode_id': row['episode_id'], 'completed': result['completed'],
                'termination': result['termination'], 'cost_usd': result['cost_usd'],
                'error': result['runtime_error']}, flush=True)
            if not result['completed']:
                raise SystemExit('Smoke paused: inspect the saved episode and resume its cached calls')
    finally:
        backend.close()


if __name__ == '__main__':
    main()
