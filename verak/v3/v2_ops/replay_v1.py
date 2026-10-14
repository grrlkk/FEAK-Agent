"""Replay all 92 archived Luna pilot trajectories without GPU, scorer, or API calls."""
from copy import deepcopy
import json
from pathlib import Path

from ..agent.runner import run_episode
from ..common import file_sha, load_config, read_json, sha_text, write_json
from ..corrupt.document import BareunBank, Document, source_document
from ..env import RevisionEnv
from ..env.analysis import ParagraphAnalyzer
from ..phase2 import restore_profile
from ..reward.total import rewards
from ..view_data import load_episode_examples


class NoLiveAnalyzer:
    def profile(self, text):
        raise AssertionError('v1 replay must use saved Bareun observations only')


class SavedParagraphs(ParagraphAnalyzer):
    def __init__(self, config, roots):
        self.config, self.roots = config, roots
        self.calls, self.hits = [], 0

    def profile(self, text, neighbors=()):
        key = sha_text(json.dumps([text, sha_text(json.dumps(neighbors, ensure_ascii=False))], ensure_ascii=False))
        path = next((r / (key + '.json') for r in self.roots if (r / (key + '.json')).exists()), None)
        if path is None:
            raise AssertionError('Missing immutable paragraph cache: ' + key)
        value = read_json(path)
        if value['text'] != text:
            raise AssertionError('Saved analyzer input differs')
        self.hits += 1
        return restore_profile(value['profile'])


class SavedScore:
    def __init__(self, row):
        self.values = {entry['text_hash']: entry['result'] for entry in row['score_calls']}
        self.hits = 0

    def score(self, question, text):
        key = sha_text(text)
        if key not in self.values:
            raise AssertionError('Replay requested an unsaved scorer input')
        self.hits += 1
        return deepcopy(self.values[key])


class SavedTeacher:
    name = 'teacher'
    context_limit = 8192
    generation_reserve = 1024

    def __init__(self, row):
        self.row, self.model, self.index, self.records = row, row['model'], 0, []

    def generate(self, messages, *, episode_id, role, turn):
        if self.index == len(self.row['calls']):
            if self.row['runtime_error']:
                raise RuntimeError(self.row['runtime_error']['message'])
            raise AssertionError('Unexpected additional teacher turn')
        response = self.row['calls'][self.index]
        if response['messages'] != messages:
            raise AssertionError(f'v1 inference context changed at {self.row["corpus_episode_id"]}/{role}/{turn}')
        if response['role'] != role or response['turn'] != turn:
            raise AssertionError('v1 teacher turn routing changed')
        self.index += 1
        self.records.append(response)
        return deepcopy(response)


def replay(config=None, *, output=None):
    config = deepcopy(config) if config is not None else load_config()
    config['method_version'] = 'v1'
    root = config['paths']['phase7_teacher_output']
    design = read_json(root / 'design.json')
    if len(design['pilot_ids']) != 92:
        raise AssertionError('Expected the exact saved 92-essay pilot')
    with (config['paths']['active_corrupt'] / 'agent_train.jsonl').open() as stream:
        corpus = {r['episode_id']: r for r in map(json.loads, stream)}
    examples = {e.id: e for e in load_episode_examples(config, 'agent_train')}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    cache_roots = [config['paths'][key] for key in ('phase7_teacher_output', 'phase7_pilot2_output',
        'phase7_pilot_output', 'phase6_output', 'phase5_output', 'phase3b_output', 'phase3_output')]
    analysis = SavedParagraphs(config, [r / 'bareun_paragraphs' for r in cache_roots])
    partial = read_json(root / 'completed_global_evaluation.json')['results']
    details, calls = [], 0
    for episode_id in design['pilot_ids']:
        path = root / 'luna_low/episodes' / (episode_id.replace(':', '_') + '.json')
        archived, row = read_json(path), corpus[episode_id]
        bank = BareunBank(config, analyzer=NoLiveAnalyzer(), cache_dir=root / 'bareun_units',
            read_cache_dirs=[r / 'bareun_units' for r in cache_roots[1:]])
        source = source_document(config, examples[row['source_id']], bank)
        for record in row['records']:
            if record['op'] == 'G_OFFTOPIC':
                source_document(config, examples[record['params']['donor']['source_id']], bank)
        episode = {key: row[key] for key in ('episode_id', 'source_id', 'question', 'genre', 'level',
            'records', 'corrupted_score', 'preexisting_spell_spans')}
        episode.update(source=source, document=Document.restore(row['corrupted_layout'], bank))
        backend = SavedTeacher(archived)
        env = RevisionEnv(config, mode='two_stage', analysis=analysis,
            scorer=SavedScore(archived), tokenizer=tokenizer)
        result = run_episode(env, episode, backend)
        for key in ('completed', 'runtime_error', 'initial_layout', 'stage1_layout', 'final_layout',
                    'final_text', 'termination', 'steps', 'checks', 'reward'):
            if result[key] != archived[key]:
                raise AssertionError(f'v1 replay differs: {episode_id}, {key}')
        for role in ('global', 'korean'):
            keys = ('action', 'args', 'valid', 'error_code', 'before_hash', 'after_hash', 'changed_sids')
            project = lambda values: [{k: a[k] for k in keys} for a in values]
            if project(result['actions_by_role'][role]) != project(archived['actions_by_role'][role]):
                raise AssertionError('v1 action behavior changed: ' + episode_id)
        if backend.index != len(archived['calls']):
            raise AssertionError('Not all saved teacher calls were replayed')
        global_only = None
        if not archived['completed']:
            expected = partial['luna_low:' + episode_id]['global_only_reward']
            q_middle = expected['quality']['after']
            global_only = rewards(source, env.corrupted, env.stage1, row['records'], genre=row['genre'],
                q_corrupted=expected['quality']['before'], q_stage1=q_middle, q_final=q_middle,
                config=config['reward'], mode='two_stage', stage1=env.stage1,
                stage1_actions=env.actions['global'], stage2_actions=[],
                preexisting_spell_spans=row.get('preexisting_spell_spans', ()))['global']
            if global_only != expected:
                raise AssertionError('Saved completed GLOBAL-stage reward differs')
        calls += backend.index
        details.append({'episode_id': episode_id, 'source_sha256': file_sha(path),
            'completed': archived['completed'], 'reward_exactly_equal': True,
            'global_only_reward_checked': global_only is not None, 'teacher_turns_replayed': backend.index})
    result = {'passed': True, 'version': 'v1', 'trajectories': len(details),
        'completed_full_reward_matches': sum(r['completed'] for r in details),
        'historical_failure_preserved': sum(not r['completed'] for r in details),
        'teacher_turns_replayed': calls, 'paragraph_cache_hits': analysis.hits,
        'comparison': 'exact Python equality for all reward fields and recorded action outcomes',
        'paid_calls': 0, 'new_bareun_calls': 0, 'new_scorer_calls': 0, 'gpu_used': False,
        'details': details}
    if output:
        write_json(Path(output), result)
    return result


if __name__ == '__main__':
    config = load_config()
    result = replay(config, output=config['paths']['repo'] / 'verak/v3/outputs/v2_ops/v1_replay.json')
    print(json.dumps({k: v for k, v in result.items() if k != 'details'}, ensure_ascii=False))
