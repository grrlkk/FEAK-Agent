"""Post-hoc reward changes at saved delegation boundaries; no teacher calls."""
from collections import Counter
from pathlib import Path
from statistics import mean
import json

from ..common import file_sha, read_json, sha_text, write_json
from ..corrupt.document import BareunBank, Document, source_document
from ..eval.resources import Resources
from ..reward.total import rewards
from ..train.pilot import safe_id
from .accounting import FREE_ACTIONS, audit_reward
from .audit import audit_snapshots
from .data import PHASE, prepare
from .environment import AgenticEnv
from .preservation import CachedParagraphs, _paragraph_texts


class _NoLoopScorer:
    def score(self, question, text):
        raise ValueError('Saved corrupted Q must satisfy the only permitted in-loop SCORE')


class _RecordedBank(BareunBank):
    """Initial units must come from a frozen source profile or an existing cache."""
    def __init__(self, config, roots, input_files):
        super().__init__(config, cache_dir=config['paths'][PHASE + '_output'] / 'redelegation_cache/bareun_units',
                         read_cache_dirs=tuple(roots))
        self.input_files = input_files

    def tokens(self, text):
        if text not in self.memory:
            filename = sha_text(text) + '.json'
            path = next((root / filename for root in (self.cache, *self.read_cache_dirs) if (root / filename).exists()), None)
            if path is None:
                raise ValueError('Initial unit lacks its saved Bareun cache; refusing new tokenization')
            self.input_files[str(path)] = file_sha(path)
        return super().tokens(text)


def replay(config, row, corpus, examples, analysis, input_files):
    """Rebuild token state exactly and require all saved tool outcomes/layouts."""
    root = config['paths'][PHASE + '_output']
    roots = [root / 'bareun_units', root.parent / 'agentic_pilot_v3/bareun_units'] + [
        config['paths'][key] / 'bareun_units' for key in ('phase7_pilot2_output', 'phase7_pilot_output',
        'phase6_output', 'phase5_output', 'phase3b_output', 'phase3_output')]
    bank = _RecordedBank(config, roots, input_files)
    def source_for(source_id):
        example = examples[source_id]
        path = config['paths']['phase2_output'] / 'bareun_profiles' / f'{example.source_line}.json'
        input_files[str(path)] = file_sha(path)
        return source_document(config, example, bank)
    source = source_for(corpus['source_id'])
    for record in corpus['records']:
        if record['op'] == 'G_OFFTOPIC':
            source_for(record['params']['donor']['source_id'])
    corrupted = Document.restore(corpus['corrupted_layout'], bank)
    graph_path = root / 'graphs' / (safe_id(row['episode_id']) + '.json')
    saved_graph = read_json(graph_path)
    input_files[str(graph_path)] = file_sha(graph_path)
    if corrupted.snapshot() != saved_graph['input_layout'] or sha_text(corrupted.text) != saved_graph['input_hash']:
        raise ValueError('Initial corruption and frozen graph input do not match')
    episode = {**corpus, 'document': corrupted, 'source': source}
    env = AgenticEnv(episode, saved_graph['discourse'], analysis=analysis, scorer=_NoLoopScorer(), version=3)
    before = {s['before_action_count']: s for s in row['sequences']}
    after = {s['after_action_count']: s for s in row['sequences'] if s.get('after_action_count') is not None}
    snapshots = {}
    for index, action in enumerate(row['actions']):
        if index in before:
            seq = before[index]
            if env.document.snapshot() != seq['before_layout']:
                raise ValueError(f"Before-layout replay mismatch at delegation {seq['delegation']}")
            snapshots[(seq['delegation'], 'before')] = env.document.clone()
        actual = env.step({'action': action['action'], 'args': action['args']}, action['role'])
        fields = ('valid', 'before_hash', 'after_hash', 'before_rows', 'after_rows', 'changed_sids')
        if any(actual[key] != action[key] for key in fields):
            raise ValueError(f'Saved action replay mismatch at action {index}')
        if action['valid'] and json.dumps(actual['result'], sort_keys=True) != json.dumps(action['result'], sort_keys=True):
            raise ValueError(f'Saved factual tool result mismatch at action {index}')
        if action['valid'] and action['action'] == 'DELEGATE':
            env.begin_editor(action['args'])
        if index + 1 in after:
            seq = after[index + 1]
            if env.document.snapshot() != seq['after_layout']:
                raise ValueError(f"After-layout replay mismatch at delegation {seq['delegation']}")
            snapshots[(seq['delegation'], 'after')] = env.document.clone()
            if seq['terminal'] == 'AUTO_REPORT':
                env.latest_report = seq['report']
            env.last_result = {'editor_return': seq['report']}
    if env.document.snapshot() != row['final_layout'] or env.document.text != row['final_text']:
        raise ValueError('Final-layout replay mismatch')
    return source, corrupted, env.document, snapshots


def snapshot_reward(config, corpus, source, corrupted, document, actions, q_final):
    charged = [action for action in actions if action['action'] not in FREE_ACTIONS]
    return rewards(source, corrupted, document, corpus['records'], genre=corpus['genre'],
        q_corrupted=corpus['corrupted_score']['mean'], q_final=q_final, config=config['reward'],
        actions=charged, tau=config['similarity']['tau'],
        preexisting_spell_spans=corpus.get('preexisting_spell_spans', ()))['combined']


def summarize_cases(cases):
    available = [case for case in cases if case['status'] == 'available']
    completed = [case for case in available if case['episode_completed']]
    def summary(selected, eligible):
        return {'eligible_redelegations': len(eligible), 'evaluated': len(selected),
            'episodes': len({case['episode_id'] for case in selected}),
            'unavailable': len(eligible) - len(selected),
            'mean_delta': {key: mean(case['delta'][key] for case in selected) if selected else None
                           for key in ('R', 'R_rec', 'R_q', 'R_over', 'R_step')},
            'reward_increased': sum(case['delta']['R'] > 1e-9 for case in selected),
            'reward_unchanged': sum(abs(case['delta']['R']) <= 1e-9 for case in selected),
            'reward_decreased': sum(case['delta']['R'] < -1e-9 for case in selected)}
    return {'all_returned_or_interrupted': summary(available, cases),
        'completed_episodes_only': summary(completed, [case for case in cases if case['episode_completed']]),
        'by_level': {level: summary([c for c in available if c['level'] == level], [c for c in cases if c['level'] == level])
                     for level in ('L1', 'L2', 'L3', 'L4')},
        'by_editor': {role: summary([c for c in available if c['role'] == role], [c for c in cases if c['role'] == role])
                     for role in ('composition', 'cohesion')},
        'by_editor_use': {kind: summary([c for c in available if c['editor_use'] == kind], [c for c in cases if c['editor_use'] == kind])
                          for kind in ('repeated_same_editor', 'first_use_of_other_editor')}}


def redelegation_rewards(config, *, allow_local_bareun=False):
    """Run only after policy generation; local scorer is lazy and never a teacher."""
    if not config[PHASE].get('relevance_protection'):
        raise ValueError('Re-delegation snapshots require the corrected protected v3 run')
    design, train, _, examples = prepare(config)
    root = config['paths'][PHASE + '_output']
    input_files = {str(config['paths']['active_corrupt'] / 'agent_train.jsonl'):
                   file_sha(config['paths']['active_corrupt'] / 'agent_train.jsonl')}
    rows, texts, saved_ids = [], set(), []
    for eid in design['corrupted_ids']:
        path = root / 'episodes' / (safe_id(eid) + '.json')
        if not path.exists():
            continue
        saved_ids.append(eid)
        row = read_json(path)
        if not any(s.get('audit_before_redelegation') for s in row.get('sequences', [])):
            continue
        input_files[str(path)] = file_sha(path)
        rows.append(row)
        for layout in [row['final_layout'], train[eid]['corrupted_layout']] + [s[key] for s in row['sequences']
                for key in ('before_layout', 'after_layout') if s.get(key) is not None]:
            texts.update(_paragraph_texts(layout))
    analysis = CachedParagraphs(config, [root / 'bareun_paragraphs', root.parent / 'agentic_pilot_v3/bareun_paragraphs',
        config['paths']['phase7_teacher_output'] / 'bareun_paragraphs', root.parent / 'observation_test/bareun_paragraphs'],
        texts, allow_local_bareun=allow_local_bareun)
    resources, score_requests = Resources(config, examples, output_key=PHASE + '_output'), []
    cases, references = [], []
    try:
        for row in rows:
            corpus = train[row['episode_id']]
            reference = {'episode_id': row['episode_id'], 'status': 'unavailable', 'passed': False}
            failure, documents = None, None
            try:
                errors = []
                audit_snapshots(row, lambda passed, label, item: errors.append(label) if not passed else None)
                if errors:
                    raise ValueError('Snapshot provenance failed: ' + ', '.join(errors))
                source, corrupted, final, documents = replay(config, row, corpus, examples, analysis, input_files)
                if row.get('reward'):
                    saved = audit_reward(row)['reward']['combined']
                    reconstructed = snapshot_reward(config, corpus, source, corrupted, final, row['actions'], saved['quality']['after'])
                    differences = {key: reconstructed[key] - saved[key] for key in ('R', 'R_rec', 'R_q', 'R_over', 'R_step')}
                    passed = all(abs(value) < 1e-7 for value in differences.values())
                    reference.update(status='available', passed=passed, differences=differences,
                                     saved_Q=saved['quality']['after'], policy_actions=len(row['actions']))
                    if not passed:
                        raise ValueError('Reconstructed final combined reward differs from the saved reward')
                else:
                    reference.update(status='saved_reward_unavailable', passed=None,
                                     replay_passed=True, policy_actions=len(row['actions']))
            except (ValueError, KeyError, OSError, RuntimeError) as exc:
                failure = {'type': type(exc).__name__, 'message': str(exc)}
                reference['error'] = failure
            references.append(reference)
            known_scores = {corrupted.text: {'mean': corpus['corrupted_score']['mean'], 'origin': 'saved_corrupted_score'}} if documents else {}
            if documents and row.get('reward'):
                known_scores[final.text] = {'mean': row['reward']['combined']['quality']['after'], 'origin': 'saved_final_score'}
            seen_roles = set()
            for sequence in row['sequences']:
                kind = 'repeated_same_editor' if sequence['role'] in seen_roles else 'first_use_of_other_editor'
                seen_roles.add(sequence['role'])
                if not sequence.get('audit_before_redelegation'):
                    continue
                case = {'episode_id': row['episode_id'], 'episode_completed': row['completed'], 'level': corpus['level'],
                        'delegation': sequence['delegation'], 'role': sequence['role'], 'editor_use': kind,
                        'terminal': sequence['terminal'], 'audit_action_indices': sequence['redelegation_audit_action_indices'],
                        'before_action_count': sequence['before_action_count'], 'after_action_count': sequence['after_action_count'],
                        'status': 'unavailable'}
                try:
                    if failure:
                        raise ValueError('Episode reconstruction unavailable: ' + failure['message'])
                    if sequence.get('after_layout') is None:
                        raise ValueError('Editor turn interrupted before a saved return snapshot')
                    for endpoint in ('before', 'after'):
                        document = documents[(sequence['delegation'], endpoint)]
                        if document.text not in known_scores:
                            score = resources.score(corpus['question'], document.text)
                            known_scores[document.text] = {**score, 'origin': 'frozen_local_scorer'}
                            score_requests.append({'episode_id': row['episode_id'], 'text_sha256': sha_text(document.text),
                                                   'score': score})
                        score = known_scores[document.text]
                        count = sequence[endpoint + '_action_count']
                        value = snapshot_reward(config, corpus, source, corrupted, document, row['actions'][:count], score['mean'])
                        case[endpoint] = {'reward': value, 'score': score, 'text_sha256': sha_text(document.text),
                                          'layout': sequence[endpoint + '_layout']}
                    case.update(status='available', delta={key: case['after']['reward'][key] - case['before']['reward'][key]
                                for key in ('R', 'R_rec', 'R_q', 'R_over', 'R_step')})
                    case['weighted_component_delta'] = {key: case['after']['reward']['weighted_components'][key] -
                        case['before']['reward']['weighted_components'][key] for key in ('recovery', 'quality', 'overedit', 'steps')}
                except (ValueError, KeyError, OSError, RuntimeError) as exc:
                    case['error'] = {'type': type(exc).__name__, 'message': str(exc)}
                cases.append(case)
                write_json(root / 'redelegation_cases' / (safe_id(row['episode_id']) + f"_{sequence['delegation']}.json"), case)
            print({'redelegation_episode': row['episode_id'], 'reconstruction': reference['status'],
                   'passed': reference['passed']}, flush=True)
        scorer_fingerprint = resources.scorer.fingerprint if resources.scorer is not None else None
    finally:
        resources.close()
    for item in analysis.used:
        input_files[item['path']] = item['sha256']
    reference_failures = [r for r in references if r['passed'] is False]
    result = {'method': {'scope': 'corrupted essays only; no combined reward imputed for unlabelled real essays',
        'trigger': 'second or later accepted delegation with valid AUDIT after the previous editor return',
        'endpoints': 'saved exact stable-ID layouts; before excludes current DELEGATE, after includes it and editor actions/REPORT',
        'reconstruction': 'replay frozen tool transitions using saved Bareun profiles; exact action results and boundary layouts required',
        'reward': 'unchanged combined recovery + quality - overedit - charged step terms; existing read-tool exemptions',
        'interpretation': 'descriptive within-episode change, not a causal effect; unavailable snapshots/rewards are never zero-filled',
        'helper_sha256': file_sha(__file__), 'runtime_manifest_sha256': file_sha(root / 'pilot_runtime.json'),
        'reward_module_sha256': {name: file_sha(Path(__file__).parents[1] / 'reward' / name)
                                 for name in ('total.py', 'recovery.py', 'overedit.py')},
        'reward_config': config['reward'], 'similarity_tau': config['similarity']['tau'],
        'scorer_fingerprint': scorer_fingerprint},
        'summary': {**summarize_cases(cases), 'coverage': {
            'intended_corrupted_episodes': len(design['corrupted_ids']), 'saved_corrupted_episodes': len(saved_ids),
            'unattempted_ids': sorted(set(design['corrupted_ids']) - set(saved_ids)),
            'episodes_with_redelegation_after_audit': len(rows), 'eligible_redelegations': len(cases)}},
        'cases': cases, 'input_files': input_files,
        'reference_checks': references,
        'reference_checks_passed': not reference_failures if any(r['status'] == 'available' for r in references) else None,
        'model_api_calls': 0, 'local_scorer_calls': len(score_requests),
        'local_scorer_cache_hits': sum(bool(r['score']['cache_hit']) for r in score_requests), 'score_requests': score_requests,
        'local_bareun_profile_calls': len(analysis.calls), 'paragraph_cache_hits': analysis.hits,
        'paragraph_reuse': dict(Counter(r['reuse'] for r in analysis.used))}
    write_json(root / 'redelegation_rewards.json', result)
    if reference_failures:
        raise ValueError('Saved reward/action reconstruction failed; inspect redelegation_rewards.json')
    return result
