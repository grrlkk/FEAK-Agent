"""Provisional CPU measurements followed by mandatory root-owned GPU reference R."""
import importlib
import time

from ..common import file_sha, read_json, write_json
from ..reward.total import _breakdown
from ..train.teacher_bulk import atomic_new
from ..v2_ops.config import PHASE
from ..v2_ops.local import load_environment
from ..v2_ops.reward import rewards_v2
from ..v2_ops.teacher import RecoveryJudge
from .teacher import Resources, prepare, recover_global
from .collection import entries


def global_reward(config, candidate, raw, resources, score, *, identity_allowed=False):
    source = resources.source(candidate)
    corrupted = resources.restore(candidate['corrupted_layout'])
    middle = resources.restore(raw['stage1_layout'])
    recovered = recover_global(config, candidate, raw, resources)['global']
    quality = None
    if identity_allowed:
        from ..global_boost.identity import identity_quality
        quality = identity_quality(config, candidate['genre'], candidate['question'], corrupted.text, middle.text)
    if quality is not None:
        records = [r for r in recovered['per_record'] if r['level'] == 'GLOBAL']
        reward = _breakdown([r['main'] for r in records], records, quality, recovered['overedit'],
                            len(raw['actions_by_role']['global']), config['reward'])
        return reward, [], {'quality': quality, 'absolute_Q_measured': False}
    values = [score(config, candidate['question'], doc.text) for doc in (corrupted, middle)]
    reward = rewards_v2(source, corrupted, middle, candidate['records'], config=config,
        genre=candidate['genre'], q_corrupted=values[0]['mean'], q_stage1=values[1]['mean'],
        q_final=values[1]['mean'], stage1=middle,
        stage1_actions=raw['actions_by_role']['global'], stage2_actions=[],
        judge=RecoveryJudge(config, candidate), preexisting_spell_spans=candidate['preexisting_spell_spans'])['global']
    return reward, values, None


def measured_row(raw, path, reward, values, *, source, identity=None):
    return {**raw, 'raw_episode_path': str(path), 'raw_episode_sha256': file_sha(path),
        'quality_scores': values, 'reward': None, 'global_only_reward': reward,
        'reward_scope': 'GLOBAL only; KOREAN/combined unmeasured', 'score_source': source,
        'provisional': source != 'gpu_reference', 'identity_quality': identity,
        'new_gpu_calls_by_component': 0}


def run(config, *, wait_for_gpu=True):
    # The durable teacher may have started before scoring implementation updates.
    from . import cpu_score
    client = importlib.reload(cpu_score)
    root = config['paths'][PHASE + '_output']
    load_environment(config)
    design, corpus = prepare(config)
    resources = Resources(config)
    errors, done = [], 0
    for item in entries(config):
        attempt, episode_id, path = item['attempt'], item['episode_id'], item['path']
        destination = item['provisional_path']
        if destination.exists():
            if read_json(destination)['raw_episode_sha256'] != file_sha(path):
                raise ValueError('Saved raw episode changed after provisional scoring')
            continue
        raw = read_json(path)
        if raw.get('stage1_layout') is None:
            continue
        try:
            reward, values, identity = global_reward(config, corpus[episode_id], raw, resources,
                lambda cfg, q, text: client.provisional_score(cfg, q, text,
                    requester=f'insertion:{attempt}:{episode_id}'), identity_allowed=True)
            atomic_new(destination, measured_row(raw, path, reward, values,
                source='provisional_cpu_approximation', identity=identity))
            done += 1
        except Exception as exc:
            errors.append({'attempt': attempt, 'episode_id': episode_id,
                           'type': type(exc).__name__, 'message': str(exc)})
        write_json(root / 'scoring_status.json', {'stage': 'provisional_cpu', 'scored_this_run': done,
            'terminal_errors': errors, 'paid_calls': 0, 'gpu_used': False, 'training': False})
    result = {'scored_this_run': done, 'errors': [], 'terminal_errors': errors,
        'paid_calls': 0, 'gpu_used': False, 'training': False, 'reward_scope': 'global_only_provisional'}
    write_json(root / 'scoring_status.json', result)
    from .gpu_handoff import publish_manifest, finalize, publish_approval
    publish_manifest(config, errors=errors)
    if wait_for_gpu:
        write_json(root / 'scoring_status.json', {**result, 'stage': 'waiting_for_root_gpu_reference'})
        marker = client.shared_root(config) / 'gpu_rescore/insertion_complete.json'
        while not marker.exists():
            time.sleep(10)
        final = finalize(config)
        write_json(root / 'scoring_status.json', {**result, 'stage': 'waiting_for_200_essay_audit',
            'gpu_finalization': final})
        while not publish_approval(config).get('published'):
            time.sleep(10)
    return result
