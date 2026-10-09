"""Queue only the two GLOBAL score states of immutable saved teacher episodes."""
import time

from ..common import file_sha, pair_key, read_json, write_json
from ..corrupt.qc import snapshot_text
from ..train.teacher_bulk import atomic_new
from ..insertion_boost.cpu_score import cached_gpu_score
from .config import PHASE
from .prepare import corpus


def enqueue(config):
    root = config['paths'][PHASE + '_output']
    shared = config['paths'].get('global_boost_shared_root', root)
    queue = shared.parent / 'cpu_scorer/requests'
    queue.mkdir(parents=True, exist_ok=True)
    rows = corpus(config)
    saved = root / 'score_prefetch.json'
    done = read_json(saved)['episodes'] if saved.exists() else {}
    added = 0
    for path in sorted(root.glob('attempt_*/episodes/*.json')):
        key = str(path.relative_to(root))
        digest = file_sha(path)
        if key in done:
            if done[key]['sha256'] != digest:
                raise ValueError('Prefetched teacher episode changed')
            continue
        result = read_json(path)
        if result.get('stage1_layout') is None:
            done[key] = {'sha256': digest, 'requests': [], 'reason': 'no_completed_global_stage'}
            continue
        row = rows[result['corpus_episode_id']]
        # No quality delta can arise from the identical scorer input. Keep an
        # explicit identity proof in measurement; the final GPU manifest still
        # contains this input, as required by the later GPU-reference audit.
        if row['corrupted_text'].encode('utf-8') == snapshot_text(result['stage1_layout']).encode('utf-8'):
            done[key] = {'sha256': digest, 'requests': [], 'reason': 'identical_scorer_input',
                         'pair_key': pair_key(row['question'], row['corrupted_text'])}
            continue
        requests, cached = [], []
        for text in (row['corrupted_text'], snapshot_text(result['stage1_layout'])):
            request_key = pair_key(row['question'], text)
            if cached_gpu_score(config, row['question'], text) is not None:
                cached.append(request_key)
                continue
            target = queue / (request_key + '.json')
            if not target.exists():
                try:
                    atomic_new(target, {'question': row['question'], 'text': text,
                        'requester': 'global_boost', 'requested_at': time.time(), 'gpu_used': False})
                    added += 1
                except FileExistsError:
                    pass
            previous = read_json(target)
            if previous['question'] != row['question'] or previous['text'] != text:
                raise ValueError('Shared CPU score queue identity mismatch')
            requests.append(request_key)
        done[key] = {'sha256': digest, 'requests': requests, 'saved_gpu_cache': cached}
    value = {'episodes': done, 'added_this_scan': added, 'at': time.time(),
             'states': ['corrupted', 'stage1'], 'gpu_used': False, 'paid_calls': 0}
    write_json(saved, value)
    return {'episodes_seen': len(done), 'added_this_scan': added}


def watch(config):
    root = config['paths'][PHASE + '_output']
    while True:
        result = enqueue(config)
        status = read_json(root / 'status.json') if (root / 'status.json').exists() else {}
        if status.get('stage') in {'measure', 'report', 'complete', 'failed'}:
            return result
        time.sleep(30)
