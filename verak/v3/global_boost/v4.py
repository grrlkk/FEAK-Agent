"""Diverse organization teacher data; preserves the v1 policy and reward contract."""
from collections import Counter
import json
import os
from pathlib import Path
import signal
import sqlite3
import time

from ..common import file_sha, read_json, write_json
from ..phase2 import read_jsonl
from .config import PHASE, OPERATORS
from .expansion import root_for


def stop_legacy(config):
    """Drain only our old concentrated expansion; preserve every billed response."""
    root = root_for(config)
    stopped = []
    for filename, command in [('expansion_worker.json', b'expand'),
                              ('finalizer_worker.json', b'finalize')]:
        metadata = root / filename
        if not metadata.exists():
            continue
        worker = read_json(metadata)
        pid = worker['pid']
        proc = Path('/proc') / str(pid)
        if not (proc / 'cmdline').exists() or not (proc / 'cmdline').read_bytes():
            stopped.append({'pid': pid, 'already_exited': True})
            continue
        cmdline = (proc / 'cmdline').read_bytes().split(b'\0')
        if b'verak.v3.cli.global_boost' not in cmdline or command not in cmdline:
            raise RuntimeError('Refuse to signal a process outside the old GLOBAL component')
        deadline = time.monotonic() + 700
        while True:
            db = sqlite3.connect(root / 'api/ledger.sqlite', timeout=10)
            try:
                db.execute('BEGIN IMMEDIATE')
                pending = db.execute("SELECT COUNT(*) FROM calls WHERE status='pending'").fetchone()[0]
                if command == b'finalize' or not pending:
                    os.kill(pid, signal.SIGTERM)
                    for _ in range(50):
                        if not proc.exists() or (proc / 'stat').read_text().split()[2] == 'Z':
                            break
                        time.sleep(.1)
                    else:
                        raise RuntimeError('Owned legacy worker did not stop')
                    stopped.append({'pid': pid, 'command': command.decode(), 'pending_at_stop': pending})
                    break
            finally:
                db.rollback()
                db.close()
            if time.monotonic() > deadline:
                raise TimeoutError('No safe old GLOBAL worker drain window')
            time.sleep(.1)
    value = {'stage': 'v4_legacy_drained', 'workers': stopped, 'at': time.time(),
             'gpu_used': False, 'training': False, 'original_traces_preserved': True}
    write_json(root / 'v4/legacy_drain.json', value)
    write_json(root / 'expansion_status.json', value)
    return value


def compare_saved(config):
    """Same92, record-level main recovery; no model, scorer, or Bareun calls."""
    paths = config['paths']
    comparison = paths['repo'] / 'verak/v3/outputs/phase7_teacher'
    sol_root = paths['repo'] / 'verak/v3/outputs/phase7_pilot2'
    ids = read_json(comparison / 'design.json')['pilot_ids']
    assert len(ids) == len(set(ids)) == 92
    corpus_path = paths['active_corrupt'] / 'agent_train.jsonl'
    rows = {r['episode_id']: r for r in read_jsonl(corpus_path)}
    results, provenance = {}, {}
    for condition, directory in [('sol_low', sol_root), ('luna_low', comparison / 'luna_low')]:
        operators, failures = {}, []
        for episode_id in ids:
            path = directory / 'episodes' / (episode_id.replace(':', '_') + '.json')
            raw = read_json(path) if path.exists() else {}
            reward = raw.get('reward') or {}
            known = {v['record_id']: v for v in reward.get('combined', {}).get('per_record', [])}
            # A historical KOREAN failure does not erase a completed GLOBAL stage.
            global_path = comparison / 'completed_global_rewards' / (condition+'_'+episode_id.replace(':', '_')+'.json')
            if global_path.exists():
                known.update({v['record_id']: v for v in read_json(global_path)['global_only_reward']['per_record']})
            for record in rows[episode_id]['records']:
                op = record['op']
                count = operators.setdefault(op, {'records': 0, 'fully_recovered': 0, 'known': 0,
                    'unknown': 0, 'partial': 0, 'failed_episode_records': 0})
                count['records'] += 1
                value = known.get(record['record_id'])
                if value is not None:
                    count['known'] += 1
                    count['fully_recovered'] += value['main'] == 1
                    count['partial'] += 0 < value['main'] < 1
                else:
                    count['unknown'] += 1
                count['failed_episode_records'] += not raw.get('completed', False)
            if not raw.get('completed', False):
                failures.append({'episode_id': episode_id, 'error': raw.get('runtime_error'),
                    'has_saved_global_reward': global_path.exists(), 'missing_episode': not path.exists()})
            provenance[str(path)] = file_sha(path) if path.exists() else None
            if global_path.exists():
                provenance[str(global_path)] = file_sha(global_path)
        for count in operators.values():
            count['full_rate_all_records'] = count['fully_recovered']/count['records']
            count['full_rate_known'] = count['fully_recovered']/count['known'] if count['known'] else None
        results[condition] = {'operators': operators, 'failed_episodes': failures,
                              'completed_episodes': 92-len(failures)}
    triggers = {}
    for op in OPERATORS:
        a,b = (results[c]['operators'][op] for c in ('sol_low','luna_low'))
        assert a['records'] == b['records']
        difference = a['full_rate_all_records']-b['full_rate_all_records']
        triggers[op] = {'difference_percentage_points': difference*100,
                        'sol_rescue_authorized': difference >= .15-1e-12}
    value = {'cohort_size': 92, 'results': results, 'triggers': triggers,
        'failure_handling': 'Denominator includes all intended records; unknowns earn no recovery. Saved completed-GLOBAL rewards are retained after a KOREAN failure. Partial recovery is not full recovery.',
        'input_sha256': provenance, 'corpus_path': str(corpus_path), 'corpus_sha256': file_sha(corpus_path),
        'paid_calls': 0, 'gpu_calls': 0, 'bareun_calls': 0}
    write_json(root_for(config) / 'v4/sol_luna_same92.json', value)
    return value
