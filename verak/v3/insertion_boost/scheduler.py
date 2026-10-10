"""File-only CPU queue priority for the GLOBAL data feeding the next RFT phase."""
import os
import time

from ..common import read_json, write_json
from .cpu_score import shared_root


def run(config):
    shared = shared_root(config)
    root = shared / 'cpu_scorer'
    seen = {}
    previous = None
    while not (shared / 'report_complete.json').exists() and not (root / 'stop.json').exists():
        priority = []
        for path in (root / 'requests').glob('*.json'):
            if (root / 'responses' / path.name).exists():
                continue
            if path.name not in seen:
                seen[path.name] = read_json(path).get('requester', '')
            if seen[path.name].startswith('global_boost'):
                priority.append(path.stem)
        policy = {'stage': 'provisional_bulk_and_200_essay_audit',
            'priority': 'GLOBAL additions required for the upcoming RFT training slot',
            'quality_use': 'provisional_only', 'final_selection_source': 'gpu_reference'}
        if priority:
            policy['allowed_keys'] = sorted(priority)
        if policy != previous:
            write_json(root / 'work_policy.json', policy)
            write_json(root / 'scheduler_status.json', {'pid': os.getpid(), 'stage': 'running',
                'priority_pending': len(priority), 'at': time.time(), 'gpu_used': False, 'paid_calls': 0})
            previous = policy
        time.sleep(5)
    write_json(root / 'scheduler_status.json', {'pid': os.getpid(), 'stage': 'stopped',
        'at': time.time(), 'gpu_used': False, 'paid_calls': 0})
    return {'stopped': True, 'gpu_used': False, 'paid_calls': 0}
