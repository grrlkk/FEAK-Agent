"""Preserve the initial v3 run and carry its spending into the corrected run."""
from collections import Counter
from pathlib import Path
import shutil
import sqlite3

from ..common import file_sha, read_json, write_json
from .data import PHASE, prepare
from . import graph


def initialize(config):
    """Copy a closed ledger once; never reset a corrected run's paid history."""
    if not config[PHASE].get('relevance_protection'):
        return None
    root = config['paths'][PHASE + '_output']
    previous = root.parent / 'agentic_pilot_v3'
    marker = root / 'initial_run.json'
    ledger = previous / 'api/ledger.sqlite'
    if marker.exists():
        value = read_json(marker)
        if file_sha(ledger) != value['ledger_sha256']:
            raise ValueError('The superseded v3 ledger changed after budget carryover')
        if not (root / 'api/ledger.sqlite').exists():
            raise ValueError('Corrected ledger missing; refuse to reset spending')
        return value
    if (root / 'api').exists():
        raise ValueError('Partial or unrecognized budget migration; inspect before dispatch')
    with sqlite3.connect('file:' + str(ledger) + '?mode=ro', uri=True) as old:
        rows = old.execute('SELECT id,stage,status,reserved,confirmed FROM calls ORDER BY id').fetchall()
        if any(r[2] == 'pending' for r in rows):
            raise ValueError('Stop and drain the superseded pilot before migration')
        confirmed, reserved = sum(r[4] for r in rows), sum(r[3] for r in rows)
        if confirmed + reserved >= config[PHASE]['max_cost_usd']:
            raise ValueError('No combined A budget remains')
        (root / 'api/requests').mkdir(parents=True)
        with sqlite3.connect(root / 'api/ledger.sqlite') as target:
            old.backup(target)
    requests = {}
    for path in sorted((previous / 'api/requests').glob('*.json')):
        shutil.copy2(path, root / 'api/requests' / path.name)
        requests[path.name] = file_sha(path)
    episodes = [read_json(p) for p in (previous / 'episodes').glob('*.json')]
    value = {'output': str(previous), 'ledger_path': str(ledger), 'ledger_sha256': file_sha(ledger),
             'calls': sum(r[2] != 'blocked_before_send' for r in rows),
             'last_call_id': max((r[0] for r in rows), default=0),
             'confirmed_usd': confirmed, 'reserved_usd': reserved, 'pending': 0,
             'combined_cap_usd': config[PHASE]['max_cost_usd'],
             'cohorts': {cohort: {'saved': sum(r['cohort'] == cohort for r in episodes),
                 'completed': sum(r['cohort'] == cohort and r['completed'] for r in episodes)}
                 for cohort in ('corrupted', 'real')},
             'request_sha256': requests,
             'reason': 'User stopped initial v3 and requested relevance protection plus a fresh pilot rerun.',
             'budget_method': 'Closed original ledger copied once; original charges and uncertainty remain '
                 'in the corrected ledger under one atomic $7 reservation cap. Original outputs stay in place.',
             'rerun_method': 'Fresh pilot_protected_* request namespace; graph extraction requests reused.'}
    write_json(marker, value)
    return value


def reuse_graphs(config):
    """Derive protection from the saved raw pair, without analyzer or model calls."""
    carried = initialize(config)
    if carried is None:
        raise ValueError('Graph protection requires the corrected v3 configuration')
    design, train, dev, examples = prepare(config)
    root, previous = config['paths'][PHASE + '_output'], Path(carried['output'])
    counts = Counter()
    provenance = []
    ids = set(design['quality_corrupted_ids'] + design['quality_source_ids'] +
              design['corrupted_ids'] + design['real_ids'])
    for path in sorted((previous / 'graphs').glob('*.json')):
        value = read_json(path)
        if value['id'] not in ids:
            continue
        source_sha = file_sha(path)
        if value['status'] == 'completed':
            left, right = [r['discourse'] for r in value['runs']]
            discourse, intersections = graph.intersection(left, right)
            if discourse != value['discourse'] or intersections != value['intersection_counts']:
                raise ValueError('Original graph is not the exact saved intersection')
            discourse['relevance_protection'] = graph.relevance_protection(left, right, value['sentence_ids'].values())
            graph.validate(discourse, value['sentence_ids'].values(), value['paragraph_ids'].values(), allow_protection=True)
            value['discourse'] = discourse
            graph.apply_protection(value['graph'], discourse)
            counts['valid_graphs'] += 1
            counts['protected_sentences'] += len(discourse['relevance_protection']['protected_ids'])
            overrides = len(discourse['relevance_protection']['overridden_off_topic'])
            counts['overridden_sentences'] += overrides
            counts['graphs_with_overrides'] += bool(overrides)
        else:
            counts['preserved_invalid_graphs'] += 1
        value['reused_from'] = str(path)
        value['reused_sha256'] = source_sha
        target = root / 'graphs' / path.name
        if target.exists() and read_json(target) != value:
            raise ValueError('Corrected graph changed after derivation')
        if not target.exists():
            write_json(target, value)
        provenance.append({'id': value['id'], 'source': str(path), 'source_sha256': source_sha,
                           'derived': str(target), 'derived_sha256': file_sha(target)})
    if {r['id'] for r in provenance} != ids:
        raise ValueError('A required saved graph is missing; refuse new extraction silently')
    write_json(root / 'graph_reuse.json', {'new_api_calls': 0, 'counts': dict(counts), 'graphs': provenance})
    from .experiment import quality
    quality(config, design, dev)
    write_json(root / 'graphs_status.json', {'requested': len(ids), 'new_api_calls': 0,
               'reused_graphs': len(provenance), 'errors': [], 'counts': dict(counts)})
    return dict(counts)
