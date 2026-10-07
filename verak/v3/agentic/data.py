"""Offline question repair and immutable cohort/baseline manifests."""
from collections import Counter
import json
import shutil

from ..common import load_config, read_json, write_json, file_sha, sha_text
from ..phase2 import read_jsonl, write_jsonl
from ..view_data import load_episode_examples
from ..train.pilot import safe_id

PHASE = 'agentic_pilot'


def config_for(model='gpt-6-luna'):
    config = load_config()
    config['paths'][PHASE + '_output'] = config['paths']['repo'] / 'verak/v3/outputs' / PHASE
    config[PHASE] = {'model': model, 'phase_api_ceiling': 12000, 'max_cost_usd': 6., 'max_concurrent_requests': 4}
    return config


def question_audit(config):
    root = config['paths'][PHASE + '_output']
    result = {'api_calls': 0, 'question_id_definition': 'Q: followed by existing stable question_hash', 'splits': {}}
    for split in ('agent_train', 'agent_dev'):
        path = config['paths']['active_corrupt'] / (split + '.jsonl')
        examples = {e.id: e for e in load_episode_examples(config, split)}
        rows = read_jsonl(path)
        counts = Counter()
        for row in rows:
            ex = examples[row['source_id']]
            if row.get('question') and row['question'] != ex.question:
                raise ValueError('Conflicting question text: ' + row['episode_id'])
            if row.get('question_hash') and row['question_hash'] != ex.question_hash:
                raise ValueError('Conflicting question hash')
            expected = {'question': ex.question, 'question_id': 'Q:' + ex.question_hash}
            for item, label in [(row, 'essay')] + [(r, 'record') for r in row['records']]:
                counts[label + '_count'] += 1
                for key, value in expected.items():
                    if item.get(key) and item[key] != value:
                        raise ValueError('Conflicting question metadata')
                    if not item.get(key):
                        counts[label + '_' + key + '_fixed'] += 1
                        item[key] = value
            row['question_hash'] = ex.question_hash
        backup = root / 'question_audit/originals' / path.name
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, backup)
        if any(k.endswith('_fixed') for k in counts):
            write_jsonl(path, rows)
        result['splits'][split] = {**counts, 'unique_questions': len({r['question_id'] for r in rows}),
                                   'original_sha256': file_sha(backup), 'current_sha256': file_sha(path)}
    output = root / 'question_audit.json'
    if output.exists():
        # Keep first-run repair counts, but verify every later invocation.
        previous = read_json(output)
        if any(previous['splits'][s]['current_sha256'] != result['splits'][s]['current_sha256'] for s in result['splits']):
            raise ValueError('Audited corpus changed unexpectedly')
        return previous
    write_json(output, result)
    return result


def prepare(config):
    root = config['paths'][PHASE + '_output']
    if not (root / 'question_audit.json').exists():
        raise ValueError('Run the offline question audit first')
    train = {r['episode_id']: r for r in read_jsonl(config['paths']['active_corrupt'] / 'agent_train.jsonl')}
    dev = {r['episode_id']: r for r in read_jsonl(config['paths']['active_corrupt'] / 'agent_dev.jsonl')}
    examples = {e.id: e for split in ('agent_train', 'agent_dev') for e in load_episode_examples(config, split)}
    pilot = read_json(config['paths']['phase7_teacher_output'] / 'design.json')['pilot_ids']
    real = read_json(config['paths']['phase6_output'] / 'design.json')['real_ids']
    quality = sorted(i for i, r in dev.items() if any(x['op'] == 'G_OFFTOPIC' for x in r['records']))
    sources = sorted({dev[i]['source_id'] for i in quality})
    assert len(pilot) == 92 and len(real) == 30 and set(pilot) <= train.keys()
    baseline = {i: config['paths']['phase7_teacher_output'] / 'luna_low/episodes' / (safe_id(i) + '.json') for i in pilot}
    baseline.update({i: config['paths']['repo'] / 'verak/v3/outputs/observation_test/current/episodes' / (safe_id(i) + '.json') for i in real})
    design = {'corrupted_ids': pilot, 'real_ids': real, 'quality_corrupted_ids': quality, 'quality_source_ids': sources,
              'quality_population': 'all active agent_dev G_OFFTOPIC essays and their distinct source essays',
              'baseline_files': {i: {'path': str(p), 'sha256': file_sha(p)} for i, p in baseline.items()},
              'corpus_sha256': {s: file_sha(config['paths']['active_corrupt'] / (s + '.jsonl')) for s in ('agent_train', 'agent_dev')},
              'model': 'gpt-6-luna', 'reasoning': 'low', 'context': 8192, 'output': 1024, 'budget_usd': 6.,
              'held_out': False, 'priority': ['question_audit', 'graphs_quality', 'corrupted', 'real', 'markers', 'relevance'],
              'completion_definition': 'Orchestrator FINISH reached and reward saved for corrupted essays',
              'decision_rule': 'both cohorts completion >= .90; paired 95% CI upper bound for combined R difference >= 0',
              'low_tool_call_rate': .05, 'bootstrap_seed': 89, 'bootstrap_samples': 10000}
    target = root / 'design.json'
    if target.exists() and read_json(target) != design:
        raise ValueError('Frozen design/baseline changed')
    if not target.exists():
        write_json(target, design)
    return design, train, dev, examples
