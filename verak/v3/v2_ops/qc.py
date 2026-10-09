"""Resumable Sol labeling and instance QC. No scorer or GPU imports."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json

from feak_tc.runtime.openai import APIUnavailable, CallBudgetExceeded
from ..common import file_sha, read_json, write_json
from ..train.teacher_bulk import atomic_new, collection_lock
from .local import load_environment
from .candidates import build
from .config import PHASE, OPERATORS, require_v2
from .data import source_pools
from .judges import contract_key, label_contract, qc_contract, passes_qc, validate
from .paid import V2API


def batches(values, size):
    return [values[i:i+size] for i in range(0, len(values), size)]


def request(api, root, messages, schema, stage, *, effort='high'):
    key = contract_key(messages, schema)
    path = root / 'judge_batches' / stage / (key + '.json')
    if path.exists():
        result = read_json(path)
        if result['status'] != 'completed':
            raise RuntimeError('Preserved failed judge batch; no new sampling: ' + key)
        validate(result['verdicts'], schema)
        return result
    try:
        # Changing future labeling effort must not resample an earlier batch.
        with api.db() as db:
            prior = db.execute("SELECT status,path FROM calls WHERE stage=? AND item_id=? "
                               "AND status!='blocked_before_send' ORDER BY id", (stage, key)).fetchall()
        complete = next((p for s, p in prior if s == 'completed'), None)
        if prior and complete is None:
            raise RuntimeError('Preserved prior sent batch outcome; no repeated label/QC sampling')
        response = read_json(complete) if complete else api.request(messages, schema=schema, stage=stage, item_id=key,
                               effort=effort, max_output=4096)
        value = json.loads(response['raw'])
        validate(value, schema)
        result = {'status': 'completed', 'phase_call': response['phase_call'],
                  'contract_sha256': key, 'verdicts': value}
        atomic_new(path, result)
        return result
    except CallBudgetExceeded:
        raise
    except Exception as exc:
        atomic_new(path, {'status': 'error', 'error': type(exc).__name__ + ': ' + str(exc),
                          'contract_sha256': key})
        raise


def planned_labels(config):
    root = config['paths'][PHASE + '_output']
    plan = read_json(root / 'source_plan.json')
    pools, _ = source_pools(config)
    out = []
    for split in ('agent_train', 'agent_dev'):
        sources = {e.id: (e, source) for e, source, _ in pools[split]}
        for entry in plan['plans']['G_DEL_LINK'][split]:
            example, source = sources[entry['source_id']]
            if entry['source_hash'] != example.essay_hash:
                raise ValueError('Frozen labeling source changed')
            out.append(({**entry, 'split': split, 'question': example.question}, source))
    return out


def dispatch_batches(values, worker, concurrency):
    """Bound dispatch and join already-sent requests when a cap stops scheduling."""
    iterator, stopped = iter(values), None
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        active = set()
        while True:
            while len(active) < concurrency and stopped is None:
                batch = next(iterator, None)
                if batch is None:
                    break
                active.add(pool.submit(worker, batch))
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                active.remove(future)
                try:
                    future.result()
                except Exception as exc:
                    stopped = exc
        if stopped is not None:
            raise stopped


def label(config, api):
    root = config['paths'][PHASE + '_output']
    # Batch membership is fixed before filtering cache hits, so partial writes replay exactly.
    def one(batch):
        paths = [root / 'labels' / (row['source_id'].replace(':', '_') + '.json') for row, _ in batch]
        if all(p.exists() for p in paths):
            return
        messages, schema = label_contract(batch)
        try:
            result = request(api, root, messages, schema, 'v2_labels', effort=config[PHASE]['label_effort'])
        except (APIUnavailable, CallBudgetExceeded):
            raise
        except Exception:
            return  # Failed batch remains unknown, with its immutable error artifact.
        for (row, _), path in zip(batch, paths):
            if path.exists():
                continue
            atomic_new(path, {'source_id': row['source_id'], 'source_hash': row['source_hash'],
                'contract_sha256': result['contract_sha256'], 'phase_call': result['phase_call'],
                'labels': [{'sid': sid, 'label': value} for sid, value in result['verdicts'][row['source_id']].items()]})
        print(json.dumps({'stage': 'v2_labels', 'batch': result['contract_sha256'],
                          'budget': api.accounting()}), flush=True)
    dispatch_batches(batches(planned_labels(config), config[PHASE]['label_batch_size']), one,
                     config[PHASE]['max_concurrent_requests'])


def candidate_paths(config):
    root = config['paths'][PHASE + '_output']
    grouped = {op: sorted((root / 'candidates' / op).glob('*/*.json')) for op in OPERATORS}
    return [grouped[op][i] for i in range(max(map(len, grouped.values()), default=0))
            for op in OPERATORS if i < len(grouped[op])]


def judge(config, api):
    root = config['paths'][PHASE + '_output']
    manifest_path = root / 'qc_plan.json'
    entries = [{'path': str(p), 'sha256': file_sha(p)} for p in candidate_paths(config)]
    manifest = {'source_plan_sha256': file_sha(root / 'source_plan.json'), 'candidates': entries,
                'batch_size': config[PHASE]['qc_batch_size']}
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError('QC candidates or batch contracts changed after freezing')
    write_json(manifest_path, manifest)
    def one(batch):
        rows = [read_json(entry['path']) for entry in batch]
        paths = [root / 'qc' / (row['episode_id'].replace(':', '_') + '.json') for row in rows]
        if all(p.exists() for p in paths):
            return
        messages, schema = qc_contract(rows)
        try:
            result = request(api, root, messages, schema, 'v2_instance_qc')
        except (APIUnavailable, CallBudgetExceeded):
            raise
        except Exception:
            return
        for entry, row, path in zip(batch, rows, paths):
            if not path.exists():
                verdict = result['verdicts'][row['episode_id']]
                atomic_new(path, {'episode_id': row['episode_id'], 'candidate_sha256': entry['sha256'],
                    'operator': row['operator'], 'split': row['split'], 'verdict': verdict,
                    'passed': passes_qc(verdict, row['operator']),
                    'contract_sha256': result['contract_sha256'], 'phase_call': result['phase_call']})
        print(json.dumps({'stage': 'v2_instance_qc', 'batch': result['contract_sha256'],
                          'budget': api.accounting()}), flush=True)
    dispatch_batches(batches(entries, config[PHASE]['qc_batch_size']), one,
                     config[PHASE]['max_concurrent_requests'])


def summary(config):
    root = config['paths'][PHASE + '_output']
    plan = read_json(root / 'source_plan.json')
    results = {}
    for op in OPERATORS:
        counts, splits = Counter(), {}
        for split in ('agent_train', 'agent_dev'):
            part = Counter(planned=len(plan['plans'][op][split]))
            for entry in plan['plans'][op][split]:
                name = entry['source_id'].replace(':', '_') + '.json'
                path = root / 'candidates' / op / split / name
                if not path.exists():
                    failure = root / 'candidate_failures' / op / split / name
                    part['mechanical_failure' if failure.exists() else 'pending_candidate'] += 1
                    continue
                part['generated'] += 1
                row = read_json(path)
                verdict = root / 'qc' / (row['episode_id'].replace(':', '_') + '.json')
                if not verdict.exists():
                    part['pending_qc'] += 1
                    continue
                judged = read_json(verdict)
                if judged['candidate_sha256'] != file_sha(path):
                    raise ValueError('Stale instance QC')
                part['judged'] += 1
                part['passed' if judged['passed'] else 'rejected'] += 1
            counts.update(part)
            splits[split] = dict(part)
        n, passed = counts['planned'], counts['passed']
        pending = counts['pending_candidate'] + counts['pending_qc']
        lower, upper = passed / n, (passed + pending) / n
        threshold = config[PHASE]['qc_min_yield']
        results[op] = {'counts': dict(counts), 'by_split': splits,
            'qc_pass_rate_judged': passed / counts['judged'] if counts['judged'] else None,
            'usable_source_yield': lower if not pending else None,
            'yield_lower_bound': lower, 'yield_upper_bound': upper,
            'decision': 'drop' if upper < threshold else 'retain' if lower >= threshold else 'undetermined'}
    result = {'operators': results, 'threshold': config[PHASE]['qc_min_yield'],
        'threshold_denominator': 'all 380 sampled source essays per operator; also report QC pass/judged separately',
        'unknown_policy': 'unjudged candidates are not failures; retain/drop only when bounds establish the threshold'}
    write_json(root / 'qc_summary.json', result)
    return result


def run(config, *, max_api_calls, paid_approved=False):
    require_v2(config)
    if not paid_approved:
        raise PermissionError('Wait for the user’s explicit Proceed before v2 paid calls')
    root = config['paths'][PHASE + '_output']
    with collection_lock(root):
        load_environment(config)
        api = V2API(config, max_api_calls, kind='sol', paid_approved=paid_approved)
        stage, error = 'labels', None
        try:
            api.client()  # Fail before reserving/looping if the live client is unavailable.
            api.settle_interrupted()
            label(config, api)
            stage = 'local_candidates'
            for op in OPERATORS:
                build(config, operator=op)
            stage = 'instance_qc'
            judge(config, api)
            stage = 'completed'
        except Exception as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
        finally:
            api.close()
            status = {'stage': stage, 'error': error, 'budget': api.accounting(), 'qc': summary(config)}
            write_json(root / 'qc_status.json', status)
        return status
