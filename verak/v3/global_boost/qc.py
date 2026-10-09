"""Unchanged Phase3b Sol high prompt, schema and both-true instance filter."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import socket

from feak_tc.runtime.openai import CallBudgetExceeded
from ..common import file_sha, read_json, write_json
from ..corrupt.api import QC_PROMPT, QCResponse
from ..corrupt.qc import qc_payload, validate_judgments
from ..train.teacher_bulk import atomic_new, collection_lock
from ..v2_ops.local import load_environment
from .config import PHASE, OPERATORS
from .paid import BoostAPI
from .prepare import corpus, safe_id


def run(config, max_api_calls=20000):
    root = config['paths'][PHASE + '_output']
    with collection_lock(root):
        load_environment(config)
        rows = corpus(config)
        api = BoostAPI(config, max_api_calls, kind='sol')
        api.settle_interrupted()
        # Fail before touching any candidate when the sandbox cannot resolve the
        # provider. Such a pre-send failure is not a failed model judgment.
        socket.getaddrinfo('api.openai.com', 443)
        api.client()
        for path in (root / 'qc_errors').glob('*.json'):
            old = read_json(path)
            if old['type'] not in {'gaierror', 'APIUnavailable'}:
                continue
            with api.db() as db:
                sent = db.execute('SELECT COUNT(*) FROM calls WHERE item_id=?', (old['episode_id'],)).fetchone()[0]
            if sent:
                raise ValueError('Preflight failure unexpectedly has an accounted send; inspect manually')
            archive = root / 'pre_send_failures' / path.name
            archive.parent.mkdir(parents=True, exist_ok=True)
            path.rename(archive)
        targets = [row for row in rows.values() if not (root / 'qc' / (safe_id(row['episode_id']) + '.json')).exists()
                   and not (root / 'qc_errors' / (safe_id(row['episode_id']) + '.json')).exists()]
        errors, completed = [], []

        def one(row):
            episode_id = row['episode_id']
            path = root / 'qc' / (safe_id(episode_id) + '.json')
            try:
                response = api.request([{'role': 'system', 'content': QC_PROMPT},
                    {'role': 'user', 'content': json.dumps(qc_payload(row), ensure_ascii=False, sort_keys=True)}],
                    stage='global_boost_qc', item_id=episode_id, effort='high', max_output=8192,
                    schema=QCResponse.model_json_schema())
                judgment = QCResponse.model_validate_json(response['raw']).model_dump()
                validate_judgments(row, judgment)
                passed = all(v['damage_real'] and v['original_is_fix'] for v in judgment['judgments'])
                atomic_new(path, {'episode_id': episode_id, 'judgment': judgment, 'passed': passed,
                    'phase_call': response['phase_call'], 'candidate_hash': row['corrupted_hash'],
                    'source_plan_sha256': file_sha(root / 'source_plan.json')})
                return {'episode_id': episode_id, 'passed': passed}
            except Exception as exc:
                error = {'episode_id': episode_id, 'type': type(exc).__name__, 'message': str(exc)}
                if not isinstance(exc, CallBudgetExceeded):
                    atomic_new(root / 'qc_errors' / path.name, error)
                raise

        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                iterator, active = iter(targets), {}
                while True:
                    while len(active) < 2 and not any(e['type'] == 'CallBudgetExceeded' for e in errors):
                        row = next(iterator, None)
                        if row is None:
                            break
                        active[pool.submit(one, row)] = row['episode_id']
                    if not active:
                        break
                    done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                    for future in done:
                        episode_id = active.pop(future)
                        try:
                            completed.append(future.result())
                        except Exception as exc:
                            errors.append({'episode_id': episode_id, 'type': type(exc).__name__, 'message': str(exc)})
                    write_json(root / 'qc_progress.json', {'finished_this_run': len(completed),
                        'errors': errors, 'in_flight': list(active.values()), 'budget': api.accounting()})
                    print(json.dumps({'qc_saved': len(completed), 'errors': len(errors),
                                      'budget': api.accounting()}), flush=True)
        finally:
            api.close()
        result = summarize(config)
        write_json(root / 'qc_status.json', {**result, 'errors_this_run': errors, 'budget': api.accounting()})
        return result


def summarize(config):
    root = config['paths'][PHASE + '_output']
    rows = corpus(config)
    result = {op: {'generated': 0, 'judged': 0, 'passed': 0, 'unknown': 0} for op in OPERATORS}
    for row in rows.values():
        values = result[row['operator']]
        values['generated'] += 1
        path = root / 'qc' / (safe_id(row['episode_id']) + '.json')
        if not path.exists():
            values['unknown'] += 1
            continue
        verdict = read_json(path)
        if verdict['candidate_hash'] != row['corrupted_hash']:
            raise ValueError('Stale candidate QC')
        values['judged'] += 1
        values['passed'] += int(verdict['passed'])
    for values in result.values():
        values['pass_rate_judged'] = values['passed'] / values['judged'] if values['judged'] else None
    write_json(root / 'qc_summary.json', result)
    return result
