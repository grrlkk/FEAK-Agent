"""One offline recoverability judgment per deletion record, no replacement essays."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json

from ..common import file_sha, read_json, write_json, sha_text
from ..phase2 import read_jsonl, write_jsonl

FILTER_PROMPT = '''한국어 글 수정 자료의 복구 가능성을 판정한다. JSON만 출력한다.
remaining_essay만 수정자에게 주어진다. deleted_sentence는 판정자용 참고이며 수정자는 볼 수 없다.
recoverable: 삭제 문장이 맡던 뒷문장에 대한 뒷받침 역할을 남은 글만으로 다시 쓸 수 있는가?
글 다른 곳에 이미 있는 내용을 옮기고 연결하고 다시 표현하는 것은 가능하다.
남은 글 어디에도 없는 사실·이유·사례·숫자·이름·출처·경험을 하나라도 새로 알아야 한다면 false다.
일반 상식이나 그럴듯한 추측도 새 내용이면 허용하지 않는다. 삭제 문장을 그대로 복사할 수
있는지가 아니라 같은 뒷받침 역할을 기존 내용만으로 복구할 수 있는지 판단한다.
note는 짧게 쓰고, 판단의 근거가 남은 글에 어디 있는지 또는 무엇이 사라졌는지 설명한다.'''
FILTER_SCHEMA = {'type': 'object', 'properties': {'recoverable': {'type': 'boolean'}, 'note': {'type': 'string'}},
                 'required': ['recoverable', 'note'], 'additionalProperties': False}


def record_key(row, record):
    return row['episode_id'] + ':' + record['record_id']


def filter_rows(rows, judgments):
    kept, dropped = [], []
    for row in rows:
        keys = [record_key(row, r) for r in row['records'] if r['op'] == 'G_DELETE_SUPPORT']
        if any(key not in judgments for key in keys):
            raise ValueError('Recoverability judgment missing; never silently keep unknown rows')
        for key in keys:
            value = judgments[key]
            if set(value) != {'recoverable', 'note'} or type(value['recoverable']) is not bool or not isinstance(value['note'], str):
                raise ValueError('Invalid recoverability judgment')
        if any(not judgments[key]['recoverable'] for key in keys):
            dropped.append(row['episode_id'])
        else:
            kept.append(row)
    return kept, dropped


def corpus_stats(rows):
    levels = Counter(r['level'] for row in rows for r in row['records'])
    local = sum(levels[k] for k in ('WORD', 'SENTENCE', 'TEXT'))
    return {'essays': len(rows), 'sources': len({r['source_id'] for r in rows}),
        'levels': dict(Counter(r['level'] for r in rows)), 'genres': dict(Counter(r['genre'] for r in rows)),
        'operators': dict(Counter(r['op'] for row in rows for r in row['records'])),
        'record_levels': dict(levels),
        'local_shares': {k: levels[k]/local if local else 0 for k in ('WORD', 'SENTENCE', 'TEXT')}}


def run_filter(config, api):
    output = config['paths']['phase6_output']/'filter'
    output.mkdir(parents=True, exist_ok=True)
    original = config['paths']['metadata']/'corrupt'
    corpora = {s: read_jsonl(original/(s+'.jsonl')) for s in ('agent_train', 'agent_dev')}
    manifest = {'source_hashes': {s: file_sha(original/(s+'.jsonl')) for s in corpora},
                'prompt_hash': sha_text(FILTER_PROMPT), 'schema': FILTER_SCHEMA,
                'model': 'gpt-6.1-sol', 'reasoning_effort': 'high'}
    path = output/'manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Filter manifest changed')
    write_json(path, manifest)
    jobs = [(row, r) for rows in corpora.values() for row in rows for r in row['records'] if r['op'] == 'G_DELETE_SUPPORT']
    if len(jobs) > 260:
        raise ValueError('More deletion records than authorized filter calls')
    def judge(job):
        row, rec = job
        payload = {'question': row['question'], 'remaining_essay': row['corrupted_text'],
                   'deleted_sentence': rec['recovery_target']['original']}
        response = api.request([{'role': 'system', 'content': FILTER_PROMPT},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
            stage='filter', item_id=record_key(row, rec), effort='high', schema=FILTER_SCHEMA)
        judgment = json.loads(response['raw'])
        filter_rows([{**row, 'records': [rec]}], {record_key(row, rec): judgment})
        return {'key': record_key(row, rec), 'episode_id': row['episode_id'], 'record_id': rec['record_id'],
                'split': row['split'], 'judgment': judgment, 'phase_call': response['phase_call']}
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        work, pending = iter(jobs), set()
        def fill():
            while len(pending) < 4:
                job = next(work, None)
                if job is None:
                    break
                pending.add(pool.submit(judge, job))
        fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.remove(future)
                results.append(future.result())
                write_jsonl(output/'judgments.jsonl', sorted(results, key=lambda r: r['key']))
                if len(results) % 10 == 0 or len(results) == len(jobs):
                    print({'stage': 'filter', 'completed': len(results), 'total': len(jobs),
                           **api.accounting()}, flush=True)
            fill()
    judgments = {r['key']: r['judgment'] for r in results}
    stats = {}
    for split, rows in corpora.items():
        kept, dropped = filter_rows(rows, judgments)
        write_jsonl(config['paths']['active_corrupt']/(split+'.jsonl'), kept)
        stats[split] = {'before': corpus_stats(rows), 'after': corpus_stats(kept), 'dropped_ids': dropped,
                        'filtered_sha256': file_sha(config['paths']['active_corrupt']/(split+'.jsonl'))}
    write_json(output/'metrics.json', stats)
    return stats
