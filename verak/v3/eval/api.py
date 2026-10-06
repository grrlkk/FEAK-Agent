"""Phase 6 shared, crash-safe cost reservations; at most four live requests."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import socket
from threading import local, Lock
import time

from feak_tc.agent.schemas import OpenAIConfig
from feak_tc.runtime.openai import CallBudgetExceeded
from ..api import EnvironmentJSONClient
from ..common import read_json, write_json, sha_text
from ..reconstruction_api import usage_cost


class Phase6API:
    model = 'gpt-6.1-sol'

    def __init__(self, config, max_api_calls, *, adapter_factory=EnvironmentJSONClient, phase='phase6'):
        self.phase = phase
        self.settings = config[phase]
        if not 0 <= max_api_calls <= self.settings['phase_api_ceiling']:
            raise ValueError(f'Invalid --max-api-calls for {phase}')
        self.limit = max_api_calls
        self.output = config['paths'][phase+'_output']/'api'
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output/'ledger.sqlite'
        self.factory = adapter_factory
        self.local = local()
        self.clients, self.clients_lock = [], Lock()
        with self.db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY, stage TEXT, item_id TEXT, fingerprint TEXT,
                status TEXT, reserved REAL, confirmed REAL DEFAULT 0,
                path TEXT, created REAL, finished REAL)''')
            db.execute('CREATE INDEX IF NOT EXISTS request_key ON calls(fingerprint)')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=60)
        try:
            with db:
                yield db
        finally:
            db.close()

    def accounting(self):
        with self.db() as db:
            rows = db.execute('SELECT stage,status,reserved,confirmed FROM calls').fetchall()
        return {'calls': sum(r[1] != 'blocked_before_send' for r in rows),
            'client_attempts': len(rows),
            'blocked_before_send': sum(r[1] == 'blocked_before_send' for r in rows),
            'confirmed_usd': sum(r[3] for r in rows),
            'reserved_usd': sum(r[2] for r in rows),
            'pending': sum(r[1] == 'pending' for r in rows),
            'by_stage': {s: {'calls': sum(r[0] == s and r[1] != 'blocked_before_send' for r in rows),
                'confirmed_usd': sum(r[3] for r in rows if r[0] == s)} for s in sorted({r[0] for r in rows})}}

    def reserve(self, stage, item_id, fingerprint, bound):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute('SELECT stage,status,reserved,confirmed FROM calls').fetchall()
            charged_rows = [r for r in rows if r[1] != 'blocked_before_send']
            if len(charged_rows) >= self.limit:
                raise CallBudgetExceeded(f'{self.phase} --max-api-calls exhausted')
            if stage == 'filter' and sum(r[0] == 'filter' for r in charged_rows) >= self.settings['filter_api_ceiling']:
                raise CallBudgetExceeded('Recoverability filter 260-call cap reached')
            if sum(r[1] == 'pending' for r in rows) >= self.settings['max_concurrent_requests']:
                return None
            if sum(r[2]+r[3] for r in rows)+bound > self.settings['max_cost_usd']:
                raise CallBudgetExceeded(f"{self.phase} ${self.settings['max_cost_usd']} cap: insufficient unreserved budget")
            cursor = db.execute('INSERT INTO calls(stage,item_id,fingerprint,status,reserved,created) VALUES(?,?,?,?,?,?)',
                (stage, item_id, fingerprint, 'pending', bound, time.time()))
            return cursor.lastrowid

    def client(self):
        if not hasattr(self.local, 'adapter'):
            cfg = OpenAIConfig(model=self.model, reasoning_effort='low', max_output_tokens=2048,
                timeout_s=600, max_input_chars=1000000, max_calls_total=12000, max_calls_per_sample=12000)
            self.local.adapter = self.factory(cfg)
            self.local.adapter._load()
            with self.clients_lock:
                self.clients.append(self.local.adapter)
        return self.local.adapter._client

    def request(self, messages, *, stage, item_id, effort='low', max_output=2048, schema=None):
        contract = {'stage': stage, 'item_id': item_id, 'model': self.model,
                    'reasoning_effort': effort, 'max_output_tokens': max_output,
                    'messages': messages, 'schema': schema}
        fingerprint = sha_text(json.dumps(contract, ensure_ascii=False, sort_keys=True))
        cache = self.output/'cache'/(fingerprint+'.json')
        if cache.exists():
            return {**read_json(cache), 'replayed': True}
        with self.db() as db:
            found = db.execute("SELECT path FROM calls WHERE fingerprint=? AND status='completed' ORDER BY id LIMIT 1", (fingerprint,)).fetchone()
        if found:
            record = read_json(Path(found[0]))
            write_json(cache, record)
            return {**record, 'replayed': True}
        client = self.client()  # Check the environment key before reserving any cost.
        # A sandbox DNS failure is not an API request. Fail before dispatch/reservation.
        socket.getaddrinfo('api.openai.com', 443)
        payload = json.dumps(messages, ensure_ascii=False) + json.dumps(schema)
        bound = ((len(payload.encode('utf-8'))+4096)*2.5 + max_output*10)/1e6
        for attempt in range(4):
            while (call_id := self.reserve(stage, item_id, fingerprint, bound)) is None:
                time.sleep(.2)
            path = self.output/'requests'/f'{call_id:06}.json'
            record = {**contract, 'fingerprint': fingerprint, 'phase_call': call_id,
                      'bound_usd': bound, 'replayed': False}
            started = time.monotonic()
            retryable = False
            try:
                kwargs = {'model': self.model, 'input': messages, 'reasoning': {'effort': effort},
                    'max_output_tokens': max_output, 'truncation': 'disabled', 'store': False}
                if schema:
                    kwargs['text'] = {'format': {'type': 'json_schema', 'name': stage,
                                               'strict': True, 'schema': schema}}
                response = client.responses.create(**kwargs)
                record.update(raw=response.output_text, status=response.status,
                    response_id=response.id, response_model=response.model,
                    usage=response.usage.model_dump() if response.usage else None)
                if record['usage']:
                    record['cost'] = usage_cost(self.model, record['usage'])
                if record['status'] != 'completed':
                    record['error_type'] = 'IncompleteResponse'
            except Exception as exc:
                status = getattr(exc, 'status_code', None)
                record.update(status='error', error_type=type(exc).__name__, http_status=status)
                retryable = status in (429, 500, 502, 503, 504, 520) or 'Timeout' in type(exc).__name__
                if 'Timeout' in type(exc).__name__:
                    record['timeout_reservation_usd'] = bound
            record['elapsed_s'] = time.monotonic()-started
            write_json(path, record)
            with self.db() as db:
                db.execute('UPDATE calls SET status=?,reserved=?,confirmed=?,path=?,finished=? WHERE id=?',
                    (record['status'], record.get('timeout_reservation_usd', 0),
                     record.get('cost', {}).get('confirmed_usd', 0), str(path), time.time(), call_id))
            if record['status'] == 'completed':
                write_json(cache, record)
                return record
            if not retryable or attempt == 3:
                raise RuntimeError(f"{self.phase} API call {call_id}: {record.get('error_type', record['status'])}")
            time.sleep((10, 40, 120)[attempt])

    def close(self):
        for client in self.clients:
            client.close()
        write_json(self.output/'accounting.json', self.accounting())


class Phase6Teacher:
    name = 'teacher'
    model = 'gpt-6.1-sol'
    context_limit = 250000

    def __init__(self, api, condition, *, max_output=1024):
        self.api, self.condition = api, condition
        self.max_output = max_output

    def generate(self, messages, *, episode_id, role, turn):
        response = self.api.request(messages, stage=self.condition,
            item_id=f'{episode_id}:{role}:{turn}', max_output=self.max_output)
        return {**response, 'episode_id': episode_id, 'role': role, 'turn': turn}
