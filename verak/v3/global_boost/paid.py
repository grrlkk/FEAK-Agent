"""One aggregate, crash-safe $12 ledger for Sol QC and both Luna attempts."""
from copy import copy, deepcopy
import json
import time
import uuid

from feak_tc.runtime.openai import CallBudgetExceeded
from ..common import read_json, sha_text
from ..eval.api import Phase6API
from ..train.teacher_bulk import BulkAPI, atomic_new
from .config import PHASE


class BoostAPI(BulkAPI):
    def __init__(self, config, max_api_calls, *, kind='luna', **kwargs):
        if kind not in {'sol', 'luna'}:
            raise ValueError('Unknown paid stage model')
        cfg = deepcopy(config)
        self.kind = kind
        self.item_namespace = cfg[PHASE].get('api_item_namespace')
        self.qc_hold_usd = cfg[PHASE].get('qc_hold_usd', 0.)
        # Frozen candidate batches have separate files, never separate budgets.
        cfg['paths'][PHASE + '_output'] = cfg['paths'].get(
            'global_boost_shared_root', cfg['paths'][PHASE + '_output'])
        self.allowed_models = {cfg[PHASE]['model'], cfg[PHASE]['sol_model']}
        cfg[PHASE]['model'] = cfg[PHASE]['sol_model'] if kind == 'sol' else cfg[PHASE]['model']
        super().__init__(cfg, max_api_calls, phase=PHASE, **kwargs)

    def reserve(self, stage, item_id, fingerprint, bound):
        if self.kind != 'sol' or not self.qc_hold_usd:
            return super().reserve(stage, item_id, fingerprint, bound)
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            n, pending, committed = db.execute("SELECT SUM(status!='blocked_before_send'), "
                "SUM(status='pending'), SUM(reserved+confirmed) FROM calls").fetchone()
            if (n or 0) >= self.limit:
                raise CallBudgetExceeded(f'{self.phase} --max-api-calls exhausted')
            if (pending or 0) >= self.settings['max_concurrent_requests']:
                return None
            if (committed or 0) + bound + self.qc_hold_usd > self.settings['max_cost_usd']:
                raise CallBudgetExceeded('GLOBAL QC stopped to preserve outstanding Luna attempt funds')
            return db.execute('INSERT INTO calls(stage,item_id,fingerprint,status,reserved,created) '
                'VALUES(?,?,?,?,?,?)', (stage, item_id, fingerprint, 'pending', bound, time.time())).lastrowid

    def settle_interrupted(self):
        if not self.item_namespace:
            return super().settle_interrupted()
        # An expansion QC batch may run while the immutable original teacher is
        # making live requests. Only reconcile this exclusively owned namespace.
        reconciled = []
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute("SELECT id,stage,item_id,fingerprint,reserved,confirmed FROM calls "
                "WHERE status='pending' AND instr(item_id,?)>0", (self.item_namespace,)).fetchall()
            for call_id, stage, item_id, fingerprint, reserved, confirmed in rows:
                outcome = self._durable_outcome(call_id, stage, item_id, fingerprint, reserved)
                outcome.setdefault('confirmed', confirmed)
                db.execute('UPDATE calls SET status=?,reserved=?,confirmed=?,path=?,finished=? WHERE id=?',
                    (outcome['status'], outcome['reserved'], outcome['confirmed'], outcome['path'], time.time(), call_id))
                reconciled.append({'phase_call': call_id, 'stage': stage, 'item_id': item_id,
                    'previous_reserved': reserved, **outcome})
        if reconciled:
            atomic_new(self.output / ('interrupted_' + uuid.uuid4().hex + '.json'),
                {'requests': reconciled, 'at': time.time(), 'namespace': self.item_namespace})
        return reconciled

    def client(self):
        if getattr(getattr(self.local, 'adapter', None), '_client', None) is None:
            self.local.__dict__.pop('adapter', None)
        return super().client()

    def request(self, messages, *, stage, item_id, effort='low', max_output=1024, schema=None):
        contract = {'stage': stage, 'item_id': item_id, 'model': self.model,
            'reasoning_effort': effort, 'max_output_tokens': max_output, 'messages': messages, 'schema': schema}
        fingerprint = sha_text(json.dumps(contract, ensure_ascii=False, sort_keys=True))
        with self.db() as db:
            prior = db.execute("SELECT id,status FROM calls WHERE fingerprint=? AND status!='blocked_before_send' ORDER BY id",
                               (fingerprint,)).fetchall()
        if prior and not any(status == 'completed' for _, status in prior):
            raise RuntimeError(f'Preserved prior API outcome {prior}; never regenerate')
        return Phase6API.request(self, messages, stage=stage, item_id=item_id, effort=effort,
                                 max_output=max_output, schema=schema)

    def _durable_outcome(self, call_id, stage, item_id, fingerprint, reserved):
        try:
            model = read_json(self.output / 'requests' / f'{call_id:06}.json')['model']
        except (OSError, ValueError, KeyError):
            model = self.model
        restored = copy(self)
        if model in self.allowed_models:
            restored.model = model
        return super(BoostAPI, restored)._durable_outcome(call_id, stage, item_id, fingerprint, reserved)
