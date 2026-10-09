"""Shared $10 reservation ledger for v2 Sol/Luna, gated before API construction."""
from copy import copy, deepcopy
from pathlib import Path

import json
from ..common import read_json, sha_text
from ..eval.api import Phase6API
from ..train.teacher_bulk import BulkAPI
from .config import PHASE, require_v2


class V2API(BulkAPI):
    def __init__(self, config, max_api_calls, *, kind, paid_approved=False, **kwargs):
        require_v2(config)
        if not paid_approved:
            raise PermissionError('The user requires explicit Proceed before any v2 paid call')
        if kind not in {'sol', 'luna'} or max_api_calls <= 0:
            raise ValueError('Paid v2 calls require a model kind and positive --max-api-calls')
        cfg = deepcopy(config)
        self.allowed_models = {cfg[PHASE]['model'], cfg[PHASE]['sol_model']}
        cfg[PHASE]['model'] = cfg[PHASE]['sol_model'] if kind == 'sol' else cfg[PHASE]['model']
        super().__init__(cfg, max_api_calls, phase=PHASE, **kwargs)

    def client(self):
        # The parent leaves its adapter allocated if initialization raises. Do not
        # treat that uninitialized adapter as a usable client on a later request.
        if getattr(getattr(self.local, 'adapter', None), '_client', None) is None:
            self.local.__dict__.pop('adapter', None)
        client = super().client()
        if not callable(getattr(getattr(client, 'responses', None), 'create', None)):
            raise RuntimeError('Responses client unavailable before API dispatch')
        return client

    def request(self, messages, *, stage, item_id, effort='low', max_output=1024, schema=None):
        contract = {'stage': stage, 'item_id': item_id, 'model': self.model,
            'reasoning_effort': effort, 'max_output_tokens': max_output, 'messages': messages, 'schema': schema}
        fingerprint = sha_text(json.dumps(contract, ensure_ascii=False, sort_keys=True))
        with self.db() as db:
            prior = db.execute("SELECT id,status FROM calls WHERE fingerprint=? AND status!='blocked_before_send' ORDER BY id",
                               (fingerprint,)).fetchall()
        if prior and not any(status == 'completed' for _, status in prior):
            raise RuntimeError(f'Preserved prior sent API outcome: {prior}; no regeneration')
        return Phase6API.request(self, messages, stage=stage, item_id=item_id, effort=effort,
                                 max_output=max_output, schema=schema)

    def _durable_outcome(self, call_id, stage, item_id, fingerprint, reserved):
        # A restart may leave either model's last request in this shared ledger.
        path = self.output / 'requests' / f'{call_id:06}.json'
        try:
            model = read_json(path)['model']
        except (OSError, ValueError, KeyError):
            model = self.model
        restored = copy(self)
        if model in self.allowed_models:
            restored.model = model
        return super(V2API, restored)._durable_outcome(call_id, stage, item_id, fingerprint, reserved)
