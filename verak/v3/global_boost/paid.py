"""One aggregate, crash-safe $12 ledger for Sol QC and both Luna attempts."""
from copy import copy, deepcopy
import json

from ..common import read_json, sha_text
from ..eval.api import Phase6API
from ..train.teacher_bulk import BulkAPI
from .config import PHASE


class BoostAPI(BulkAPI):
    def __init__(self, config, max_api_calls, *, kind='luna', **kwargs):
        if kind not in {'sol', 'luna'}:
            raise ValueError('Unknown paid stage model')
        cfg = deepcopy(config)
        self.allowed_models = {cfg[PHASE]['model'], cfg[PHASE]['sol_model']}
        cfg[PHASE]['model'] = cfg[PHASE]['sol_model'] if kind == 'sol' else cfg[PHASE]['model']
        super().__init__(cfg, max_api_calls, phase=PHASE, **kwargs)

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
