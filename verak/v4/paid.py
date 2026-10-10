"""Independent B/C/D hard caps using the existing durable Responses ledger."""
from copy import copy
import json

from verak.v3.eval.api import Phase6API
from verak.v3.train.teacher_bulk import BulkAPI
from verak.v3.v2_ops.local import load_environment
from .common import ROOT, load_config, read_json, sha_text

CAPS = {'B': 5., 'C': 4., 'D': 15.}


class PrepAPI(BulkAPI):
    def client(self):
        if getattr(getattr(self.local, 'adapter', None), '_client', None) is None:
            self.local.__dict__.pop('adapter', None)
        return super().client()

    def request(self, messages, *, stage, item_id, effort='low', max_output=2048, schema=None):
        contract = {'stage': stage, 'item_id': item_id, 'model': self.model,
                    'reasoning_effort': effort, 'max_output_tokens': max_output,
                    'messages': messages, 'schema': schema}
        fingerprint = sha_text(json.dumps(contract, ensure_ascii=False, sort_keys=True))
        with self.db() as db:
            prior = db.execute("SELECT id,status FROM calls WHERE fingerprint=? AND status!='blocked_before_send' ORDER BY id",
                               (fingerprint,)).fetchall()
        # Completed legacy requests remain readable. Newly dispatched v4 editor
        # calls use the canonical frozen policy; v3/RFT uses separate clients.
        if 'teacher' in stage and not any(status == 'completed' for _, status in prior):
            from .policy_prompts import assert_editor_request
            assert_editor_request(messages)
        if prior and not any(status == 'completed' for _, status in prior):
            raise RuntimeError(f'Preserved prior API outcome {prior}; no silent regeneration')
        return Phase6API.request(self, messages, stage=stage, item_id=item_id,
                                 effort=effort, max_output=max_output, schema=schema)

    def _durable_outcome(self, call_id, stage, item_id, fingerprint, reserved):
        saved = copy(self)
        path = self.output / 'requests' / f'{call_id:06}.json'
        if path.exists():
            model = read_json(path).get('model')
            if model in self.allowed_models:
                saved.model = model
        return super(PrepAPI, saved)._durable_outcome(call_id, stage, item_id, fingerprint, reserved)


def api_for(component, kind='luna', *, max_calls=20000):
    if component not in CAPS or kind not in {'sol', 'luna'}:
        raise ValueError('Unknown authorized preparation component/model')
    config = load_config()
    load_environment(config)
    phase = 'v4_prep_' + component.lower()
    luna = read_json(config['paths']['phase4_output'] / 'models.json')['luna_model']
    config[phase] = {'model': 'gpt-6.1-sol' if kind == 'sol' else luna,
                     'max_cost_usd': CAPS[component], 'max_concurrent_requests': 4,
                     'phase_api_ceiling': 20000}
    config['paths'][phase+'_output'] = ROOT / component
    api = PrepAPI(config, max_calls, phase=phase)
    api.allowed_models = {luna, 'gpt-6.1-sol'}
    return api
