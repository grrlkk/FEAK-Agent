"""Budget-only amendment: existing contracts, reservations and failed outcomes."""
import json

import pytest

from feak_tc.runtime.openai import CallBudgetExceeded
from verak.v4 import scale3_common as common

PRIOR = .660314125


def contracts(root):
    common.freeze(root/'B2/contract.json', {'component': 'B2', 'cap_usd': 50.-PRIOR,
        'cumulative_content_cap_usd': 50., 'smoke': {'prior_confirmed_usd': PRIOR}})
    common.freeze(root/'B3/contract.json', {'component': 'B3', 'cap_usd': 30.,
        'cumulative_content_cap_usd': None})


def amendment(root, **changes):
    value = {'schema_version': 1, 'component': 'B2',
        'original_contract_sha256': common.file_sha(root/'B2/contract.json'),
        'original_cumulative_content_cap_usd': 50., 'new_cumulative_content_cap_usd': 100.,
        'prior_smoke_confirmed_usd': PRIOR, 'authorization': 'explicit_user_B2_budget_only_20261011',
        'collection_contract_unchanged': True, 'context_failures_retry': False}
    common.freeze(root/'B2/budget_amendment.json', value | changes)


def local_api(root, monkeypatch, component='B2', kind='luna'):
    """Use the real local SQLite reservation implementation, never an API client."""
    import verak.v3.v2_ops.local as local
    common.freeze(root/'phase4/models.json', {'luna_model': 'gpt-6-luna'})
    monkeypatch.setattr(common, 'load_config', lambda: {'paths': {'phase4_output': root/'phase4'}})
    monkeypatch.setattr(local, 'load_environment', lambda config: None)
    monkeypatch.setattr(common, 'component_contract', lambda component, root:
                        common.read_json(root/component/'contract.json'))
    api = common.api_for(component, kind, root=root)
    def no_client():
        raise AssertionError('Budget tests must never initialize or call a network client')
    monkeypatch.setattr(api, 'client', no_client)
    return api


def test_default_50_and_b3_30_are_unchanged(tmp_path, monkeypatch):
    contracts(tmp_path)
    before = {c: (tmp_path/c/'contract.json').read_bytes() for c in ('B2', 'B3')}
    old = local_api(tmp_path, monkeypatch)
    b3 = local_api(tmp_path, monkeypatch, 'B3')
    assert old.settings['max_cost_usd'] == 49.339685875
    assert b3.settings['max_cost_usd'] == 30.
    assert common.component_accounting('B2', tmp_path)['cumulative_content_cap_usd'] == 50.
    amendment(tmp_path)
    assert common.effective_budget('B2', tmp_path)['component_cap_usd'] == 99.339685875
    assert old.settings['max_cost_usd'] == 49.339685875  # Clean resume is required.
    unchanged = common.effective_budget('B3', tmp_path)
    assert unchanged['component_cap_usd'] == 30. and unchanged['budget_amendment_path'] is None
    assert common.component_accounting('B3', tmp_path)['cumulative_content_cap_usd'] is None
    assert all((tmp_path/c/'contract.json').read_bytes() == data for c, data in before.items())
    old.close(); b3.close()


@pytest.mark.parametrize('change', [
    {'schema_version': True}, {'component': 'B3'}, {'original_contract_sha256': 'wrong'},
    {'original_cumulative_content_cap_usd': 100.}, {'new_cumulative_content_cap_usd': 101.},
    {'prior_smoke_confirmed_usd': 0.}, {'authorization': 'not_authorized'},
    {'collection_contract_unchanged': False}, {'context_failures_retry': True},
    {'context_failures_retry': 0}, {'extra_policy_change': 'retry_context_failures'},
])
def test_amendment_fails_closed_on_any_unapproved_change(tmp_path, change):
    contracts(tmp_path); amendment(tmp_path, **change)
    with pytest.raises(ValueError, match='amendment'):
        common.effective_budget('B2', tmp_path)


def test_amendment_rejects_changed_original_or_b3_amendment(tmp_path):
    contracts(tmp_path); amendment(tmp_path)
    path = tmp_path/'B2/contract.json'
    value = common.read_json(path); value['changed_field'] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='amendment'):
        common.effective_budget('B2', tmp_path)
    common.freeze(tmp_path/'B3/budget_amendment.json', {})
    with pytest.raises(ValueError, match='No B3'):
        common.effective_budget('B3', tmp_path)


def test_resumed_api_enforces_100_including_pending_and_unknown_reservations(tmp_path, monkeypatch):
    contracts(tmp_path)
    old = local_api(tmp_path, monkeypatch)
    with old.db() as db:
        db.execute("INSERT INTO calls(stage,item_id,status,reserved,confirmed) VALUES(?,?,?,?,?)",
                   ('teacher', 'saved', 'completed', 0., 49.25))
    with pytest.raises(CallBudgetExceeded):
        old.reserve('teacher', 'blocked', 'old_bound', .1)
    amendment(tmp_path)
    new = local_api(tmp_path, monkeypatch, kind='sol')
    assert new.settings['max_cost_usd'] == 99.339685875
    assert new.path == old.path
    reservation = new.reserve('teacher', 'new', 'new_bound', .1)
    account = common.component_accounting('B2', tmp_path)
    assert account['cumulative_content_cap_usd'] == 100.
    assert 50. < account['cumulative_confirmed_and_reserved_usd'] < 100.
    assert account['budget_amendment_sha256'] == common.file_sha(tmp_path/'B2/budget_amendment.json')
    with new.db() as db:
        db.execute("UPDATE calls SET confirmed=? WHERE item_id='saved'", (new.settings['max_cost_usd']-.15,))
        # Unknown charged outcome retains its reservation even when not pending.
        db.execute("UPDATE calls SET status='error' WHERE id=?", (reservation,))
    with pytest.raises(CallBudgetExceeded):
        new.reserve('teacher', 'would_exceed', 'large_bound', .06)
    assert new.reserve('teacher', 'fits', 'small_bound', .04) is not None
    assert common.component_accounting('B2', tmp_path)['cumulative_confirmed_and_reserved_usd'] == pytest.approx(99.99)
    with new.db() as db:
        db.execute("UPDATE calls SET reserved=reserved+.02 WHERE item_id='fits'")
    with pytest.raises(ValueError, match='effective cap'):
        common.component_accounting('B2', tmp_path)
    assert not old.clients and not new.clients
    old.close(); new.close()


def test_budget_extension_never_retries_saved_context_error(tmp_path, monkeypatch):
    from verak.v4 import policy_prompts_v43 as prompts
    contracts(tmp_path); amendment(tmp_path)
    api = local_api(tmp_path, monkeypatch)
    monkeypatch.setattr(prompts, 'verify_frozen', lambda: {})
    contract = {'stage': 'v43_scale3_content_plan', 'item_id': 'saved_context_error',
        'model': api.model, 'reasoning_effort': 'low', 'max_output_tokens': 2048,
        'messages': [], 'schema': None}
    fingerprint = common.sha_text(json.dumps(contract, ensure_ascii=False, sort_keys=True))
    with api.db() as db:
        db.execute("INSERT INTO calls(stage,item_id,fingerprint,status,reserved,confirmed) VALUES(?,?,?,?,?,?)",
                   (contract['stage'], contract['item_id'], fingerprint, 'error', 0., .01))
    before = common.accounting(api.path)
    with pytest.raises(RuntimeError, match='no silent regeneration'):
        api.request([], stage='scale3_content_plan', item_id=contract['item_id'])
    assert common.accounting(api.path) == before
    assert not api.clients
    api.close()
