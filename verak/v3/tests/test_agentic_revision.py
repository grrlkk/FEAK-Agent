"""The corrected run cannot discard earlier A spending or replay old teachers."""
from copy import deepcopy

import pytest

from feak_tc.runtime.openai import CallBudgetExceeded
from verak.v3.agentic.data import PHASE, config_for
from verak.v3.agentic.revision import initialize
from verak.v3.agentic.runner import pilot_stage
from verak.v3.eval.api import Phase6API


def prior_run(tmp_path, *, pending=False):
    config = config_for(version=3)
    config['paths'][PHASE + '_output'] = tmp_path / 'agentic_pilot_v3_protected'
    previous = deepcopy(config)
    previous['paths'][PHASE + '_output'] = tmp_path / 'agentic_pilot_v3'
    api = Phase6API(previous, 10, phase=PHASE)
    call = api.reserve('pilot_final_corrupted', 'old', 'old', .4)
    if not pending:
        with api.db() as db:
            db.execute("UPDATE calls SET status='error',confirmed=.3,reserved=.1 WHERE id=?", (call,))
    return config, api


def test_carryover_retains_confirmed_and_uncertain_cost_and_never_resets_new_calls(tmp_path):
    config, old = prior_run(tmp_path)
    carried = initialize(config)
    assert carried['confirmed_usd'] == .3 and carried['reserved_usd'] == .1
    api = Phase6API(config, 10, phase=PHASE)
    with pytest.raises(CallBudgetExceeded):
        api.reserve('pilot_protected_corrupted', 'new', 'new', 6.61)
    new = api.reserve('pilot_protected_corrupted', 'new', 'new', .5)
    initialize(config)
    with api.db() as db:
        assert db.execute('SELECT count(*) FROM calls').fetchone()[0] == 2
        assert db.execute('SELECT reserved FROM calls WHERE id=?', (new,)).fetchone()[0] == .5
    assert old.accounting()['calls'] == 1
    assert api.accounting()['confirmed_usd'] == .3 and api.accounting()['reserved_usd'] == .6


def test_live_or_changed_original_ledger_blocks_carryover(tmp_path):
    config, old = prior_run(tmp_path, pending=True)
    with pytest.raises(ValueError, match='Stop and drain'):
        initialize(config)
    assert not (config['paths'][PHASE + '_output'] / 'api').exists()
    with old.db() as db:
        db.execute("UPDATE calls SET status='error'")
    initialize(config)
    with old.db() as db:
        db.execute('UPDATE calls SET reserved=.2')
    with pytest.raises(ValueError, match='superseded v3 ledger changed'):
        initialize(config)


def test_fresh_teacher_namespace_keeps_original_behavior_reproducible():
    assert pilot_stage(config_for(version=3)) == 'pilot_protected_'
    assert pilot_stage(config_for(version=2)) == 'pilot_final_'
