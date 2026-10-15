"""Approved cost reconciliation, bounded retry chains, and circuit-breaker tests."""

from concurrent.futures import ThreadPoolExecutor
import json
import pytest

from verak.v3.common import file_sha, read_json
from verak.v3.corrupt.full_judging import CostLedger, CostUnavailable
from verak.v3.corrupt.resilient_judging import (MAX_CALLS, ReliabilityGate, RetryLedger,
                                              collect_responses, accounting, retryable)


def pilot():
    return {"calls":100,"cost_nanos":1_000_000_000,"ids":["pilot"]}


def raw(sid, attempt, *, http_status=None, timeout=False, success=False, usage=None):
    value={"sample_id":sid,"phase_call":attempt["call"],"attempt_id":attempt["attempt_id"],
           "status":"completed" if success else "error", "usage":usage,
           "model":"gpt-6.1-sol","response_model":"gpt-6.1-sol" if success else None,
           "http_status":http_status,"error_type":"APITimeoutError" if timeout else "InternalServerError",
           "prompt_sha256":"synthetic"}
    if success:
        value.pop("error_type")
        value["raw"]=json.dumps({"judgments":[{"record_id":sid+":R1","damage_real":True,
                                  "original_is_fix":True,"note":"synthetic"}]})
    return value


def test_legacy_migration_reconciles_only_5xx_and_preserves_original(tmp_path):
    path=tmp_path/'ledger.json'
    old=CostLedger(path,pilot(),max_api_calls=3033)
    calls=[]
    for sid,kind in [('server','server'),('timeout','timeout'),('done','done')]:
        number=old.reserve(sid,100_000_000)
        record=raw(sid,{'call':number,'attempt_id':sid},http_status=503 if kind=='server' else None,
                   timeout=kind=='timeout',success=kind=='done',
                   usage={'input_tokens':100,'output_tokens':10} if kind=='done' else None)
        record.pop('attempt_id')
        old.settle(sid,record); calls.append(record)
    before=file_sha(path)
    new=RetryLedger(path,pilot(),max_api_calls=MAX_CALLS,prior_calls=calls)
    s=new.snapshot()
    assert s['confirmed_cost_usd']==pytest.approx(1.0003)
    assert s['timeout_reserved_usd']==pytest.approx(.1)
    assert s['cost_usd']==pytest.approx(1.1003)
    assert s['zero_usage_5xx']==1 and s['calls']==103
    assert file_sha(tmp_path/'cost_ledger_before_retry_approval.json')==before
    again=RetryLedger(path,pilot(),max_api_calls=MAX_CALLS,prior_calls=calls)
    assert again.snapshot()==s


@pytest.mark.parametrize('status',[500,502,503,504,520])
def test_5xx_without_usage_is_zero_but_reported_usage_is_charged(status):
    assert accounting({'http_status':status},100_000_000)==(0,'unreported_5xx_zero_by_user_decision')
    cost,category=accounting({'http_status':status,'usage':{'input_tokens':100,'output_tokens':10}},100_000_000)
    assert cost==300_000 and category=='confirmed_usage'


@pytest.mark.parametrize('failure', [
    {'http_status':401, 'error_type':'AuthenticationError'},
    {'http_status':429, 'error_type':'RateLimitError'},
    {'error_type':'APIConnectionError'},
])
def test_other_unreported_errors_are_not_charged_or_retried(failure):
    assert accounting(failure,100_000_000)==(0,'no_confirmed_usage_non_retryable')
    assert not retryable(failure)


def test_retries_have_independent_entries_and_completed_candidate_is_never_rejudged(tmp_path):
    ledger=RetryLedger(tmp_path/'ledger.json',pilot(),max_api_calls=MAX_CALLS)
    first=ledger.reserve('x',100_000_000)
    failed=raw('x',first,http_status=503)
    ledger.settle(first['attempt_id'],failed)
    second=ledger.reserve('x',100_000_000)
    done=raw('x',second,success=True,usage={'input_tokens':100,'output_tokens':10})
    ledger.settle(second['attempt_id'],done)
    assert first['attempt_id']!=second['attempt_id'] and second['attempt']==2
    assert ledger.snapshot()['calls']==102
    with pytest.raises(ValueError,match='Completed'): ledger.reserve('x',100_000_000)
    with pytest.raises(ValueError,match='Completed'): ledger.reserve('pilot',100_000_000)
    responses,history=collect_responses([{'episode_id':'x','records':[{'record_id':'x:R1'}]}],[],[done,failed],ledger)
    assert set(responses)=={'x'} and len(history['x'])==2
    with pytest.raises(ValueError,match='duplicate|completed'):
        collect_responses([{'episode_id':'x','records':[{'record_id':'x:R1'}]}],[],[failed,done,done],ledger)


def test_timeout_reservation_survives_a_successful_retry(tmp_path):
    ledger=RetryLedger(tmp_path/'ledger.json',pilot(),max_api_calls=MAX_CALLS)
    first=ledger.reserve('x',150_000_000)
    ledger.settle(first['attempt_id'],raw('x',first,timeout=True))
    second=ledger.reserve('x',150_000_000)
    ledger.settle(second['attempt_id'],raw('x',second,success=True,usage={'input_tokens':100,'output_tokens':10}))
    state=ledger.snapshot()
    assert state['confirmed_cost_usd']==pytest.approx(1.0003)
    assert state['timeout_reserved_usd']==pytest.approx(.15)
    assert state['cost_usd']==pytest.approx(1.1503)


def test_three_retries_is_four_total_attempts(tmp_path):
    ledger=RetryLedger(tmp_path/'ledger.json',pilot(),max_api_calls=MAX_CALLS)
    for n in range(1,5):
        entry=ledger.reserve('x',100_000_000)
        assert entry['attempt']==n
        ledger.settle(entry['attempt_id'],raw('x',entry,http_status=520))
    with pytest.raises(ValueError,match='exhausted'): ledger.reserve('x',100_000_000)
    assert ledger.snapshot()['calls']==104


def test_concurrent_reservations_include_timeout_hold(tmp_path):
    p=pilot();p['cost_nanos']=49_700_000_000
    ledger=RetryLedger(tmp_path/'ledger.json',p,max_api_calls=MAX_CALLS)
    first=ledger.reserve('timeout',100_000_000)
    ledger.settle(first['attempt_id'],raw('timeout',first,timeout=True))
    def reserve(i):
        try:return ledger.reserve(str(i),100_000_000)
        except CostUnavailable:return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        values=list(pool.map(reserve,range(4)))
    assert sum(v is not None for v in values)==2
    assert ledger.snapshot()['outstanding_nanos']==200_000_000


def test_circuit_breaker_strict_threshold_persistence_and_resume(tmp_path):
    clock=[1000.]
    gate=ReliabilityGate(tmp_path/'reliability.json',now=lambda:clock[0])
    for _ in range(40): gate.record({'status':'completed'})
    for _ in range(10): gate.record({'http_status':503})
    assert gate.seconds_remaining()==0  # Exactly 20% is not "exceeds 20%".
    gate.record({'http_status':520})
    assert gate.seconds_remaining()==600
    state=read_json(tmp_path/'reliability.json')
    assert state['events'][0]['5xx']==11 and state['recent']==[]
    clock[0]+=120
    resumed=ReliabilityGate(tmp_path/'reliability.json',now=lambda:clock[0])
    assert resumed.seconds_remaining()==480
    clock[0]+=481
    resumed.record({'status':'completed'})
    assert resumed.seconds_remaining()==0 and len(resumed.state['events'])==1


@pytest.mark.parametrize('calls,cost',[(MAX_CALLS+1,50),(100,50.01)])
def test_authorized_limits_cannot_be_exceeded(tmp_path,calls,cost):
    with pytest.raises(ValueError): RetryLedger(tmp_path/'ledger.json',pilot(),max_api_calls=calls,max_cost_usd=cost)
