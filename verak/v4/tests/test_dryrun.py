"""Technical dry-run contracts: selection, labels, strict execution, and GPU isolation."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from verak.v4.dryrun_data import choose, audit_sample
from verak.v4.dryrun_student import run_case, metrics
from verak.v4.dryrun_common import require_gpu_approval, ROOT
from verak.v4.environment_v43 import normalize_row
from verak.v4.tests.test_smoke42 import FakeBoundaryAnalysis, row
from verak.v3.train.sft_data import source_split


def candidate(source, attempt, preferred='2', *, keep=True, revision=1, korean=1):
    return {'source_id': source, 'attempt': attempt, 'preferred': preferred,
            'selection': {'export_keep': keep}, 'valid_calls': {'revision': revision, 'korean': korean}}


def test_selects_only_existing_eligible_preferred_and_source_distinct():
    rows = [candidate('a', 1), candidate('a', 2), candidate('b', 1),
            candidate('b', 2, keep=False), candidate('c', 1, keep=False)]
    chosen = choose(rows, count=2)
    for role in chosen:
        assert chosen[role]['a']['attempt'] == 2
        assert chosen[role]['b']['attempt'] == 1
        assert 'c' not in chosen[role]
    with pytest.raises(ValueError, match='only 2'):
        choose(rows, count=3)


def test_role_targets_and_shared_source_holdout_no_cross_role_leak():
    rows = [candidate(str(i), 1) for i in range(205)] + [candidate('revision_only', 1, korean=0)]
    selected = choose(rows, count=200)
    mapped = {'global': selected['revision'], 'korean': selected['korean']}
    split = source_split(mapped, {s: {'source_id': s} for values in mapped.values() for s in values}, seed=871)
    held = set().union(*(set(v['validation_sources']) for v in split['roles'].values()))
    train = set().union(*(set(v['train_sources']) for v in split['roles'].values()))
    assert not held & train
    assert all(len(v['validation_sources']) == 10 for v in split['roles'].values())


@pytest.fixture
def tok():
    from verak.v4.dryrun_common import tokenizer
    return tokenizer()


def prepared():
    value = normalize_row(row('고칠 표현이다.'), FakeBoundaryAnalysis())
    value.update(genre='explanatory')
    return value


def task():
    return {'item_id': 'private:example:1', 'owner': 'revision', 'action': 'EDIT',
            'location': ['S1'], 'instruction': '표현을 고친다.', 'needs_search': 'no'}


STOP = json.dumps({'action': 'STOP', 'status': 'done', 'summary': '작업을 보고한다.', 'issues': []}, ensure_ascii=False)


class Backend:
    def __init__(self, raw):
        self.raw = iter(raw); self.roles = []
        from verak.v4.dryrun_common import tokenizer
        self.tok = tokenizer()
    def generate(self, messages, *, episode_id, role, turn):
        self.roles.append(role)
        return {'raw': next(self.raw), 'finish_reason': 'stop', 'usage': {'prompt_tokens':
            len(self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))}}


def test_strict_student_keeps_multi_json_invalid_and_full_handoff(tmp_path, tok):
    edit = json.dumps({'action': 'EDIT', 'sentence': 'S1', 'old': '고칠', 'new': '고친'}, ensure_ascii=False)
    backend = Backend([edit+'\n'+STOP, edit, STOP, STOP])
    result = run_case(prepared(), {'items': [task()]}, backend, tok,
                      FakeBoundaryAnalysis(), output_root=tmp_path)
    assert result['complete'] and result['status'] == 'completed'
    assert result['calls'][0]['raw'] == edit+'\n'+STOP
    assert result['calls'][0]['action']['valid'] is False
    assert 'canonical_action' not in result['calls'][0]
    assert result['calls'][1]['action']['valid'] is True
    assert result['final_text'] == '고친 표현이다.'
    assert result['states']['revision_final'] == result['states']['korean_initial']
    assert all(result['handoff_audit'][k] for k in
               ('document_identical', 'whole_essay_scope', 'handoff_present', 'journal_present', 'notices_present'))
    prefix = json.loads(Path(result['calls'][0]['prefix']['path']).read_text())
    assert len(prefix['prefix_input_ids']) == prefix['prefix_tokens']
    summary = metrics([result], planned=1)
    assert summary['roles']['revision']['valid_action_rate'] == 2/3
    assert summary['roles']['korean']['STOP_status'] == {'done': 1}
    assert run_case(prepared(), {'items': [task()]}, Backend([]), tok,
                    FakeBoundaryAnalysis(), output_root=tmp_path) == result


def test_context_overflow_saved_exactly_never_sent_or_truncated(tmp_path, tok):
    class LongPrefix:
        def apply_chat_template(self, messages, **kwargs): return list(range(7169))
        def encode(self, text, **kwargs): return tok.encode(text, **kwargs)
    backend = Backend([])
    result = run_case(prepared(), {'items': [task()]}, backend, LongPrefix(),
                      FakeBoundaryAnalysis(), output_root=tmp_path)
    assert not backend.roles and not result['calls']
    assert len(result['context_failures']) == 2
    assert result['delegations'][0]['ending'] == result['korean_stage']['ending'] == 'context_limit'
    for entry in result['context_failures']:
        saved = json.loads(Path(entry['path']).read_text())
        assert saved['prefix_input_ids'] == list(range(7169))
        assert saved['current_text'] == prepared()['text']


def test_invalid_role_action_and_step_budget_remain_visible(tmp_path, tok):
    backend = Backend([STOP]+['{"action":"DELETE","sentence":"S1"}']*14)
    result = run_case(prepared(), {'items': [task()]}, backend, tok,
                      FakeBoundaryAnalysis(), output_root=tmp_path)
    assert result['termination']['korean'] == 'step_limit'
    assert result['final_text'] == prepared()['text']
    assert metrics([result], planned=1)['roles']['korean']['invalid_action_types'] == {'DELETE': 14}


def test_local_backend_exception_is_saved_with_exact_failed_prefix(tmp_path, tok):
    class FailedBackend:
        def generate(self, messages, **kwargs):
            raise RuntimeError('local transport failure fixture')
    result = run_case(prepared(), {'items': [task()]}, FailedBackend(), tok,
                      FakeBoundaryAnalysis(), output_root=tmp_path)
    assert result['status'] == 'error' and not result['calls']
    failure = next((tmp_path/'requests').glob('*/*.failure.json'))
    value = json.loads(failure.read_text())
    assert value['retried'] is False and 'local transport failure fixture' in value['error']
    prefix = json.loads(Path(value['prefix']['path']).read_text())
    assert prefix['prefix_input_ids'] == tok.apply_chat_template(prefix['public_messages'],
        tokenize=True, add_generation_prompt=True)
    assert run_case(prepared(), {'items': [task()]}, Backend([]), tok,
                    FakeBoundaryAnalysis(), output_root=tmp_path) == result


def test_saved_action_export_prefix_labels_and_eos(tok):
    path = ROOT/'data/revision.train.jsonl.gz'
    if not path.exists():
        pytest.skip('Local dry-run dataset absent')
    with gzip.open(path, 'rt') as stream:
        sample = json.loads(next(stream))
    good = audit_sample(sample, tok)
    assert good['end_turn_token'] == tok.eos_token_id
    assert good['prefix_tokens'] <= 7168 and good['total_tokens'] <= 8192
    for index, value in [(0, sample['input_ids'][0]), (-1, -100)]:
        broken = deepcopy(sample); broken['labels'][index] = value
        with pytest.raises(ValueError): audit_sample(broken, tok)
    broken = deepcopy(sample); broken['messages'][0]['content'] += '\nNEW RULE'
    with pytest.raises(ValueError): audit_sample(broken, tok)


@pytest.mark.parametrize('observed,status', [(None, 'missing'), (True, 'missing'), (-1, 'missing'),
                                          (45, 'mismatch'), (100, 'match')])
def test_vllm_template_count_audit(observed, status):
    from verak.v4.dryrun_student import prompt_token_audit
    assert prompt_token_audit(100, {'prompt_tokens': observed})['status'] == status


def test_template_mismatch_preserves_raw_and_stops_without_edit(tmp_path, tok):
    class WrongUsage(Backend):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            result['usage']['prompt_tokens'] += 1
            return result
    result = run_case(prepared(), {'items': [task()]}, WrongUsage([STOP, STOP]), tok,
                      FakeBoundaryAnalysis(), output_root=tmp_path)
    assert len(result['calls']) == 2 and not result['complete']
    assert all(c['raw'] == STOP and c['action']['action'] == 'NOT_EXECUTED_TEMPLATE' for c in result['calls'])
    assert result['final_text'] == prepared()['text']
    assert metrics([result], planned=1)['chat_template_audit']['status_counts'] == {'mismatch': 2}


@pytest.mark.parametrize('output', ['N/A\n', 'not-a-PID\n', '12\nN/A\n', '0\n'])
def test_gpu_idle_malformed_does_not_mean_idle(monkeypatch, output):
    from verak.v4 import dryrun_service as service
    monkeypatch.setattr(service.subprocess, 'check_output', lambda *a, **k: output)
    with pytest.raises(RuntimeError, match='malformed PID'):
        service.gpu1_idle()


def test_gpu_idle_foreign_pid_and_uuid_match_are_checked(monkeypatch):
    from verak.v4 import dryrun_service as service
    monkeypatch.setattr(service.subprocess, 'check_output', lambda *a, **k: '1234\n')
    with pytest.raises(RuntimeError, match='occupied'):
        service.gpu1_idle()
    monkeypatch.setattr(service.subprocess, 'check_output', lambda *a, **k: 'GPU-example-uuid\n')
    assert service.gpu1_uuid('example-uuid')['uuid_match']
    with pytest.raises(RuntimeError, match='UUID does not match'):
        service.gpu1_uuid('different-device')


def test_gpu_guard_rejects_unreviewed_or_wrong_physical_device(tmp_path, monkeypatch):
    from verak.v4 import dryrun_common as common
    (tmp_path/'data').mkdir()
    (tmp_path/'data/manifest.json').write_text('{}')
    (tmp_path/'preflight.json').write_text('{}')
    approval = {'approved': True, 'physical_gpu': 1,
        'data_manifest_sha256': common.file_sha(tmp_path/'data/manifest.json'),
        'preflight_sha256': common.file_sha(tmp_path/'preflight.json')}
    common.write_json(tmp_path/'gpu_approval.json', approval)
    for value in ('', '0', '0,1'):
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', value)
        with pytest.raises(ValueError, match='exactly physical GPU1'):
            require_gpu_approval(root=tmp_path)
    approval['physical_gpu'] = 0
    common.write_json(tmp_path/'gpu_approval.json', approval)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    with pytest.raises(ValueError, match='reviewed GPU1'):
        require_gpu_approval(root=tmp_path)
