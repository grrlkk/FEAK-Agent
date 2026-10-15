import json
import pytest

from verak.v5 import common


def test_reference_score_fallback_is_read_only_and_checks_identity(tmp_path, monkeypatch):
    old, new = tmp_path/'old', tmp_path/'new'
    monkeypatch.setattr(common, 'REPO', tmp_path)
    common.atomic_new(tmp_path/'verak/v4/outputs/train44/smoke/scorer_identity.json', {'fingerprint': 'reference'})
    question, text = '질문', '원문'
    name = common.sha_text(json.dumps([question, text], ensure_ascii=False))+'.json'
    path = old/'scores'/name
    value = {'question': question, 'text': text, 'result': {'Q': 3.5}, 'fingerprint': 'reference'}
    common.atomic_new(path, value)
    scorer = common.ReferenceScores(root=new, fallback=old)
    assert scorer(question, text) == {'Q': 3.5}
    assert scorer.scorer is None and not new.exists()
    value['fingerprint'] = 'different'
    common.write_json(path, value)
    with pytest.raises(ValueError, match='fingerprint'):
        scorer(question, text)


def test_default_reference_score_root_remains_pipe1(monkeypatch, tmp_path):
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    assert common.ReferenceScores().root == tmp_path
