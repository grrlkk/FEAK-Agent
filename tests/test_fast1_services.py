import json
import math

from verak.v5 import fast1_services as services


def test_score_delegates_to_reference_scores_and_preserves_result(monkeypatch, tmp_path):
    calls = []

    class FakeReferenceScores:
        def __init__(self, *, root, fallback):
            calls.append(('init', root, fallback))
            (root).mkdir(parents=True, exist_ok=True)
            (root / 'scorer_identity.json').write_text(json.dumps({'fingerprint': 'frozen-fp'}))

        def __call__(self, question, text):
            calls.append((question, text))
            return {'expected': [5.0] * 8, 'cache_hit': True, 'cache_key': 'pair-key'}

    monkeypatch.setattr(services.common, 'ReferenceScores', FakeReferenceScores)
    monkeypatch.setattr(services, 'SCORES_ROOT', tmp_path / 'shared')
    monkeypatch.setattr(services, 'REFERENCE_FALLBACK', tmp_path / 'pipe1')
    monkeypatch.setattr(services, '_score', None)

    result = services._score_call('q', 'normalized essay')

    assert result == {'result': {'expected': [5.0] * 8, 'cache_hit': True,
                                 'cache_key': 'pair-key'}, 'fingerprint': 'frozen-fp'}
    assert calls[-1] == ('q', 'normalized essay')


def test_perplexity_uses_prompt_logprobs_and_caches(monkeypatch):
    class FakeTokenizer:
        def encode(self, prompt, add_special_tokens=True):
            return [1, 2, 3]

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({'usage': {'prompt_tokens': 3}, 'choices': [
                {'logprobs': {'token_logprobs': [None, -0.1, -0.2, -8.0],
                              'text_offset': [0, 1, 2, 6]}}]}).encode()

    monkeypatch.setattr(services.urllib.request, 'urlopen', lambda *args, **kwargs: Response())
    monkeypatch.setattr(services, '_ppl_cache', {})
    monkeypatch.setattr(services.common, 'tokenizer', lambda: FakeTokenizer())
    monkeypatch.setattr(services, '_policy_tokenizer', None)

    first = services._completion_perplexities(['prompt'])[0]
    second = services._completion_perplexities(['prompt'])[0]

    assert math.isclose(first['perplexity'], math.exp(0.15))
    assert first['cache_hit'] is False
    assert second['cache_hit'] is True
