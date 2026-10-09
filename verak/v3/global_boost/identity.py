"""Exact zero quality delta from identical scorer inputs, never invented scores."""
import json

from ..common import sha_text
from ..score.kanana import SCORER_VERSION, SYSTEM_PROMPT


def identity_quality(config, genre, question, before_text, after_text, *, scorer_contract=None):
    contract = scorer_contract if scorer_contract is not None else {
        'version': SCORER_VERSION, 'settings': config['scorer'],
        'base': str(config['paths']['policy_base']), 'adapter': str(config['paths']['scorer_adapter'])}
    def serialized(text):
        # KananaScorer.prepare_input uses these exact messages, followed by the
        # same pinned tokenizer/chat template for both states. Equal UTF-8
        # request bytes imply equal serialized model inputs and the same cache.
        value = {'contract': contract, 'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f'질문: {question}\n에세이: {text}'}]}
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    before, after = serialized(before_text), serialized(after_text)
    if before.encode('utf-8') != after.encode('utf-8'):
        return None
    settings = config['reward']['quality']
    return {'value': 0., 'delta': 0.,
        'noise_floor': settings['noise_floor_by_genre'].get(genre, settings['noise_floor']),
        'before': None, 'after': None, 'measurement': 'identical_serialized_scorer_input',
        'identity': {'before_input_sha256': sha_text(before), 'after_input_sha256': sha_text(after),
            'question_sha256': sha_text(question), 'essay_sha256': sha_text(before_text),
            'scorer_contract': contract}}
