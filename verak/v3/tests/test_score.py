from types import SimpleNamespace

import pytest
import torch

from verak.v3.common import sha_text, write_json
from verak.v3.score.kanana import (DIGIT_IDS, InputTooLong, KananaScorer, ScoreCache,
    ScoreParseError, expected_from_digit_logits, parse_first_line, teacher_positions)


def test_expectation_uses_nine_raw_logits_and_preserves_small_mass():
    logits = torch.zeros((8, 9), dtype=torch.float64)
    expected, integers = expected_from_digit_logits(logits)
    assert expected == pytest.approx([5.] * 8)
    assert integers == [1] * 8
    logits[:, 8] = 4
    expected, integers = expected_from_digit_logits(logits)
    assert all(8 < value < 9 for value in expected)
    assert integers == [9] * 8
    assert expected_from_digit_logits(logits - 10000)[0] == pytest.approx(expected)


def test_next_token_logit_alignment_and_single_digit_contract():
    ids = [16, 220, 17, 220, 18, 220, 19, 220, 20, 220, 21, 220, 22, 220, 23]
    assert teacher_positions(10, ids) == [9, 11, 13, 15, 17, 19, 21, 23]
    for invalid in ("", "1 2 3", "1 2 3 4 5 6 7 10", "01 2 3 4 5 6 7 8", "1 2 3 4 5 6 7 8\nfeedback"):
        with pytest.raises(ScoreParseError):
            parse_first_line(invalid)
    with pytest.raises(ScoreParseError):
        teacher_positions(10, [1721] + ids[1:])  # 01 is not the digit 1.
    assert parse_first_line("1 2 3 4 5 6 7 8") == list(range(1, 9))


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        mapping = {str(i): i + 15 for i in range(1, 10)} | {" ": 220, "\n": 198}
        return [mapping[x] for x in text]

    def decode(self, ids, skip_special_tokens=True):
        inverse = {i + 15: str(i) for i in range(1, 10)} | {220: " ", 198: "\n", 0: "", 99: ""}
        return "".join(inverse[int(x)] for x in ids)

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]

    def __call__(self, prompt, **kwargs):
        assert kwargs["truncation"] is False
        length = 4000 if "TOO_LONG" in prompt else 10
        return {"input_ids": torch.ones((1, length), dtype=torch.long),
                "attention_mask": torch.ones((1, length), dtype=torch.long)}


class Model:
    device = torch.device("cpu")

    def __init__(self):
        self.generations = 0
        self.forwards = 0

    def eval(self):
        return self

    def generate(self, input_ids, **kwargs):
        assert kwargs["do_sample"] is False and kwargs["num_beams"] == 1
        assert kwargs["temperature"] is None and kwargs["top_k"] is None
        assert "output_scores" not in kwargs
        self.generations += 1
        suffix = Tokenizer().encode("1 2 3 4 5 6 7 8\n")
        output = torch.cat([input_ids, torch.tensor([suffix])], dim=1)
        assert kwargs["stopping_criteria"](output, None).item()
        return output

    def __call__(self, input_ids, attention_mask, use_cache):
        self.forwards += 1
        assert use_cache is False
        logits = torch.zeros((1, input_ids.shape[1], 1800))
        logits[:, :, 1721] = 100  # Invalid 01 token must contribute zero mass.
        positions = teacher_positions(10, input_ids[0, 10:].tolist())
        for i, position in enumerate(positions):
            logits[0, position, DIGIT_IDS[i]] = 7
        return SimpleNamespace(logits=logits)


def make_scorer(tmp_path):
    write_json(tmp_path / "genres.json", {"questions": {sha_text("question"): {"genre": "설명"}}})
    config = {"paths": {"metadata": tmp_path, "output": tmp_path / "output",
                         "policy_base": tmp_path / "base", "scorer_adapter": tmp_path / "adapter"},
              "scorer": {"max_input_tokens": 3072, "max_new_tokens": 32, "cache": True}}
    model = Model()
    return KananaScorer(config, model=model, tokenizer=Tokenizer()), model


def test_scorer_determinism_is_not_a_cache_test_and_cache_key_uses_text(tmp_path):
    scorer, model = make_scorer(tmp_path)
    first = scorer.score("question", "essay", use_cache=False)
    second = scorer.score("question", "essay", use_cache=False)
    assert first.expected == second.expected and first.integers == list(range(1, 9))
    assert model.generations == model.forwards == 2
    assert not first.cache_hit and not second.cache_hit
    hit = scorer.score("question", "essay")
    assert hit.cache_hit and model.generations == 2
    changed = scorer.score("question", "essay ")
    assert changed.cache_key != first.cache_key and model.generations == 3
    assert first.genre == "설명" and first.mean == sum(first.expected) / 8
    scorer.close()


def test_long_input_fails_before_generation_and_cache_does_not_cross_models(tmp_path):
    scorer, model = make_scorer(tmp_path)
    with pytest.raises(InputTooLong) as info:
        scorer.score("question", "TOO_LONG")
    assert info.value.tokens == 4000 and model.generations == 0
    scorer.close()
    with pytest.raises(ValueError, match="different model"):
        ScoreCache(tmp_path / "output/score_cache.sqlite", "wrong-fingerprint")
