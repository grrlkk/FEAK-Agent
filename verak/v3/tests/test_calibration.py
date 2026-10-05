from types import SimpleNamespace

import pytest

from verak.v3.calibration import noise_statistics, summarize_noise, validate_surface, whitespace_variant


class Analyzer:
    def __init__(self, style="해라체", sentences=1):
        self.style, self.sentences = style, sentences

    def profile(self, text):
        return SimpleNamespace(sentences=[SimpleNamespace(style_candidates=[self.style])]
                              * self.sentences)


def test_whitespace_probe_changes_exactly_one_space_outside_markers():
    original = "#@사람 이름# 학생은 물을 관찰했다."
    changed, position = whitespace_variant(original)
    assert len(changed) == len(original) + 1
    assert changed[:position] + changed[position+1:] == original
    assert "#@사람 이름#" in changed
    assert position >= len("#@사람 이름#")


def test_paraphrase_surface_contract_preserves_numbers_markers_register():
    original = "#@이름# 학생은 3가지 수질 지표를 조사했고, 그 내용을 보고서에 정확하게 기록했다."
    candidate = original.replace("정확하게", "꼼꼼하게")
    assert validate_surface(original, candidate, ["해라체"], Analyzer())["sentences_after"] == 1
    with pytest.raises(ValueError, match="Numeric"):
        validate_surface(original, candidate.replace("3가지", "4가지"), ["해라체"], Analyzer())
    with pytest.raises(ValueError, match="markers"):
        validate_surface(original, candidate.replace("#@이름#", "#@지명#"), ["해라체"], Analyzer())
    with pytest.raises(ValueError, match="register"):
        validate_surface(original, candidate, ["해라체"], Analyzer(style="하십시오체"))
    with pytest.raises(ValueError, match="exactly one"):
        validate_surface(original, candidate, ["해라체"], Analyzer(sentences=2))
    with pytest.raises(ValueError, match="Length"):
        validate_surface(original, candidate + " 이 내용은 추가 설명이다.", ["해라체"], Analyzer())


def test_noise_floor_uses_signed_delta_std_not_std_of_absolute_delta():
    records = [{"id": str(i), "genre": "설명", "variant": variant, "delta": value}
               for variant in ("whitespace", "paraphrase") for i, value in enumerate((-1., 1.))]
    summary = summarize_noise(records, 2)
    assert summary["noise_floor"] == 2
    assert summary["variants"]["paraphrase"]["overall"]["std_abs_delta"] == 0
    assert summary["variants"]["paraphrase"]["overall"]["mean_abs_delta"] == 1
    assert summary["variants"]["paraphrase"]["overall"]["ddof"] == 0
    with pytest.raises(ValueError, match="incomplete"):
        summarize_noise(records[:-1], 2)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_noise(records + [records[-1]], 2)
