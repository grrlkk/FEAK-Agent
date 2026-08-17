from collections import Counter

import pytest

from feak_tc.corruption.shuffle_sensitivity import (
    build_shuffle_sensitivity_chains,
    summarize_shuffle_measurements,
)
from feak_tc.diagnose.stub import split_sentences


ESSAY = (
    "첫째 문장이다. 둘째 문장은 앞 내용을 설명한다. "
    "셋째 문장은 구체적인 사례를 제시한다. 넷째 문장은 의미를 정리한다. "
    "마지막 문장은 결론을 제시한다."
)


def test_build_shuffle_sensitivity_preserves_sentence_multiset():
    chains = build_shuffle_sensitivity_chains(
        [{"record_id": "essay-1", "question": "질문", "text": ESSAY}],
        levels=("single", "partial", "full"),
        replicates=2,
        seed=7,
    )
    assert len(chains) == 6
    expected = Counter(split_sentences(ESSAY))
    for chain in chains:
        assert chain["states"][0] != chain["states"][1]
        assert Counter(split_sentences(chain["states"][1])) == expected


def test_summarize_shuffle_measurements_reports_paired_target_drop():
    chains = build_shuffle_sensitivity_chains(
        [{"record_id": "essay-1", "text": ESSAY}],
        levels=("full",),
        replicates=2,
        seed=7,
    )
    measurements = {}
    for chain in chains:
        before = [5.0] * 8
        after = [5.0] * 8
        after[4] = 4.2
        measurements[(chain["record_id"], 0)] = {"rf_corrected": before}
        measurements[(chain["record_id"], 1)] = {"rf_corrected": after}
    report = summarize_shuffle_measurements(chains, measurements)
    full = report["by_level"]["full"]
    assert report["pairs"] == 2
    assert full["target_drop"]["median"] == pytest.approx(0.8)
    assert full["above_threshold"]["0.5"] == 2
    assert full["target_specificity_median"] == pytest.approx(0.8)
