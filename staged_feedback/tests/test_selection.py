from types import SimpleNamespace

import pytest

from staged_feedback.selection import diverse_sample


def test_new_cohort_has_one_per_stratum_and_three_genres_reproducibly():
    candidates, genres = [], {}
    for number in range(9):
        identifier = f"synthetic-{number}"
        candidates.append(SimpleNamespace(essay=SimpleNamespace(essay_id=identifier), total=number,
            provenance={"essay_id": identifier}))
        genres[identifier] = ("설명", "논증", "정서")[number % 3]
    first, manifest = diverse_sample(candidates, genres, seed=73)
    second, _ = diverse_sample(candidates, genres, seed=73)
    assert first == second
    assert {genres[c.essay.essay_id] for c in first} == {"설명", "논증", "정서"}
    assert [c["stratum"] for c in manifest["selected"]] == ["low", "middle", "high"]
    assert first[0].total <= manifest["cutpoints"][0] < first[1].total <= manifest["cutpoints"][1] < first[2].total


def test_missing_genre_diversity_fails_without_silent_substitution():
    candidates = [SimpleNamespace(essay=SimpleNamespace(essay_id=str(n)), total=n,
                                  provenance={"essay_id": str(n)}) for n in range(9)]
    with pytest.raises(ValueError, match="three distinct genres"):
        diverse_sample(candidates, {str(n): "설명" for n in range(9)}, seed=73)
