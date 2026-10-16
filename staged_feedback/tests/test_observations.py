def feedback(quote):
    return [{"issue_id": "x", "target_quote": quote}]


def test_split_merge_not_whole_sentence_deletion(observer):
    original = "나는 책을 읽고 친구와 이야기했다."
    split = "나는 책을 읽었다. 친구와 이야기했다."
    obs = observer.compare(original, split, feedback=feedback("책을 읽고"))
    assert len(obs["changed_sentences"][0]["after_sentences"]) == 2
    assert not any(f["kind"] == "student_sentence_deleted" for f in obs["flags"])
    merged = observer.compare(split, original, feedback=feedback("친구와"))
    assert not any(f["kind"] == "student_sentence_deleted" for f in merged["flags"])


def test_previous_sentence_tracking_does_not_expand_over_equal_block(observer):
    before = "하나였다. 둘이었다. 셋이었다."
    first = "하나이다. 둘이었다. 셋이었다."
    obs1 = observer.compare(before, first)
    obs2 = observer.compare(first, first, previous_changed_ranges=obs1["next_changed_ranges"])
    assert len(obs2["next_changed_ranges"]) == 1
    obs3 = observer.compare(first, "하나이다. 둘이다. 셋이었다.", previous_changed_ranges=obs2["next_changed_ranges"])
    assert not any(f["kind"] == "earlier_revision_changed" for f in obs3["flags"])


def test_outside_target_changes_and_unknown_quote_are_annotations(observer):
    observed = observer.compare("책을 읽었다. 공을 찼다.", "책을 읽었다. 공을 던졌다.", feedback=feedback("책을 읽었다"))
    assert any(f["kind"] == "outside_feedback_target" for f in observed["flags"])
    unknown = observer.compare("읽었다.", "독서했다.", feedback=feedback("없는 인용"))
    assert unknown["unlocated_feedback_issue_ids"] == ["x"]


def test_marker_deletion_and_malformed_placeholder_observed(observer):
    observed = observer.compare("#@이름#이 왔다.", "__ANON_broken 왔다.")
    marker = next(f for f in observed["flags"] if f["kind"] == "anonymization_changed")
    assert marker["before"] == ["#@이름#"] and marker["malformed_placeholder_present"]


def test_layout_change_is_recorded(observer):
    obs = observer.compare("책을 읽었다.\n친구를 만났다.", "책을 읽었다.\n\n친구를 만났다.")
    assert not obs["changed_sentences"]
    assert obs["layout_changes"]


def test_insertion_at_untargeted_sentence_start_is_observed(observer):
    obs = observer.compare("책을 읽었다. 공을 찼다.", "책을 읽었다. 신나게 공을 찼다.", feedback=feedback("책을 읽었다"))
    assert any(f["kind"] == "outside_feedback_target" and f["sentence"] == "공을 찼다." for f in obs["flags"])


def test_deleted_duplicate_does_not_borrow_later_survivor(observer):
    obs = observer.compare("동일하다. 차이가 난다. 동일하다.", "차이가 난다. 동일하다.")
    deleted = [f for f in obs["flags"] if f["kind"] == "student_sentence_deleted"]
    assert len(deleted) == 1 and deleted[0]["before_span"][0] == 0


def test_multiple_feedback_quotes_for_one_issue_keep_all_targets(observer):
    obs = observer.compare("책을 읽었다. 공을 찼다.", "책을 읽는다. 공을 찬다.", feedback=feedback("책")+feedback("공"))
    assert len(obs["feedback_targets"]["x"]) == 2
    assert not any(f["kind"] == "outside_feedback_target" for f in obs["flags"])
