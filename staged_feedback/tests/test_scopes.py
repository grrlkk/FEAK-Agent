import json

from conftest import ScriptedLLM, sentences, issue
from test_engine import execute
from staged_feedback.models import FeedbackItem, Issue, Restore
from staged_feedback.scopes import document_units, revision_scopes, restore_sentences, assemble


def feedback(key="i1", ids=None, quote="읽을수"):
    return FeedbackItem(issue_id=key, target_quote=quote, target_ids=ids or [],
                        what="띄어쓰기", how="읽을 수", student_explanation="띄어 씁니다.")


def test_scope_excludes_remote_sentences_and_preserves_every_outside_byte(tmp_path, observer):
    original = "맨 앞 비밀. 앞 맥락. 책을 읽을수 있다. 뒤 맥락. 맨 뒤 비밀."
    class Scoped(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            response = super().request(role, payload, schema, context=context)
            if role == "feedback":
                response.items[0].target_ids = ["s3"]
            return response
    result, _, llm, _ = execute(tmp_path, observer, text=original, llm=Scoped())
    request = next(c["payload"] for c in llm.calls if c["role"] == "reviser")
    assert set(request) == {"source_text", "context_before", "context_after", "feedback"}
    assert request["source_text"] == "책을 읽을수 있다."
    assert request["context_before"] == "앞 맥락." and request["context_after"] == "뒤 맥락."
    assert "비밀" not in json.dumps(request, ensure_ascii=False)
    assert result["final_text"] == original.replace("읽을수", "읽을 수")


def test_macro_selects_paragraph_and_merges_overlapping_micro():
    original = "첫 문단이다.\n  책을 읽을수 있다. 같이 읽는다.\n끝 문단이다."
    issues = {"m": Issue(id="m", level="macro", macro_type="restructure", description="재배열", evidence_quotes=["책"], priority=1), "i1": issue()}
    scopes, missing = revision_scopes(original, [feedback("m", ["p2"]), feedback(ids=["s2"])], issues, sentences)
    assert not missing and len(scopes) == 1
    assert scopes[0]["source_text"] == "책을 읽을수 있다. 같이 읽는다."
    assert len(scopes[0]["feedback"]) == 2
    assert scopes[0]["context_before"] == "첫 문단이다."
    assert scopes[0]["context_after"] == "끝 문단이다."


def test_nonadjacent_targets_never_pull_intervening_sentence_into_scope():
    original = "읽는다. 그대로 둔다. 읽는다."
    scopes, missing = revision_scopes(original, [feedback(ids=["s1", "s3"], quote="읽는다.")], {"i1": issue()}, sentences)
    assert not missing and len(scopes) == 2
    assert all(s["source_text"] == "읽는다." for s in scopes)
    revised = assemble(original, [{"start": s["start"], "end": s["end"], "replacement": "즐겁게 읽는다."} for s in scopes])
    assert revised == "즐겁게 읽는다. 그대로 둔다. 즐겁게 읽는다."


def test_adjacent_paragraph_targets_allow_reordering():
    original = "결론이다.\n이유이다.\n그대로 둔다."
    macro = Issue(id="m", level="macro", macro_type="restructure", description="이유 먼저", evidence_quotes=["결론"], priority=1)
    scopes, _ = revision_scopes(original, [feedback("m", ["p1", "p2"])], {"m": macro}, sentences)
    assert len(scopes) == 1 and scopes[0]["source_text"] == "결론이다.\n이유이다."


def test_ambiguous_quote_is_reported_without_guessing_or_losing_text():
    scopes, missing = revision_scopes("읽는다. 읽는다.", [feedback(quote="읽는다.")], {"i1": issue()}, sentences)
    assert not scopes and len(missing) == 1


def test_micro_paragraph_id_uses_unique_quote_to_find_sentence_scope():
    original = "앞 문장이다. 책을 읽을수 있다. 뒤 문장이다."
    scopes, missing = revision_scopes(original, [feedback(ids=["p1"])], {"i1": issue()}, sentences)
    assert not missing and len(scopes) == 1
    assert scopes[0]["source_text"] == "책을 읽을수 있다."
    assert scopes[0]["target_ids"] == ["s2"]


def test_restore_replacement_deletion_and_split_exact_previous_sentences():
    previous = "처음이다. 학생 말이다. 끝이다."
    cases = [
        ("처음이다. 엉뚱한 말이다. 끝이다.", ["s2"], None),
        ("처음이다. 끝이다.", [], "s2"),
        ("처음이다. 엉뚱하다. 새 말이다. 끝이다.", ["s2", "s3"], None),
    ]
    for current, current_ids, anchor in cases:
        restored, records = restore_sentences(previous, current, [Restore(previous_sentence_ids=["s2"],
            current_sentence_ids=current_ids, insert_before_sentence_id=anchor, reason="불필요한 변경")], sentences)
        assert restored == previous
        assert records[0]["status"] == "applied"


def test_invalid_restore_reference_preserves_candidate_and_records_problem():
    candidate = "새 문장이다."
    result, records = restore_sentences("원래 문장이다.", candidate,
        [Restore(previous_sentence_ids=["s99"], current_sentence_ids=["s1"], reason="총괄 지시")], sentences)
    assert result == candidate and records[0]["status"] == "unavailable"


def test_coordinator_restore_becomes_next_turn_input(tmp_path, observer):
    class Restoring(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            response = super().request(role, payload, schema, context=context)
            if role == "coordinator_review" and context["turn"] == 1:
                response.restore = [Restore(previous_sentence_ids=["s1"], current_sentence_ids=["s1"], reason="요구하지 않은 변경")]
            return response
    result, _, llm, _ = execute(tmp_path, observer, llm=Restoring(turns=2, revisions=["엉뚱하게 썼다.", lambda t:t.replace("읽을수", "읽을 수")]))
    assert result["turns"][0]["candidate_text"].startswith("엉뚱하게")
    assert result["turns"][0]["revised_text"] == result["initial_text"]
    second = [c for c in llm.calls if c["role"] == "reviser"][1]
    assert second["payload"]["source_text"] == "책을 읽을수 있다."
    assert result["final_text"] == "책을 읽을 수 있다. 함께 연습한다."


def test_remaining_turns_means_future_turns_and_can_extend_plan(tmp_path, observer):
    class Extended(ScriptedLLM):
        def request(self, role, payload, schema, *, context):
            response = super().request(role, payload, schema, context=context)
            if role == "coordinator_initial":
                response.remaining_turns = 2
            return response
    result, _, llm, _ = execute(tmp_path, observer, llm=Extended(turns=3))
    assert len(result["turns"]) == 3
    assert result["turns"][0]["remaining_before"] == 2
    assert result["turns"][0]["remaining_after_consumption"] == 1
    assert result["turns"][0]["remaining_after_review"] == 2
    assert result["remaining_turns"] == 0
    for call in llm.calls:
        assert "turn_budget" not in call["payload"]
