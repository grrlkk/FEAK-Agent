import json
from types import SimpleNamespace

import pytest

import feak_tc.diagnose.kanana as kanana_module
from feak_tc.diagnose import Diagnosis, RUBRIC_KEYS, StubDiagnoser
from feak_tc.diagnose.constants import FEAK_FEATURE_NAMES
from feak_tc.diagnose.kanana import KananaDiagnoser
from feak_tc.mvp.batch import iter_text_records, run_batch
from feak_tc.mvp.heuristic import build_result, select
from feak_tc.mvp.patch import apply_patch
from feak_tc.mvp.propose import propose
from feak_tc.mvp.schemas import Candidate, Transition
from feak_tc.mvp.surface import normalize_surface_text
from feak_tc.mvp.targeting import select_target_span
from feak_tc.mvp.transition import compute_transition, edit_ratio, source_token_retention
from feak_tc.mvp import serializable_one_step

NON_STOP_ACTIONS = {
    "ADD_DETAIL",
    "DELETE_OR_FOCUS",
    "COMPRESS",
    "RESTRUCTURE",
    "STYLE_REFINE",
}


def test_stub_diagnoser_uses_kanana_rubric_shape():
    diag = StubDiagnoser().diagnose("인권은 인간이 가지는 기본적인 권리이다.")

    assert list(diag.rubrics) == RUBRIC_KEYS
    assert len(diag.features) == 29
    assert len(diag.weak_rubrics) == 3


def test_one_step_mvp_returns_decision_and_candidates():
    output = serializable_one_step(
        "인권은 인간이 가지는 기본적인 권리이다. 우리는 서로의 권리를 존중해야 한다.",
        diagnoser=StubDiagnoser(),
    )

    assert output["before"]["weak_rubrics"]
    assert len(output["results"]) == 5
    assert output["decision"]["decision"] in {"accept", "reject_all", "stop"}
    for result in output["results"]:
        assert result["candidate"]["action_type"]
        assert result["transition"]["target_rubric"]
        assert isinstance(result["heuristic_score"], float)


def test_kanana_diagnoser_extracts_features_before_loading_model():
    calls = []

    class FakeCorrection:
        def predict(self, means, stds, features):
            return means, means

    class OrderedDiagnoser(KananaDiagnoser):
        def _extract_features(self, text):
            calls.append("features")
            return {name: 0.0 for name in FEAK_FEATURE_NAMES}

        def _ensure_loaded(self):
            calls.append("model")
            return {
                "build_user_prompt": lambda question, text, keywords: text,
                "score_example": lambda *args: {
                    "raw_scores": [[1, 2, 3, 4, 1, 2, 3, 4]],
                    "feedbacks": [],
                },
                "model": object(),
                "tokenizer": object(),
                "helper": object(),
                "system_prompt": "system",
                "config": object(),
                "correction": FakeCorrection(),
            }

    OrderedDiagnoser().diagnose("인권은 기본권이다.")

    assert calls == ["features", "model"]


def test_kanana_diagnoser_normalizes_only_scoring_prompt_text():
    captured = {}

    class FakeCorrection:
        def predict(self, means, stds, features):
            return means, means

    class CapturingDiagnoser(KananaDiagnoser):
        def _extract_features(self, text):
            captured["feature_text"] = text
            return {name: 0.0 for name in FEAK_FEATURE_NAMES}

        def _ensure_loaded(self):
            def build_user_prompt(question, text, keywords):
                captured["prompt_text"] = text
                return text

            return {
                "build_user_prompt": build_user_prompt,
                "score_example": lambda *args: {
                    "raw_scores": [[1, 2, 3, 4, 1, 2, 3, 4]],
                    "feedbacks": [],
                },
                "model": object(),
                "tokenizer": object(),
                "helper": object(),
                "system_prompt": "system",
                "config": object(),
                "correction": FakeCorrection(),
            }

    raw_text = "  첫 문장이다.   다음이다.  \n    둘째   문장이다.  \n"
    diag = CapturingDiagnoser().diagnose(raw_text)

    assert diag.text == raw_text
    assert captured["feature_text"] == raw_text
    assert captured["prompt_text"] == "첫 문장이다. 다음이다.\n둘째 문장이다."
    assert diag.metadata["scoring_text_normalized"] is True


def test_kanana_feature_subprocess_uses_offline_stable_environment(monkeypatch, tmp_path):
    captured = {}

    def fake_run(*args, **kwargs):
        captured["env"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout='__FEAK_FEATURES_JSON__{"feature": 1.5}\n',
            stderr="",
        )

    monkeypatch.setattr(kanana_module, "_ensure_package_importable", lambda package_path: None)
    monkeypatch.setattr(kanana_module, "FEATURE_LOCK_PATH", tmp_path / "feature.lock")
    monkeypatch.setattr(kanana_module.subprocess, "run", fake_run)

    features = KananaDiagnoser()._extract_features("인권은 기본권이다.")

    assert features == {"feature": 1.5}
    assert captured["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert captured["env"]["OMP_NUM_THREADS"] == "1"
    assert captured["env"]["MKL_NUM_THREADS"] == "1"
    assert captured["env"]["MPLCONFIGDIR"] == "/tmp/feak_matplotlib"


def test_bareun_surface_normalizer_applies_allowed_categories(monkeypatch):
    monkeypatch.setenv("BAREUN_API_KEY", "test-key")

    def fake_post_json(url, *, payload, headers, timeout):
        assert url == "https://api.bareun.ai/bareun.RevisionService/CorrectError"
        assert headers["api-key"] == "test-key"
        assert payload["document"]["content"] == "이주노동자는기본적인 권리가 있다."
        return {
            "revised": "이주 노동자는 기본적인 권리가 있다.",
            "revised_blocks": [{"revisions": [{"category": "SPACING"}], "nested": []}],
            "tokens_count": 5,
            "language": "ko_KR",
        }

    monkeypatch.setattr("feak_tc.mvp.surface._post_json", fake_post_json)

    result = normalize_surface_text(
        "이주노동자는기본적인 권리가 있다.",
        cfg={"surface_normalizer": {"mode": "bareun"}},
    )

    assert result is not None
    assert result.applied is True
    assert result.rejected is False
    assert result.normalized_text == "이주 노동자는 기본적인 권리가 있다."


def test_bareun_surface_normalizer_rejects_standard_word_changes(monkeypatch):
    monkeypatch.setenv("BAREUN_API_KEY", "test-key")

    def fake_post_json(url, *, payload, headers, timeout):
        return {
            "revised": "오늘은 총각무를 다듬었다.",
            "revised_blocks": [{"revisions": [{"category": "STANDARD"}], "nested": []}],
        }

    monkeypatch.setattr("feak_tc.mvp.surface._post_json", fake_post_json)

    result = normalize_surface_text(
        "오늘은 알타리무를 다듬었다.",
        cfg={"surface_normalizer": {"mode": "bareun"}},
    )

    assert result is not None
    assert result.applied is False
    assert result.rejected is True
    assert result.normalized_text == "오늘은 알타리무를 다듬었다."
    assert "surface:disallowed_category:STANDARD" in result.reject_reasons
    assert result.metadata["suggested_text"] == "오늘은 총각무를 다듬었다."


def test_one_step_scores_raw_text_before_surface_normalization(monkeypatch):
    monkeypatch.setenv("BAREUN_API_KEY", "test-key")

    def fake_post_json(url, *, payload, headers, timeout):
        return {
            "revised": "이주 노동자는 기본적인 권리가 있다.",
            "revised_blocks": [{"revisions": [{"category": "SPACING"}], "nested": []}],
        }

    monkeypatch.setattr("feak_tc.mvp.surface._post_json", fake_post_json)

    class RecordingDiagnoser:
        def __init__(self):
            self.texts = []

        def diagnose(self, text):
            self.texts.append(text)
            return Diagnosis(
                text=text,
                rubrics={key: 5.0 for key in RUBRIC_KEYS},
                features={name: 0.0 for name in FEAK_FEATURE_NAMES},
                weak_rubrics=["task_1", "content_1", "content_2"],
                metadata={"call_index": len(self.texts) - 1},
            )

    diagnoser = RecordingDiagnoser()
    output = serializable_one_step(
        "이주노동자는기본적인 권리가 있다.",
        diagnoser=diagnoser,
        cfg={
            "surface_normalizer": {"mode": "bareun"},
            "proposer": {"mode": "deterministic"},
            "patcher": {"mode": "deterministic"},
        },
    )

    assert diagnoser.texts[:2] == [
        "이주노동자는기본적인 권리가 있다.",
        "이주 노동자는 기본적인 권리가 있다.",
    ]
    assert output["before"]["text"] == "이주노동자는기본적인 권리가 있다."
    assert output["action_before"]["text"] == "이주 노동자는 기본적인 권리가 있다."
    assert output["surface_normalization"]["original_text"] == "이주노동자는기본적인 권리가 있다."
    assert output["before"]["metadata"]["surface_normalizer"]["provider"] == "bareun"
    assert output["action_before"]["metadata"]["weak_rubrics_source"] == "pre_surface_diagnosis"


def test_llm_proposer_accepts_schema_valid_json(monkeypatch):
    diag = StubDiagnoser().diagnose("인권은 기본적인 권리이다. 서로 존중해야 한다.")

    def fake_request_json(**kwargs):
        return {
            "candidates": [
                {
                    "action_type": "ADD_DETAIL",
                    "target_rubric": "content_2",
                    "target_span": "인권은 기본적인 권리이다.",
                    "instruction": "인권의 구체적 사례를 한 문장 덧붙인다.",
                }
            ]
        }

    monkeypatch.setattr("feak_tc.mvp.propose.request_json", fake_request_json)

    candidates = propose(diag, cfg={"proposer": {"mode": "llm"}})

    assert len(candidates) == 5
    assert candidates[0].metadata["source"] == "llm_proposer"
    assert {candidate.action_type for candidate in candidates} == NON_STOP_ACTIONS


def test_deterministic_proposer_honors_n_per_action_without_stop():
    diag = StubDiagnoser().diagnose("인권은 기본적인 권리이다. 서로 존중해야 한다.")

    candidates = propose(diag, n_per_action=2, cfg={"proposer": {"mode": "deterministic"}})

    assert len(candidates) == 10
    assert {candidate.action_type for candidate in candidates} == NON_STOP_ACTIONS
    for action_type in NON_STOP_ACTIONS:
        assert sum(1 for candidate in candidates if candidate.action_type == action_type) == 2


def test_auto_proposer_falls_back_when_llm_unavailable(monkeypatch):
    diag = StubDiagnoser().diagnose("인권은 기본적인 권리이다. 서로 존중해야 한다.")

    def fake_request_json(**kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("feak_tc.mvp.propose.request_json", fake_request_json)

    candidates = propose(diag, cfg={"proposer": {"mode": "auto"}})

    assert len(candidates) == 5
    assert all(candidate.metadata["source"] == "deterministic_proposer" for candidate in candidates)
    assert all("llm_error" in candidate.metadata for candidate in candidates)


def test_llm_patcher_applies_structured_patch(monkeypatch):
    candidate = Candidate(
        action_type="ADD_DETAIL",
        target_rubric="content_2",
        target_span="인권은 기본적인 권리이다.",
        instruction="구체적 사례를 덧붙인다.",
    )

    def fake_request_json(**kwargs):
        return {
            "operation": "insert_after",
            "target_span": "인권은 기본적인 권리이다.",
            "before": "인권은 기본적인 권리이다.",
            "after": "예를 들어 자유롭게 의견을 말할 권리가 이에 해당한다.",
            "reason": "설명구체성 보완",
        }

    monkeypatch.setattr("feak_tc.mvp.patch.request_json", fake_request_json)

    patched = apply_patch(
        "인권은 기본적인 권리이다.",
        candidate,
        cfg={"patcher": {"mode": "llm"}},
    )

    assert "자유롭게 의견을 말할 권리" in patched.new_text
    assert patched.patch.operation == "insert_after"


def test_targeter_prefers_rubric_relevant_sentence_over_shortest():
    text = (
        "좋다. "
        "인권은 인간이 태어날 때부터 가지는 기본적인 권리이다. "
        "예를 들어 헌법은 자유를 보호한다."
    )

    target = select_target_span(text, "content_2", action_type="ADD_DETAIL")

    assert target == "인권은 인간이 태어날 때부터 가지는 기본적인 권리이다."


def test_targeter_uses_question_overlap_without_topic_markers():
    text = (
        "짧다. "
        "도시의 쓰레기 문제는 주민의 생활 환경을 악화시킨다. "
        "예를 들어 분리배출 교육을 확대할 수 있다."
    )
    question = "도시 쓰레기 문제의 원인과 해결 방안을 서술하세요."

    target = select_target_span(text, "content_2", action_type="ADD_DETAIL", question=question)

    assert target == "도시의 쓰레기 문제는 주민의 생활 환경을 악화시킨다."


def test_deterministic_add_detail_uses_small_topic_specific_insert():
    text = "인권은 인간이 태어날 때부터 가지는 기본적인 권리이다. 우리는 서로의 권리를 존중해야 한다."
    candidate = Candidate(
        action_type="ADD_DETAIL",
        target_rubric="content_2",
        target_span="인권은 인간이 태어날 때부터 가지는 기본적인 권리이다.",
        instruction="구체 사례를 덧붙인다.",
    )

    patched = apply_patch(text, candidate, cfg={"patcher": {"mode": "deterministic"}})

    assert patched.patch.after == "예를 들어, 표현의 자유와 안전하게 살 권리가 이에 해당한다."
    assert edit_ratio(text, patched.new_text) < 0.5
    assert source_token_retention(text, patched.new_text) == 1.0


def test_batch_runner_writes_jsonl_logs(tmp_path):
    source = tmp_path / "essays.jsonl"
    output = tmp_path / "mvp.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "essay_id": "essay-a",
                        "text": "인권은 인간이 가지는 기본적인 권리이다. 우리는 서로의 권리를 존중해야 한다.",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "essay_id": "essay-b",
                        "text": "권리는 중요하다. 예를 들어 표현의 자유가 있다.",
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    summary = run_batch(
        input_path=source,
        output_path=output,
        diagnoser=StubDiagnoser(),
        cfg={"proposer": {"mode": "deterministic"}, "patcher": {"mode": "deterministic"}},
        limit=1,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary["total"] == 1
    assert summary["ok"] == 1
    assert summary["error"] == 0
    assert rows[0]["record_id"] == "essay-a"
    assert rows[0]["status"] == "ok"
    assert len(rows[0]["output"]["results"]) == 5
    assert rows[0]["output"]["decision"]["decision"] in {"accept", "reject_all", "stop"}


def test_batch_runner_reads_json_files_from_directory(tmp_path):
    source_dir = tmp_path / "raw"
    output = tmp_path / "mvp.jsonl"
    source_dir.mkdir()
    (source_dir / "essay.json").write_text(
        json.dumps(
            {
                "essay_id": "essay-dir",
                "essay": {"text": "인권은 기본적인 권리이다. 서로 존중해야 한다."},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = run_batch(
        input_path=source_dir,
        output_path=output,
        diagnoser=StubDiagnoser(),
        cfg={"proposer": {"mode": "deterministic"}, "patcher": {"mode": "deterministic"}},
    )

    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert summary["total"] == 1
    assert row["record_id"] == "essay-dir"
    assert row["status"] == "ok"


def test_batch_reader_extracts_essay_from_scoring_jsonl_without_keywords(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(
            {
                "user": (
                    "질문: 인권의 뜻과 특징에 대해 서술하세요\n"
                    "에세이: 인권은 사람이 태어나면서 가지는 기본적인 권리이다. "
                    "서로의 권리를 존중해야 한다.\n"
                    "핵심 키워드: 인간(사람), 당연, 권리, 존중(침해)"
                ),
                "assistant": "1 2 3 4 5 6 7 8",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records = list(iter_text_records(source))

    assert len(records) == 1
    assert records[0].text == "인권은 사람이 태어나면서 가지는 기본적인 권리이다. 서로의 권리를 존중해야 한다."
    assert "핵심 키워드" not in records[0].text
    assert records[0].metadata["question"] == "인권의 뜻과 특징에 대해 서술하세요"
    assert records[0].metadata["keywords"] == "인간(사람), 당연, 권리, 존중(침해)"


def test_batch_runner_applies_record_question_without_keywords_to_diagnoser(tmp_path):
    source = tmp_path / "train.jsonl"
    output = tmp_path / "mvp.jsonl"
    source.write_text(
        json.dumps(
            {
                "user": (
                    "질문: 인권의 뜻과 특징에 대해 서술하세요\n"
                    "에세이: 인권은 사람이 태어나면서 가지는 기본적인 권리이다. "
                    "서로의 권리를 존중해야 한다.\n"
                    "핵심 키워드: 인간(사람), 당연, 권리, 존중(침해)"
                ),
                "assistant": "1 2 3 4 5 6 7 8",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class ContextDiagnoser:
        def __init__(self):
            self.question = "default question"
            self.keywords = None

        def diagnose(self, text):
            return Diagnosis(
                text=text,
                rubrics={key: 5.0 for key in RUBRIC_KEYS},
                features={},
                weak_rubrics=["task_1", "content_1", "content_2"],
                metadata={"question": self.question, "keywords": self.keywords},
            )

    diagnoser = ContextDiagnoser()
    run_batch(
        input_path=source,
        output_path=output,
        diagnoser=diagnoser,
        cfg={"proposer": {"mode": "deterministic"}, "patcher": {"mode": "deterministic"}},
    )

    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row["output"]["before"]["metadata"]["question"] == "인권의 뜻과 특징에 대해 서술하세요"
    assert row["input"]["metadata"]["keywords"] == "인간(사람), 당연, 권리, 존중(침해)"
    assert row["output"]["before"]["metadata"]["keywords"] is None
    assert diagnoser.question == "default question"
    assert diagnoser.keywords is None


def test_batch_reader_skips_short_records_with_min_chars(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"essay_id": "short", "text": "짧은 글"}, ensure_ascii=False),
                json.dumps({"essay_id": "long", "text": "인권은 기본적인 권리이다. 서로의 권리를 존중해야 한다."}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )

    records = list(iter_text_records(source, min_chars=20))

    assert [record.record_id for record in records] == ["long"]


def test_non_stop_no_effect_candidate_is_rejected():
    candidate = Candidate(
        action_type="COMPRESS",
        target_rubric="content_2",
        target_span="인권은 기본적인 권리이다.",
        instruction="반복 표현을 압축한다.",
    )
    result = build_result(candidate, _transition(action_type="COMPRESS", target_rubric="content_2"))

    assert result.rejected
    assert "no_effect" in result.reject_reasons


def test_selects_stop_when_only_stop_remains_viable():
    no_effect = Candidate(
        action_type="DELETE_OR_FOCUS",
        target_rubric="content_3",
        target_span="인권은 기본적인 권리이다.",
        instruction="초점을 맞춘다.",
    )
    stop = Candidate(
        action_type="STOP",
        target_rubric="content_3",
        target_span="",
        instruction="수정하지 않는다.",
    )
    results = [
        build_result(no_effect, _transition(action_type="DELETE_OR_FOCUS", target_rubric="content_3")),
        build_result(stop, _transition(action_type="STOP", target_rubric="content_3")),
    ]

    decision = select(results)

    assert decision.decision == "stop"
    assert decision.chosen_index == 1


def _transition(action_type: str, target_rubric: str) -> Transition:
    return Transition(
        action_type=action_type,
        target_rubric=target_rubric,
        target_gain=0.0,
        non_target_drop=0.0,
        target_gap_reduction=0.0,
        evidence_match=0.9,
        edit_ratio=0.0,
        goal_preservation=1.0,
        emb_sim=1.0,
    )


def test_validity_rejects_sentence_fragment_replace():
    from feak_tc.mvp.schemas import Patch
    from feak_tc.mvp.validity import patch_validity_violations

    text = "미국은 이민자의 나라이다. 많은 이민자들을 받아들여 성장한 나라이기 때문이다. 따라서 다양성이 중요하다."
    candidate = Candidate(
        action_type="COMPRESS",
        target_rubric="expression_1",
        target_span="많은 이민자들을 받아들여 성장한 나라이기 때문이다.",
        instruction="압축한다.",
    )
    candidate.patch = Patch(
        operation="replace",
        target_span=candidate.target_span,
        before="많은 이민자들을 받아들여 성장한 나라이기 때문이다.",
        after="많은 이민자들을 받아",
        reason="압축",
    )
    candidate.new_text = text.replace(candidate.patch.before, candidate.patch.after, 1)

    violations = patch_validity_violations(text, candidate)

    assert "validity:sentence_fragment" in violations


def test_validity_accepts_complete_sentence_insert():
    from feak_tc.mvp.schemas import Patch
    from feak_tc.mvp.validity import patch_validity_violations

    text = "모든 사람들은 태어날 때부터 인간답게 살 권리를 가지고 있다. 우리는 이를 존중해야 한다."
    candidate = Candidate(
        action_type="ADD_DETAIL",
        target_rubric="content_2",
        target_span="모든 사람들은 태어날 때부터 인간답게 살 권리를 가지고 있다.",
        instruction="예시를 추가한다.",
    )
    candidate.patch = Patch(
        operation="insert_after",
        target_span=candidate.target_span,
        before=candidate.target_span,
        after="예를 들어, 교육을 받을 권리와 의료 서비스에 접근할 권리가 포함된다.",
        reason="예시 추가",
    )
    candidate.new_text = text.replace(
        candidate.patch.before, f"{candidate.patch.before} {candidate.patch.after}", 1
    )

    assert patch_validity_violations(text, candidate) == []


def test_validity_rejects_essay_collapse():
    from feak_tc.mvp.schemas import Patch
    from feak_tc.mvp.validity import patch_validity_violations

    text = "문화를 바라보는 시각에는 세 가지가 있다. 첫째는 문화절대주의이다. 둘째는 문화상대주의이다. 셋째는 문화보편주의이다."
    deleted = "첫째는 문화절대주의이다. 둘째는 문화상대주의이다. 셋째는 문화보편주의이다."
    candidate = Candidate(
        action_type="DELETE_OR_FOCUS",
        target_rubric="content_3",
        target_span=deleted,
        instruction="삭제한다.",
    )
    candidate.patch = Patch(
        operation="delete",
        target_span=deleted,
        before=deleted,
        after="",
        reason="삭제",
    )
    candidate.new_text = text.replace(deleted, "", 1).strip()

    violations = patch_validity_violations(text, candidate)

    assert "validity:essay_collapse" in violations


def test_target_gain_min_hard_constraint():
    candidate = Candidate(
        action_type="COMPRESS",
        target_rubric="content_2",
        target_span="인권은 기본적인 권리이다.",
        instruction="압축한다.",
    )
    transition = Transition(
        action_type="COMPRESS",
        target_rubric="content_2",
        target_gain=0.0,
        non_target_drop=1.0,
        target_gap_reduction=0.0,
        evidence_match=0.9,
        edit_ratio=0.1,
        goal_preservation=0.95,
        emb_sim=0.95,
    )
    cfg = {"hard_constraints": {"target_gain_min": 1.0}}

    result = build_result(candidate, transition, cfg)

    assert result.rejected
    assert "target_gain" in result.reject_reasons


def test_heuristic_score_normalizes_rubric_scale():
    from feak_tc.mvp.heuristic import heuristic_score

    transition = Transition(
        action_type="ADD_DETAIL",
        target_rubric="content_2",
        target_gain=4.0,
        non_target_drop=0.0,
        target_gap_reduction=0.0,
        evidence_match=0.0,
        edit_ratio=0.0,
        goal_preservation=0.0,
        emb_sim=1.0,
    )
    cfg = {
        "rubric_score_range": 8.0,
        "weights": {
            "target_gain": 2.0,
            "target_gap_reduction": 1.0,
            "evidence_match": 0.5,
            "goal_preservation": 0.5,
            "non_target_drop": -2.0,
            "edit_ratio": -0.5,
        },
    }

    # gain 4 on a 1-9 scale -> 0.5 normalized, weighted 2.0 -> 1.0
    assert abs(heuristic_score(transition, cfg) - 1.0) < 1e-9


def test_gap_reduction_uses_elite_band():
    features_before = {name: 0.0 for name in FEAK_FEATURE_NAMES}
    features_after = dict(features_before)
    features_before.update({"word_Cnt": 100.0, "char_Cnt": 400.0, "2-gram_NDW": 50.0})
    features_after.update({"word_Cnt": 150.0, "char_Cnt": 600.0, "2-gram_NDW": 80.0})
    rubrics = {key: 3.0 for key in RUBRIC_KEYS}
    before = Diagnosis(text="원문이다.", rubrics=dict(rubrics), features=features_before, weak_rubrics=["content_2"])
    after = Diagnosis(text="수정본이다.", rubrics=dict(rubrics), features=features_after, weak_rubrics=["content_2"])
    candidate = Candidate(
        action_type="ADD_DETAIL",
        target_rubric="content_2",
        target_span="원문이다.",
        instruction="구체화한다.",
    )
    elite_stats = {
        "word_Cnt": {"low": 150.0, "high": 250.0},
        "char_Cnt": {"low": 600.0, "high": 1000.0},
        "2-gram_NDW": {"low": 80.0, "high": 140.0},
    }

    transition = compute_transition(before, after, candidate, elite_stats=elite_stats)

    # Every target feature moved from below the elite band to its lower edge.
    assert transition.target_gap_reduction > 0.4

    overshoot = dict(features_before)
    overshoot.update({"word_Cnt": 400.0, "char_Cnt": 1600.0, "2-gram_NDW": 230.0})
    after_over = Diagnosis(text="과수정본이다.", rubrics=dict(rubrics), features=overshoot, weak_rubrics=["content_2"])

    over_transition = compute_transition(before, after_over, candidate, elite_stats=elite_stats)

    # Overshooting past the band must score worse than landing inside it.
    assert over_transition.target_gap_reduction < transition.target_gap_reduction


def test_transition_uses_rf_corrected_scores_when_both_present(monkeypatch):
    from feak_tc.mvp import transition as transition_module

    monkeypatch.setattr(
        transition_module,
        "_embedding_model",
        lambda: (None, {"error": "sentence-transformers unavailable"}),
    )
    integer_rubrics = {key: 4.0 for key in RUBRIC_KEYS}
    before_rf = dict(integer_rubrics)
    after_rf = dict(integer_rubrics)
    before_rf["content_2"] = 4.20
    after_rf["content_2"] = 4.65
    before_rf["task_1"] = 4.80
    after_rf["task_1"] = 4.55
    before = Diagnosis(
        text="도시는 쓰레기 문제를 해결해야 한다.",
        rubrics=dict(integer_rubrics),
        features={},
        weak_rubrics=["content_2"],
        metadata={"rf_corrected_score": [before_rf[key] for key in RUBRIC_KEYS]},
    )
    after = Diagnosis(
        text="도시는 쓰레기 문제를 해결해야 한다. 예시를 덧붙였다.",
        rubrics=dict(integer_rubrics),
        features={},
        weak_rubrics=["content_2"],
        metadata={"rf_corrected_score": [after_rf[key] for key in RUBRIC_KEYS]},
    )
    candidate = Candidate(
        action_type="ADD_DETAIL",
        target_rubric="content_2",
        target_span="도시는 쓰레기 문제를 해결해야 한다.",
        instruction="예시를 추가한다.",
    )

    result = compute_transition(before, after, candidate, elite_stats={})
    same_text_result = compute_transition(before, before, candidate, elite_stats={})

    assert result.target_gain == pytest.approx(0.45)
    assert result.non_target_drop == pytest.approx(0.25)
    assert candidate.metadata["score_basis"] == "rf_corrected"
    assert same_text_result.target_gain == 0.0
    assert same_text_result.non_target_drop == 0.0


def test_transition_falls_back_to_integer_scores_when_continuous_missing(monkeypatch):
    from feak_tc.mvp import transition as transition_module

    monkeypatch.setattr(
        transition_module,
        "_embedding_model",
        lambda: (None, {"error": "sentence-transformers unavailable"}),
    )
    before_rubrics = {key: 4.0 for key in RUBRIC_KEYS}
    after_rubrics = dict(before_rubrics)
    after_rubrics["content_2"] = 5.0
    before_rubrics["task_1"] = 5.0
    after_rubrics["task_1"] = 3.0
    misleading_rf = {key: 7.5 for key in RUBRIC_KEYS}
    before = Diagnosis(
        text="원문이다.",
        rubrics=before_rubrics,
        features={},
        weak_rubrics=["content_2"],
        metadata={"rf_corrected_score": [misleading_rf[key] for key in RUBRIC_KEYS]},
    )
    after = Diagnosis(
        text="수정본이다.",
        rubrics=after_rubrics,
        features={},
        weak_rubrics=["content_2"],
    )
    candidate = Candidate(
        action_type="ADD_DETAIL",
        target_rubric="content_2",
        target_span="원문이다.",
        instruction="예시를 추가한다.",
    )

    result = compute_transition(before, after, candidate, elite_stats={})

    assert result.target_gain == 1.0
    assert result.non_target_drop == 2.0
    assert candidate.metadata["score_basis"] == "integer"


def test_transition_similarity_falls_back_and_logs_method(monkeypatch):
    from feak_tc.mvp import transition as transition_module

    monkeypatch.setattr(
        transition_module,
        "_embedding_model",
        lambda: (None, {"error": "sentence-transformers unavailable"}),
    )
    before = Diagnosis(
        text="도시는 쓰레기 문제를 해결해야 한다.",
        rubrics={key: 5.0 for key in RUBRIC_KEYS},
        features={},
        weak_rubrics=["content_2"],
    )
    after = Diagnosis(
        text="도시는 쓰레기 문제를 해결해야 한다. 예를 들어 분리배출 교육을 확대할 수 있다.",
        rubrics={key: 6.0 for key in RUBRIC_KEYS},
        features={},
        weak_rubrics=["content_2"],
    )
    candidate = Candidate(
        action_type="ADD_DETAIL",
        target_rubric="content_2",
        target_span="도시는 쓰레기 문제를 해결해야 한다.",
        instruction="예시를 추가한다.",
    )

    result = transition_module.compute_transition(before, after, candidate, elite_stats={})

    assert result.goal_preservation == result.emb_sim
    assert result.goal_preservation == source_token_retention(before.text, after.text)
    assert candidate.metadata["similarity"]["method"] == "token_fallback"
