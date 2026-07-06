import json

from feak_tc.diagnose import RUBRIC_KEYS, StubDiagnoser
from feak_tc.diagnose.constants import FEAK_FEATURE_NAMES
from feak_tc.diagnose.kanana import KananaDiagnoser
from feak_tc.mvp.batch import run_batch
from feak_tc.mvp.heuristic import build_result, select
from feak_tc.mvp.patch import apply_patch
from feak_tc.mvp.propose import propose
from feak_tc.mvp.schemas import Candidate, Transition
from feak_tc.mvp.targeting import select_target_span
from feak_tc.mvp.transition import edit_ratio, source_token_retention
from feak_tc.mvp import serializable_one_step


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
    assert len(output["results"]) == 6
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

    assert len(candidates) == 6
    assert candidates[0].metadata["source"] == "llm_proposer"
    assert {candidate.action_type for candidate in candidates} == {
        "ADD_DETAIL",
        "DELETE_OR_FOCUS",
        "COMPRESS",
        "RESTRUCTURE",
        "STYLE_REFINE",
        "STOP",
    }


def test_auto_proposer_falls_back_when_llm_unavailable(monkeypatch):
    diag = StubDiagnoser().diagnose("인권은 기본적인 권리이다. 서로 존중해야 한다.")

    def fake_request_json(**kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("feak_tc.mvp.propose.request_json", fake_request_json)

    candidates = propose(diag, cfg={"proposer": {"mode": "auto"}})

    assert len(candidates) == 6
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
    assert len(rows[0]["output"]["results"]) == 6
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
