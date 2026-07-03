from feak_tc.diagnose import RUBRIC_KEYS, StubDiagnoser
from feak_tc.diagnose.constants import FEAK_FEATURE_NAMES
from feak_tc.diagnose.kanana import KananaDiagnoser
from feak_tc.mvp.patch import apply_patch
from feak_tc.mvp.propose import propose
from feak_tc.mvp.schemas import Candidate
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
