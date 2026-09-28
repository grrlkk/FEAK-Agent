"""Deterministic wiring fixture, explicitly not a writing-quality evaluator."""

from feak_tc.diagnose.constants import RUBRIC_KEYS

from .scorer import ScoreState
from .schemas import Criterion, PlanResponse, Revision, RevisionPlan, RevisionVerdict


class OfflineScorer:
    def score(self, text):
        # Deliberately flat: acceptance must not depend on a score increase.
        return ScoreState({key: 5.0 for key in RUBRIC_KEYS}, {"source": "offline_stub", "rescaled": False})


class OfflineRoles:
    def __init__(self):
        self.planned = False

    def plan(self, prompt, text, scores):
        if self.planned:
            return PlanResponse(plan=None, reason="모델 없는 연결 검사를 마침")
        self.planned = True
        return PlanResponse(plan=RevisionPlan(
            target_rubric="expression_2", target_span=text,
            problem="모델 없는 연결 검사", goal="검사용 표시를 한 번 추가",
            must_preserve=["기존 글의 모든 내용"]), reason="모의 계획")

    def revise(self, prompt, text, plan, feedback=None):
        return Revision(revised_text=text + "\n[모델 없는 연결 검사]", summary_of_change="모의 변경")

    def verify(self, prompt, before, after, plan):
        criterion = Criterion(label="PASS", reason="모의 판정: 품질 평가 아님")
        return RevisionVerdict(goal_achievement=criterion, necessity=criterion,
                               preservation=criterion, global_benefit=criterion, decision="ACCEPT")
