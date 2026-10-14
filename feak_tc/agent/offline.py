"""Deterministic wiring fixture, explicitly not a writing-quality evaluator."""

from feak_tc.diagnose.constants import RUBRIC_KEYS

from .scorer import ScoreState
from .editing import apply_patch, text_units
from .schemas import (Criterion, EditScope, PlanResponse, ReplacementPatch,
                      Revision, RevisionPlan, RevisionVerdict, RubricCheck)


class OfflineScorer:
    def score(self, text):
        # Deliberately flat: acceptance must not depend on a score increase.
        return ScoreState({key: 5.0 for key in RUBRIC_KEYS}, {"source": "offline_stub", "rescaled": False})


class OfflineRoles:
    def __init__(self):
        self.planned = False

    def plan(self, prompt, text, scores):
        checks = [RubricCheck(rubric=key, finding="no_actionable_issue", reason="모델 없는 연결 검사")
                  for key in RUBRIC_KEYS]
        if self.planned:
            return PlanResponse(outcome="no_actionable_issue", priority_checks=checks,
                                plan=None, reason="모델 없는 연결 검사를 마침")
        self.planned = True
        checks[-1].finding = "actionable"
        return PlanResponse(outcome="edit", priority_checks=checks, plan=RevisionPlan(
            target_rubric="expression_2", target_span=text,
            problem="모델 없는 연결 검사", goal="검사용 표시를 한 번 추가",
            must_preserve=["기존 글의 모든 내용"], evidence_spans=[text],
            reader_impact="모의 진단: 독해 품질 평가 아님", context_check="모의 문맥 확인",
            operation="insert", edit_scope=EditScope(start_unit="U0001", end_unit=text_units(text)[-1].unit_id)),
            reason="모의 계획")

    def revise(self, prompt, text, plan, feedback=None):
        units = text_units(text)
        original = text[units[0].start:units[-1].end]
        patch = ReplacementPatch(**plan.edit_scope.model_dump(), expected_text=original,
                                 replacement=original + "\n[모델 없는 연결 검사]")
        return Revision(revised_text=apply_patch(text, plan, patch), summary_of_change="모의 변경", patch=patch)

    def verify(self, prompt, before, after, plan):
        criterion = Criterion(label="PASS", reason="모의 판정: 품질 평가 아님")
        return RevisionVerdict(goal_achievement=criterion, necessity=criterion,
                               preservation=criterion, global_benefit=criterion, decision="ACCEPT")
