# FEAK-Agent 작업 지침

## 현재 기준 — 2026-09-28

사용자가 지정한 `paper_docs/FEAK_RV_PILOT_IMPLEMENTATION.md`가 현재 방법론이다.
원본은 `/home/chanwoo/FEAK_RV_PILOT_IMPLEMENTATION.md`, 저장소 사본도 로컬 보관한다.
실행과 구현 계약은 `../docs/RV_PILOT.md`를 읽는다.
이 지시는 과거 `FEAK_TC_METHOD_FINAL.md`, `PILOT_BRIEF.md`, 2축 RV·경로 가드보다 우선한다.

- 새 진입점: `scripts/run_pilot.py`. 현재 알고리즘: `feak_tc/agent/`.
- 사람 블라인드 평가: `scripts/run_pilot_review.py`, `feak_tc/review/`, `../docs/HUMAN_REVIEW.md`.
  모델 호출 없이 저장 후보를 사용한다. 현재 네 기준을 유지하고 모델·점수·기존 판정·예상 정답은
  평가자 API에 전달하지 않는다. 참여 코드·사람 답변·원본 연결표는 실험 결과로 로컬 보관한다.
- Planner → Reviser → 4기준 RV → Controller. 한 번에 한 문제, 최대 3 iterations.
- 2026-09-29 사용자 요청: Kanana 점수에 따른 목표 선택과 원문 근거를 갖춘 수정 계획을 명시한다.
  Planner의 첫 호출은 점수 없이 원문을 진단한다. 코드는 actionable 중 Kanana 점수가 가장 낮은
  항목을 선택하고, 둘째 호출은 그 고정된 문제에 대한 계획을 만든다. 둘 다 Planner 내부 단계다.
  Reviser는 지정 범위의 패치만 생성하고 전체 수정본은 코드로 조립한다. 출력 계약은 RV_PILOT.md 참고.
  plan=null의 수정 불필요·정보 부족과 Reviser의 수정 불가를 구분해 기록한다.
  이 확장은 RV의 입력·프롬프트·채택 기준을 변경하지 않는다.
- RV: goal_achievement, necessity, preservation, global_benefit의 PASS/FAIL/UNCERTAIN.
- 모두 PASS만 ACCEPT, FAIL 우선 REJECT. FAIL 없이 UNCERTAIN이면 fresh context로 1회 재검증.
- 거절하면 같은 원문·plan에서 이유와 후보를 받아 최대 1회 재수정. 또 거절되면 STOP.
- 기존 essay_scoring_llm의 채점기·보정기·자질 계산은 그대로 호출한다. 학습하지 않는다.
- 사용자 확정: 현재 점수 체계 유지. 기본은 RF 보정 연속값이며 최종 1–9 등급과 soft mean/std도 기록.
  0–10 환산, clipping, 임의 반올림을 하지 않는다.
- 사용자 확정: GPT API를 쓰되 비싼 모델은 피한다. 기본 gpt-5-mini, reasoning low.
  더 비싼 모델로 자동 전환하지 않는다. 호출 예산·출력 길이·사용량을 기록한다.
- 점수와 자질은 RV에 주지 않는다. 점수는 상태·계획·비교용 로그에만 쓰고 채택에 관여하지 않는다.
- RV 입력: 문항, 전후 글, goal, must_preserve, diff만. 이전 RV 판단·근거·iteration·변경 요약은 금지.
- RV 학습, RAG, 다중 후보 탐색, memory, 경로 가드, rollback을 새 루프에 추가하지 않는다.

## 보관·검증

- 과거 루프: feak_tc/legacy/agent/. 공통 모델 실행: feak_tc/runtime/.
- scripts/run_agent.py와 기존 웹은 과거 루프 전용. 최신 방법론으로 소개하지 않는다.
- mvp/, rv/, corruption/과 데이터 도구는 과거 실험 재현용이다.
  rv/의 과거 4축 라벨도 새 기준과 다르므로 혼용하지 않는다.
- 정리 전 소스·로컬 문서 백업: .cleanup_archive/2026-09-28-rv-pilot/.
- 원문·후보·실패 기록·모델·데이터·결과를 보존한다. 삭제/이동 전 사용처·복구 방법을 확인한다.
- 후보별 JSONL을 즉시 flush하고 원시 API 출력도 기록한다.
- 테스트: python -m pytest -q. 실제 실행 전: python scripts/check_env.py.
- offline-smoke와 examples/pilot_samples.jsonl은 실행 확인용. 후자는 직접 작성한 합성 예제다.
- 실행 성공을 사람 평가, held-out 성능, 품질 향상, RV 정확도의 근거로 주장하지 않는다.

## Git/GitHub

- 명시적 요청 없이는 main에 직접 push하지 않는다.
- topic branch에서 변경·검증하고 그 브랜치만 push한다. 검증 후 gh CLI로 PR 생성·병합한다.
- 기존 미병합 작업에 이어지는 변경은 기존 작업 브랜치를 PR base로 사용한다.
- 논문 원고·데이터·모델·백업·실험 결과·키는 명시적 업로드 요청 없이는 커밋하지 않는다.
- push 전에 git status와 git diff --cached로 staged 내용을 확인한다.
