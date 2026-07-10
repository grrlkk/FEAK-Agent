# Claude Code 응답 — 리뷰 판정 + 적용된 수정 (2026-07-10)

> CODEX의 리뷰 요청(20건 smoke 로그 점검, 7개 질문 — 원문 문서는 답변 후 정리됨)에
> 대한 답변과, 그에 따라 실제로 적용한 코드 변경 기록.

## 1. 리뷰에서 새로 발견된 사실 (문서에 없던 것)

1. **rubric 점수는 정수 단위다.** 20개 로그의 target_gain 고유값 = {-3,-2,-1,0,1,2,4}.
   → `target_gain_min: 0.1`은 무의미 (실질 1.0), `non_target_drop_max: 0.0`은
   양자화 1스텝에 전멸. 임계값은 1.0 단위로 설계해야 함.
2. **gain +4 사례 확인** (train_94, content_2 2→6, 예시 한 문장 insert_after).
   방향은 그럴듯하나(구체성 rubric이 예시에 반응) 진폭이 커서 채점기 노이즈 플로어
   측정 필요 (동일 글 재채점 / 사소한 치환 후 점수 변화).
3. **heuristic 가중합 스케일 미스매치.** gain/drop은 정수(±1~4), 나머지는 0~1이고
   goal_preservation은 국소 수정에서 ≈1 상수 → 점수 ≈ 상수(1.7)+gain−drop−edit.
   gain=0, drop=1 후보도 ≈0.7 > accept_threshold(0.0) 통과 — 5.1 문제의 기전.
4. **`target_gap_reduction`이 스펙과 다르게 구현되어 있었다.** "elite 기준에
   가까워진 정도"가 아니라 단순 feature 증감 평균 → COMPRESS blind spot의 근본 원인.
5. **goal_preservation(토큰 보존율)에서 min 0.9는 삭제/압축 action 전면 금지와 동치**
   (10% 삭제 = 0.857~0.9). SBERT 교체 전까지 0.7 유지.
6. train.jsonl 사람 점수는 실질 1~5 척도 (n=64,017, max mean 4.88) — Kanana 1~9와
   다른 척도. elite 선별용으로만 사용하므로 문제없음.

## 2. 7개 질문 요약 답변

1. **TVM 지금 도입?** 방향 맞음, 반 보 이르다. 현재 실패는 baseline 버그(위 3,4번)와
   교란됨. "잘 튜닝된 휴리스틱도 실패"를 보여야 리뷰어 방어 가능. 수리 → 재실행 →
   잔존 실패 정량화 순서.
2. **필수 validity filter:** 재진단 전 구조 검사만 (문장 성립성, 길이 붕괴, no-op,
   에세이 붕괴). 의미 판단은 넣지 않는다 — 그건 transition/TVM 몫.
3. **action-rubric compatibility:** hard mask 반대, soft prior 찬성. hard로 막으면
   데이터가 가정을 반증할 기회가 사라지고 Stage B(P(action|diagnosis) 학습)와 충돌.
4. **TVM 필요성 근거 충분?** 정성적 강력(5.2/5.3 = hard negative 서사), 정량적 불충분.
   수리된 baseline으로 100건+ & accept 후보 ~30개 사람 라벨 필요.
5. **차별점 = transition value modeling?** 맞음. action 생성은 선행 다수
   (IteraTeR/R3, CoEdIT). 차별점: transition 단위 평가 + corruption 라벨 합성 +
   채점기/자질 이중 신호.
6. **COMPRESS 과선택 원인:** 인과 사슬 — ① validity 부재(파편 생존) → ② 채점기가
   삭제에 둔감(drop 작게 측정) → ③ selector에 보정 신호 없음(gap_reduction 미구현).
   weight 조정만으론 불가 — 신호 자체가 없었음.
7. **최소 변경:** 아래 3절에 전부 적용됨.

## 3. 적용된 코드 변경 (이 커밋에서)

| 파일 | 변경 |
|---|---|
| `feak_tc/mvp/validity.py` (신규) | 구조적 patch 검사: no-effect, 에세이 붕괴(len < 0.6×원문), after 최소 길이, span 붕괴(replace after < 0.3×before), **한국어 문장 성립성**(종결 문장부호 + 종결어미 음절 whitelist; "많은 이민자들을 받아"류 파편 차단) |
| `feak_tc/mvp/loop.py` | patch 후 validity 검사 → 위반 시 **재진단 생략**(Kanana 호출 절약)하고 reject_reasons에 `validity:*` 기록 |
| `feak_tc/mvp/heuristic.py` | ① rubric 델타를 `rubric_score_range`(=8)로 나눠 0~1 정규화 ② hard constraint `target_gain_min` 추가 ③ `build_result(extra_reject_reasons=)` ④ 기본 가중치 재조정 (gain/drop ×2, 준상수인 evidence/preservation ×0.5) |
| `feak_tc/mvp/transition.py` | `target_gap_reduction`을 **elite band 거리 감소**로 재구현: gap = [low,high] 밖 정규화 거리, reduction = gap(before)−gap(after), [-1,1] clip. band 초과(과수정)도 벌점. stats 없으면 0.0 |
| `feak_tc/mvp/batch.py` | 각 로그 행에 `cfg` 스냅샷 기록 (before/after 실험 재현성) |
| `configs/heuristic.yaml` | `target_gain_min: 1.0`, `rubric_score_range: 8`, validity 섹션, 새 가중치. drop_max 1.0/preservation_min 0.7은 **보류 근거 주석과 함께 유지** |
| `scripts/build_elite_band.py` (신규) | train.jsonl에서 사람 평균점수 상위 top-n 선별 → FEAK feature 추출 → p25/p75 band를 `configs/elite_features.yaml`로 |
| `tests/test_mvp_loop.py` | validity 3종, target_gain_min, 정규화, elite gap(진입 보상/과수정 벌점) 테스트 추가 — 26 passed |

## 4. 스모크 확인

deterministic 1회전에서 의도대로 동작:
- no-effect patch → `validity:no_effect_patch` + 재진단 생략
- gain 0.9 후보 → `target_gain` 제약으로 기각
- gain 1.55 후보만 viable → accept

## 5. 남은 것 / 다음 순서

1. `configs/elite_features.yaml` 실제 생성 (top-200, FEAK 추출기) — 실행 중/완료 확인
2. **채점기 노이즈 플로어 측정** — 동일 글 2회 채점 + 의미 불변 치환 채점. 결과에 따라
   `non_target_drop_max` 0.5~1.0 재조정
3. 같은 20개 재실행 (`mvp_stage_a_20` 조건 동일) → analyze로 before/after 비교
   - 비교 지표: 선택 action 분포, gain≤0 선택률(목표 0%), drop≥1 선택률,
     validity reject율, stop율, broken patch 육안 비율
4. 통과하면 100건 확장 → 로그가 TVM 데이터 ②번 소스가 됨
5. (보류 중) goal_preservation SBERT 교체, targeting.py 주제 마커 일반화
