# Corruption 생성 — 운영 결정 기록 (2026-07-22)

`docs/SPEC_CORRUPTION_TVM.md`의 설계(action 역연산, 재측정 필수, GBM 먼저)는
그대로 따르고, 스펙이 비워둔 운영 파라미터를 사용자와 확정했다. 구현은 아직
시작 안 함 — 다음 작업자는 이 결정대로 진행하면 된다.

## 확정된 결정

1. **진행 순서**: 파일럿 50편 먼저 → 폐기율·GPU 시간·API 비용 실측 →
   파일럿 쌍으로 GBM 학습 확인 → 본 생성 스케일업.
2. **체인 설계**: 깊이 3 (x0→x1→x2→x3), 스텝당 corruption 연산자 1개,
   글당 체인 1개. 연산자는 5종에서 중복 없이 3개 무작위 샘플
   → 글당 선호쌍 C(4,2)=6개, action 라벨 분포 균등 유지.
3. **본 생성 규모**: 1,000편 (쌍 ~6,000, 재측정 ~4,000회). 파일럿 실측 후
   재조정 가능. TVM 본학습용 증량은 GBM 결과 보고 결정.
4. **소스 선별**: train.jsonl(6.4만)에서 grader 2인 평균 상위 ~25%,
   질문(164종) 층화, 길이 필터. held-out 평가용 글은 생성 전에 분리.
   Stage A 100편과 겹침 여부 기록.
5. **실행 방식**: LLM(gpt-4o-mini) 위주 + 규칙 보조 (스펙 리스크 §6).
   Stage A patcher처럼 span 고정으로 "어느 문장을 어떻게 뭉갰는지" 기록
   → 역방향 action의 target_span 라벨 확보.
6. **validity 정책**: 구조 가드(문장 파편, essay_min_ratio)는 적용,
   의미보존(gp)·정의 문장 가드는 의도적 훼손이므로 미적용.
7. **폐기 기준**: 의도한 축(아래 표)의 soft score가 실측에서 안 떨어지면
   그 스텝 폐기. 임계는 채점 노이즈(`SCORER_NOISE_M_SWEEP.md`, m=10) 대비
   유의미한 하락으로 — 구체 수치는 파일럿에서 확정.

## 연산자 ↔ 역방향 action ↔ 주 하락 예상 축

| corruption (순방향) | 역방향 action | 주 하락 예상 rubric |
|---|---|---|
| 구체 사례·근거 삭제 | ADD_DETAIL | content_2(설명구체성), content_1 |
| 주제 무관 문장 삽입 | DELETE_OR_FOCUS | content_3(설명적절성), organization_2(글통일성) |
| 반복·장황화 주입 | COMPRESS | 길이·반복 feature 계열 (rubric 축은 파일럿에서 확인) |
| 문장 순서 섞기·접속 제거 | RESTRUCTURE | organization_1(문장연결성) |
| 종결어미 단일화·어휘 반복 | STYLE_REFINE | expression_1·2(어휘·어법적절성) |

예상 축이 실측과 다르면 표를 갱신할 것 (스펙 §2 ③: "뭉갰으니 나빠졌겠지" 가정 금지).

## 다음 작업 (구현 순서)

1. 소스 선별 스크립트 (grader 평균, 층화, held-out 분리)
2. corruption 연산자 모듈 (`feak_tc/corruption/`, LLM 프롬프트 + span 기록)
3. 파일럿 50편 실행 → 폐기율/비용 리포트 → 이 문서에 결과 추가
