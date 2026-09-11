# FEAK-TC 작업 지침

## 먼저 읽을 문서

사용자가 확정한 최종 문서는 `paper_docs/`에 있다.

1. [최종 방법론](paper_docs/FEAK_TC_METHOD_FINAL.md)
2. [로컬 전체 루프 실행 지침](../docs/TRAINING_FREE_AGENT.md) — 현재 구현·실행
3. [RV 데이터 생성 및 학습 지침](paper_docs/FEAK_TC_RV_DATA_GENERATION.md) — 향후 학습 재개 시
4. [서론·관련연구](paper_docs/FEAK_TC_INTRO_RELATED_2026-09-09.md)
5. [기존 데이터 진행 현황](중간정리/RV_DATA_STATUS_2026-09-08.md) — 실제 보유 파일과 과거 완료 여부

구현·학습 판단에서는 최종 방법론과 데이터 지침이 과거 초안·실험 메모보다 우선한다.
서론의 개념 설명과 세부 모델 출력이 다르게 읽히면 최종 방법론의 명시적 정의를 따른다.
논문 원고는 로컬 보관이므로 누락된 환경에서는 임의로 복원하지 말고 해당 사실을 알린다.

## 현재 확정된 방향

- FEAK-TC는 한국어 글쓰기의 반복 수정과 경로 제어를 연구한다.
- 2026-09-11 사용자 지시: 신규 학습을 보류하고 전체 루프 실행을 먼저 완성한다.
- 진단기만 기존 학습된 Kanana를 사용한다. Planner·Generator·RV·경로 가드는 일반 로컬 LLM을
  사용한다. 기본은 캐시된 Qwen2.5-7B-Instruct 4비트 추론이며 외부 LLM API를 호출하지 않는다.
- 개별 모듈의 학습·교체는 루프 실행 및 평가 이후의 선택 사항이다. 기존 TVM의 추가 개발은 종료했다.
- RV 입력은 과제, 수정 전 글, 수정 요구·보존 조건, 수정 후 글이다.
- RV 출력은 `target_fulfillment`와 `preservation`의 두 축이며 각 축은 pass/partial/fail이다.
- `action_consistency`와 `edit_appropriateness`를 별도 학습 출력으로 추가하지 않는다.
- FEAK는 기존 채점기와 독립 자질 계산기를 사용한다.
- RAG는 train에서 선별한 rubric별 우수 사례를 계획에 제공한다.
- Planner·Generator·경로 가드는 기존 LLM을 사용한다. 별도 policy·Planner 학습은 하지 않는다.
- Controller는 채택, 거절·재계획, checkpoint 복구, 종료를 구분한다.

## 설계와 구현 상태

현재 실행 진입점은 `scripts/run_agent.py`, 반복 controller는 `feak_tc/agent/`다.
기존 `feak_tc/mvp/`의 patch·validity·transition·heuristic을 재사용한다.
국소 RV의 두 축, 재계획, checkpoint 복구, 경로 가드, 종료 예산을 연결한다.
`feak_tc/rv/`와 `configs/rv_*.yaml`은 기존 4축 파일럿을 재사용하기 위해 보존한 코드·설정이다.
이 파일들의 존재를 최종 2축 RV 구현 완료로 해석하지 않는다.

기존 corruption 1,000 transition과 RV v2의 227개 선택 후보는 보존한다.
227개는 사람 검수 완료 데이터가 아니다. 최종 지침의 첨부 36편 샘플과도 별도 집합이다.
최종 2축 데이터와 RV 학습은 이번 범위 밖이다. 선택적인 train-only 사례 검색은 제공하지만
실제 corpus가 없으면 사례 없이 계획하며, 예제 실행 성공을 품질 개선이나 최초성 입증으로 해석하지 않는다.

## 데이터 및 구현 원칙

- 원문, 생성 후보, 판정 출처와 실패 이력을 보존한다.
- 원본·유사·파생 글의 split을 먼저 고정하고, 평가 글을 RAG 저장소에 넣지 않는다.
- 학습 데이터 구성에 쓴 전문가 피드백을 최종 에이전트 평가의 운영 입력으로 제공하지 않는다.
- 생성 유형, 참고 복원문, FEAK gain과 기존 judge 라벨을 RV 입력에 넣지 않는다.
- 생성 의도와 FEAK 점수 상승을 성공 라벨로 자동 확정하지 않는다.
- 판정 불가는 partial로 바꾸지 않고 해당 축의 label mask로 처리한다.
- 같은 실패 유형을 얻으려는 반복 생성·전수 재판정은 원인과 종료 기준 없이 재개하지 않는다.
- 기존 채점기·자질 계산 코드는 수정하지 않고 호출한다.
- LLM의 구조화 출력은 schema로 검사하고, 설정은 `configs/`에서 관리한다.
- 먼저 stub과 작은 검수 세트로 검증한다. 기존 데이터·모델·환경을 이유 없이 다시 만들지 않는다.

## 코드·이력 찾기

- [저장소 README](../README.md): 실행·폴더·보존한 진입점
- [기존 MVP 스펙](docs/IMPLEMENTATION_MVP.md): 현재 한 단계 코드의 역사적 설명
- [과거 실험 안내](중간정리/README.md): 결과 근거. 과거 ‘다음 작업’은 현재 지시가 아님
- [정리·복구 기록](../docs/CLEANUP_2026-09-08.md): 제거한 경로와 백업

## Git/GitHub 흐름

- 사용자가 명시적으로 요청하지 않는 한 main에 직접 push하지 않는다.
- topic branch에서 변경하고 검증한 뒤 해당 브랜치만 원격에 push한다.
- main 반영은 GitHub PR로 진행한다. 필요한 PR 생성과 병합은 검증 후 gh CLI로 수행한다.
- 기존 작업 브랜치에 이어지는 정리는 그 브랜치를 PR base로 사용하여,
  관련 없는 미병합 작업을 함께 main에 반영하지 않는다.
- 논문 원고와 로컬 데이터·모델·백업은 명시적 업로드 요청 없이는 커밋하지 않는다.
