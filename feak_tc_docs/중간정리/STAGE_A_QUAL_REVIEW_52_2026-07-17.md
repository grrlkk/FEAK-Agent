# Stage A accept 52건 질적 검토

작성일: 2026-07-17
대상 로그: `experiments/results/mvp_stage_a_100_soft_m10.jsonl`

## 검토 범위

- DELETE_OR_FOCUS 12건 전수
- `goal_preservation < 0.9` 9건 전수 (8건은 DOF와 중복, 나머지는 train_141)
- 52건 전체 자동 스캔: 신규 숫자/통계 표현, 문장 중복(근사 포함), target_span 존재·수정 여부
- 표본 수동 검토: ADD_DETAIL 5, COMPRESS 3, RESTRUCTURE 2, STYLE_REFINE 1

## 판정 결과 요약

| action | accept | 결함 확인 | 비고 |
|---|---:|---:|---|
| DELETE_OR_FOCUS | 12 | 11~12 | 깨끗한 성공 0건 |
| ADD_DETAIL | 24 | 3 확정 | 통계 환각 3건, 나머지 삽입 위치는 정상 |
| COMPRESS | 5 | 1 | train_136 문장 중복 |
| RESTRUCTURE | 5 | 0 (경미) | 실제로는 STYLE 수준 미세 수정, label 불일치 |
| STYLE_REFINE | 6 | 0 | 표본 정상 |

## DELETE_OR_FOCUS 12건 전수 판정

실패 유형 (중복 해당 있음):

**A. 엉뚱한 문장 삭제 — target_span은 그대로 남고 다른 문장이 삭제됨 (7건)**
train_79, 85, 94, 123, 246, 1015, 1030.
특히 첫 문장(주제문)을 삭제하는 경향이 강하다. 85, 246, 1030은
주제문이 사라져 글이 "예를 들면...", "이는..."으로 시작한다.

**B. 문장 일부만 삭제해 비문 생성 (4건)**
train_79 (두 문장이 융합되어 비문), 126, 937, 1033
(문장 전반부만 지워 주어 없는 문장으로 글이 시작).

**C. target은 지웠으나 지시어/접속어가 붕괴 (2건)**
train_1017 ("그래서"가 가리킬 문장이 사라짐), 1021 (경미, "크게 ...로
나뉜다"의 주어 소실).

12건 중 borderline 수용 가능은 train_1021 하나뿐이다. 사실상 DOF의
실질 성공률은 0/12다. patcher LLM이 "삭제" 지시를 받으면 target_span이
아니라 첫 문장이나 문장 절반을 지우는 패턴이 지배적이다.

## ADD_DETAIL 통계 환각 3건 (자동 스캔 확정)

| record | 추가된 허위 내용 | gp |
|---|---|---:|
| train_70 | "최근 조사에 따르면 외국인 노동자의 60%가 임금 체불" | 1.000 |
| train_97 | "2022년 조사에 따르면 60%가 차별을 경험" | 1.000 |
| train_120 | "2021년 기준 약 3억 1천만 명이 영양실조" | 1.000 |

세 건 모두 proposer instruction이 "구체적인 사례나 **통계 자료**를
추가한다"였다. 즉 proposer가 환각을 유도하고 patcher가 수치를 지어낸다.
세 건 모두 `goal_preservation = 1.0` — 기존 지표는 환각을 전혀 보지
못한다. 나머지 ADD_DETAIL 21건은 신규 수치 없음(원문 유래 또는 일반
서술 추가)이며, 표본 검토(train_76, 1024)에서 삽입 위치도 정상이었다.

## COMPRESS 중복 1건

train_136: 압축된 문장을 삽입했지만 원래 문장을 지우지 않아 유사 문장이
연속으로 두 번 나온다. bigram overlap ≈ 0.7이라 0.8 임계 근사중복
검출을 통과했다 — repetition guard 임계는 0.6 수준이 필요하다.
train_141, 245는 깨끗한 성공.

## 가드가 필요한가에 대한 답

필요하다. 근거:

1. accept 52건 중 최소 15건(29%)이 결함 — DOF 11~12, 환각 3, 중복 1.
2. 이 결함들은 현재 지표의 사각지대다. 환각 3건은 gp=1.0, DOF 실패들은
   gp 0.78~0.95로 hard constraint(0.7)를 전부 통과했다. 점수 지표를
   조정해서 잡을 수 있는 문제가 아니다.
3. 반면 검출은 값싸다. 이번 자동 스캔(정규식 + span 포함 검사 +
   근사중복)만으로 환각 3/3, 중복 1/1을 오탐 없이 잡았고, DOF
   실패도 "after에 target_span이 그대로 존재"만으로 7/12가 걸린다.

단, DOF는 guard(사후 거부)보다 **생성 방식 수정**이 근본 해법이다.
삭제는 LLM에게 전문 재작성을 시키지 말고 target_span을 기계적으로
제거한 뒤 앞뒤 문장의 접속어만 LLM이 수선하게 하면 "엉뚱한 문장
삭제" 실패 유형 자체가 사라진다. guard는 그 위의 안전망으로 둔다.

## 다음 작업

1. Guard 3종 구현 (모두 LLM 불필요, 룰베이스):
   - patch/target consistency: 비삭제 action은 span 부근 수정 여부,
     DOF는 span 제거 여부 + 비대상 문장 보존 여부 검사
   - repetition: 문장 근사중복 (bigram overlap ≥ 0.6)
   - fabrication: before에 없는 숫자·연도·"조사/통계/연구에 따르면" 검출
     (ADD_DETAIL 한정 아님, 전 action 적용)
2. DOF patcher를 기계적 span 삭제 + 접속어 수선 방식으로 교체
3. 같은 100건 로그의 후보 500개에 guard 재적용 → pre/post 비교
4. proposer instruction에서 "통계 자료 추가" 류 지시 금지 (환각 유도 차단)
