# Stage A BGE-M3 100건 재실행 결과 (3-run 비교 완성)

작성일: 2026-07-20
로그: `experiments/results/mvp_stage_a_100_bge_m10.jsonl`

## 실행 조건

spanfix(07-19)와 동일 + BGE-M3 유사도(`FEAK_EMBEDDING_DEVICE=cpu`) +
`goal_preservation_min 0.95`. 100/100 ok, 500/500 후보가 BAAI/bge-m3로
유사도 계산됨 (fallback 0건).

## 3-run 비교

| | baseline 07-16 | spanfix 07-19 | **bge 07-20** |
|---|---:|---:|---:|
| accept | 52 | 48 | **56** |
| accept 중 스캔 결함 | 21 (환각3·중복1 포함) | 0 | **0** |
| 채택 gp 최소 | 0.733 | 0.733 | **0.960** |
| similarity | token_fallback | token_fallback | **bge-m3** |

선택 action 분포 (bge run): ADD_DETAIL 26, COMPRESS 12, DOF 8,
STYLE_REFINE 6, RESTRUCTURE 4.

## 해석

1. **accept 48 → 56 (+8)**: token 잔존율이 정당한 바꿔쓰기를 벌하던
   것이 사라진 효과가 크다. 특히 COMPRESS 7→12로 증가 — 압축은 원문
   토큰을 많이 바꾸므로 기존 지표에서 gp가 부당하게 낮았다.
   (proposer/patcher temperature > 0이므로 run 간 후보 자체의 변동도
   일부 섞여 있음에 유의.)
2. **56건 자동 스캔 무결점**: 환각·중복·DOF span 오류 0건.
   guard는 이번에도 13건(숫자 6, 인용 1, 중복 6)을 걸렀다.
3. **DOF 채택 8건 중 신규 4건 수동 확인**: 전부 깨끗한 삭제.
   `goal_preservation < 0.9` 리스크 지표는 0/56 — BGE 스케일에서
   0.95 hard constraint가 의미 훼손 후보를 사전에 거른다.
4. 남은 품질 이슈는 여전히 proposer targeting (train_631이 또
   정의 문장을 삭제 target으로 선택 — 패치는 정확하나 내용상 손실
   가능).

## 결론

patcher span 고정 + validity guard + BGE-M3 유사도 조합으로,
같은 입력에서 **accept가 baseline보다 많으면서(52→56) 검출 가능한
결함은 21→0**이 됐다. Stage A 파이프라인 신뢰성 작업은 여기서 일단락.

## 다음 작업

1. proposer targeting 개선: 질문 핵심어를 정의하는 문장을 삭제
   target에서 제외
2. corruption 데이터 생성 (`docs/SPEC_CORRUPTION_TVM.md`) 진입 —
   이제 transition 로그가 TVM 학습 데이터로 쓸 만큼 깨끗함
3. (선택) gp 0.95 임계 재검증: bad 표본 12건이 작으므로 다음 질적
   검토 때 라벨 추가
