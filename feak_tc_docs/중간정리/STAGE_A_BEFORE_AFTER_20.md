# 중간정리 — validity/heuristic 수리 전후 20건 비교 (2026-07-10)

> `CLAUDE_REVIEW_RESPONSE.md`의 수정 사항을 적용한 뒤, 같은 20개 에세이를
> 동일 조건으로 재실행해 비교한 결과.
> 비교 도구: `scripts/compare_mvp_logs.py` (신규)

## 실행 조건 (양쪽 동일)

```text
input: data/data_jsonl/train.jsonl (limit 20, min_chars 250)
diagnoser: kanana (GPU 3, m=3, 4bit)   ← 원본 실행의 m은 로그 soft_std로 역추적 확인
proposer/patcher: llm (gpt-4o-mini), n_per_action: 1
BEFORE: experiments/results/mvp_stage_a_20.jsonl    (수리 전)
AFTER : experiments/results/mvp_stage_a_20_v2.jsonl (수리 후 + elite band)
```

AFTER에만 적용된 것: validity filter, target_gain_min 1.0, rubric 정규화 가중합,
elite-gap 기반 target_gap_reduction (`configs/elite_features.yaml`, 상위 200편 p25-p75),
로그 행별 cfg 스냅샷.

## 핵심 지표

| 지표 | BEFORE | AFTER | 판정 |
|---|---|---|---|
| decision | accept 19 / stop 1 | accept 13 / reject_all 7 | 선택이 보수화됨 |
| **gain≤0 선택률** | **8/19 (42%)** | **0/13 (0%)** | ✅ 목표 달성 |
| drop≥1 선택률 | 7/19 (37%) | 4/13 (31%) | ⚠ 소폭 개선 (drop_max 1.0 유지 중이라 예상대로) |
| preserve<0.9 선택률 | 5/19 (26%) | 1/13 (8%) | ✅ |
| COMPRESS 과선택 | 8회 | 2회 | ✅ blind spot 차단 |
| 선택 분포 | ADD 6 / DEL 3 / COMP 8 / STYLE 2 | ADD 10 / DEL 1 / COMP 2 | ADD 중심으로 재편 |

## reject 사유 변화

```text
                        BEFORE   AFTER
target_gain                0       76    ← 새 제약이 지배적 필터
non_target_drop           19       19
validity:sentence_fragment 0        1    ← "여자들의 인권을 침해하기 때문이다, 먼저."
validity:after_too_short   0        1
validity:no_effect_patch   0        1
goal_preservation          0        1
no_effect                  1        1
```

## elite-gap 신호 작동 확인

- `target_gap_reduction`: 100개 후보 중 90개 비영(범위 -0.274 ~ +0.188).
  음수 = elite band에서 멀어짐(과수정 포함) — 이전 구현에선 존재하지 않던 신호.
- 선택된 13개 전부 gain ≥ 1, 그중 gain 3짜리 2건(train_76, train_94)은
  gap_reduction도 양수 — 채점기·자질 이중 신호가 같은 방향.

## 채점기 노이즈 측정 결과 (별도 실험, `scripts/measure_scorer_noise.py`)

- 채점기는 temperature 0.7 샘플링 m개 앙상블(soft self-consistency) + RF 보정 구조.
  패키지 설계 기본값 m=20, 지금까지 MVP는 m=1~3으로 실행해 왔음.
- m=1 측정: 동일 텍스트 재채점 최대 ±2, 공백 정리 ±3, 동의어 1개 치환 ±2.
  m=3이면 대략 ±1.2 수준으로 추정 (±2/√3).
- **함의: target_gain=1은 m=3에서 노이즈와 구분이 어렵다.** 선택된 13건 중
  gain=1이 8건 — 이 중 일부는 노이즈 accept일 수 있음.
- 결과 파일: `experiments/results/scorer_noise.json`

## 남은 문제 (다음 결정 사항)

1. **m 트레이드오프**: m=20이면 노이즈 ±0.5 수준으로 떨어지지만 진단당 생성 비용 6.7배.
   절충안 — (a) 스텝당 진단 수를 action prior로 줄이고 m을 올린다,
   (b) gain=1 후보는 재채점 1회로 검증(2-of-2 통과 시 accept), (c) m=10로 중간 타협.
   **결정 필요.**
2. `non_target_drop_max` 1.0 유지 중 → drop=1 후보 4건 통과(31%).
   노이즈 ±1.2에서 drop=1은 판별 불가. m을 올린 뒤 0.5로 조이는 게 순서.
3. reject_all 7건(35%)의 의미: STOP 생성 제거 후 "전 후보 기각 = 사실상 stop".
   7건 모두가 정말 수정 불필요한 글인지 육안 확인 필요 — gain_min이 너무 보수적이면
   개선 여지가 있는 글도 멈춤. (노이즈 문제와 동전의 양면)
4. STYLE_REFINE 0회 / RESTRUCTURE 0회 선택: 두 action은 rubric을 ±1 이상 움직이기
   어려움. 실제 효과가 없는 건지, 채점기가 문체·구조 변화에 둔감한 건지
   corruption 데이터에서 검증할 것 (STYLE/RESTRUCTURE 역연산 corruption의
   rubric 재측정 결과가 판별해줌).
5. 이번 20건은 표본이 작다. 위 1~2 결정 후 100건으로 확장 → 그 로그가
   TVM 학습 데이터 ②번 소스의 첫 배치가 됨.

## 재현 명령

```bash
# 재실행 (AFTER 조건)
python scripts/run_mvp_batch.py \
  --input data/data_jsonl/train.jsonl \
  --output experiments/results/mvp_stage_a_20_v2.jsonl \
  --overwrite --limit 20 --min-chars 250 \
  --diagnoser kanana --proposer-mode llm --patcher-mode llm \
  --n-per-action 1 --kanana-m 3 --device-id 3

# 비교
python scripts/compare_mvp_logs.py \
  experiments/results/mvp_stage_a_20.jsonl \
  experiments/results/mvp_stage_a_20_v2.jsonl

# elite band 재구축 (feature 정의가 바뀌면)
python scripts/build_elite_band.py --input data/data_jsonl/train.jsonl --top-n 200

# 노이즈 재측정 (m 결정 시 --kanana-m 바꿔가며)
python scripts/measure_scorer_noise.py --device-id 3 --n-essays 3
```
