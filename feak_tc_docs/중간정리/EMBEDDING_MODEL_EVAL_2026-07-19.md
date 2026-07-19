# 임베딩 모델 비교 평가와 BGE-M3 채택

작성일: 2026-07-19

## 목적

`goal_preservation`/`emb_sim`이 500/500 후보에서 `token_fallback`(토큰
잔존율)으로 계산되고 있었다. 표면 겹침 지표는 정당한 바꿔쓰기를 벌하고
문맥 붕괴를 놓친다. 의미 기반 임베딩으로 교체하되, 어떤 모델이 우리
태스크에서 실제로 나은지 라벨 데이터로 판정했다.

## 평가 설계

질적 검토로 라벨이 확정된 수정 쌍(before → after)을 사용:

- **bad 12**: baseline(07-16) accept 중 문맥 붕괴·비문 확정
  (DOF 실패 11 + COMPRESS 중복 1)
- **good 47**: spanfix(07-19) accept 중 스캔 무결점 (borderline
  train_1021 제외)
- **hallu 3**: 통계 환각 (임베딩이 못 잡는 게 정상인지 확인용)

모델이 good과 bad를 유사도로 얼마나 분리하는지(AUC, 최적 임계 정확도) 측정.

## 결과

| 방법 | max_seq | AUC | 최적 임계 정확도 |
|---|---:|---:|---:|
| token_fallback (기존) | - | 0.762 | 0.814 |
| jhgan/ko-sbert-sts | 128 | 0.647 | 0.797 |
| **BAAI/bge-m3** | 8192 | **0.819** | **0.881** |
| nlpai-lab/KURE-v1 | 8192 | 0.796 | 0.864 |

- **고전 한국어 SBERT는 기존 fallback보다도 나빴다.** max_seq 128이라
  에세이(130~250+ 토큰) 뒷부분이 잘려, 뒷부분 수정에 장님이 된다.
  후보 목록에서 제외했다.
- BGE-M3 기준 good 47건 전부 ≥ 0.9637, bad는 절반이 0.95 미만.
- 환각 3건은 모든 모델에서 높은 유사도(BGE 평균 0.951) — 예상대로
  임베딩은 환각을 못 보며, validity guard가 계속 필요하다는 재확인.

## 적용한 변경

1. `feak_tc/mvp/transition.py`: `EMBEDDING_MODEL_CANDIDATES`를
   `(BAAI/bge-m3, nlpai-lab/KURE-v1)`로 교체. `FEAK_EMBEDDING_DEVICE`
   env로 장치 지정 가능(shard GPU의 VRAM이 빠듯하면 `cpu` 지정).
2. `configs/heuristic_stage_a_soft.yaml`: `goal_preservation_min`
   0.7 → **0.95** (BGE-M3 스케일 보정). token_fallback으로 떨어지는
   환경에서는 0.7로 되돌려야 함(yaml 주석 참조).
3. 세 모델 모두 로컬 캐시에 다운로드 완료
   (`~/.cache/huggingface/hub`). `local_files_only=True` 로드 확인.

## 한계

- bad 표본이 12건으로 작다. 0.95 임계는 잠정값이며, 다음 100건 실행
  로그로 재검증 필요.
- AUC 0.82는 완전 분리가 아니다(bad 최고 0.996). gp는 단독 판정기가
  아니라 여러 신호 중 하나로 유지한다.

## 다음 작업

1. BGE-M3 + 0.95 임계로 100건 재실행, gp 분포와 accept 변화 확인
2. proposer targeting 개선 (질문 핵심어 정의 문장 삭제 방지)
3. corruption 데이터 생성 단계 진입
