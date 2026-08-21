# TVM Stage-1 4090 서버 실행 가이드

> 목적: 전체 FEAK 데이터와 로컬 Hugging Face cache를 복사하지 않고, TVM 학습·평가에
> 필요한 최소 입력만 4090 서버로 보내 동일 실험을 재현한다.

## 옮겨야 하는 것

코드는 topic branch `tvm-stage1-cross-backbone`을 원격에서 clone한다. Git 제외 데이터는
아래 두 파일만 복사한다. 합계는 약 15MB다.

| 파일 | 크기 | SHA-256 |
|---|---:|---|
| `experiments/results/corruption_g1_rulev5_1000_training.jsonl` | 7.7MB | `993ac439c82d04292b1f3ba7eac20eb34907829932c6aa28bd6d8fbfceca1094` |
| `experiments/results/corruption_g1_rulev5_1000_bge_m3_states_verified.npz` | 7.3MB | `20a32220e79d8308a873005b9bbc87bedbad6f008aae62cfb0cf5d49fe33e6a3` |

`configs/elite_features.yaml`, `configs/tvm_stage1.yaml`과 학습 코드는 Git branch에 들어간다.
Qwen/Kanana base model은 복사하지 않고 서버에서 다시 내려받는다. BGE-M3 base model도
필요 없다. 위 NPZ에 검증된 state embedding이 이미 들어 있다.

두 파일을 묶은 로컬 전송본은
`experiments/results/tvm_stage1_minimal_inputs_2026-08-21.tar.gz`(8.5MB,
SHA-256 `0e7f3a18bf48ba3ec485fbe1cceac12d282e6f8b5340f0f615c93f890976c2b9`)다.
서버의 repo root에서 `tar -xzf`하면 원래 경로로 풀린다.

## 서버 준비

```bash
git clone --branch tvm-stage1-cross-backbone \
  https://github.com/grrlkk/FEAK-Agent.git
cd FEAK-Agent
pip install -e '.[tvm,g2,dev]'

mkdir -p experiments/results
# scp/rsync로 위 두 파일을 experiments/results/에 둔다.
sha256sum \
  experiments/results/corruption_g1_rulev5_1000_training.jsonl \
  experiments/results/corruption_g1_rulev5_1000_bge_m3_states_verified.npz

hf download Qwen/Qwen2.5-7B-Instruct --exclude '*.gguf' '*.bin'
hf download kakaocorp/kanana-1.5-8b-instruct-2505 --exclude '*.gguf' '*.bin'
pytest -q
```

## 최장 입력 smoke

`--limit 20`은 operator별 최장 샘플을 우선 선택하므로 24GB 4090 메모리 stress test다.

```bash
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python scripts/train_tvm_stage1.py \
    --model-key qwen --feature-variant scorer_free --learning-rate 2e-6 \
    --output-dir experiments/results/tvm_stage1_smoke/qwen --limit 20

env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python scripts/train_tvm_stage1.py \
    --model-key kanana --feature-variant scorer_free --learning-rate 2e-6 \
    --output-dir experiments/results/tvm_stage1_smoke/kanana --limit 20
```

## 12-run validation sweep

한 4090에서는 순차 실행을 권장한다. 스크립트는 완료된 run을 검증 후 건너뛰므로 SSH가
끊겨도 같은 명령으로 재개할 수 있다. test split은 이 단계에서 평가하지 않는다.

```bash
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python scripts/run_tvm_stage1_sweep.py \
    --runs-root experiments/results/tvm_stage1

python scripts/select_tvm_stage1.py \
  --runs-root experiments/results/tvm_stage1 \
  --output experiments/results/tvm_stage1/selection.json
```

조건은 `Qwen/Kanana × full/scorer_free × LR 1/2/3e-6`이며 모두 같은 seed, essay-disjoint
`793/106/101` split, 1 epoch, pair batch 1, gradient accumulation 8, max length 1536을 쓴다.

## 선택 후 test 1회와 baseline

```bash
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python scripts/evaluate_selected_tvm_stage1.py \
    --selection-manifest experiments/results/tvm_stage1/selection.json

python scripts/run_tvm_stage1_baselines.py \
  --selection-manifest experiments/results/tvm_stage1/selection.json \
  --report-out experiments/results/tvm_stage1/baselines.json \
  --predictions-out experiments/results/tvm_stage1/baseline_predictions.jsonl

python scripts/summarize_tvm_stage1.py \
  --selection-manifest experiments/results/tvm_stage1/selection.json \
  --baseline-report experiments/results/tvm_stage1/baselines.json \
  --baseline-predictions experiments/results/tvm_stage1/baseline_predictions.jsonl \
  --output experiments/results/tvm_stage1/summary.json
```

## 다시 가져올 최소 산출물

분석만 필요하면 `selection.json`, `summary.json`, `baselines.json`, 12개
`validation_report.json`, 선택된 4개 `test_report.json`과 prediction JSONL만 가져오면 된다.
실제 controller에서 모델을 쓰려면 선택된 네 run의 `adapter/`도 가져온다. Qwen smoke 기준
adapter 하나는 약 78MB다. 각 run의 중복 `tokenizer/`는 로컬 base tokenizer가 있으므로
복사하지 않아도 된다.

이 synthetic test는 사람 선호 검증을 대체하지 않는다. `scorer_free`도 corruption의
채택·필터가 Kanana 측정에 의존하므로 최종 결론은 별도 blind human preference에서 낸다.
