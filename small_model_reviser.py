#!/usr/bin/env python3
"""모델, 프롬프트, 원문을 아래에 입력하고 실행하는 단일 파일 수정기.

설치: python -m pip install torch 'transformers==4.57.6' accelerate
실행: python small_model_reviser.py
첫 실행에는 선택한 모델을 Hugging Face에서 내려받습니다.
RUN_ALL_MODELS=True이면 같은 원문을 네 모델에 각각 전달합니다.
"""

from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import secrets
import sys


# ===== 여기에서 모델, 프롬프트, 원문을 바꾸세요. =====
MODELS = {
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "exaone": "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "kanana-3b": "kakaocorp/kanana-2-3b-instruct",
    "kanana-1.3b": "kakaocorp/kanana-2-1.3b-instruct",
}
MODEL_NAME = "kanana-1.3b"  # qwen / exaone / kanana-3b / kanana-1.3b
RUN_ALL_MODELS = False     # True: 네 모델을 하나씩 실행하여 비교

DO_SAMPLE = True          # False: 가장 확률 높은 토큰만 선택(greedy)
TEMPERATURE = 0.7         # 샘플링일 때만 적용. 0 대신 DO_SAMPLE=False를 사용하세요.
TOP_P = 0.95              # 샘플링 후보의 누적 확률. 1.0이면 제한 없음
TOP_K = 0                 # 샘플링 후보 수. 0이면 제한 없음
SEED = None               # None: 매 실행 새 시드. 정수: 같은 환경에서 재현용

PROMPT = """다음 원문을 자연스럽고 명확한 한국어로 수정해 주세요.
원문의 의미와 주장을 유지하고, 글에 없는 사실이나 사례를 새로 만들지 마세요.
맞춤법, 띄어쓰기, 어색한 표현과 문장 연결을 다듬어 주세요.
설명이나 머리말 없이 수정한 글 전체만 출력해 주세요."""

# 실행 예시용 원문입니다. 여기에 수정할 글을 붙여 넣으세요.
ORIGINAL_TEXT = """나는 학교에서 책을 읽는 시간이 필요하다고 생각한다.
왜냐하면 책을 읽으면 여러가지 생각을 할수 있기 때문이다.
그래서 학교는 학생들이 책을 읽을수 있는 시간을 마련해 주면 좋겟다."""

MAX_NEW_TOKENS = 2048
DEVICE = "auto"  # auto: CUDA가 있으면 cuda:0, 없으면 cpu. 직접 cuda:1 등도 지정 가능
OUTPUT_ROOT = Path(__file__).resolve().parent / "verak/outputs/small_model_reviser"
# ===================================================


def log(message):
    print(message, file=sys.stderr, flush=True)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generation_options():
    if type(DO_SAMPLE) is not bool:
        raise ValueError("DO_SAMPLE은 True 또는 False여야 합니다.")
    if type(MAX_NEW_TOKENS) is not int or MAX_NEW_TOKENS <= 0:
        raise ValueError("MAX_NEW_TOKENS는 양수인 정수여야 합니다.")
    if SEED is not None and (type(SEED) is not int or not 0 <= SEED < 2**32):
        raise ValueError("SEED는 None 또는 0 이상 2**32 미만인 정수여야 합니다.")
    if DO_SAMPLE:
        if (type(TEMPERATURE) not in (int, float)
                or not math.isfinite(TEMPERATURE) or TEMPERATURE <= 0):
            raise ValueError("샘플링 온도는 양수여야 합니다. 결정적 생성은 DO_SAMPLE=False로 설정하세요.")
        if type(TOP_P) not in (int, float) or not 0 < TOP_P <= 1:
            raise ValueError("TOP_P는 0 초과 1 이하여야 합니다.")
        if type(TOP_K) is not int or TOP_K < 0:
            raise ValueError("TOP_K는 0 이상인 정수여야 합니다.")
    return {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "temperature": float(TEMPERATURE) if DO_SAMPLE else None,
        "top_p": float(TOP_P) if DO_SAMPLE else None,
        "top_k": TOP_K if DO_SAMPLE else None,
        "num_beams": 1,
        "num_return_sequences": 1,
    }


def sampling_seed():
    return (SEED if SEED is not None else secrets.randbits(32)) if DO_SAMPLE else None


def revise(model_id, messages, *, seed=None):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

    options = generation_options()
    seed = (sampling_seed() if seed is None else seed) if DO_SAMPLE else None

    device = torch.device(
        ("cuda:0" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else DEVICE
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    # Kanana 1.3B의 kanana2_tiny 구조는 공식 저장소의 전용 구현을 사용합니다.
    # https://huggingface.co/kakaocorp/kanana-2-1.3b-instruct/blob/main/config.json
    trust_remote_code = model_id == MODELS["kanana-1.3b"]
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,     # EXAONE: 수정문을 바로 생성
        thinking_mode="no_think",  # Kanana 2: 수정문을 바로 생성
        truncation=False,
    )
    input_tokens = inputs["input_ids"].shape[-1]
    context_limit = config.max_position_embeddings
    if input_tokens + MAX_NEW_TOKENS > context_limit:
        raise ValueError(
            f"입력 {input_tokens} + 최대 출력 {MAX_NEW_TOKENS} 토큰이 "
            f"모델 한도 {context_limit}을 넘습니다. 원문이나 MAX_NEW_TOKENS를 줄이세요."
        )

    model = None
    try:
        log(f"모델 로딩: {model_id} ({device}, {dtype})")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            device_map={"": str(device)},
        ).eval()
        inputs = inputs.to(device)
        log(f"생성 중: 입력 {input_tokens} 토큰 / 최대 출력 {MAX_NEW_TOKENS} 토큰 "
            f"/ do_sample={DO_SAMPLE}, temperature={options['temperature']}, seed={seed}")
        # 모델 로딩에 사용된 난수와 생성 난수를 분리합니다.
        if seed is not None:
            set_seed(seed)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                **options,
                pad_token_id=tokenizer.pad_token_id,
            )
        # 입력 프롬프트를 제외하고 새로 생성한 부분만 반환합니다.
        token_ids = generated[0, input_tokens:].tolist()
        answer = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if not answer:
            raise RuntimeError("모델이 빈 응답을 생성했습니다.")
        eos_ids = model.generation_config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        finish_reason = "eos" if token_ids and token_ids[-1] in eos_ids else "length"
        return {
            "model_id": model_id,
            "text": answer,
            "device": str(device),
            "dtype": str(dtype),
            "input_tokens": input_tokens,
            "output_tokens": len(token_ids),
            "finish_reason": finish_reason,
            "last_token_id": token_ids[-1],
            "last_token": tokenizer.decode(token_ids[-1:], skip_special_tokens=False),
            "eos_token_ids": eos_ids,
            "generation": options,
            "seed": seed,
            "model_revision": getattr(config, "_commit_hash", None),
        }
    finally:
        # 다음 모델을 불러오기 전에 모델과 입력/출력 텐서를 해제합니다.
        model = None
        inputs = None
        generated = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    if not PROMPT.strip() or not ORIGINAL_TEXT.strip():
        raise ValueError("파일 상단의 PROMPT와 ORIGINAL_TEXT에 내용을 입력하세요.")
    if not RUN_ALL_MODELS and MODEL_NAME not in MODELS:
        raise ValueError(f"MODEL_NAME은 {', '.join(MODELS)} 중 하나여야 합니다.")
    options = generation_options()
    seed = sampling_seed()
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import accelerate  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "필요한 패키지를 설치하세요: "
            "python -m pip install torch 'transformers==4.57.6' accelerate"
        ) from exc

    names = list(MODELS) if RUN_ALL_MODELS else [MODEL_NAME]
    messages = [{"role": "user", "content": f"{PROMPT}\n\n[원문]\n{ORIGINAL_TEXT}"}]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = OUTPUT_ROOT / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    save_json(output_dir / "input.json", {
        "prompt": PROMPT,
        "original_text": ORIGINAL_TEXT,
        "messages": messages,
        "models": {name: MODELS[name] for name in names},
        **options,
        "requested_seed": SEED,
        "seed": seed,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    })
    log(f"저장 경로: {output_dir}")

    failures = 0
    for name in names:
        try:
            result = revise(MODELS[name], messages, seed=seed)
            save_json(output_dir / f"{name}.json", result)
            (output_dir / f"{name}.txt").write_text(result["text"] + "\n", encoding="utf-8")
            print(f"\n===== {name}: {MODELS[name]} =====\n{result['text']}", flush=True)
            if result["finish_reason"] == "length":
                log(f"{name}: 출력 길이 한도에 도달했습니다. 필요하면 MAX_NEW_TOKENS를 늘리세요.")
        except Exception as exc:
            failures += 1
            save_json(output_dir / f"{name}.error.json", {
                "model_id": MODELS[name], "error_type": type(exc).__name__, "error": str(exc),
            })
            log(f"{name} 실행 실패: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        log(str(exc))
        raise SystemExit(1)
