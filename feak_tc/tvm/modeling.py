"""Quantized decoder reward model with a scalar sequence-classification head."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def resolve_model_snapshot(
    model_name: str,
    *,
    local_files_only: bool = True,
) -> Path:
    """Resolve a Hub ID once so Transformers cannot make hidden network calls."""

    source = Path(model_name)
    if source.exists():
        return source.resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(model_name, local_files_only=local_files_only)
    ).resolve()


def load_tokenizer(model_name: str, *, local_files_only: bool = True) -> Any:
    from transformers import AutoTokenizer

    snapshot = resolve_model_snapshot(
        model_name,
        local_files_only=local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(f"tokenizer for {model_name} has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_reward_model(
    model_name: str,
    *,
    tokenizer: Any,
    load_in_4bit: bool,
    gradient_checkpointing: bool,
    lora: Mapping[str, Any] | None = None,
    adapter_path: str | Path | None = None,
    local_files_only: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Load a scalar reward model and optionally attach/train a LoRA adapter."""

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

    snapshot = resolve_model_snapshot(
        model_name,
        local_files_only=local_files_only,
    )
    if load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("4-bit TVM loading requires CUDA")
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    quantization_config = None
    device_map = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        device_map = {"": 0}
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot,
        num_labels=1,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    model.config.use_cache = False

    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )
        model.eval()
        return model, _model_info(
            model_name, snapshot, model, compute_dtype, load_in_4bit
        )

    if lora is None:
        raise ValueError("LoRA configuration is required when training a TVM")
    if load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        bias="none",
        target_modules=list(lora.get("target_modules", DEFAULT_LORA_TARGETS)),
        modules_to_save=["score"],
    )
    model = get_peft_model(model, config)
    model.train()
    return model, _model_info(model_name, snapshot, model, compute_dtype, load_in_4bit)


def trainable_parameter_summary(model: Any) -> dict[str, int | float]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable": int(trainable),
        "total": int(total),
        "trainable_fraction": float(trainable / total if total else 0.0),
    }


def _model_info(
    model_name: str,
    snapshot: Path,
    model: Any,
    compute_dtype: Any,
    load_in_4bit: bool,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "snapshot": str(snapshot),
        "revision": snapshot.name,
        "architecture": model.base_model.model.config.architectures
        if hasattr(model, "base_model")
        else model.config.architectures,
        "model_type": model.config.model_type,
        "compute_dtype": str(compute_dtype).removeprefix("torch."),
        "load_in_4bit": bool(load_in_4bit),
        "parameters": trainable_parameter_summary(model),
    }
