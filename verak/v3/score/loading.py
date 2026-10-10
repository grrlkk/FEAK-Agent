"""Adapted from essay_scoring_llm.soft_sc.load_model_and_tokenizer.

Same frozen base + LoRA and NF4 recipe; explicit one-GPU mapping, local files
only, and no import of the scorer package's .env-loading __init__.
"""

from __future__ import annotations

import os


def load_frozen_model(base_path, adapter_path, *, gpu=1, seed=42, load_in_4bit=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available() or gpu >= torch.cuda.device_count():
        raise RuntimeError(f"Requested scorer GPU {gpu} is unavailable")
    torch.cuda.set_device(gpu)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = None
    if load_in_4bit:
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(str(base_path), local_files_only=True,
        quantization_config=quantization, device_map={"": f"cuda:{gpu}"},
        torch_dtype=dtype, attn_implementation="eager")
    model = PeftModel.from_pretrained(base, str(adapter_path), local_files_only=True,
                                      is_trainable=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), local_files_only=True,
                                               use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
