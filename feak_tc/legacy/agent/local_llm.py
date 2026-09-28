"""A single cached, untrained Transformers model shared by all revision roles."""

import json
from typing import Any

from feak_tc.mvp.llm import LLMResponseError

from .schemas import LocalModelConfig


class LocalCallBudgetExceeded(RuntimeError):
    pass


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise LLMResponseError("Local model did not return a complete JSON object") from exc
    if not isinstance(value, dict):
        raise LLMResponseError("Local model response must be a JSON object")
    return value


class LocalJSONClient:
    """Matches the existing patcher's JSON callback; never calls a remote API."""

    def __init__(self, cfg: LocalModelConfig):
        self.cfg = cfg
        self.calls = 0
        self.records: list[dict[str, Any]] = []
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if self.cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check GPU access before running the local agent")
        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "device_map": {"": self.cfg.device},
            "torch_dtype": torch.float16 if self.cfg.device.startswith("cuda") else torch.float32,
        }
        if self.cfg.load_in_4bit:
            if not self.cfg.device.startswith("cuda"):
                raise ValueError("This 4-bit runtime requires a CUDA device")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(self.cfg.model, **kwargs).eval()

    def __call__(self, *, system: str, user: str, **unused) -> dict[str, Any]:
        if self.calls >= self.cfg.max_calls:
            raise LocalCallBudgetExceeded("Local LLM call budget exhausted")
        self._load()
        import torch

        prompt = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.cfg.device)
        input_tokens = inputs["input_ids"].shape[1]
        if input_tokens > self.cfg.max_input_tokens:
            raise LLMResponseError(f"Input has {input_tokens} tokens; refusing to truncate the essay")
        self.calls += 1
        options = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": self.cfg.temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if self.cfg.temperature > 0:
            options["temperature"] = self.cfg.temperature
        # Scorer sampling and generator sampling do not alter each other's RNG state.
        # manual_seed seeds every CUDA device, including the Kanana GPU.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(self.cfg.seed + self.calls)
            output = self._model.generate(**inputs, **options)
        generated = output[0, input_tokens:]
        raw = self._tokenizer.decode(generated, skip_special_tokens=True)
        self.records.append({
            "call": self.calls, "model": self.cfg.model,
            "input_tokens": input_tokens, "output_tokens": len(generated), "raw": raw,
        })
        return parse_json_object(raw)
