"""Adapter for the sibling essay_scoring_llm Kanana scorer."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from .base import Diagnosis, select_weak_rubrics
from .constants import scores_to_rubric_dict


LOW_MEMORY_CONFIG_DEFAULTS = {
    "generate_feedback": False,
    "chunk_m": 1,
}

FEATURE_LOCK_PATH = Path(os.getenv("FEAK_FEATURE_LOCK_PATH", "/tmp/feak_feature_extract.lock"))


class KananaDiagnoser:
    """Diagnoser backed by essay_scoring_llm.

    The heavy Kanana model is loaded lazily once and reused. The package is not
    imported at module import time so normal MVP tests can run without GPU deps.
    """

    def __init__(
        self,
        question: str = "다음 글을 평가하세요.",
        keywords: Optional[str] = None,
        weak_top_n: int = 3,
        package_path: Optional[str] = None,
        config_kwargs: Optional[dict[str, Any]] = None,
        feature_timeout_s: int = 300,
        feature_retries: int = 2,
    ) -> None:
        self.question = question
        self.keywords = keywords
        self.weak_top_n = weak_top_n
        self.package_path = package_path or os.getenv("ESSAY_SCORING_LLM_PATH")
        self.config_kwargs = config_kwargs or {}
        self.feature_timeout_s = feature_timeout_s
        self.feature_retries = feature_retries
        self._loaded: Optional[dict[str, Any]] = None

    def diagnose(self, text: str) -> Diagnosis:
        features = self._extract_features(text)
        loaded = self._ensure_loaded()
        scoring_text = _normalize_scoring_text(text)
        user = loaded["build_user_prompt"](self.question, scoring_text, self.keywords)
        try:
            scored = loaded["score_example"](
                loaded["model"],
                loaded["tokenizer"],
                loaded["helper"],
                loaded["system_prompt"],
                user,
                loaded["config"],
            )
        except RuntimeError as exc:
            if _is_cuda_memory_error(exc):
                raise RuntimeError(
                    "Kanana generation failed because CUDA memory is too tight. "
                    "Free GPU memory with `nvidia-smi`, use a less busy GPU with "
                    "`--device-id`, or stop unrelated GPU processes before retrying."
                ) from exc
            raise
        raw_scores = scored["raw_scores"]
        if not raw_scores:
            raise RuntimeError("Kanana scorer returned no valid raw_scores.")

        raw = np.asarray(raw_scores, dtype=np.float64)
        means = np.nanmean(raw, axis=0)
        stds = np.nanstd(raw, axis=0)
        corrected, final = loaded["correction"].predict(means, stds, features)
        rubrics = scores_to_rubric_dict(final)
        weak = select_weak_rubrics(rubrics, top_n=self.weak_top_n)
        return Diagnosis(
            text=text,
            rubrics=rubrics,
            features=features,
            weak_rubrics=weak,
            metadata={
                "diagnoser": "kanana",
                "question": self.question,
                "feature_device": "isolated_subprocess",
                "scoring_text_normalized": True,
                "soft_mean": [float(x) for x in means],
                "soft_std": [float(x) for x in stds],
                "rf_corrected_score": [float(x) for x in corrected],
                "feedback": scored.get("feedbacks", []),
            },
        )

    def _ensure_loaded(self) -> dict[str, Any]:
        if self._loaded is not None:
            return self._loaded

        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        _ensure_package_importable(self.package_path)
        from essay_scoring_llm.config import ScoringConfig
        from essay_scoring_llm.correction import CorrectionModule
        from essay_scoring_llm.soft_sc import (
            SYSTEM_PROMPT,
            build_user_prompt,
            load_model_and_tokenizer,
            score_example,
        )

        config_values = {**LOW_MEMORY_CONFIG_DEFAULTS, **self.config_kwargs}
        config = ScoringConfig(**config_values)
        model, tokenizer, helper = load_model_and_tokenizer(config)
        correction = CorrectionModule(config.correction_models, config.rounder_thresholds)
        self._loaded = {
            "config": config,
            "model": model,
            "tokenizer": tokenizer,
            "helper": helper,
            "correction": correction,
            "build_user_prompt": build_user_prompt,
            "score_example": score_example,
            "system_prompt": SYSTEM_PROMPT,
        }
        return self._loaded

    def _extract_features(self, text: str) -> dict[str, float]:
        _ensure_package_importable(self.package_path)

        script = """
import json
import sys

from essay_scoring_llm.feak import extract_feak_features

features = extract_feak_features(sys.stdin.read())
print("__FEAK_FEATURES_JSON__" + json.dumps(features, ensure_ascii=False))
"""
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("MPLCONFIGDIR", "/tmp/feak_matplotlib")
        feature_cuda = os.getenv("FEAK_FEATURE_CUDA_VISIBLE_DEVICES")
        if feature_cuda is not None:
            env["CUDA_VISIBLE_DEVICES"] = feature_cuda
        if self.package_path:
            package_path = str(Path(self.package_path).expanduser())
            env["PYTHONPATH"] = (
                package_path
                if not env.get("PYTHONPATH")
                else package_path + os.pathsep + env["PYTHONPATH"]
            )

        completed = None
        with _feature_extraction_lock():
            for attempt in range(self.feature_retries + 1):
                try:
                    completed = subprocess.run(
                        [sys.executable, "-c", script],
                        input=text,
                        text=True,
                        capture_output=True,
                        check=False,
                        env=env,
                        timeout=self.feature_timeout_s,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        "FEAK feature extraction timed out in isolated subprocess "
                        f"after {self.feature_timeout_s} seconds."
                    ) from exc
                if completed.returncode == 0:
                    break
                if attempt < self.feature_retries:
                    time.sleep(1.0 + attempt)

        if completed is None or completed.returncode != 0:
            stdout = completed.stdout if completed is not None else ""
            stderr = completed.stderr if completed is not None else ""
            returncode = completed.returncode if completed is not None else "unknown"
            raise RuntimeError(
                "FEAK feature extraction failed in isolated subprocess.\n"
                f"returncode: {returncode}\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )

        prefix = "__FEAK_FEATURES_JSON__"
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith(prefix):
                raw_features = json.loads(line.removeprefix(prefix))
                return {str(key): float(value) for key, value in raw_features.items()}

        raise RuntimeError(
            "FEAK feature extraction finished, but no feature JSON was found.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


@contextmanager
def _feature_extraction_lock() -> Iterator[None]:
    """Serialize the legacy FEAK extractor across parallel GPU shards."""

    import fcntl

    FEATURE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEATURE_LOCK_PATH.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _normalize_scoring_text(text: str) -> str:
    lines = text.strip().splitlines()
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in lines)


def _ensure_package_importable(package_path: Optional[str]) -> None:
    try:
        __import__("essay_scoring_llm")
        return
    except ModuleNotFoundError:
        pass

    candidates = []
    if package_path:
        candidates.append(Path(package_path).expanduser())
    candidates.append(Path("/home/chanwoo/essay_scoring_llm"))
    for candidate in candidates:
        if (candidate / "essay_scoring_llm").is_dir():
            sys.path.insert(0, str(candidate))
            __import__("essay_scoring_llm")
            return
    raise ModuleNotFoundError(
        "essay_scoring_llm is not importable. Install it with "
        "`pip install -e /home/chanwoo/essay_scoring_llm` or set ESSAY_SCORING_LLM_PATH."
    )


def _is_cuda_memory_error(exc: RuntimeError) -> bool:
    text = str(exc)
    markers = (
        "CUDA out of memory",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "cuda runtime error",
    )
    return any(marker in text for marker in markers)
