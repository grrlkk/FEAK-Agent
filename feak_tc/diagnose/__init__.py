"""Diagnoser factory."""

from __future__ import annotations

from .base import Diagnosis, Diagnoser, select_weak_rubrics
from .constants import FEAK_FEATURE_NAMES, RUBRIC_KEYS, RUBRIC_NAMES_KO
from .stub import StubDiagnoser


def get_diagnoser(kind: str = "stub", **kwargs) -> Diagnoser:
    if kind == "stub":
        return StubDiagnoser(**kwargs)
    if kind == "kanana":
        from .kanana import KananaDiagnoser

        return KananaDiagnoser(**kwargs)
    if kind == "feak_kobert":
        from .feak_kobert import FeakKobertDiagnoser

        return FeakKobertDiagnoser(**kwargs)
    raise ValueError(f"Unknown diagnoser kind: {kind}")


__all__ = [
    "Diagnosis",
    "Diagnoser",
    "FEAK_FEATURE_NAMES",
    "RUBRIC_KEYS",
    "RUBRIC_NAMES_KO",
    "StubDiagnoser",
    "get_diagnoser",
    "select_weak_rubrics",
]
