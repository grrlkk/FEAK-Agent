"""Deterministic, nested sets of whitespace-only score probes."""

from dataclasses import asdict, dataclass
import math
import random
import re

VERSION = "one-extra-space-distinct-boundaries-v1"
MARKER = re.compile(r"#@[^#\r\n]+#")


@dataclass(frozen=True)
class WhitespaceProbe:
    text: str
    position: int | None


@dataclass(frozen=True)
class AveragedScoreResult:
    mean: float
    genre: str
    k: int
    seed: int
    positions: list[int | None]
    members: list[dict]
    member_seconds: list[float]
    elapsed_s: float

    def to_dict(self):
        return asdict(self)


def whitespace_probes(text, k, *, seed=13):
    """Original plus k-1 single-space insertions; no edits inside a marker.

    Shuffle the complete, ordered list with one fixed seed before taking a
    prefix. Thus k=1/3/5 use nested sets, independent of request order or Q.
    Horizontal whitespace runs count as one boundary, even for the old
    calibration whitespace treatment. Newlines and paragraph breaks stay intact.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if not text.strip():
        raise ValueError("Text must be nonblank")
    spans = [m.span() for m in MARKER.finditer(text)]
    positions = [m.start() for m in re.finditer(r"(?<=\S)[ \t]+(?=\S)", text)
                 if not any(a <= m.start() < b for a, b in spans)]
    if len(positions) < k - 1:
        raise ValueError("Not enough distinct whitespace boundaries for requested k")
    random.Random(seed).shuffle(positions)
    return [WhitespaceProbe(text, None)] + [
        WhitespaceProbe(text[:p] + " " + text[p:], p) for p in positions[:k-1]]


def average_prefix(members, k):
    if type(k) is not int or not 1 <= k <= len(members):
        raise ValueError("Invalid averaging prefix")
    values = [row["mean"] for row in members[:k]]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite component Q")
    return sum(values) / k
