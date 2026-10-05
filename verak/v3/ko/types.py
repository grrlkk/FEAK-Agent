"""Observations and explicitly uncertain heuristic readings; source text is immutable."""

from dataclasses import asdict, dataclass, field

from verak.src.schemas import Token
from .levels import ANNOTATION_LEVELS, COHESION_CHANGE_LEVELS, UNCERTAINTY_LEVELS


@dataclass
class ConnInfo:
    token_id: str
    form: str
    span: list[int]
    candidates: list[str]
    classification: str = "AMBIGUOUS"
    level: str = "WORD"


@dataclass
class ConjInfo:
    form: str
    relation: str
    level: str = "SENTENCE"


@dataclass
class SubjInfo:
    realized: bool = False
    surface: str | None = None
    marker: str = "NONE"
    lemma: str | None = None
    span: list[int] | None = None
    entity_id: str | None = None


@dataclass
class AnteInfo:
    status: str = "none"
    targets: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)


@dataclass
class Annotation:
    sid: str
    paragraph: str
    style: str
    connectives: list[ConnInfo]
    initial_conj: ConjInfo | None
    subject: SubjInfo
    topic: str | None
    antecedent: AnteInfo
    polarity: str
    modality: str | None
    uncertain: list[str]
    text: str
    start: int
    end: int
    tokens: list[Token]
    subject_candidates: list[SubjInfo] = field(default_factory=list)
    multi_unit: bool = False
    final_endings: list[dict] = field(default_factory=list)
    focus_particles: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Edge:
    kind: str
    src: str
    dst: str
    label: str
    confidence: str = "LOW"
    level: str = "SENTENCE"


@dataclass
class Structure:
    text: str
    annotations: list[Annotation]
    edges: list[Edge]
    dominant_style: str
    analyzer: str
    analyzer_version: str

    def to_dict(self):
        result = asdict(self)
        for ann in result["annotations"]:
            # Uncertainty can span all three levels: tag each constituent, never
            # incorrectly collapse this list to a single linguistic level.
            ann["field_levels"] = dict(ANNOTATION_LEVELS)
            ann["field_levels"]["uncertain"] = {
                name: UNCERTAINTY_LEVELS[name] for name in ann["uncertain"]}
        result["field_levels"] = {"dominant_style": "TEXT", "edges": "SENTENCE"}
        result["cohesion_change_levels"] = dict(COHESION_CHANGE_LEVELS)
        return result
