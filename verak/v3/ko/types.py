"""Observations and explicitly uncertain heuristic readings; source text is immutable."""

from dataclasses import asdict, dataclass, field

from verak.src.schemas import Token


@dataclass
class ConnInfo:
    token_id: str
    form: str
    span: list[int]
    candidates: list[str]


@dataclass
class ConjInfo:
    form: str
    relation: str


@dataclass
class SubjInfo:
    realized: bool = False
    surface: str | None = None
    marker: str = "NONE"
    lemma: str | None = None
    span: list[int] | None = None


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


@dataclass(frozen=True)
class Edge:
    kind: str
    src: str
    dst: str
    label: str


@dataclass
class Structure:
    text: str
    annotations: list[Annotation]
    edges: list[Edge]
    dominant_style: str
    analyzer: str
    analyzer_version: str

    def to_dict(self):
        return asdict(self)
