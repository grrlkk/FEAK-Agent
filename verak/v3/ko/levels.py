"""Three equally weighted linguistic levels; contracts, not Phase 3 operators."""

ANNOTATION_LEVELS = {
    "sid": "SENTENCE", "paragraph": "SENTENCE", "text": "SENTENCE",
    "start": "SENTENCE", "end": "SENTENCE", "tokens": "WORD",
    "style": "TEXT", "final_endings": "TEXT", "multi_unit": "SENTENCE",
    "connectives": "WORD", "focus_particles": "WORD", "polarity": "WORD",
    "modality": "WORD", "initial_conj": "SENTENCE", "subject": "SENTENCE",
    "subject_candidates": "SENTENCE", "topic": "SENTENCE", "antecedent": "SENTENCE",
}
UNCERTAINTY_LEVELS = {**ANNOTATION_LEVELS, "style_shift": "TEXT",
                      "modality_ill_formed": "WORD", "relation_target": "SENTENCE"}
COHESION_CHANGE_LEVELS = {
    "ec_relation": "WORD", "polarity": "WORD", "modality": "WORD", "focus_particle": "WORD",
    "reference": "SENTENCE", "topic": "SENTENCE", "initial_conjunction": "SENTENCE",
    "relation_target": "SENTENCE", "style": "TEXT", "dominant_style": "TEXT",
}


def relation_eligible(conn):
    return conn.classification == "UNAMBIGUOUS" and len(conn.candidates) == 1


def dependent_eligible(edge):
    return edge.kind in {"REF", "TOPIC"} and edge.confidence == "HIGH"


def corruption_target_eligible(annotation):
    return not annotation.multi_unit


def cohesion_change(kind, **details):
    """Typed record factory for later feedback; unknown types fail closed."""
    return {**details, "type": kind, "level": COHESION_CHANGE_LEVELS[kind]}
