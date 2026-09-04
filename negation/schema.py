"""Record schema for negation variants.

A ``NegationVariant`` is the single unit of output for stage 1 of the graded
negation-sensitivity project.  Every generator returns a list of these, and the
JSONL emitted by the CLI is one serialised record per line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Family / enum-ish constants.  Kept as plain strings so records stay trivially
# JSON-serialisable, but centralised so typos surface at import time.
# ---------------------------------------------------------------------------

FAM_SYNTACTIC = "A_syntactic"
FAM_QUANTIFIER = "B_quantifier"
FAM_NEG_ADVERB = "C_neg_adverb"
FAM_AFFIXAL = "D_affixal"
FAM_ANTONYM = "E_antonym"
FAM_IMPLICIT = "F_implicit"
FAM_PREPOSITIONAL = "G_prepositional"
FAM_HEDGED = "H_hedged"
FAM_INTENSIFIED = "I_intensified"
FAM_CONTRASTIVE = "J_contrastive"
FAM_CANCELLATION = "K1_cancellation"
FAM_COMPOUND = "K2_compound"
FAM_SCOPE_POSITION = "L_scope_position"

FAMILIES: frozenset[str] = frozenset(
    {
        FAM_SYNTACTIC,
        FAM_QUANTIFIER,
        FAM_NEG_ADVERB,
        FAM_AFFIXAL,
        FAM_ANTONYM,
        FAM_IMPLICIT,
        FAM_PREPOSITIONAL,
        FAM_HEDGED,
        FAM_INTENSIFIED,
        FAM_CONTRASTIVE,
        FAM_CANCELLATION,
        FAM_COMPOUND,
        FAM_SCOPE_POSITION,
    }
)

SCOPE_SUBJECT = "subject"
SCOPE_PREDICATE = "predicate"
SCOPE_CLAUSE = "clause"
SCOPE_TARGETS: frozenset[str] = frozenset({SCOPE_SUBJECT, SCOPE_PREDICATE, SCOPE_CLAUSE})

INT_NEUTRAL = "neutral"
INT_HEDGED = "hedged"
INT_INTENSIFIED = "intensified"
INT_PARTIAL = "partial"
INTENSITY_HINTS: frozenset[str] = frozenset(
    {INT_NEUTRAL, INT_HEDGED, INT_INTENSIFIED, INT_PARTIAL}
)

#: Hard cap on how many negation operations may be stacked in one variant.
MAX_DEPTH = 2


@dataclass
class NegationVariant:
    """One generated negated variant of a base sentence, with provenance."""

    base_id: str
    base_sentence: str
    variant: str
    family: str
    subtype: str
    cue_tokens: list[str]
    cue_char_spans: list[tuple[int, int]]
    cue_count: int
    net_negation: int
    scope_target: str
    intensity_hint: str
    generator: str
    #: Number of stacked negation operations (1 = atomic, 2 = composed).  Never
    #: exceeds :data:`MAX_DEPTH`; used by the composition layer, not exported as
    #: part of the required schema but useful downstream.
    depth: int = 1
    #: Hook for stage 2 (LLM verification).  ``None`` means "not yet checked".
    verified: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family: {self.family!r}")
        if self.scope_target not in SCOPE_TARGETS:
            raise ValueError(f"unknown scope_target: {self.scope_target!r}")
        if self.intensity_hint not in INTENSITY_HINTS:
            raise ValueError(f"unknown intensity_hint: {self.intensity_hint!r}")
        if self.depth > MAX_DEPTH:
            raise ValueError(f"depth {self.depth} exceeds hard cap {MAX_DEPTH}")
        # cue_count is derived, never hand-maintained.
        self.cue_count = len(self.cue_char_spans)

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-safe dict (tuples become lists)."""
        d = asdict(self)
        d["cue_char_spans"] = [list(s) for s in self.cue_char_spans]
        return d
