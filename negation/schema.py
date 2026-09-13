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
FAM_MODAL = "M_modal"

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
        FAM_MODAL,
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

# ---------------------------------------------------------------------------
# Operations.  Stage 1 only ever went affirmative-in / negated-out; with
# arbitrary input the same machinery has to run in three directions.
# ---------------------------------------------------------------------------

#: Affirmative input -> negated output, or a negated input gaining a further
#: cue (the K1/K2 case).
OP_NEGATE = "negate"
#: Negated input -> affirmative output: a cue is removed, not added.
OP_AFFIRM = "affirm"
#: Negated input -> same cue count, different scope
#: (*"Not everyone passed"* <-> *"Everyone did not pass"*).
OP_RESCOPE = "rescope"
OPERATIONS: frozenset[str] = frozenset({OP_NEGATE, OP_AFFIRM, OP_RESCOPE})

#: Hard cap on how many negation operations may be stacked in one variant,
#: measured from a fully affirmative baseline.  This is the stage-1 notion and
#: is what :attr:`NegationVariant.depth` reports.
MAX_DEPTH = 2

#: Hard cap on how many operations a single record may apply *to its own
#: input* -- the stage-2 notion, reported by :attr:`NegationVariant.op_depth`.
#: Every generator in the arbitrary-input layer is held to exactly this.
MAX_OP_DEPTH = 1


class DepthViolation(AssertionError):
    """A record applied more operations to its input than it declared.

    Raised rather than filtered: a generator that miscounts its own operations
    is a bug in the generator, and silently dropping the record would hide it.
    """


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
    #: Number of stacked negation operations counted from a fully affirmative
    #: baseline (1 = atomic, 2 = composed).  Never exceeds :data:`MAX_DEPTH`.
    depth: int = 1
    # -- input description, filled in from the InputProfile ------------------
    #: Which direction this record runs in: see :data:`OPERATIONS`.
    operation: str = OP_NEGATE
    #: Polarity of the *input* sentence, ``affirmative`` or ``negated``.
    input_polarity: str = "affirmative"
    #: How many cues the input already carried -- the ``N`` in the depth-1 rule.
    input_cue_count: int = 0
    #: Clause type of the input: declarative / interrogative / imperative /
    #: exclamative.
    clause_type: str = "declarative"
    #: Voice of the input: active / passive.
    voice: str = "active"
    #: Modality of the input clause: none / epistemic / deontic / ability.
    modality: str = "none"
    #: Index into the input's clause heads of the clause this record targets.
    target_clause_idx: int = 0
    #: Negation operations applied *relative to the input sentence*.  Distinct
    #: from :attr:`depth`, which counts from an affirmative baseline: negating
    #: *"The task is impossible"* is ``depth=2`` (two cues from affirmative) but
    #: ``op_depth=1`` (one operation applied to the sentence we were given).
    #: Every arbitrary-input generator declares ``op_depth=1``.
    op_depth: int = 1
    #: Hook for stage 2 (LLM verification).  ``None`` means "not yet checked".
    verified: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family: {self.family!r}")
        if self.scope_target not in SCOPE_TARGETS:
            raise ValueError(f"unknown scope_target: {self.scope_target!r}")
        if self.intensity_hint not in INTENSITY_HINTS:
            raise ValueError(f"unknown intensity_hint: {self.intensity_hint!r}")
        if self.operation not in OPERATIONS:
            raise ValueError(f"unknown operation: {self.operation!r}")
        if self.depth > MAX_DEPTH:
            raise ValueError(f"depth {self.depth} exceeds hard cap {MAX_DEPTH}")
        if self.op_depth < 1:
            raise ValueError(f"op_depth {self.op_depth} must be at least 1")
        # cue_count is derived, never hand-maintained.
        self.cue_count = len(self.cue_char_spans)

    @property
    def max_output_cues(self) -> int:
        """The most cues this record is allowed to carry: ``N + op_depth``.

        Checked against the *re-parsed* variant in
        :func:`negation.filters.check_operation_depth`, so both sides of the
        comparison use the same notion of a cue.
        """
        return self.input_cue_count + self.op_depth

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-safe dict (tuples become lists)."""
        d = asdict(self)
        d["cue_char_spans"] = [list(s) for s in self.cue_char_spans]
        return d
