"""The one record every corpus is normalised to.

Corpora disagree about almost everything -- CD-SCO is CoNLL columns, SFU and
BioScope are two different XML vocabularies, the NLI sets are sentence pairs
with no spans at all -- so a loader's job is to answer the same questions about
each: what was the sentence, which tokens were the cue, which tokens were the
scope, and which of our families would have produced that cue.

The unit is one **annotation instance**, not one sentence.  A sentence with
three negations in CD-SCO becomes three records sharing a ``sentence_id``, which
is also the unit a cue/scope detector trains on.  Grouping back to sentences is
what ``sentence_id`` is for, and the multi-cue statistics depend on it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: What the annotation actually marks.  Kept distinct because the whole reason
#: SFU and BioScope are interesting is that they annotate speculation
#: *separately* from negation, and collapsing the two would destroy exactly the
#: signal the intensity regressor needs.
ANN_NEGATION = "negation"
ANN_SPECULATION = "speculation"
ANNOTATION_TYPES: frozenset[str] = frozenset({ANN_NEGATION, ANN_SPECULATION})

#: Gold spans came from the corpus; pair-level corpora have none at all.
PROV_GOLD = "gold"
PROV_PAIR_ONLY = "pair-level-only"
PROVENANCES: frozenset[str] = frozenset({PROV_GOLD, PROV_PAIR_ONLY})

#: Used when a cue has no counterpart in our 13-family taxonomy.
UNMAPPED = "unmapped"


@dataclass
class AnnotatedSentence:
    """One negation (or speculation) annotation, normalised across corpora."""

    sentence: str
    tokens: list[str]
    #: Token-index half-open spans.  A list because cues and scopes are both
    #: routinely discontinuous: CD-SCO's *neither ... nor* is one cue in two
    #: pieces, and a scope interrupted by its own cue is two pieces.
    cue_spans: list[tuple[int, int]]
    scope_spans: list[tuple[int, int]]
    #: One of the 13 families from :mod:`negation.schema`, or ``"unmapped"``.
    cue_family: str
    source_dataset: str
    split: str

    #: Groups the annotations that share a sentence.  Required for the
    #: multi-cue and K1/K2 statistics.
    sentence_id: str = ""
    #: ``negation`` or ``speculation``.
    annotation_type: str = ANN_NEGATION
    #: Surface string of the cue as the corpus recorded it.  For an affixal cue
    #: this is the affix (*in*), not the token it sits on (*infrequent*).
    cue_text: str = ""
    #: Whether the spans are the corpus's own or simply absent.
    provenance: str = PROV_GOLD
    #: Loader-specific extras that do not belong in the common schema -- the
    #: NLI pair's premise and gold label, BioScope's subcorpus, and so on.
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.annotation_type not in ANNOTATION_TYPES:
            raise ValueError(f"unknown annotation_type: {self.annotation_type!r}")
        if self.provenance not in PROVENANCES:
            raise ValueError(f"unknown provenance: {self.provenance!r}")
        for name in ("cue_spans", "scope_spans"):
            for start, end in getattr(self, name):
                if not 0 <= start < end <= len(self.tokens):
                    raise ValueError(
                        f"{name} span ({start}, {end}) out of range for "
                        f"{len(self.tokens)} tokens in {self.sentence_id!r}"
                    )

    @property
    def cue_token_indices(self) -> list[int]:
        return [i for start, end in self.cue_spans for i in range(start, end)]

    @property
    def scope_token_indices(self) -> list[int]:
        return [i for start, end in self.scope_spans for i in range(start, end)]

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cue_spans"] = [list(s) for s in self.cue_spans]
        d["scope_spans"] = [list(s) for s in self.scope_spans]
        return d
