"""The K1-vs-K2 decision: scope containment, never cue counting.

Two negation cues in one sentence mean opposite things depending on whether the
second falls *inside* the clause the first commands:

* contained  -> the inner negation is itself negated, the two cancel, and the
  sentence is (roughly) affirmative: *"is not impossible"*, *"never fails to
  sort"*.  ``net_negation = 0``.
* separate   -> two independent negations of two different predications, and
  the sentence is doubly negative: *"does not sort the array and does not print
  the result"*.  ``net_negation = 2``.

Counting cues cannot tell these apart -- both have exactly two -- which is why
every K record in this pipeline is classified by :func:`classify_double` and
nothing else.  The containment test uses
:func:`negation.nlp_core.clause_scope`, which descends the dependency tree but
stops at clause-boundary edges (``conj``, ``advcl``, ``ccomp``, ...); a plain
``subtree`` test would wrongly report a coordinated second clause as contained.
"""

from __future__ import annotations

from spacy.tokens import Token

from .nlp_core import clause_scope
from .schema import FAM_CANCELLATION, FAM_COMPOUND


def classify_double(outer_anchor: Token, inner_anchor: Token) -> str:
    """Family for a two-cue variant: K1 if contained, K2 if not.

    ``outer_anchor`` is the clause head the first (clausal) negator attaches to;
    ``inner_anchor`` is the token bearing the second cue.
    """
    if inner_anchor.i in clause_scope(outer_anchor):
        return FAM_CANCELLATION
    return FAM_COMPOUND


def net_negation_for(family: str) -> int:
    """``0`` for a cancellation, ``2`` for a compound negation."""
    if family == FAM_CANCELLATION:
        return 0
    if family == FAM_COMPOUND:
        return 2
    raise ValueError(f"not a double-negation family: {family!r}")  # pragma: no cover
