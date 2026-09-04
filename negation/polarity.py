"""Polarity inspection: what negative material a clause already carries.

Two distinctions the rest of the pipeline depends on:

*explicit* negation
    An overt clausal cue (*not*, *never*, *no*, *cannot*, ...).  A clause that
    already has one is not a valid input for families A or C -- adding a second
    *not* to the same slot produces ``*"will not not fail"``.

*lexical* negation
    A content word with inherent negative polarity (*impossible*, *unlikely*,
    *fails*).  A clause containing one is still a valid input for clausal
    negation, but the result is a **cancellation**, not a plain negation, so
    families A and C stand aside and K1 emits the record with
    ``net_negation = 0``.
"""

from __future__ import annotations

from typing import Iterable

from spacy.tokens import Doc, Token

from .lexicons import LEXICALLY_NEGATIVE

#: Overt negation cues, matched on lowercase surface form.
EXPLICIT_CUES: frozenset[str] = frozenset(
    {
        "not", "n't", "never", "no", "none", "nobody", "no-one", "noone",
        "nothing", "nowhere", "neither", "nor", "cannot", "cant", "without",
        "hardly", "barely", "rarely", "seldom", "scarcely",
    }
)


def _tokens_in(doc: Doc, scope: Iterable[int] | None) -> list[Token]:
    if scope is None:
        return list(doc)
    return [doc[i] for i in sorted(scope)]


def explicit_negation_in(doc: Doc, scope: Iterable[int] | None = None) -> list[Token]:
    """Overt negation cues inside ``scope`` (whole doc when ``scope`` is None)."""
    return [
        t
        for t in _tokens_in(doc, scope)
        if t.lower_ in EXPLICIT_CUES or t.dep_ == "neg"
    ]


def lexical_negatives_in(doc: Doc, scope: Iterable[int] | None = None) -> list[Token]:
    """Inherently negative content words inside ``scope``."""
    return [
        t
        for t in _tokens_in(doc, scope)
        if t.lower_ in LEXICALLY_NEGATIVE or t.lemma_.lower() in LEXICALLY_NEGATIVE
    ]


def is_affirmative(doc: Doc, scope: Iterable[int] | None = None) -> bool:
    """True when ``scope`` carries neither an overt cue nor a negative content word."""
    return not explicit_negation_in(doc, scope) and not lexical_negatives_in(doc, scope)
