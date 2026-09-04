"""Generator abstract base class, registry, and record construction helpers."""

from __future__ import annotations

import abc
from typing import Optional, Type

from spacy.tokens import Doc, Token

from .schema import (
    INT_NEUTRAL,
    MAX_DEPTH,
    NegationVariant,
    SCOPE_CLAUSE,
    SCOPE_PREDICATE,
    SCOPE_SUBJECT,
)
from .splice import Edit, apply_edits, normalize_spacing, recase_like

# The base sentence id travels on the Doc so that generators keep the
# ``applies(doc)`` / ``generate(doc)`` signatures the design calls for.
if not Doc.has_extension("base_id"):
    Doc.set_extension("base_id", default="")


class Generator(abc.ABC):
    """One linguistic rule for producing negated variants.

    Subclasses must be cheap in :meth:`applies` -- dependency-label and POS
    checks only.  The driver calls ``applies`` on every registered generator for
    every sentence, so any string building there is wasted work on the sentences
    that do not license the rule.
    """

    #: Family constant from :mod:`negation.schema`.
    family: str = ""
    #: Stable identifier written into every record's ``generator`` field.
    name: str = ""
    #: How many negation operations this generator stacks.
    depth: int = 1

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.depth > MAX_DEPTH:
            raise ValueError(f"{cls.__name__}.depth exceeds hard cap {MAX_DEPTH}")

    @abc.abstractmethod
    def applies(self, doc: Doc) -> bool:
        """Cheap structural precondition check. No string building."""

    @abc.abstractmethod
    def generate(self, doc: Doc) -> list[NegationVariant]:
        """Produce every variant this rule licenses for ``doc``."""

    # -- shared construction helper -------------------------------------
    def build(
        self,
        doc: Doc,
        edits: list[Edit],
        *,
        subtype: str,
        net_negation: int = 1,
        scope_target: str = SCOPE_PREDICATE,
        intensity_hint: str = INT_NEUTRAL,
        family: Optional[str] = None,
        depth: Optional[int] = None,
    ) -> Optional[NegationVariant]:
        """Splice ``edits`` into ``doc``'s text and wrap the result in a record.

        ``cue_tokens`` and ``cue_char_spans`` are read straight off the splice,
        so they are always in output order and always agree with each other; a
        generator marks cue pieces when it builds its edits and never maintains
        a second list by hand.

        Returns ``None`` when the edits do not actually change anything, which
        keeps degenerate rules from emitting copies of the base sentence.
        """
        text, cues = apply_edits(doc.text, edits)
        text, cues = normalize_spacing(text, cues)
        text = recase_like(doc.text, text)
        if text.strip() == doc.text.strip():
            return None
        spans = [(c[1], c[2]) for c in cues]
        # Read the tokens back out of the finished string so they reflect any
        # recasing the splice applied ("no" -> "No" at sentence start).
        cue_tokens = [text[start:end] for start, end in spans]
        return NegationVariant(
            base_id=doc._.base_id,
            base_sentence=doc.text,
            variant=text,
            family=family or self.family,
            subtype=subtype,
            cue_tokens=cue_tokens,
            cue_char_spans=spans,
            cue_count=len(spans),
            net_negation=net_negation,
            scope_target=scope_target,
            intensity_hint=intensity_hint,
            generator=self.name,
            depth=depth if depth is not None else self.depth,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: list[Generator] = []


def register(cls: Type[Generator]) -> Type[Generator]:
    """Class decorator: instantiate ``cls`` once and add it to the registry."""
    if not cls.family or not cls.name:
        raise ValueError(f"{cls.__name__} must declare both `family` and `name`")
    _REGISTRY.append(cls())
    return cls


def registry() -> tuple[Generator, ...]:
    """All registered generators, in registration order."""
    return tuple(_REGISTRY)


def clear_registry() -> None:  # pragma: no cover - test utility
    _REGISTRY.clear()


# ---------------------------------------------------------------------------
# Scope labelling
# ---------------------------------------------------------------------------


def scope_target_for(token: Token) -> str:
    """Label whether ``token`` sits in the subject, the predicate, or the clause.

    Walking up the head chain is enough: anything dominated by an ``nsubj``
    (or passive/clausal subject) is subject-internal; a clause head itself
    scopes over the whole clause.
    """
    cur = token
    seen = {cur.i}
    while True:
        if cur.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass"):
            return SCOPE_SUBJECT
        if cur.dep_ == "ROOT" or cur.head.i == cur.i:
            return SCOPE_CLAUSE if cur.i == token.i else SCOPE_PREDICATE
        cur = cur.head
        if cur.i in seen:  # pragma: no cover - malformed parse guard
            return SCOPE_PREDICATE
        seen.add(cur.i)
