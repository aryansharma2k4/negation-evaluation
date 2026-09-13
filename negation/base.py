"""Generator abstract base class, registry, and record construction helpers."""

from __future__ import annotations

import abc
from typing import Optional, Type

from spacy.tokens import Doc, Token

from .classify import (
    CLAUSE_DECLARATIVE,
    CLAUSE_EXCLAMATIVE,
    CLAUSE_IMPERATIVE,
    CLAUSE_TYPES,
    InputProfile,
    classify_input,
)
from .schema import (
    INT_NEUTRAL,
    MAX_DEPTH,
    MAX_OP_DEPTH,
    NegationVariant,
    OP_NEGATE,
    SCOPE_CLAUSE,
    SCOPE_PREDICATE,
    SCOPE_SUBJECT,
)
from .nlp_core import clause_index_at
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
    #: How many negation operations this generator stacks, counted from a fully
    #: affirmative baseline.
    depth: int = 1
    #: How many operations this generator applies *to its own input*.  Stage-1
    #: generators that compose two cues onto an affirmative declarative declare
    #: 2 here as well; every arbitrary-input generator declares 1, which is the
    #: depth-1 contract.
    op_depth: int = 1
    #: Which layer this generator belongs to.  ``1`` is the original
    #: affirmative-declarative battery, whose output is frozen by the
    #: regression test; ``2`` is the arbitrary-input layer.
    stage: int = 1
    #: The direction this generator runs in; see :data:`negation.schema.OPERATIONS`.
    operation: str = OP_NEGATE
    #: Clause types this generator is licensed for; the driver skips it on any
    #: other input without even calling ``applies``.
    #:
    #: The default excludes interrogatives on purpose.  Stage 1 was written for
    #: declaratives and imperatives, and its clausal families put the negator
    #: straight after the finite slot -- which in an inverted question is the
    #: fronted auxiliary, yielding ``*"Does not it compile?"``.  A question wants
    #: the negator after the *subject*, so interrogatives are handled by the
    #: generators that know that, and every other generator declines them.
    clause_types: frozenset[str] = frozenset(
        {CLAUSE_DECLARATIVE, CLAUSE_IMPERATIVE, CLAUSE_EXCLAMATIVE}
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.depth > MAX_DEPTH:
            raise ValueError(f"{cls.__name__}.depth exceeds hard cap {MAX_DEPTH}")
        unknown = set(cls.clause_types) - CLAUSE_TYPES
        if unknown:
            raise ValueError(f"{cls.__name__}.clause_types has unknown {unknown}")
        if cls.stage == 2 and cls.op_depth != MAX_OP_DEPTH:
            raise ValueError(
                f"{cls.__name__}.op_depth is {cls.op_depth}; every arbitrary-input "
                f"generator must apply exactly {MAX_OP_DEPTH} operation to its input"
            )

    def licensed_for(self, doc: Doc) -> bool:
        """Whether this generator may be offered ``doc`` at all.

        Checked by the driver before ``applies``, so a generator's ``applies``
        never has to re-state which clause types it was written for.
        """
        return classify_input(doc).clause_type in self.clause_types

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
        operation: Optional[str] = None,
        target_clause_idx: Optional[int] = None,
        modality: Optional[str] = None,
        generator: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> Optional[NegationVariant]:
        """Splice ``edits`` into ``doc``'s text and wrap the result in a record.

        ``cue_tokens`` and ``cue_char_spans`` are read straight off the splice,
        so they are always in output order and always agree with each other; a
        generator marks cue pieces when it builds its edits and never maintains
        a second list by hand.

        The input-describing fields (``clause_type``, ``voice``,
        ``input_polarity``, ``input_cue_count``, ``modality``) are read off the
        cached :class:`~negation.classify.InputProfile` rather than passed in,
        so every generator reports them consistently and no generator can
        forget to.  ``modality`` may be overridden, because it is the one such
        field an edit can change: replacing *can* with *may not* turns an
        ability claim into a permission one, and the record describes the
        variant, not the input it came from.

        Returns ``None`` when the edits do not actually change anything, which
        keeps degenerate rules from emitting copies of the base sentence.
        """
        profile: InputProfile = classify_input(doc)
        if not edits:
            return None
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
            generator=generator or self.name,
            confidence=confidence,
            depth=depth if depth is not None else self.depth,
            operation=operation or self.operation,
            input_polarity=profile.polarity,
            input_cue_count=profile.existing_count,
            clause_type=profile.clause_type,
            voice=profile.voice,
            modality=modality or profile.modality,
            target_clause_idx=(
                target_clause_idx
                if target_clause_idx is not None
                else clause_index_at(doc, min(e.start for e in edits))
            ),
            op_depth=self.op_depth,
        )


def mark_cue_tokens(tokens: list[Token], edits: list[Edit]) -> list[Edit]:
    """Re-splice ``tokens`` as themselves, flagged as cues.

    A cue already in the input does not move, so nothing would record where it
    landed in the output.  Splicing it in as itself makes the splicer report it
    alongside the new material, with correct post-edit offsets and no manual
    arithmetic -- the trick :class:`~negation.generators.k_double.LexicalCancellation`
    uses, shared here because affirmation, rescoping and clause targeting all
    need it.

    A token an edit already covers has that edit's pieces flagged instead, so no
    second, overlapping edit is produced.
    """
    out = list(edits)
    for token in tokens:
        start, end = token.idx, token.idx + len(token.text)
        for position, edit in enumerate(out):
            if edit.start <= start < edit.end:
                out[position] = Edit(
                    edit.start, edit.end, tuple((t, True) for t, _ in edit.pieces)
                )
                break
        else:
            out.append(Edit.replace(start, end, token.text, cue=True))
    return out


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
