"""Rescoping -- move a negation without adding or removing one.

``L_scope_position`` emits the two scope readings of a quantified clause as
separate records when the *input* is affirmative:

    Not everyone passed the exam.     (NEG > FORALL: some did, some did not)
    Everyone did not pass the exam.   (FORALL > NEG: nobody passed)

When the input is already one of those, neither operation stage 1 knows about
applies.  Negating again would stack a second cue, and affirming would throw the
negation away; what is wanted is the *other* reading of the negation already
there.  That is the third operation: same cue count in, same cue count out, a
different ``scope_target``.

Both directions are implemented, and each is a single operation even though the
edit has two halves -- the negator is removed from one position and realised in
the other, which is one relocation, not a negation plus an affirmation.  The
removal half is :func:`negation.generators.affirm.clausal_negator_removal`, so
the do-support that a predicate-scope negator was propping up is collapsed
properly on the way out: *"Everyone did not pass"* rescopes to *"Not everyone
passed"*, never to *"Not everyone did pass"*.

Only quantified subjects license this.  Without a scope-bearing quantifier the
two positions are paraphrases rather than distinct readings, and relocating the
negator would be a null edit dressed up as a record.
"""

from __future__ import annotations

from typing import Optional

from spacy.tokens import Doc, Token

from ..base import Generator, register
from ..classify import InputProfile, classify_input
from ..frames import detect_frame, negator_edits
from ..nlp_core import root_of
from ..schema import (
    FAM_SCOPE_POSITION,
    NegationVariant,
    OP_RESCOPE,
    SCOPE_PREDICATE,
    SCOPE_SUBJECT,
)
from ..splice import Edit, lower_initial
from .affirm import clausal_negator_removal
from .l_scope_position import quantified_subject


class Rescope(Generator):
    """Base class: relocate the input's single negator, cue count unchanged."""

    family = FAM_SCOPE_POSITION
    stage = 2
    op_depth = 1
    operation = OP_RESCOPE

    def profile(self, doc: Doc) -> InputProfile:
        return classify_input(doc)

    def _licensed(self, doc: Doc) -> bool:
        """One negation, one quantified subject, one clause to move it within."""
        profile = self.profile(doc)
        if not profile.is_negated or profile.existing_count != 1:
            return False
        if profile.clause_count != 1:
            return False
        return quantified_subject(doc) is not None


def _preposed_negator(subject: Token) -> Optional[Token]:
    """The negator sitting in front of ``subject``, if there is one.

    Scanned over the whole subject subtree rather than its immediate children:
    with a predeterminer the parser hangs *Not* off that instead of off the noun
    (*"__Not__ all the students"* -> ``neg`` on *all*), so a children-only check
    sees the quantified subject but misses its negation.
    """
    for token in subject.subtree:
        if token.dep_ == "neg" and token.i < subject.i:
            return token
    return None


@register
class SubjectToPredicateScope(Rescope):
    """NEG > QUANT becomes QUANT > NEG.

    *"Not everyone passed the exam."* -> *"Everyone did not pass the exam."*

    The two are not paraphrases -- the first says some passed and some did not,
    the second says none passed -- which is exactly why both are worth having as
    records with the same ``base_id``.
    """

    name = "rescope_subject_to_predicate_v1"

    def _target(self, doc: Doc) -> Optional[tuple[Token, Token]]:
        if not self._licensed(doc):
            return None
        subject = quantified_subject(doc)
        root = root_of(doc)
        if subject is None or root is None:
            return None
        neg = _preposed_negator(subject)
        if neg is None or detect_frame(root) is None:
            return None
        return subject, neg

    def applies(self, doc: Doc) -> bool:
        return self._target(doc) is not None

    def generate(self, doc: Doc) -> list[NegationVariant]:
        target = self._target(doc)
        if target is None:
            return []
        subject, neg = target
        root = root_of(doc)
        frame = detect_frame(root) if root is not None else None
        if frame is None:
            return []
        edits = [Edit.replace(neg.idx, neg.idx + len(neg.text), "")]
        edits.extend(negator_edits(frame, "not"))
        record = self.build(
            doc,
            edits,
            subtype="subject_to_predicate_scope",
            net_negation=1,
            scope_target=SCOPE_PREDICATE,
        )
        return [record] if record else []


@register
class PredicateToSubjectScope(Rescope):
    """QUANT > NEG becomes NEG > QUANT.

    *"Everyone did not pass the exam."* -> *"Not everyone passed the exam."*

    Removing the clausal negator also collapses the do-support it was propping
    up, so the promoted verb gets its tense back rather than stranding *did*.
    """

    name = "rescope_predicate_to_subject_v1"

    def _target(self, doc: Doc) -> Optional[tuple[Token, Token]]:
        if not self._licensed(doc):
            return None
        subject = quantified_subject(doc)
        root = root_of(doc)
        if subject is None or root is None:
            return None
        neg = _preposed_negator(subject)
        if neg is not None:
            return None  # already subject-scoped; the other direction's input
        for child in root.children:
            if child.dep_ == "neg" and child.i > subject.i:
                return subject, child
        return None

    def applies(self, doc: Doc) -> bool:
        return self._target(doc) is not None

    def generate(self, doc: Doc) -> list[NegationVariant]:
        target = self._target(doc)
        if target is None:
            return []
        subject, neg = target
        removal = clausal_negator_removal(neg)
        if removal is None:
            return []
        first = min(subject.subtree, key=lambda t: t.i)
        edits = [
            Edit.compound(
                first.idx,
                first.idx + len(first.text),
                [("not ", True), (lower_initial(first.text), False)],
            ),
            *removal,
        ]
        record = self.build(
            doc,
            edits,
            subtype="predicate_to_subject_scope",
            net_negation=1,
            scope_target=SCOPE_SUBJECT,
        )
        return [record] if record else []
