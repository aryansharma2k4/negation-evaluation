"""L_scope_position -- the same two negations, in the two scope orders.

With a quantified subject, where the negator lands changes what the sentence
means, not just how it is phrased:

    Not everyone passed the exam.     (NEG > FORALL: some did, some did not)
    Everyone did not pass the exam.   (FORALL > NEG: nobody passed)

Both are emitted, distinguished by ``subtype`` and by ``scope_target``
(``subject`` vs ``predicate``).  This pair is the whole reason the schema
carries ``scope_target``: a similarity model that treats them as paraphrases is
exactly what the project is measuring.
"""

from __future__ import annotations

from typing import Optional

from spacy.tokens import Doc, Token

from ..base import Generator, register
from ..frames import detect_frame, negator_edits
from ..lexicons import SCOPE_BEARING_QUANTIFIERS
from ..nlp_core import clause_scope, root_of, subject_of
from ..polarity import explicit_negation_in, lexical_negatives_in
from ..schema import (
    FAM_SCOPE_POSITION,
    NegationVariant,
    SCOPE_PREDICATE,
    SCOPE_SUBJECT,
)
from ..splice import Edit, lower_initial


def quantified_subject(doc: Doc) -> Optional[Token]:
    """The root's subject, if it is quantified by a scope-bearing quantifier."""
    root = root_of(doc)
    if root is None:
        return None
    subject = subject_of(root)
    if subject is None:
        return None
    if subject.lower_ in SCOPE_BEARING_QUANTIFIERS:
        return subject
    for child in subject.children:
        if child.dep_ in ("det", "predet") and child.lower_ in SCOPE_BEARING_QUANTIFIERS:
            return subject
    return None


@register
class ScopePositionPair(Generator):
    """Emit the subject-scope and predicate-scope readings as separate records."""

    family = FAM_SCOPE_POSITION
    name = "scope_position_v1"

    def applies(self, doc: Doc) -> bool:
        root = root_of(doc)
        if root is None or quantified_subject(doc) is None:
            return False
        if detect_frame(root) is None:
            return False
        scope = clause_scope(root)
        return not explicit_negation_in(doc, scope) and not lexical_negatives_in(doc, scope)

    def generate(self, doc: Doc) -> list[NegationVariant]:
        root = root_of(doc)
        subject = quantified_subject(doc)
        if root is None or subject is None:
            return []
        frame = detect_frame(root)
        if frame is None:
            return []

        out: list[NegationVariant] = []

        # NEG > QUANT: the negator is preposed onto the quantified subject.
        first = min(subject.subtree, key=lambda t: t.i)
        subject_edit = Edit.compound(
            first.idx,
            first.idx + len(first.text),
            [("not ", True), (lower_initial(first.text), False)],
        )
        record = self.build(
            doc,
            [subject_edit],
            subtype="subject_scope",
            net_negation=1,
            scope_target=SCOPE_SUBJECT,
        )
        if record:
            out.append(record)

        # QUANT > NEG: ordinary clausal negation, quantifier left alone.
        edits = negator_edits(frame, "not")
        record = self.build(
            doc,
            edits,
            subtype="predicate_scope",
            net_negation=1,
            scope_target=SCOPE_PREDICATE,
        )
        if record:
            out.append(record)
        return out
