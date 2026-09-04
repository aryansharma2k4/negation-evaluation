"""G_prepositional -- privative prepositions.

Two rules, both anchored on real dependency structure rather than string search:

``comitative_swap``  an existing ``with`` prepositional phrase flips to its
                     privative counterpart (*"runs with a cache"* -> *"runs
                     without a cache"* / *"in the absence of a cache"*).
``possession_swap``  a possession predicate is recast as a privative copular
                     one (*"has redundancy"* -> *"is without redundancy"*,
                     *"is devoid of redundancy"*, *"is free of redundancy"*).

The second rule needs a copula it must build itself, so the *be* form is
agreed with the subject and the original tense (see :func:`frames.be_form_for`).
"""

from __future__ import annotations

from spacy.tokens import Doc, Token

from ..base import Generator, register, scope_target_for
from ..frames import DO_SUPPORT, be_form_for, detect_frame
from ..lexicons import POSSESSION_VERBS
from ..nlp_core import clause_scope, root_of
from ..polarity import explicit_negation_in
from ..schema import FAM_PREPOSITIONAL, NegationVariant, SCOPE_PREDICATE
from ..splice import Edit

#: Privatives that can stand in for a comitative *with*.
_WITH_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("without", "without"),
    ("in the absence of", "in_the_absence_of"),
)

#: Privatives that can head the complement of a rebuilt copula.
_COPULAR_PRIVATIVES: tuple[tuple[str, str], ...] = (
    ("without", "without"),
    ("devoid of", "devoid_of"),
    ("free of", "free_of"),
)


def _with_preps(doc: Doc) -> list[Token]:
    return [t for t in doc if t.dep_ == "prep" and t.lower_ == "with"]


def _possession_object(root: Token) -> Token | None:
    if root.lemma_ not in POSSESSION_VERBS:
        return None
    for child in root.children:
        if child.dep_ in ("dobj", "attr"):
            return child
    return None


@register
class ComitativeToPrivative(Generator):
    """Flip a ``with`` PP to a privative one, leaving its object in place."""

    family = FAM_PREPOSITIONAL
    name = "prep_without_v1"

    def applies(self, doc: Doc) -> bool:
        return bool(_with_preps(doc)) and not explicit_negation_in(doc)

    def generate(self, doc: Doc) -> list[NegationVariant]:
        out: list[NegationVariant] = []
        for prep in _with_preps(doc):
            for replacement, subtype in _WITH_REPLACEMENTS:
                edit = Edit.replace(
                    prep.idx, prep.idx + len(prep.text), replacement, cue=True
                )
                record = self.build(
                    doc,
                    [edit],
                    subtype=subtype,
                    net_negation=1,
                    scope_target=scope_target_for(prep),
                )
                if record:
                    out.append(record)
        return out


@register
class PossessionToPrivative(Generator):
    """Recast a possession predicate as a privative copular predicate.

    *"The system has redundancy."* -> *"The system is without redundancy."*
    """

    family = FAM_PREPOSITIONAL
    name = "prep_privative_copula_v1"

    def applies(self, doc: Doc) -> bool:
        root = root_of(doc)
        if root is None or _possession_object(root) is None:
            return False
        frame = detect_frame(root)
        if frame is None or frame.subtype != DO_SUPPORT:
            return False
        return not explicit_negation_in(doc, clause_scope(root))

    def generate(self, doc: Doc) -> list[NegationVariant]:
        root = root_of(doc)
        if root is None:
            return []
        frame = detect_frame(root)
        if frame is None:
            return []
        be = be_form_for(frame)
        out: list[NegationVariant] = []
        for privative, subtype in _COPULAR_PRIVATIVES:
            pieces = [(be, False), (" ", False), (privative, True)]
            edit = Edit.compound(root.idx, root.idx + len(root.text), pieces)
            record = self.build(
                doc,
                [edit],
                subtype=subtype,
                net_negation=1,
                scope_target=SCOPE_PREDICATE,
            )
            if record:
                out.append(record)
        return out
