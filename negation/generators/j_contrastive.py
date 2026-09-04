"""J_contrastive -- contrastive/oppositional framing of a copular complement.

*anything but*, *far from* and *the opposite of* negate by asserting the
complement's opposite rather than denying the predication.  They are all
copular-complement operators, so the generator requires a genuine ``acomp`` or
``attr`` under a copula and splices in front of that complement's subtree.

*the opposite of* is restricted to nominal complements: *"is the opposite of
impossible"* is at best marked, while *"is the opposite of a solution"* is not.
"""

from __future__ import annotations

from spacy.tokens import Doc, Token

from ..base import Generator, register
from ..frames import COPULA, detect_frame
from ..lexicons import CONTRASTIVE_FRAMES
from ..nlp_core import clause_scope, root_of, subtree_char_span
from ..polarity import explicit_negation_in, lexical_negatives_in
from ..schema import FAM_CONTRASTIVE, NegationVariant, SCOPE_PREDICATE
from ..splice import Edit

#: Frames limited to nominal complements.
_NOMINAL_ONLY = frozenset({"the_opposite_of"})


def _complement(root: Token) -> Token | None:
    for child in root.children:
        if child.dep_ in ("acomp", "attr"):
            return child
    return None


@register
class ContrastiveFrame(Generator):
    """Prefix a copular complement with a contrastive operator.

    *"The task is impossible."* -> *"The task is anything but impossible."*
    """

    family = FAM_CONTRASTIVE
    name = "contrastive_frame_v1"

    def applies(self, doc: Doc) -> bool:
        root = root_of(doc)
        if root is None:
            return False
        frame = detect_frame(root)
        if frame is None or frame.subtype != COPULA:
            return False
        if _complement(root) is None:
            return False
        scope = clause_scope(root)
        return not explicit_negation_in(doc, scope) and not lexical_negatives_in(doc, scope)

    def generate(self, doc: Doc) -> list[NegationVariant]:
        root = root_of(doc)
        if root is None:
            return []
        comp = _complement(root)
        if comp is None:
            return []
        start, _ = subtree_char_span(comp)
        nominal = comp.pos_ in ("NOUN", "PROPN", "PRON")
        out: list[NegationVariant] = []
        for phrase, subtype in CONTRASTIVE_FRAMES:
            if subtype in _NOMINAL_ONLY and not nominal:
                continue
            edit = Edit.insert(start, phrase + " ", cue=True)
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
