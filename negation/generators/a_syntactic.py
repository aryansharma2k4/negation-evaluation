"""A_syntactic -- clausal negation in the finite-verb complex.

This is the foundation family: it realises the plain negator *not* in whichever
of the three English slots the clause makes available (copula, existing
auxiliary, or a do-support auxiliary it has to introduce).  Every composed
family downstream (H, I, K, L) reuses the same frames.
"""

from __future__ import annotations

from spacy.tokens import Doc

from ..base import Generator, register
from ..frames import COPULA, DO_SUPPORT, EXISTING_AUX, detect_frame, negator_edits
from ..nlp_core import clause_scope, root_of
from ..polarity import explicit_negation_in, lexical_negatives_in
from ..schema import FAM_SYNTACTIC, NegationVariant, SCOPE_CLAUSE


class _SyntacticBase(Generator):
    family = FAM_SYNTACTIC
    #: Frame subtype this generator handles.
    frame_subtype = ""

    def applies(self, doc: Doc) -> bool:
        root = root_of(doc)
        if root is None:
            return False
        frame = detect_frame(root)
        if frame is None or frame.subtype != self.frame_subtype:
            return False
        scope = clause_scope(root)
        # An already-negated clause has no free negator slot, and a clause with
        # a negative content word yields a cancellation -- K1's job, not ours.
        return not explicit_negation_in(doc, scope) and not lexical_negatives_in(doc, scope)

    def generate(self, doc: Doc) -> list[NegationVariant]:
        root = root_of(doc)
        if root is None:
            return []
        frame = detect_frame(root)
        if frame is None or frame.subtype != self.frame_subtype:
            return []
        edits = negator_edits(frame, "not")
        record = self.build(
            doc,
            edits,
            subtype=self.frame_subtype,
            net_negation=1,
            scope_target=SCOPE_CLAUSE,
        )
        return [record] if record else []


@register
class CopulaNegation(_SyntacticBase):
    """Copular clause: insert *not* immediately after the finite *be*.

    *"The task is impossible."* -> *"The task is not impossible."*
    """

    name = "syn_copula_v1"
    frame_subtype = COPULA


@register
class ExistingAuxNegation(_SyntacticBase):
    """Clause already has an auxiliary: the negator attaches to it.

    *"She has finished."* -> *"She has not finished."*;
    *"The model can handle noise."* -> *"The model cannot handle noise."*
    (*can* is the one auxiliary whose negated form is lexicalised rather than
    written as two words.)
    """

    name = "syn_existing_aux_v1"
    frame_subtype = EXISTING_AUX


@register
class DoSupportNegation(_SyntacticBase):
    """Bare finite lexical verb: introduce *do* to carry the negator.

    English will not let *not* attach to a lexical verb, so the verb's tense,
    person and number features are moved onto a periphrastic *do* and the verb
    reverts to its lemma:

    *"The function sorts the array."* -> *"The function does not sort the array."*
    *"The function sorted the array."* -> *"The function did not sort the array."*
    *"Sort the array."*                -> *"Do not sort the array."*
    """

    name = "syn_do_support_v1"
    frame_subtype = DO_SUPPORT
