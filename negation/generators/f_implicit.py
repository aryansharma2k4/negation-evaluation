"""F_implicit -- negation carried by a verbal trigger rather than a negator.

These are the NegEx-style implicit negations: the polarity lives in the matrix
verb (*fails to*, *refuses to*, *is unable to*, *lacks*) and the original
predicate is demoted into its complement.  Because the trigger takes over the
finite slot, the original verb loses its tense and agreement -- the same
morphological bookkeeping do-support needs, run in reverse: the trigger is
inflected to the features the lexical verb was carrying, and the lexical verb
drops to its lemma (as a bare infinitive after *to*, or a gerund after *deny*).

Only bare finite lexical verbs are restructured.  Clauses that already have an
auxiliary would need the trigger to be threaded through the auxiliary chain,
which this generator does not attempt rather than risk a malformed record.
"""

from __future__ import annotations

from spacy.tokens import Doc, Token

from ..base import Generator, register
from ..frames import DO_SUPPORT, agreeing_finite, be_form_for, detect_frame
from ..lexicons import IMPLICIT_TRIGGERS, LACK_TRIGGER, POSSESSION_VERBS
from ..nlp_core import clause_scope, present_participle, root_of
from ..polarity import explicit_negation_in, lexical_negatives_in
from ..schema import FAM_IMPLICIT, NegationVariant, SCOPE_CLAUSE, SCOPE_PREDICATE
from ..splice import Edit


def _has_object(verb: Token) -> bool:
    return any(c.dep_ in ("dobj", "dative", "attr", "oprd") for c in verb.children)


@register
class ImplicitTriggerRewrite(Generator):
    """Rebuild the predicate under a negative-implicative matrix verb.

    *"The function sorts the array."*
      -> *"The function fails to sort the array."*
      -> *"The function is unable to sort the array."*
    *"The system has redundancy."* -> *"The system lacks redundancy."*
    """

    family = FAM_IMPLICIT
    name = "implicit_trigger_v1"

    def applies(self, doc: Doc) -> bool:
        root = root_of(doc)
        if root is None:
            return False
        frame = detect_frame(root)
        if frame is None or frame.subtype != DO_SUPPORT:
            return False
        scope = clause_scope(root)
        return not explicit_negation_in(doc, scope) and not lexical_negatives_in(doc, scope)

    def generate(self, doc: Doc) -> list[NegationVariant]:
        root = root_of(doc)
        if root is None:
            return []
        frame = detect_frame(root)
        if frame is None or frame.subtype != DO_SUPPORT:
            return []

        start, end = root.idx, root.idx + len(root.text)
        out: list[NegationVariant] = []

        for trigger in IMPLICIT_TRIGGERS:
            if trigger.lemma == root.lemma_:
                continue  # "fails to fail"
            if trigger.frame == "to_inf":
                pieces = [
                    (agreeing_finite(trigger.lemma, frame), True),
                    (f" to {root.lemma_}", False),
                ]
            elif trigger.frame == "copular":
                # The copula has to agree with a subject; an imperative has
                # none, and "*Is unable to sort the array" is the result.
                if frame.subject is None:
                    continue
                pieces = [
                    (be_form_for(frame), False),
                    (" ", False),
                    (trigger.predicate, True),
                    (f" to {root.lemma_}", False),
                ]
            elif trigger.frame == "gerund":
                pieces = [
                    (agreeing_finite(trigger.lemma, frame), True),
                    (f" {present_participle(root.lemma_)}", False),
                ]
            else:  # pragma: no cover - "np" is handled by LackRewrite
                continue
            record = self.build(
                doc,
                [Edit.compound(start, end, pieces)],
                subtype=trigger.subtype,
                net_negation=1,
                scope_target=SCOPE_CLAUSE,
            )
            if record:
                out.append(record)
        return out


@register
class LackRewrite(Generator):
    """Possession predicate -> *lacks*.

    *have/possess/contain/include* + object is negated lexically rather than
    syntactically: *"The system has redundancy."* -> *"The system lacks
    redundancy."*  The object NP is untouched, which is why this trigger needs
    its own generator -- it is the one implicit trigger that takes an NP
    complement instead of a clause.
    """

    family = FAM_IMPLICIT
    name = "implicit_lack_v1"

    def applies(self, doc: Doc) -> bool:
        root = root_of(doc)
        if root is None or root.lemma_ not in POSSESSION_VERBS:
            return False
        if not _has_object(root):
            return False
        frame = detect_frame(root)
        if frame is None or frame.subtype != DO_SUPPORT:
            return False
        scope = clause_scope(root)
        return not explicit_negation_in(doc, scope) and not lexical_negatives_in(doc, scope)

    def generate(self, doc: Doc) -> list[NegationVariant]:
        root = root_of(doc)
        if root is None:
            return []
        frame = detect_frame(root)
        if frame is None:
            return []
        surface = agreeing_finite(LACK_TRIGGER.lemma, frame)
        edit = Edit.replace(root.idx, root.idx + len(root.text), surface, cue=True)
        record = self.build(
            doc,
            [edit],
            subtype=LACK_TRIGGER.subtype,
            net_negation=1,
            scope_target=SCOPE_PREDICATE,
        )
        return [record] if record else []
