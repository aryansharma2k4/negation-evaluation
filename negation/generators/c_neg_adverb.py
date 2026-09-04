"""C_neg_adverb -- negation carried by a negative or near-negative adverb.

Two subtypes, and the difference matters for the graded-sensitivity work:

``absolute``  *never* -- a full negation of the event.
``partial``   *hardly / barely / rarely / seldom / scarcely* -- downward
              entailing but not flatly negative, so records are tagged
              ``intensity_hint="partial"``.

Unlike *not*, these adverbs attach directly to a lexical verb, so no do-support
is introduced: *"He rarely arrives early"*, not *"*He does rarely arrive early"*.
The adverb lands after the finite auxiliary when there is one and immediately
before the lexical verb otherwise.
"""

from __future__ import annotations

from spacy.tokens import Doc

from ..base import Generator, register
from ..frames import COPULA, DO_SUPPORT, EXISTING_AUX, detect_frame
from ..lexicons import NEGATIVE_ADVERBS
from ..nlp_core import clause_scope, root_of
from ..polarity import explicit_negation_in, lexical_negatives_in
from ..schema import FAM_NEG_ADVERB, NegationVariant, SCOPE_CLAUSE
from ..splice import Edit, lower_initial

#: Adverbs already present block the insertion of a competing one.
_BLOCKING_ADVERBS = frozenset(NEGATIVE_ADVERBS) | {"not", "no", "always", "n't"}


@register
class NegativeAdverbInsertion(Generator):
    """Insert a negative or approximative-negative adverb into the verb complex."""

    family = FAM_NEG_ADVERB
    name = "negadv_insert_v1"

    def applies(self, doc: Doc) -> bool:
        root = root_of(doc)
        if root is None or detect_frame(root) is None:
            return False
        if any(t.lower_ in _BLOCKING_ADVERBS for t in doc):
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

        if frame.subtype in (COPULA, EXISTING_AUX):
            slot = frame.finite_slot
            at = slot.idx + len(slot.text)
            make = lambda adv: [Edit.insert(at, " " + adv, cue=True)]  # noqa: E731
        elif frame.subtype == DO_SUPPORT:
            at = frame.head.idx
            first = doc[0]
            # In an imperative the verb is sentence-initial; once the adverb
            # sits in front of it, it is no longer the capitalised word.
            demote = (
                [Edit.replace(first.idx, first.idx + len(first.text), lower_initial(first.text))]
                if at == first.idx
                else []
            )
            make = lambda adv: [Edit.insert(at, adv + " ", cue=True), *demote]  # noqa: E731
        else:  # pragma: no cover - detect_frame returns nothing else
            return []

        out: list[NegationVariant] = []
        for adverb, (subtype, hint) in NEGATIVE_ADVERBS.items():
            record = self.build(
                doc,
                make(adverb),
                subtype=subtype,
                net_negation=1,
                scope_target=SCOPE_CLAUSE,
                intensity_hint=hint,
            )
            if record:
                out.append(record)
        return out
