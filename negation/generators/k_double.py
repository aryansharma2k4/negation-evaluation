"""K1_cancellation and K2_compound -- the two shapes of double negation.

Both families put two negation cues in one sentence.  What separates them is
**scope containment**, decided in :mod:`negation.scope` and nowhere else:

* K1: the second cue lies inside the clause the first negator commands, so the
  negations cancel (``net_negation = 0``).
* K2: the two cues sit in separate clauses, so both survive
  (``net_negation = 2``).

Three K1 routes are implemented -- negating a clause that already contains a
negative content word (*"is not impossible"*), composing a clausal negator with
a freshly derived affixal negative (*"is not unimportant"*), and putting a
negative adverb over an implicative trigger (*"never fails to sort"*) -- plus
two K2 routes: negating each of two coordinate clauses, and negating one clause
while a lexical negative sits in the other.

Everything here composes exactly two operations; the ``depth = 2`` class
attribute is enforced against the hard cap in :class:`negation.base.Generator`.
"""

from __future__ import annotations

from typing import Optional

from spacy.tokens import Doc, Token

from ..base import Generator, register
from ..frames import (
    DO_SUPPORT,
    ClauseFrame,
    agreeing_finite,
    detect_frame,
    do_support_insertion_point,
    negator_edits,
)
from ..lexicons import IMPLICIT_TRIGGERS
from ..nlp_core import clause_scope, clause_heads, root_of
from ..polarity import explicit_negation_in, lexical_negatives_in
from ..scope import classify_double, net_negation_for
from ..schema import (
    FAM_CANCELLATION,
    FAM_COMPOUND,
    NegationVariant,
    SCOPE_CLAUSE,
)
from ..splice import Edit
from .d_affixal import derived_forms, eligible as affix_eligible

#: Cap on affixal targets per sentence so a long sentence cannot combinatorially
#: explode the output.
MAX_AFFIX_TARGETS = 2


def _clause_frames(doc: Doc) -> list[ClauseFrame]:
    frames = []
    for head in clause_heads(doc):
        frame = detect_frame(head)
        if frame is not None:
            frames.append(frame)
    return frames


@register
class LexicalCancellation(Generator):
    """Clausal negator over a negative content word already in the clause.

    *"The task is impossible."* -> *"The task is not impossible."*
    *"The function fails."*     -> *"The function does not fail."*

    The record is K1 rather than A because the two negations cancel: the
    resulting sentence is closer in meaning to the base's positive counterpart
    than to a plain denial.
    """

    family = FAM_CANCELLATION
    name = "k1_lexical_cancel_v1"
    depth = 2

    def _target(self, doc: Doc) -> Optional[tuple[ClauseFrame, Token]]:
        root = root_of(doc)
        if root is None:
            return None
        frame = detect_frame(root)
        if frame is None:
            return None
        scope = clause_scope(root)
        if explicit_negation_in(doc, scope):
            return None
        negatives = lexical_negatives_in(doc, scope)
        return (frame, negatives[0]) if negatives else None

    def applies(self, doc: Doc) -> bool:
        return self._target(doc) is not None

    def generate(self, doc: Doc) -> list[NegationVariant]:
        target = self._target(doc)
        if target is None:
            return []
        frame, negative = target
        family = classify_double(frame.anchor, negative)
        # The negative word is already in the base text, so it is re-spliced as
        # itself and flagged as a cue: the splicer then reports both cues with
        # correct post-edit offsets, with no manual offset arithmetic.  When the
        # negative word *is* the verb do-support is rewriting ("failed" ->
        # "did not fail"), that edit already covers the span, so its pieces are
        # flagged in place instead of adding a second, overlapping edit.
        edits = negator_edits(frame, "not")
        covered = any(e.start <= negative.idx < e.end for e in edits)
        if covered:
            edits = [
                Edit(e.start, e.end, tuple((text, True) for text, _ in e.pieces))
                if e.start <= negative.idx < e.end
                else e
                for e in edits
            ]
        else:
            edits.append(
                Edit.replace(
                    negative.idx,
                    negative.idx + len(negative.text),
                    negative.text,
                    cue=True,
                )
            )
        record = self.build(
            doc,
            edits,
            subtype=f"cancel_{negative.lower_}",
            net_negation=net_negation_for(family),
            scope_target=SCOPE_CLAUSE,
            family=family,
        )
        return [record] if record else []


@register
class AffixalCancellation(Generator):
    """Clausal negator composed with a freshly derived affixal negative.

    *"The result is important."* -> *"The result is not unimportant."*

    The derived word is validated exactly as in D_affixal, and the family is
    decided by asking whether the derived word's token lies in the clause the
    negator commands -- for a single-clause sentence it always does, but the
    check is what makes the classification principled rather than positional.
    """

    family = FAM_CANCELLATION
    name = "k1_affixal_cancel_v1"
    depth = 2

    def _targets(self, doc: Doc) -> list[Token]:
        root = root_of(doc)
        if root is None:
            return []
        scope = clause_scope(root)
        return [doc[i] for i in sorted(scope) if affix_eligible(doc[i])]

    def applies(self, doc: Doc) -> bool:
        root = root_of(doc)
        if root is None or detect_frame(root) is None:
            return False
        scope = clause_scope(root)
        if explicit_negation_in(doc, scope) or lexical_negatives_in(doc, scope):
            return False
        return any(derived_forms(t) for t in self._targets(doc)[:MAX_AFFIX_TARGETS])

    def generate(self, doc: Doc) -> list[NegationVariant]:
        root = root_of(doc)
        if root is None:
            return []
        frame = detect_frame(root)
        if frame is None:
            return []
        neg_edits = negator_edits(frame, "not")
        out: list[NegationVariant] = []
        for token in self._targets(doc)[:MAX_AFFIX_TARGETS]:
            for form, subtype in derived_forms(token):
                family = classify_double(frame.anchor, token)
                edit = Edit.replace(
                    token.idx, token.idx + len(token.text), form, cue=True
                )
                record = self.build(
                    doc,
                    neg_edits + [edit],
                    subtype=f"not_{subtype}",
                    net_negation=net_negation_for(family),
                    scope_target=SCOPE_CLAUSE,
                    family=family,
                )
                if record:
                    out.append(record)
        return out


@register
class TriggerCancellation(Generator):
    """Negative adverb scoping over a negative-implicative trigger.

    *"The function sorts the array."* -> *"The function never fails to sort the
    array."*

    *never* commands the trigger verb, so the two negations cancel and the
    variant entails the base -- a K1 record with ``net_negation = 0``.
    """

    family = FAM_CANCELLATION
    name = "k1_trigger_cancel_v1"
    depth = 2

    #: Only the ``to_inf`` triggers read naturally under *never*.
    _TRIGGERS = tuple(t for t in IMPLICIT_TRIGGERS if t.frame == "to_inf")

    def applies(self, doc: Doc) -> bool:
        root = root_of(doc)
        if root is None:
            return False
        frame = detect_frame(root)
        if frame is None or frame.subtype != DO_SUPPORT:
            return False
        # Restructuring the matrix verb of a multi-clause sentence would leave
        # the other clause dangling under the new trigger, so single clauses only.
        if len(clause_heads(doc)) > 1:
            return False
        # A preverbal adverbial would end up between "never" and the trigger
        # ("*never always fails to"), so those clauses are left alone.
        if do_support_insertion_point(frame) != root.idx:
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
        at = root.idx
        out: list[NegationVariant] = []
        for trigger in self._TRIGGERS:
            if trigger.lemma == root.lemma_:
                continue
            surface = agreeing_finite(trigger.lemma, frame)
            # "never" is inserted at the finite slot; the trigger takes over the
            # verb's position, so "never" ends up commanding it.
            family = classify_double(frame.anchor, root)
            edits = [
                Edit.insert(at, "never ", cue=True),
                Edit.compound(
                    root.idx,
                    root.idx + len(root.text),
                    [(surface, True), (f" to {root.lemma_}", False)],
                ),
            ]
            record = self.build(
                doc,
                edits,
                subtype=f"never_{trigger.subtype}",
                net_negation=net_negation_for(family),
                scope_target=SCOPE_CLAUSE,
                family=family,
            )
            if record:
                out.append(record)
        return out


@register
class CompoundClauseNegation(Generator):
    """Negate two coordinate/subordinate clauses independently.

    *"The function sorts the array and prints the result."*
      -> *"The function does not sort the array and does not print the result."*

    The second negator's clause head is *not* inside the first's clause scope --
    it is reached across a ``conj`` edge -- so the cues do not cancel and the
    record is K2 with ``net_negation = 2``.  Note that a naive ``subtree`` test
    would call this containment, since the conjunct verb *is* in the root's
    subtree; only the clause-boundary-aware scope gets it right.
    """

    family = FAM_COMPOUND
    name = "k2_clause_pair_v1"
    depth = 2

    def _frames(self, doc: Doc) -> list[ClauseFrame]:
        frames = _clause_frames(doc)
        if len(frames) < 2:
            return []
        first, second = frames[0], frames[1]
        if classify_double(first.anchor, second.anchor) != FAM_COMPOUND:
            return []
        for frame in (first, second):
            scope = clause_scope(frame.anchor)
            if explicit_negation_in(doc, scope):
                return []
        return [first, second]

    def applies(self, doc: Doc) -> bool:
        return bool(self._frames(doc))

    def generate(self, doc: Doc) -> list[NegationVariant]:
        frames = self._frames(doc)
        if not frames:
            return []
        first, second = frames
        edits_a = negator_edits(first, "not")
        edits_b = negator_edits(second, "not")
        family = classify_double(first.anchor, second.anchor)
        record = self.build(
            doc,
            edits_a + edits_b,
            subtype=f"{first.subtype}+{second.subtype}",
            net_negation=net_negation_for(family),
            scope_target=SCOPE_CLAUSE,
            family=family,
        )
        return [record] if record else []


@register
class CrossClauseCompound(Generator):
    """Negate one clause while an affixal negative sits in the *other* clause.

    *"The result is important and the method works."*
      -> *"The result is unimportant and the method does not work."*

    Structurally this looks like :class:`AffixalCancellation` -- one clausal
    negator, one derived negative word -- and a cue count of two would classify
    them identically.  They are different because the derived word is outside
    the negator's clause, so nothing cancels: K2, ``net_negation = 2``.
    """

    family = FAM_COMPOUND
    name = "k2_cross_clause_v1"
    depth = 2

    def _pairs(self, doc: Doc) -> list[tuple[ClauseFrame, Token]]:
        frames = _clause_frames(doc)
        if len(frames) < 2:
            return []
        pairs: list[tuple[ClauseFrame, Token]] = []
        for frame in frames:
            scope = clause_scope(frame.anchor)
            if explicit_negation_in(doc, scope) or lexical_negatives_in(doc, scope):
                continue
            for other in frames:
                if other.anchor.i == frame.anchor.i:
                    continue
                other_scope = clause_scope(other.anchor)
                for i in sorted(other_scope):
                    token = doc[i]
                    if affix_eligible(token) and derived_forms(token):
                        pairs.append((frame, token))
                        break
            if pairs:
                break
        return pairs[:1]

    def applies(self, doc: Doc) -> bool:
        return bool(self._pairs(doc))

    def generate(self, doc: Doc) -> list[NegationVariant]:
        pairs = self._pairs(doc)
        if not pairs:
            return []
        frame, token = pairs[0]
        forms = derived_forms(token)
        if not forms:
            return []
        form, subtype = forms[0]
        family = classify_double(frame.anchor, token)
        neg_edits = negator_edits(frame, "not")
        edit = Edit.replace(token.idx, token.idx + len(token.text), form, cue=True)
        record = self.build(
            doc,
            neg_edits + [edit],
            subtype=f"not+{subtype}",
            net_negation=net_negation_for(family),
            scope_target=SCOPE_CLAUSE,
            family=family,
        )
        return [record] if record else []
