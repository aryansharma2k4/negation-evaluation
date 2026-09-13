"""One clause per record, in sentences that have more than one.

Stage 1 negated the ROOT clause and left the rest alone, so a two-clause
sentence yielded exactly one clausal record however many clauses it had.  Two
gaps followed from that.  A conditional was only ever negated in its consequent
-- *"If the test passes, the build does not succeed."* -- and the far more
interesting *"If the test does **not** pass, the build succeeds."* was never
produced.  And a coordination could only be negated in its first conjunct.

This generator closes both by targeting each clause in turn, one record each,
recording which one in ``target_clause_idx``.  One clause per record is not a
stylistic choice: negating the antecedent *and* the consequent at once would be
two operations on one input, which is exactly what the depth-1 rule forbids.
``K2_compound`` is the family that does that, it declares ``op_depth = 2``, and
it stays where it is.

The root clause is offered here too, even though ``A_syntactic`` already covers
it.  The duplicate is dropped by the ordinary dedup filter, and letting the two
overlap keeps the rule stated as "every clause" rather than "every clause except
one", which is the version that stays correct when a sentence's ROOT is not its
first clause.
"""

from __future__ import annotations

from typing import Optional

from spacy.tokens import Doc, Token

from ..base import Generator, mark_cue_tokens, register
from ..classify import InputProfile, classify_input
from ..frames import ClauseFrame, detect_frame, negator_edits
from ..nlp_core import clause_heads, clause_scope
from ..polarity import explicit_negation_in
from ..schema import FAM_SYNTACTIC, NegationVariant, SCOPE_CLAUSE
from ..scope import double_labelling


@register
class ClauseTargetedNegation(Generator):
    """Negate exactly one clause of a multi-clause sentence, per record.

    *"If the test passes, the build succeeds."*
      -> *"If the test does not pass, the build succeeds."*   (clause 0)
      -> *"If the test passes, the build does not succeed."*  (clause 1)

    *"The function sorts the array and returns the result."*
      -> *"The function does not sort the array and returns the result."*
      -> *"The function sorts the array and does not return the result."*

    A clause that already carries an overt negator is skipped -- there is no
    free slot in it -- but a *lexical* negative elsewhere in the sentence is not
    a reason to skip anything; it just means the new cue is the second one, and
    :func:`~negation.scope.double_labelling` decides whether the two cancel (K1)
    or compound (K2) by the usual containment test.
    """

    family = FAM_SYNTACTIC
    name = "clause_target_negation_v1"
    stage = 2
    op_depth = 1

    def profile(self, doc: Doc) -> InputProfile:
        return classify_input(doc)

    def _targets(self, doc: Doc) -> list[tuple[int, ClauseFrame]]:
        """``(clause_index, frame)`` for every clause with a free negator slot."""
        profile = self.profile(doc)
        if profile.clause_count < 2:
            return []
        out: list[tuple[int, ClauseFrame]] = []
        for index, head in enumerate(clause_heads(doc)):
            frame = detect_frame(head)
            if frame is None:
                continue
            if explicit_negation_in(doc, clause_scope(head)):
                continue
            out.append((index, frame))
        return out

    def applies(self, doc: Doc) -> bool:
        return bool(self._targets(doc))

    def generate(self, doc: Doc) -> list[NegationVariant]:
        profile = self.profile(doc)
        existing = [doc[cue.token_idx] for cue in profile.existing_cues]
        out: list[NegationVariant] = []

        for index, frame in self._targets(doc):
            labelling = double_labelling(frame.anchor, existing)
            family, net = labelling if labelling else (self.family, 1)
            edits = mark_cue_tokens(existing, negator_edits(frame, "not"))
            record = self.build(
                doc,
                edits,
                subtype=f"clause_{frame.subtype}",
                net_negation=net,
                scope_target=SCOPE_CLAUSE,
                family=family,
                target_clause_idx=index,
            )
            if record:
                out.append(record)
        return out
