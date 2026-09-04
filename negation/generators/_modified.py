"""Shared machinery for H_hedged and I_intensified.

Both families are *compositions*: they take the clausal negation family A
licenses and modulate its force, so they reuse A's clause frames and stack to
depth 2 (the hard cap).  What they do not do is naively concatenate a modifier
onto A's output string -- ``*"does maybe not sort"`` is not English.  Each
modifier declares the syntactic slot it actually occupies:

``adverb``    a sentence adverb, placed after the first auxiliary if there is
              one and before an introduced do-support auxiliary otherwise;
``modal``     a modal that replaces the finite slot outright;
``post_aux``  a negative phrase substituting for *not* after a real auxiliary.
"""

from __future__ import annotations

from spacy.tokens import Doc

from ..base import Generator
from ..frames import (
    ClauseFrame,
    adverb_edits,
    detect_frame,
    modal_edits,
    post_aux_phrase_edits,
)
from ..lexicons import Modifier
from ..nlp_core import clause_scope, root_of
from ..polarity import explicit_negation_in, lexical_negatives_in
from ..schema import NegationVariant, SCOPE_CLAUSE


class ModifiedNegation(Generator):
    """Base class: apply each declared :class:`Modifier` to the root clause frame."""

    depth = 2
    #: Populated by subclasses.
    modifiers: tuple[Modifier, ...] = ()
    intensity: str = ""

    def applies(self, doc: Doc) -> bool:
        root = root_of(doc)
        if root is None:
            return False
        frame = detect_frame(root)
        if frame is None:
            return False
        scope = clause_scope(root)
        return not explicit_negation_in(doc, scope) and not lexical_negatives_in(doc, scope)

    def _edits_for(self, frame: ClauseFrame, modifier: Modifier):
        if modifier.frames and frame.subtype not in modifier.frames:
            return None
        if modifier.mode == "adverb":
            return adverb_edits(frame, modifier.text)
        if modifier.mode == "modal":
            return modal_edits(frame, modifier.text)
        if modifier.mode == "post_aux":
            return post_aux_phrase_edits(frame, modifier.text)
        raise ValueError(f"unknown modifier mode: {modifier.mode!r}")  # pragma: no cover

    def generate(self, doc: Doc) -> list[NegationVariant]:
        root = root_of(doc)
        if root is None:
            return []
        frame = detect_frame(root)
        if frame is None:
            return []
        out: list[NegationVariant] = []
        for modifier in self.modifiers:
            edits = self._edits_for(frame, modifier)
            if edits is None:
                continue
            record = self.build(
                doc,
                edits,
                subtype=modifier.subtype,
                net_negation=1,
                scope_target=SCOPE_CLAUSE,
                intensity_hint=self.intensity,
            )
            if record:
                out.append(record)
        return out
