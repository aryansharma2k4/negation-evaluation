"""Clause frames: the shared syntactic substrate for clausal negation.

``A_syntactic`` is the foundation of this pipeline, and families C, F, H, I, K
and L all realise their cue in the same three slots it uses.  Rather than each
generator re-deriving "where does the negator go", they all ask for a
:class:`ClauseFrame` and then request edits from the builders below.

The three frames:

``copula``        the clause head *is* a form of *be* -- negate by inserting
                  *not* right after it (*"is not impossible"*).
``existing_aux``  the head governs an ``aux``/``auxpass`` -- the negator lands
                  after that auxiliary (*"has not finished"*), with *can* taking
                  its lexicalised form *cannot*.
``do_support``    a bare finite lexical verb -- English has no slot for the
                  negator, so one is created: the tense and agreement features
                  are stripped off the verb and carried by *do*, and the verb
                  reverts to its lemma (*"sorts" -> "does not sort"*,
                  *"sorted" -> "did not sort"*).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from spacy.tokens import Doc, Token

from .lexicons import (
    CONTRACTED_NEGATIVE_AUX,
    IRREGULAR_AUX_CONTRACTIONS,
    UNCONTRACTABLE_AUX,
)
from .nlp_core import finite_aux, subject_of, third_person_singular
from .splice import Edit, Piece

COPULA = "copula"
EXISTING_AUX = "existing_aux"
DO_SUPPORT = "do_support"


@dataclass(frozen=True)
class ClauseFrame:
    """Where and how a clausal negator can be realised for one clause head."""

    head: Token
    subtype: str
    aux: Optional[Token] = None
    do_form: Optional[str] = None
    subject: Optional[Token] = None
    is_imperative: bool = False

    @property
    def doc(self) -> Doc:
        return self.head.doc

    @property
    def anchor(self) -> Token:
        """The token the negator's scope is anchored to (always the head)."""
        return self.head

    @property
    def finite_slot(self) -> Token:
        """The token carrying finiteness: the auxiliary, else the head itself."""
        return self.aux if self.aux is not None else self.head


def contracted_negative(aux_text: str) -> Optional[str]:
    """The ``n't`` form of an auxiliary, or ``None`` when it has none.

    Mostly a suffix, but not always: *can* contracts to *can't* rather than
    ``*cann't``, *will* to *won't*, and *am* has no standard contraction at all.
    Casing is copied from the input so a fronted *Does* yields *Doesn't*.
    """
    lowered = aux_text.lower()
    if lowered in UNCONTRACTABLE_AUX:
        return None
    contracted = IRREGULAR_AUX_CONTRACTIONS.get(lowered, lowered + "n't")
    if aux_text[:1].isupper():
        return contracted[:1].upper() + contracted[1:]
    return contracted


def _do_form_for(head: Token) -> Optional[str]:
    """Pick ``do``/``does``/``did`` from the head verb's own morphology.

    This is the step naive implementations skip: the auxiliary must inherit the
    tense, person and number features that the lexical verb is about to lose.
    """
    morph = head.morph
    tense = morph.get("Tense")
    verb_form = morph.get("VerbForm")

    if "Past" in tense:
        return "did"
    if "Pres" in tense or head.tag_ in ("VBZ", "VBP"):
        person = morph.get("Person")
        number = morph.get("Number")
        if head.tag_ == "VBZ" or ("3" in person and "Sing" in number):
            return "does"
        return "do"
    # Tagger sometimes leaves a finite verb featureless; fall back to the tag.
    if head.tag_ == "VBD":
        return "did"
    if head.tag_ == "VB" and "Inf" in verb_form:
        return "do"  # bare infinitive: only reachable for imperatives
    return None


def _inherited_subject(head: Token) -> Optional[Token]:
    """``head``'s own subject, or the one it shares as a coordinated verb.

    Verb-phrase coordination leaves the second conjunct subjectless in the
    dependency parse (*"sorts the array and returns the result"*), but it is
    still a finite clause that takes its own negator, so the subject is looked
    up through the ``conj`` chain.
    """
    subject = subject_of(head)
    if subject is not None:
        return subject
    cur = head
    seen = {cur.i}
    while cur.dep_ == "conj" and cur.head.i not in seen:
        cur = cur.head
        seen.add(cur.i)
        subject = subject_of(cur)
        if subject is not None:
            return subject
    return None


def detect_frame(head: Token) -> Optional[ClauseFrame]:
    """Classify ``head`` into a clause frame, or ``None`` if it takes none."""
    if head.pos_ not in ("VERB", "AUX"):
        return None

    aux = finite_aux(head)
    subject = _inherited_subject(head)

    if aux is not None:
        return ClauseFrame(head, EXISTING_AUX, aux=aux, subject=subject)

    if head.lemma_ == "be" and head.pos_ in ("AUX", "VERB"):
        # A bare "be" head is the copula (predicative "is impossible"); with an
        # aux it would already have been caught above.
        return ClauseFrame(head, COPULA, subject=subject)

    do_form = _do_form_for(head)
    if do_form is None:
        return None

    imperative = subject is None and head.tag_ == "VB"
    if subject is None and not imperative:
        return None
    return ClauseFrame(
        head, DO_SUPPORT, do_form=do_form, subject=subject, is_imperative=imperative
    )


def frames_for(doc: Doc, heads: list[Token]) -> list[ClauseFrame]:
    out = []
    for head in heads:
        frame = detect_frame(head)
        if frame is not None:
            out.append(frame)
    return out


# ---------------------------------------------------------------------------
# Edit builders.  Each returns ``(edits, cue_tokens)``.
# ---------------------------------------------------------------------------

#: Preverbal dependents that must stay to the *right* of an introduced
#: auxiliary: "he always arrives" negates as "he does not always arrive",
#: never "*he always does not arrive".
_PREVERBAL_DEPS = frozenset({"advmod", "neg", "npadvmod"})


def do_support_insertion_point(frame: ClauseFrame) -> int:
    """Character offset where an introduced ``do`` auxiliary belongs.

    Normally immediately before the lexical verb, but any preverbal adverbial
    modifier has to end up after the auxiliary, so the insertion point moves
    left past such modifiers.
    """
    head = frame.head
    at = head.idx
    for child in head.children:
        if child.i < head.i and child.dep_ in _PREVERBAL_DEPS and child.idx < at:
            subtree_start = min(t.idx for t in child.subtree)
            if frame.subject is None or subtree_start > frame.subject.idx:
                at = min(at, subtree_start)
    return at


def negator_edits(frame: ClauseFrame, negator: str = "not") -> list[Edit]:
    """Realise a plain clausal negator in ``frame``'s slot."""
    if frame.subtype == COPULA:
        end = frame.head.idx + len(frame.head.text)
        return [Edit.insert(end, " " + negator, cue=True)]

    if frame.subtype == EXISTING_AUX:
        aux = frame.aux
        assert aux is not None
        contracted = CONTRACTED_NEGATIVE_AUX.get(aux.lower_)
        if contracted is not None and negator == "not":
            return [Edit.replace(aux.idx, aux.idx + len(aux.text), contracted, cue=True)]
        end = aux.idx + len(aux.text)
        return [Edit.insert(end, " " + negator, cue=True)]

    # do_support
    head = frame.head
    at = do_support_insertion_point(frame)
    return [
        Edit.compound(at, at, [(f"{frame.do_form} ", False), (negator, True), (" ", False)]),
        Edit.replace(head.idx, head.idx + len(head.text), head.lemma_),
    ]


def adverb_edits(frame: ClauseFrame, adverb: str) -> list[Edit]:
    """Negate with a modal adverb attached.

    Placement follows the standard English rule: a sentence adverb sits *after*
    the first auxiliary when there is one (*"is certainly not impossible"*,
    *"has probably not finished"*), but *before* the do-support auxiliary, since
    ``*"does probably not sort"`` is ungrammatical while *"probably does not
    sort"* is not.
    """
    if frame.subtype in (COPULA, EXISTING_AUX):
        slot = frame.finite_slot
        end = slot.idx + len(slot.text)
        pieces: list[Piece] = [(f" {adverb}", True), (" not", True)]
        return [Edit(end, end, tuple(pieces))]

    # Adverb and negator are separated by the inserted auxiliary, so they are
    # kept as two distinct cue pieces and surface as two cue spans.
    head = frame.head
    at = do_support_insertion_point(frame)
    pieces = [
        (adverb, True),
        (f" {frame.do_form} ", False),
        ("not", True),
        (" ", False),
    ]
    return [
        Edit.compound(at, at, pieces),
        Edit.replace(head.idx, head.idx + len(head.text), head.lemma_),
    ]


def modal_edits(frame: ClauseFrame, modal: str) -> list[Edit]:
    """Negate by *replacing* the finite slot with a negated modal.

    *"might not"* and *"may not"* are modals, not adverbs: they occupy the
    finite auxiliary position rather than stacking on it.  So do-support is not
    introduced at all, an existing modal is overwritten, and a non-modal
    auxiliary (or the copula) reverts to its lemma under the new modal
    (*"has finished" -> "might not have finished"*).
    """
    if frame.subtype == DO_SUPPORT:
        head = frame.head
        at = do_support_insertion_point(frame)
        pieces: list[Piece] = [(modal, True), (" ", False), ("not", True), (" ", False)]
        return [
            Edit.compound(at, at, pieces),
            Edit.replace(head.idx, head.idx + len(head.text), head.lemma_),
        ]

    slot = frame.finite_slot
    if slot.tag_ == "MD":
        pieces = [(modal, True), (" ", False), ("not", True)]
    else:
        pieces = [(modal, True), (" ", False), ("not", True), (f" {slot.lemma_}", False)]
    return [Edit.compound(slot.idx, slot.idx + len(slot.text), tuple(pieces))]


def post_aux_phrase_edits(frame: ClauseFrame, phrase: str) -> list[Edit]:
    """Put a negative phrase in the post-auxiliary slot (*"is by no means X"*).

    Requires a real auxiliary or copula; with do-support the phrase would have
    to be postposed past the object, which this builder does not attempt.
    """
    if frame.subtype == DO_SUPPORT:
        raise ValueError("post-auxiliary phrases need an existing auxiliary")
    slot = frame.finite_slot
    end = slot.idx + len(slot.text)
    return [Edit.insert(end, " " + phrase, cue=True)]


def be_form_for(frame: ClauseFrame) -> str:
    """The form of *be* agreeing with ``frame``'s subject and tense.

    Used by families that rebuild the predicate around a copula (``F``'s
    *is unable to*, ``G``'s *is without*).
    """
    head = frame.head
    past = "Past" in head.morph.get("Tense") or head.tag_ == "VBD"
    subject = frame.subject
    plural = False
    person = "3"
    if subject is not None:
        plural = "Plur" in subject.morph.get("Number") or subject.tag_ in ("NNS", "NNPS")
        got = subject.morph.get("Person")
        if got:
            person = got[0]
        if subject.lower_ in ("i",):
            person = "1"
        elif subject.lower_ in ("you", "we", "they"):
            person, plural = "2", True
    if past:
        return "were" if plural or person == "2" else "was"
    if plural:
        return "are"
    if person == "1":
        return "am"
    if person == "2":
        return "are"
    return "is"


def agreeing_finite(lemma: str, frame: ClauseFrame) -> str:
    """Inflect ``lemma`` to carry the finiteness ``frame``'s head currently has.

    The mirror image of do-support: when family F swaps the lexical verb for a
    trigger verb, the trigger has to pick up the tense/agreement the original
    verb was carrying.
    """
    if lemma == "be":
        return be_form_for(frame)
    from .nlp_core import past_tense  # local import keeps module import order flat

    head = frame.head
    if frame.subtype == DO_SUPPORT:
        if frame.do_form == "did":
            return past_tense(lemma)
        if frame.do_form == "does":
            return third_person_singular(lemma)
        return lemma
    if "Past" in head.morph.get("Tense") or head.tag_ == "VBD":
        return past_tense(lemma)
    if head.tag_ == "VBZ":
        return third_person_singular(lemma)
    return lemma
