"""Affirmation -- run a negation backwards and take the cue out.

Stage 1's generators all shared one shape: an affirmative input gains a cue.
These run the other way.  Each one owns the inverse of a single family, and
declines unless :func:`~negation.classify.classify_input` reports a cue of that
family in the input -- so *"does not sort"* is claimed by
:class:`RemoveSyntacticNegation` and nobody else, and a sentence with no cue at
all is claimed by nobody.

Every generator here removes **exactly one** cue.  That is the depth-1 contract
in its affirming direction: a two-cue input affirms to a one-cue sentence, never
straight to a bare positive.  The cues that survive are re-spliced as themselves
and flagged, so ``cue_char_spans`` keeps pointing at the negation that is still
there rather than silently going empty.

The hard part is not deleting the cue, it is what the cue was holding up.
English do-support exists only to carry a negator, so removing the negator has
to collapse the auxiliary too and hand its tense, person and number back to the
lexical verb: *"did not sort"* -> *"sorted"*, not *"did sort"* or *"sort"*.
That is :mod:`negation.frames`'s do-support rule run in reverse, and it is why
this module reads morphology rather than editing strings.
"""

from __future__ import annotations

from typing import Optional

from spacy.tokens import Doc, Token

from ..affixes import negative_base_forms
from ..base import Generator, register, scope_target_for
from ..classify import COPULA_BE, InputProfile, classify_input
from ..frames import COPULA, DO_SUPPORT, agreeing_finite, detect_frame
from ..lexicons import (
    DO_FORM_TAGS,
    IMPLICIT_TRIGGER_LEMMAS,
    IMPLICIT_TRIGGER_PREDICATES,
    LACK_TRIGGER,
    NEGATIVE_ADVERBS,
    NEGATIVE_QUANTIFIER_BIGRAMS,
    NEGATIVE_QUANTIFIER_FLIPS,
    NEG_ADVERB_AFFIRMATIONS,
)
from ..nlp_core import (
    inflect_verb,
    match_inflection,
    matches_case,
    root_of,
)
from ..schema import (
    FAM_AFFIXAL,
    FAM_IMPLICIT,
    FAM_NEG_ADVERB,
    FAM_QUANTIFIER,
    FAM_SYNTACTIC,
    NegationVariant,
    OP_AFFIRM,
    SCOPE_CLAUSE,
    SCOPE_PREDICATE,
)
from ..splice import Edit, lower_initial

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def net_negation_after_removal(profile: InputProfile) -> int:
    """``net_negation`` for a variant that drops one of the input's cues.

    Uses the same three-valued convention as the rest of the schema: ``0`` when
    nothing negative survives, ``1`` for a single surviving negation, ``2`` when
    two independent ones remain.
    """
    remaining = max(profile.existing_count - 1, 0)
    return remaining if remaining < 2 else 2


def retain_surviving_cues(
    doc: Doc,
    profile: InputProfile,
    edits: list[Edit],
    removed: int,
) -> list[Edit]:
    """Flag every cue except ``removed`` so the record reports what is left.

    A cue the edits already cover -- the lexical verb of *"does not fail to
    sort"* is both the do-support target and a surviving implicit trigger -- has
    its existing edit re-flagged rather than gaining a second, overlapping one.
    """
    out = list(edits)
    for cue in profile.existing_cues:
        if cue.token_idx == removed:
            continue
        token = doc[cue.token_idx]
        start, end = token.idx, token.idx + len(token.text)
        for position, edit in enumerate(out):
            if edit.start <= start < edit.end:
                out[position] = Edit(
                    edit.start, edit.end, tuple((t, True) for t, _ in edit.pieces)
                )
                break
        else:
            out.append(Edit.replace(start, end, token.text, cue=True))
    return out


class Affirmation(Generator):
    """Base class: one family's cue, removed, with the depth-1 contract attached."""

    stage = 2
    op_depth = 1
    operation = OP_AFFIRM

    def profile(self, doc: Doc) -> InputProfile:
        return classify_input(doc)

    def target_cue(self, doc: Doc):
        """The single cue this generator would remove, or ``None``."""
        profile = self.profile(doc)
        if not profile.is_negated:
            return None
        cues = profile.cues_of(self.family)
        return cues[0] if cues else None

    def build_affirmation(
        self,
        doc: Doc,
        edits: list[Edit],
        removed: int,
        *,
        subtype: str,
        scope_target: str = SCOPE_CLAUSE,
    ) -> Optional[NegationVariant]:
        profile = self.profile(doc)
        return self.build(
            doc,
            retain_surviving_cues(doc, profile, edits, removed),
            subtype=subtype,
            net_negation=net_negation_after_removal(profile),
            scope_target=scope_target,
        )


def _delete(token: Token) -> Edit:
    return Edit.replace(token.idx, token.idx + len(token.text), "")


# ---------------------------------------------------------------------------
# A_syntactic, run backwards
# ---------------------------------------------------------------------------


def _do_aux_of(neg: Token) -> Optional[Token]:
    """The do-support auxiliary the negator ``neg`` is riding on, if any."""
    for child in neg.head.children:
        if child.dep_ in ("aux", "auxpass") and child.lower_ in DO_FORM_TAGS:
            return child
    return None


def _plain_aux_of(neg: Token) -> Optional[Token]:
    """The non-do auxiliary ``neg`` attaches to (*has*, *was*, *can*, *must*)."""
    auxes = [
        c
        for c in neg.head.children
        if c.dep_ in ("aux", "auxpass") and c.lower_ not in DO_FORM_TAGS
    ]
    return min(auxes, key=lambda t: t.i) if auxes else None


def clausal_negator_removal(neg: Token) -> Optional[list[Edit]]:
    """Edits that take the clausal negator ``neg`` out of its clause.

    Dispatches on the same three frames :mod:`negation.frames` negates into, run
    backwards.  Only do-support needs morphology restored -- the other two
    already have a finite form that survives the cut.  Returns ``None`` when
    ``neg`` is not a clausal negator at all (a preposed *"Not everyone"*, say).

    Shared with :mod:`negation.generators.rescope`, which has to remove a
    negator from one position before putting one back in another.
    """
    verb = neg.head
    aux = _do_aux_of(neg)
    if aux is not None and aux.i < neg.i:
        restored = inflect_verb(verb.lemma_, DO_FORM_TAGS[aux.lower_])
        return [
            # One cut from the auxiliary through the negator; the lexical verb
            # is a separate edit because material may sit between them.
            Edit.replace(aux.idx, neg.idx + len(neg.text), ""),
            Edit.replace(verb.idx, verb.idx + len(verb.text), restored),
        ]
    plain = _plain_aux_of(neg)
    if plain is not None and plain.i < neg.i:
        return [_delete(neg)]
    if verb.lemma_ == "be" and verb.pos_ in ("AUX", "VERB") and neg.i > verb.i:
        return [_delete(neg)]
    return None


@register
class RemoveSyntacticNegation(Affirmation):
    """Collapse do-support: *"does not sort"* -> *"sorts"*.

    The auxiliary exists only to carry the negator, so it goes with it, and the
    tense/person/number it was holding are handed back to the lexical verb --
    *"did not sort"* -> *"sorted"*, *"does not sort"* -> *"sorts"*.  Anything
    sitting between the auxiliary and the verb (*"does not always sort"*) is
    left where it is; only the ``do`` + negator span is cut.
    """

    family = FAM_SYNTACTIC
    name = "affirm_do_support_v1"

    def _target(self, doc: Doc) -> Optional[tuple[Token, Token]]:
        cue = self.target_cue(doc)
        if cue is None:
            return None
        neg = doc[cue.token_idx]
        if neg.head.pos_ not in ("VERB", "AUX"):
            return None
        aux = _do_aux_of(neg)
        if aux is None or aux.i > neg.i:
            return None
        return aux, neg

    def applies(self, doc: Doc) -> bool:
        return self._target(doc) is not None

    def generate(self, doc: Doc) -> list[NegationVariant]:
        target = self._target(doc)
        if target is None:
            return []
        _, neg = target
        edits = clausal_negator_removal(neg)
        if edits is None:
            return []
        record = self.build_affirmation(
            doc, edits, neg.i, subtype="collapse_do_support"
        )
        return [record] if record else []


@register
class RemoveAuxNegation(Affirmation):
    """Drop the negator off a real auxiliary: *"has not finished"* -> *"has
    finished"*, *"cannot"* -> *"can"*, *"was not sorted"* -> *"was sorted"*.

    No morphology to restore here: the auxiliary was already carrying the
    finiteness, so the negator is simply cut out.  This is also the passive
    route, since a passive clause's negator rides its ``auxpass``.
    """

    family = FAM_SYNTACTIC
    name = "affirm_aux_neg_v1"

    def _target(self, doc: Doc) -> Optional[Token]:
        cue = self.target_cue(doc)
        if cue is None:
            return None
        neg = doc[cue.token_idx]
        if neg.head.pos_ not in ("VERB", "AUX"):
            return None
        if _do_aux_of(neg) is not None:
            return None  # RemoveSyntacticNegation's job
        aux = _plain_aux_of(neg)
        if aux is None or aux.i > neg.i:
            return None
        return neg

    def applies(self, doc: Doc) -> bool:
        return self._target(doc) is not None

    def generate(self, doc: Doc) -> list[NegationVariant]:
        neg = self._target(doc)
        if neg is None:
            return []
        record = self.build_affirmation(
            doc, [_delete(neg)], neg.i, subtype="drop_aux_negator"
        )
        return [record] if record else []


@register
class RemoveCopulaNegation(Affirmation):
    """*"is not important"* -> *"is important"*.

    Kept apart from :class:`RemoveAuxNegation` because a copular clause has no
    auxiliary at all -- the finite *be* is the predicate -- so the two are
    licensed by different structures even though the edit looks the same.
    """

    family = FAM_SYNTACTIC
    name = "affirm_copula_neg_v1"

    def _target(self, doc: Doc) -> Optional[Token]:
        cue = self.target_cue(doc)
        if cue is None:
            return None
        profile = self.profile(doc)
        if profile.copula_type != COPULA_BE:
            return None
        neg = doc[cue.token_idx]
        root = root_of(doc)
        if root is None or neg.head.i != root.i:
            return None
        frame = detect_frame(root)
        if frame is None or frame.subtype != COPULA or neg.i < root.i:
            return None
        return neg

    def applies(self, doc: Doc) -> bool:
        return self._target(doc) is not None

    def generate(self, doc: Doc) -> list[NegationVariant]:
        neg = self._target(doc)
        if neg is None:
            return []
        record = self.build_affirmation(
            doc, [_delete(neg)], neg.i, subtype="drop_copula_negator"
        )
        return [record] if record else []


# ---------------------------------------------------------------------------
# B_quantifier, run backwards
# ---------------------------------------------------------------------------


@register
class QuantifierFlip(Affirmation):
    """*"No students passed"* -> *"Some students passed"*.

    Not derivable from :data:`~negation.lexicons.QUANTIFIER_SWAPS`: that map is
    many-to-one (*some*, *every* and *each* all negate to *no*), so the positive
    counterpart is chosen in
    :data:`~negation.lexicons.NEGATIVE_QUANTIFIER_FLIPS` rather than inverted.
    """

    family = FAM_QUANTIFIER
    name = "affirm_quantifier_flip_v1"

    def _target(self, doc: Doc) -> Optional[tuple[Token, int, str]]:
        cue = self.target_cue(doc)
        if cue is None:
            return None
        token = doc[cue.token_idx]
        if cue.token_idx + 1 < len(doc):
            bigram = (token.lower_, doc[cue.token_idx + 1].lower_)
            if bigram in NEGATIVE_QUANTIFIER_BIGRAMS:
                end = doc[cue.token_idx + 1]
                return token, end.idx + len(end.text), NEGATIVE_QUANTIFIER_BIGRAMS[bigram]
        flip = NEGATIVE_QUANTIFIER_FLIPS.get(token.lower_)
        if flip is None:
            return None
        return token, token.idx + len(token.text), flip

    def applies(self, doc: Doc) -> bool:
        return self._target(doc) is not None

    def generate(self, doc: Doc) -> list[NegationVariant]:
        target = self._target(doc)
        if target is None:
            return []
        token, end, flip = target
        edit = Edit.replace(token.idx, end, lower_initial(flip))
        record = self.build_affirmation(
            doc,
            [edit],
            token.i,
            subtype=f"{token.lower_}_to_{flip.replace(' ', '_')}",
            scope_target=scope_target_for(token),
        )
        return [record] if record else []


# ---------------------------------------------------------------------------
# C_neg_adverb, run backwards
# ---------------------------------------------------------------------------


@register
class NegAdverbStrip(Affirmation):
    """*"never sorts"* -> *"sorts"*; *"rarely sorts"* -> *"often sorts"*.

    Both readings are emitted for the approximative negatives, because deleting
    *rarely* and replacing it with *often* are different claims: the first drops
    the frequency assertion, the second reverses it.
    """

    family = FAM_NEG_ADVERB
    name = "affirm_neg_adverb_v1"

    def _target(self, doc: Doc) -> Optional[Token]:
        cue = self.target_cue(doc)
        if cue is None:
            return None
        token = doc[cue.token_idx]
        if token.lower_ not in NEGATIVE_ADVERBS:
            return None
        return token

    def applies(self, doc: Doc) -> bool:
        return self._target(doc) is not None

    def generate(self, doc: Doc) -> list[NegationVariant]:
        token = self._target(doc)
        if token is None:
            return []
        out: list[NegationVariant] = []
        for replacement in NEG_ADVERB_AFFIRMATIONS[token.lower_]:
            if replacement:
                subtype = f"{token.lower_}_to_{replacement}"
                surface = matches_case(replacement, token)
            else:
                subtype = f"strip_{token.lower_}"
                surface = ""
            edit = Edit.replace(token.idx, token.idx + len(token.text), surface)
            record = self.build_affirmation(doc, [edit], token.i, subtype=subtype)
            if record:
                out.append(record)
        return out


# ---------------------------------------------------------------------------
# D_affixal, run backwards
# ---------------------------------------------------------------------------


@register
class AffixalStrip(Affirmation):
    """*"unimportant"* -> *"important"*, *"useless"* -> *"useful"*.

    The stem is validated exactly as the derivation was: it has to be in WordNet
    *and* antonym-linked to the surface form, which is what stops *income* from
    affirming to *come*.  It is then re-inflected into the slot the derivative
    vacated, so a plural noun stays plural.
    """

    family = FAM_AFFIXAL
    name = "affirm_affixal_strip_v1"

    def _target(self, doc: Doc) -> Optional[Token]:
        cue = self.target_cue(doc)
        if cue is None:
            return None
        token = doc[cue.token_idx]
        return token if negative_base_forms(token) else None

    def applies(self, doc: Doc) -> bool:
        return self._target(doc) is not None

    def generate(self, doc: Doc) -> list[NegationVariant]:
        token = self._target(doc)
        if token is None:
            return []
        out: list[NegationVariant] = []
        for stem, subtype in negative_base_forms(token):
            surface = matches_case(match_inflection(stem, token), token)
            if surface.lower() == token.lower_:
                continue
            edit = Edit.replace(token.idx, token.idx + len(token.text), surface)
            record = self.build_affirmation(
                doc,
                [edit],
                token.i,
                subtype=f"strip_{subtype}",
                scope_target=scope_target_for(token),
            )
            if record:
                out.append(record)
        return out


# ---------------------------------------------------------------------------
# F_implicit, run backwards
# ---------------------------------------------------------------------------


def _infinitival_complement(trigger: Token) -> Optional[Token]:
    """The verb ``trigger`` embeds under *to*, as in *fails **to sort***."""
    for child in trigger.children:
        if child.dep_ in ("xcomp", "ccomp") and child.pos_ == "VERB":
            if any(c.lower_ == "to" and c.dep_ == "aux" for c in child.children):
                return child
    return None


def _gerund_complement(trigger: Token) -> Optional[Token]:
    """The ``-ing`` verb ``trigger`` embeds, as in *denies **sorting***."""
    for child in trigger.children:
        if child.dep_ in ("xcomp", "ccomp") and child.tag_ == "VBG":
            return child
    return None


@register
class ImplicitStrip(Affirmation):
    """Undo a NegEx-style trigger by promoting what it embedded.

    *"The function fails to sort the array."* -> *"The function sorts the array."*
    *"The system lacks redundancy."*          -> *"The system has redundancy."*

    The trigger was holding the clause's finiteness, so promoting the embedded
    verb means inflecting it to the trigger's tense and agreement -- the same
    bookkeeping :class:`~negation.generators.f_implicit.ImplicitTriggerRewrite`
    does on the way in, read off the trigger's Penn tag on the way out.  The
    ``lacks`` case is the odd one: its complement is an NP, not a clause, so the
    trigger is swapped for *have* and the object is left alone.
    """

    family = FAM_IMPLICIT
    name = "affirm_implicit_strip_v1"

    def _target(self, doc: Doc) -> Optional[tuple[Token, Optional[Token], str]]:
        """``(trigger, promoted_verb, mode)`` for the cue this can undo."""
        cue = self.target_cue(doc)
        if cue is None:
            return None
        trigger = doc[cue.token_idx]

        if trigger.lemma_.lower() == LACK_TRIGGER.lemma:
            if any(c.dep_ in ("dobj", "attr") for c in trigger.children):
                return trigger, None, "np"
            return None

        if trigger.lower_ in IMPLICIT_TRIGGER_PREDICATES:
            # "is unable to sort": the finite slot is the copula, not the
            # predicate, so the whole "is unable to" span is what gets replaced.
            complement = _infinitival_complement(trigger)
            if complement is None:
                return None
            return trigger, complement, "copular"

        if trigger.lemma_.lower() not in IMPLICIT_TRIGGER_LEMMAS:
            return None
        complement = _infinitival_complement(trigger)
        if complement is not None:
            return trigger, complement, "to_inf"
        complement = _gerund_complement(trigger)
        if complement is not None:
            return trigger, complement, "gerund"
        return None

    def applies(self, doc: Doc) -> bool:
        return self._target(doc) is not None

    def generate(self, doc: Doc) -> list[NegationVariant]:
        target = self._target(doc)
        if target is None:
            return []
        trigger, complement, mode = target

        if mode == "np":
            frame = detect_frame(trigger)
            if frame is None or frame.subtype != DO_SUPPORT:
                return []
            surface = agreeing_finite("have", frame)
            edits = [Edit.replace(trigger.idx, trigger.idx + len(trigger.text), surface)]
            record = self.build_affirmation(
                doc, edits, trigger.i, subtype="lacks_to_has",
                scope_target=SCOPE_PREDICATE,
            )
            return [record] if record else []

        assert complement is not None
        # The finite slot is the copula for "is unable to", the trigger itself
        # otherwise; either way its Penn tag says how to inflect the promotion.
        finite = _finite_slot_for(trigger) if mode == "copular" else trigger
        if finite is None:
            return []
        restored = inflect_verb(complement.lemma_, finite.tag_)
        start = min(finite.idx, trigger.idx)
        end = complement.idx + len(complement.text)
        if start >= end:
            return []
        record = self.build_affirmation(
            doc,
            [Edit.replace(start, end, restored)],
            trigger.i,
            subtype=f"strip_{mode}",
        )
        return [record] if record else []


def _finite_slot_for(predicate: Token) -> Optional[Token]:
    """The finite verb a predicative adjective hangs off (*is* in *is unable*)."""
    head = predicate.head
    if head.pos_ in ("VERB", "AUX") and head.lemma_ == "be":
        return head
    for child in head.children:
        if child.dep_ in ("aux", "auxpass") and child.lemma_ == "be":
            return child
    return None
