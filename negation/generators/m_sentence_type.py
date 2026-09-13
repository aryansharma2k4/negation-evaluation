"""Negation in the sentence types stage 1 never saw.

Stage 1's clausal families all realise the negator immediately after the finite
slot.  That is right for a declarative and for an imperative, and wrong for
everything else:

*interrogative*  the finite slot is fronted past the subject, so "after the
                 finite slot" lands before it -- ``*"Does not it compile?"``.
                 The negator belongs after the *subject*, or contracted onto
                 the fronted auxiliary.
*existential*    *"There is not a solution"* is grammatical but stilted; English
                 negates an existential on the quantifier, not the copula.
*modal*          negating a modal is scope-ambiguous in a way plain clausal
                 negation is not, and the readings are not paraphrases.
*comparative*    *"not faster than X"* and *"no faster than X"* mean different
                 things -- the first denies the comparison, the second asserts
                 equality-at-best.

Each generator here fires on exactly one of those configurations and declines
otherwise, and each applies exactly one operation to its input.
"""

from __future__ import annotations

from typing import Optional

from spacy.tokens import Doc, Token

from ..base import Generator, register
from ..classify import (
    CLAUSE_DECLARATIVE,
    CLAUSE_EXCLAMATIVE,
    CLAUSE_IMPERATIVE,
    CLAUSE_INTERROGATIVE,
    InputProfile,
    classify_input,
)
from ..frames import DO_SUPPORT, contracted_negative, detect_frame
from ..lexicons import MODAL_NEGATION_READINGS, NO_MODALITY
from ..nlp_core import clause_scope, root_of, subject_of
from ..polarity import explicit_negation_in
from ..scope import classify_double, net_negation_for
from ..schema import (
    FAM_MODAL,
    FAM_QUANTIFIER,
    FAM_SYNTACTIC,
    NegationVariant,
    SCOPE_CLAUSE,
    SCOPE_PREDICATE,
)
from ..splice import Edit, lower_initial


class SentenceTypeGenerator(Generator):
    """Base class for the arbitrary-input negation rules."""

    stage = 2
    op_depth = 1

    def profile(self, doc: Doc) -> InputProfile:
        return classify_input(doc)

    def _affirmative(self, doc: Doc) -> bool:
        """These all *add* a cue, so they decline anything already negated.

        A negated input is not out of scope for the pipeline -- it is handled by
        the affirmation and rescope families, and by K1/K2 for a second cue --
        but adding a cue here as well would be a second route to the same place.
        """
        return not self.profile(doc).is_negated


# ---------------------------------------------------------------------------
# Interrogatives
# ---------------------------------------------------------------------------


def _fronted_aux(doc: Doc) -> Optional[tuple[Token, Token]]:
    """``(auxiliary, subject)`` for a subject-auxiliary-inverted question.

    Two shapes: an auxiliary fronted past the subject (*"__Does__ it
    compile?"*, *"__Has__ she finished?"*), and a copula that is itself the
    fronted finite slot (*"__Is__ the task impossible?"*), where the parser
    makes the copula the ROOT rather than an ``aux``.
    """
    root = root_of(doc)
    if root is None:
        return None
    subject = subject_of(root)
    if subject is None:
        return None
    if root.pos_ == "AUX" and root.i < subject.i:
        return root, subject
    fronted = [
        c for c in root.children if c.dep_ in ("aux", "auxpass") and c.i < subject.i
    ]
    if not fronted:
        return None
    return min(fronted, key=lambda t: t.i), subject


#: Post-head dependents of an inverted question's subject that are really the
#: predicate.  A genuine NP-internal modifier that follows its noun is a
#: relative clause or a PP; a *following* adjective is not how English builds an
#: NP, so when the parser hangs one there it has misanalysed *"Is the task
#: impossible?"* as ``[the task impossible]`` and the adjective belongs to the
#: predicate instead.
_MISATTACHED_PREDICATE_DEPS = frozenset({"amod", "attr", "acomp", "oprd"})


def _subject_end(subject: Token) -> int:
    """Character offset just past the subject phrase.

    The whole subject subtree rather than just its head, so that in *"Does the
    parser that caches compile?"* the negator goes after *caches* -- but with
    misattached predicate material excluded, so that *"Is the task impossible?"*
    negates to *"Is the task not impossible?"* and not
    ``*"Is the task impossible not?"``.
    """
    excluded: set[int] = set()
    for child in subject.children:
        if child.i > subject.i and child.dep_ in _MISATTACHED_PREDICATE_DEPS:
            excluded.update(t.i for t in child.subtree)
    tokens = [t for t in subject.subtree if t.i not in excluded and not t.is_punct]
    last = max(tokens, key=lambda t: t.i)
    return last.idx + len(last.text)


@register
class InterrogativeNegation(SentenceTypeGenerator):
    """Negate inside a question, preserving the inversion and the *?*.

    *"Does it compile?"* -> *"Does it not compile?"* (negator after the subject)
                         -> *"Doesn't it compile?"*  (contracted onto the aux)

    Both are emitted because they are not stylistic variants of one another in
    the way the declarative *"does not"* / *"doesn't"* pair is: the contracted
    question is the ordinary way to ask it, while the uncontracted one carries a
    distinctly different, more deliberate force.
    """

    family = FAM_SYNTACTIC
    name = "interrogative_negation_v1"
    clause_types = frozenset({CLAUSE_INTERROGATIVE})

    def applies(self, doc: Doc) -> bool:
        if _fronted_aux(doc) is None:
            return False
        root = root_of(doc)
        if root is None:
            return False
        # Family A never stacks on itself: a clause that already has an overt
        # negator has no free slot, and "Hasn't she not finished?" is what
        # ignoring that produces.  A *lexical* negative in the clause is fine --
        # that is the K1 cancellation route, and it is still one operation.
        return not explicit_negation_in(doc, clause_scope(root))

    def _labelling(self, doc: Doc) -> tuple[str, int]:
        """``(family, net_negation)`` for the record this will emit.

        An affirmative question gains its first cue and is plain ``A_syntactic``.
        A question that already carries one gains a *second*, which is the K1/K2
        distinction -- and it is decided the same way it is everywhere else in
        this pipeline, by asking whether the existing cue lies inside the clause
        the new negator commands.  Adding one cue to an N-cue input is still one
        operation, so the depth-1 contract holds either way.
        """
        profile = self.profile(doc)
        root = root_of(doc)
        if not profile.is_negated or root is None:
            return FAM_SYNTACTIC, 1
        inner = doc[profile.existing_cues[0].token_idx]
        family = classify_double(root, inner)
        return family, net_negation_for(family)

    def generate(self, doc: Doc) -> list[NegationVariant]:
        target = _fronted_aux(doc)
        if target is None:
            return []
        aux, subject = target
        family, net = self._labelling(doc)
        profile = self.profile(doc)
        retained = [doc[c.token_idx] for c in profile.existing_cues]
        out: list[NegationVariant] = []

        forms: list[tuple[str, list[Edit]]] = []
        if aux.dep_ in ("aux", "auxpass"):
            # Post-subject placement needs to know where the subject ends, and
            # that is only trustworthy when the fronted element is a real
            # auxiliary over a lexical verb.  When the copula itself is fronted,
            # the parser has no verb to anchor on and splits "Is the task
            # simple?" as [the task simple] -- so that shape gets the contracted
            # form only, which needs no boundary at all.
            at = _subject_end(subject)
            forms.append(("question_post_subject", [Edit.insert(at, " not", cue=True)]))
        contracted = contracted_negative(aux.text)
        if contracted is not None:
            forms.append(
                (
                    "question_contracted",
                    [Edit.replace(aux.idx, aux.idx + len(aux.text), contracted, cue=True)],
                )
            )

        for subtype, edits in forms:
            # An existing cue is re-spliced as itself so the record reports both
            # cues with correct post-edit offsets, exactly as K1 does.
            marked = list(edits) + [
                Edit.replace(t.idx, t.idx + len(t.text), t.text, cue=True)
                for t in retained
                if not any(e.start <= t.idx < e.end for e in edits)
            ]
            record = self.build(
                doc,
                marked,
                subtype=subtype,
                net_negation=net,
                scope_target=SCOPE_CLAUSE,
                family=family,
            )
            if record:
                out.append(record)
        return out


# ---------------------------------------------------------------------------
# Imperatives
# ---------------------------------------------------------------------------


@register
class ImperativeContractedNegation(SentenceTypeGenerator):
    """*"Sort the array."* -> *"Don't sort the array."*

    The uncontracted *"Do not sort the array."* is already family A's do-support
    rule, which handles imperatives correctly because a bare imperative has no
    inversion to preserve.  Only the contraction is new, and it is worth its own
    record: a prohibition is far more often written contracted.
    """

    family = FAM_SYNTACTIC
    name = "imperative_contracted_v1"
    clause_types = frozenset({CLAUSE_IMPERATIVE})

    def _frame(self, doc: Doc):
        root = root_of(doc)
        if root is None:
            return None
        frame = detect_frame(root)
        if frame is None or not frame.is_imperative:
            return None
        return frame

    def applies(self, doc: Doc) -> bool:
        return self._affirmative(doc) and self._frame(doc) is not None

    def generate(self, doc: Doc) -> list[NegationVariant]:
        frame = self._frame(doc)
        if frame is None:
            return []
        head = frame.head
        edits = [Edit.insert(head.idx, "don't ", cue=True)]
        if head.i == 0:
            # The verb was the capitalised word; once the contraction sits in
            # front of it, it is not.  build() recases the new initial.
            edits.append(
                Edit.replace(head.idx, head.idx + len(head.text), lower_initial(head.text))
            )
        record = self.build(
            doc,
            edits,
            subtype="imperative_contracted",
            net_negation=1,
            scope_target=SCOPE_CLAUSE,
        )
        return [record] if record else []


# ---------------------------------------------------------------------------
# Existentials
# ---------------------------------------------------------------------------


def _existential_complement(doc: Doc) -> Optional[Token]:
    """The NP an existential asserts the existence of."""
    root = root_of(doc)
    if root is None:
        return None
    if not any(c.dep_ == "expl" and c.lower_ == "there" for c in root.children):
        return None
    for child in root.children:
        if child.dep_ in ("attr", "nsubj", "dobj"):
            return child
    return None


@register
class ExistentialNegation(SentenceTypeGenerator):
    """*"There is a solution."* -> *"There is no solution."*

    Quantifier negation rather than clausal: *"There is not a solution"* is
    grammatical but stilted, and the natural denial of an existential puts the
    negation on the determiner.  So the record is filed under ``B_quantifier``,
    which is the family that actually did the work.

    An existing determiner is replaced (*a solution* -> *no solution*) rather
    than prefixed, so nothing yields ``*"no a solution"``; a bare plural simply
    gets *no* in front of it.
    """

    family = FAM_QUANTIFIER
    name = "existential_no_v1"

    def _target(self, doc: Doc) -> Optional[tuple[Token, Optional[Token]]]:
        if not self.profile(doc).is_existential:
            return None
        complement = _existential_complement(doc)
        if complement is None:
            return None
        determiner = next(
            (c for c in complement.children if c.dep_ == "det" and c.i < complement.i),
            None,
        )
        return complement, determiner

    def applies(self, doc: Doc) -> bool:
        return self._affirmative(doc) and self._target(doc) is not None

    def generate(self, doc: Doc) -> list[NegationVariant]:
        target = self._target(doc)
        if target is None:
            return []
        complement, determiner = target
        if determiner is not None:
            edit = Edit.replace(
                determiner.idx, determiner.idx + len(determiner.text), "no", cue=True
            )
        else:
            edit = Edit.insert(complement.idx, "no ", cue=True)
        record = self.build(
            doc,
            [edit],
            subtype="existential_no",
            net_negation=1,
            scope_target=SCOPE_PREDICATE,
        )
        return [record] if record else []


# ---------------------------------------------------------------------------
# Modals
# ---------------------------------------------------------------------------


def _root_modal(doc: Doc) -> Optional[Token]:
    root = root_of(doc)
    if root is None:
        return None
    candidates = [root, *root.children] if root.tag_ == "MD" else list(root.children)
    for token in sorted(candidates, key=lambda t: t.i):
        if token.tag_ == "MD" and token.lower_ in MODAL_NEGATION_READINGS:
            return token
    return None


@register
class ModalReadingNegation(SentenceTypeGenerator):
    """Emit each reading a negated modal splits into, as its own record.

    *"The parser must handle noise."*
      -> *"The parser must not handle noise."*  (prohibition)
      -> *"The parser need not handle noise."*  (absence of obligation)

    These are not paraphrases: the first forbids the handling, the second merely
    stops requiring it.  Plain clausal negation collapses them into whichever
    string it happens to build, so the pair gets its own family, ``M_modal``,
    where both survive deduplication and each carries the reading in its
    subtype and the modality it expresses in ``modality``.
    """

    family = FAM_MODAL
    name = "modal_reading_v1"
    clause_types = frozenset(
        {CLAUSE_DECLARATIVE, CLAUSE_IMPERATIVE, CLAUSE_EXCLAMATIVE, CLAUSE_INTERROGATIVE}
    )

    def applies(self, doc: Doc) -> bool:
        if not self._affirmative(doc) or self.profile(doc).modality == NO_MODALITY:
            return False
        # An inverted question would need the reading placed after the subject,
        # which is InterrogativeNegation's rule, not a modal substitution.
        if self.profile(doc).clause_type == CLAUSE_INTERROGATIVE:
            return False
        return _root_modal(doc) is not None

    def generate(self, doc: Doc) -> list[NegationVariant]:
        modal = _root_modal(doc)
        if modal is None:
            return []
        out: list[NegationVariant] = []
        for reading in MODAL_NEGATION_READINGS[modal.lower_]:
            # "cannot" is one lexicalised word and is wholly the cue; "must not"
            # is a modal plus a negator, and only the negator is.
            words = reading.surface.split()
            if len(words) == 2:
                pieces = [(words[0], False), (" ", False), (words[1], True)]
            else:
                pieces = [(reading.surface, True)]
            edit = Edit.compound(modal.idx, modal.idx + len(modal.text), pieces)
            record = self.build(
                doc,
                [edit],
                subtype=reading.subtype,
                net_negation=1,
                scope_target=SCOPE_CLAUSE,
                modality=reading.modality,
            )
            if record:
                out.append(record)
        return out


# ---------------------------------------------------------------------------
# Comparatives
# ---------------------------------------------------------------------------


def _comparative_head(doc: Doc) -> Optional[Token]:
    """The comparative adjective/adverb of an explicit *than* comparison.

    The *than* phrase is required: without it *"The task is harder"* has no
    standard of comparison for *no* to deny, and *"no harder"* is left hanging.
    """
    for token in doc:
        if token.tag_ not in ("JJR", "RBR"):
            continue
        if any(c.lower_ == "than" for c in token.children):
            return token
    return None


@register
class ComparativeNegation(SentenceTypeGenerator):
    """*"faster than X"* -> *"no faster than X"*.

    The other half of the pair -- *"not faster than X"* -- is ordinary clausal
    negation and family A already emits it.  They are kept as separate records
    because they say different things: *not faster* denies the comparison and
    leaves slower or equal open, while *no faster* asserts equality at best.
    """

    family = FAM_QUANTIFIER
    name = "comparative_no_v1"

    def applies(self, doc: Doc) -> bool:
        if not self._affirmative(doc) or not self.profile(doc).is_comparative:
            return False
        return _comparative_head(doc) is not None

    def generate(self, doc: Doc) -> list[NegationVariant]:
        head = _comparative_head(doc)
        if head is None:
            return []
        record = self.build(
            doc,
            [Edit.insert(head.idx, "no ", cue=True)],
            subtype="no_comparative",
            net_negation=1,
            scope_target=SCOPE_PREDICATE,
        )
        return [record] if record else []
