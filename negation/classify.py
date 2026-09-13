"""Input classification: what kind of sentence are we being handed?

Stage 1 assumed every input was an affirmative declarative, so the generators
could go straight from a parse to a negator slot.  Once arbitrary input is
allowed, that assumption has to be replaced by an explicit description of the
input, because it decides three different things:

*which operation is even available*
    An affirmative input can be negated; a negated one can additionally be
    affirmed or rescoped.  Generators read :attr:`InputProfile.polarity` to
    know which of the three they are entitled to.

*where the cue may land*
    A question puts the negator after the subject (*"Does it not compile?"*),
    a passive puts it on the passive auxiliary, an existential prefers a
    quantifier (*"There is no solution"*) over a copular *not*.

*how much negation is already there*
    :attr:`InputProfile.existing_count` is the ``N`` in the depth-1 rule: an
    emitted variant may carry at most ``N + 1`` cues.  Counting it requires
    recognising cues from *every* family, not just *not* -- which is why this
    module reads the same lexicons the generators write with, rather than
    keeping a list of its own.

The profile is computed once per :class:`~spacy.tokens.Doc` and cached on it,
so the driver's ``applies`` sweep pays for it exactly once per sentence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

from spacy.tokens import Doc, Token

from .affixes import negative_base_forms
from .lexicons import (
    IMPLICIT_TRIGGER_LEMMAS,
    IMPLICIT_TRIGGER_PREDICATES,
    MODALITY_BY_MODAL,
    NEGATIVE_ADVERBS,
    NEGATIVE_AUX_BASES,
    NEGATIVE_QUANTIFIER_BIGRAMS,
    NEGATIVE_QUANTIFIERS,
    NO_MODALITY,
    PRIVATIVE_HEADS,
    PRIVATIVE_NEEDS_OF,
)
from .nlp_core import clause_heads, root_of, subject_of
from .schema import (
    FAM_AFFIXAL,
    FAM_IMPLICIT,
    FAM_NEG_ADVERB,
    FAM_PREPOSITIONAL,
    FAM_QUANTIFIER,
    FAM_SYNTACTIC,
)

# ---------------------------------------------------------------------------
# Enum-ish constants.  Plain strings, like the schema's, so a profile stays
# trivially serialisable and comparable in tests.
# ---------------------------------------------------------------------------

CLAUSE_DECLARATIVE = "declarative"
CLAUSE_INTERROGATIVE = "interrogative"
CLAUSE_IMPERATIVE = "imperative"
CLAUSE_EXCLAMATIVE = "exclamative"
CLAUSE_TYPES: frozenset[str] = frozenset(
    {CLAUSE_DECLARATIVE, CLAUSE_INTERROGATIVE, CLAUSE_IMPERATIVE, CLAUSE_EXCLAMATIVE}
)

VOICE_ACTIVE = "active"
VOICE_PASSIVE = "passive"
VOICES: frozenset[str] = frozenset({VOICE_ACTIVE, VOICE_PASSIVE})

POL_AFFIRMATIVE = "affirmative"
POL_NEGATED = "negated"
POLARITIES: frozenset[str] = frozenset({POL_AFFIRMATIVE, POL_NEGATED})

COPULA_NONE = "none"
COPULA_BE = "be"
COPULA_SEEM = "become/seem/appear"
COPULA_TYPES: frozenset[str] = frozenset({COPULA_NONE, COPULA_BE, COPULA_SEEM})

#: Verbs that pattern with *be* as copulas but are not *be*.
_SEEM_LEMMAS: frozenset[str] = frozenset({"become", "seem", "appear"})

#: Dependency labels that mean "a subordinate clause hangs here".
_SUBORDINATE_DEPS: frozenset[str] = frozenset({"advcl", "ccomp", "relcl", "xcomp", "acl"})

#: Complement labels a copula takes.
_PREDICATIVE_DEPS: frozenset[str] = frozenset({"acomp", "attr", "oprd"})

#: Subordinators that introduce a conditional.
_CONDITIONAL_MARKERS: frozenset[str] = frozenset({"if", "unless"})

#: Wh-words that can open an exclamative rather than a question.
_EXCLAMATIVE_OPENERS: frozenset[str] = frozenset({"what", "how"})


class ExistingCue(NamedTuple):
    """A negation cue already present in the input.

    A plain ``(token_idx, cue_text, family_guess)`` tuple, so it unpacks
    positionally as well as reading by name.  ``family_guess`` is a guess in the
    honest sense: it says which family *would have produced* this cue, which is
    what an affirmation generator needs in order to know whether it is the one
    that can undo it.
    """

    token_idx: int
    cue_text: str
    family_guess: str


@dataclass(frozen=True)
class InputProfile:
    """Everything a generator needs to know about its input before firing."""

    clause_type: str
    voice: str
    polarity: str
    existing_cues: tuple[ExistingCue, ...]
    existing_count: int
    modality: str
    clause_count: int
    has_subordinate: bool
    is_existential: bool
    is_comparative: bool
    is_conditional: bool
    copula_type: str

    def __post_init__(self) -> None:
        if self.clause_type not in CLAUSE_TYPES:
            raise ValueError(f"unknown clause_type: {self.clause_type!r}")
        if self.voice not in VOICES:
            raise ValueError(f"unknown voice: {self.voice!r}")
        if self.polarity not in POLARITIES:
            raise ValueError(f"unknown polarity: {self.polarity!r}")
        if self.copula_type not in COPULA_TYPES:
            raise ValueError(f"unknown copula_type: {self.copula_type!r}")

    @property
    def is_negated(self) -> bool:
        return self.polarity == POL_NEGATED

    def cues_of(self, family: str) -> tuple[ExistingCue, ...]:
        """The existing cues attributed to ``family``."""
        return tuple(c for c in self.existing_cues if c.family_guess == family)


# ---------------------------------------------------------------------------
# Cue detection
# ---------------------------------------------------------------------------


def _is_privative(token: Token) -> bool:
    """*without X*, *devoid of X*, *in the absence of X*.

    The heads that are ordinary words on their own (*free*, *absence*) count
    only with their ``of`` complement, so *"the build is free"* is not read as
    a negation.
    """
    if token.lower_ not in PRIVATIVE_HEADS:
        return False
    if token.lower_ not in PRIVATIVE_NEEDS_OF:
        return True
    return any(c.lower_ == "of" for c in token.children)


def _is_implicit_trigger(token: Token) -> bool:
    """A NegEx-style verbal trigger (*fails to*, *refuses to*, *lacks*).

    The copular trigger is the exception: *unable* is only the ``is unable to``
    trigger when it actually embeds a complement.  Bare *"she is unable"* is
    left to the affixal reader.
    """
    if token.pos_ in ("VERB", "AUX") and token.lemma_.lower() in IMPLICIT_TRIGGER_LEMMAS:
        return True
    if token.lower_ in IMPLICIT_TRIGGER_PREDICATES:
        return any(c.dep_ in ("xcomp", "ccomp", "acomp") for c in token.children)
    return False


def _cue_family(token: Token) -> Optional[str]:
    """Which family would have produced ``token`` as a cue, if any.

    Order matters.  *never* is a negative adverb even though ``B`` reaches it
    from *always*; *no* is a quantifier even when the parser labels it ``neg``;
    *unable* is an implicit trigger before it is an affixal negative.
    """
    lower = token.lower_
    if lower in NEGATIVE_ADVERBS:
        return FAM_NEG_ADVERB
    if lower in NEGATIVE_QUANTIFIERS:
        return FAM_QUANTIFIER
    if lower in ("not", "n't", "nt") or lower in NEGATIVE_AUX_BASES:
        return FAM_SYNTACTIC
    if token.dep_ == "neg":
        return FAM_SYNTACTIC
    if _is_privative(token):
        return FAM_PREPOSITIONAL
    if _is_implicit_trigger(token):
        return FAM_IMPLICIT
    if negative_base_forms(token):
        return FAM_AFFIXAL
    return None


def existing_cues(doc: Doc) -> tuple[ExistingCue, ...]:
    """Every negation cue already in ``doc``, across all families, in order.

    Multiword cues are reported once: *"no one"* is a single quantifier cue, not
    a quantifier plus a stray noun, so the second token is consumed rather than
    re-examined.
    """
    found: list[ExistingCue] = []
    index = 0
    while index < len(doc):
        token = doc[index]
        if index + 1 < len(doc):
            bigram = (token.lower_, doc[index + 1].lower_)
            if bigram in NEGATIVE_QUANTIFIER_BIGRAMS:
                text = doc.text[token.idx : doc[index + 1].idx + len(doc[index + 1].text)]
                found.append(ExistingCue(token.i, text, FAM_QUANTIFIER))
                index += 2
                continue
        family = _cue_family(token)
        if family is not None:
            found.append(ExistingCue(token.i, token.text, family))
        index += 1
    return tuple(found)


# ---------------------------------------------------------------------------
# Structural classification
# ---------------------------------------------------------------------------


def _is_inverted(doc: Doc, root: Optional[Token]) -> bool:
    """Subject-auxiliary inversion: the finite auxiliary precedes the subject."""
    if root is None:
        return False
    subject = subject_of(root)
    if subject is None:
        return False
    if root.pos_ == "AUX" and root.i < subject.i:
        return True  # "Is it broken?" -- the copula itself is the finite slot
    return any(c.dep_ in ("aux", "auxpass") and c.i < subject.i for c in root.children)


def clause_type_of(doc: Doc) -> str:
    """``declarative`` | ``interrogative`` | ``imperative`` | ``exclamative``.

    Punctuation is evidence but never the whole story: *"Sort the array!"* is an
    imperative, not an exclamative, and a question can be written without its
    mark.  So the structural tests -- subject-auxiliary inversion, a wh-opener,
    a subjectless bare-infinitive root -- decide, and punctuation only breaks
    the remaining ties.
    """
    text = doc.text.strip()
    root = root_of(doc)
    first = doc[0] if len(doc) else None
    inverted = _is_inverted(doc, root)
    wh_opener = first is not None and first.tag_ in ("WP", "WDT", "WRB", "WP$")

    if text.endswith("?") or (inverted and wh_opener):
        return CLAUSE_INTERROGATIVE
    if root is not None and root.tag_ == "VB" and subject_of(root) is None:
        # Subjectless bare infinitive: an imperative even under "!".
        return CLAUSE_IMPERATIVE
    if text.endswith("!"):
        return CLAUSE_EXCLAMATIVE
    if wh_opener and first is not None and first.lower_ in _EXCLAMATIVE_OPENERS and not inverted:
        return CLAUSE_EXCLAMATIVE
    if inverted:
        return CLAUSE_INTERROGATIVE
    return CLAUSE_DECLARATIVE


def voice_of(doc: Doc) -> str:
    return (
        VOICE_PASSIVE
        if any(t.dep_ in ("auxpass", "nsubjpass", "csubjpass") for t in doc)
        else VOICE_ACTIVE
    )


def modality_of(doc: Doc) -> str:
    """The modality of the root clause, or ``none``.

    Only the root clause's modal counts: a modal buried in a relative clause
    does not make the sentence modal, and the generators that read this field
    all operate on the matrix predicate.
    """
    root = root_of(doc)
    if root is None:
        return NO_MODALITY
    candidates = [root, *root.children] if root.tag_ == "MD" else list(root.children)
    for token in sorted(candidates, key=lambda t: t.i):
        if token.tag_ == "MD" and token.lower_ in MODALITY_BY_MODAL:
            return MODALITY_BY_MODAL[token.lower_]
    return NO_MODALITY


def copula_type_of(doc: Doc) -> str:
    """``be`` | ``become/seem/appear`` | ``none`` for the root predicate."""
    root = root_of(doc)
    if root is None:
        return COPULA_NONE
    if root.lemma_ == "be" and root.pos_ in ("AUX", "VERB"):
        # A "be" that governs a participle is a passive/progressive auxiliary,
        # not a copula; the copular reading needs a predicative complement.
        if any(c.dep_ in _PREDICATIVE_DEPS for c in root.children) or not any(
            c.dep_ in ("aux", "auxpass") for c in root.children
        ):
            return COPULA_BE
    if root.lemma_ in _SEEM_LEMMAS and any(
        c.dep_ in _PREDICATIVE_DEPS for c in root.children
    ):
        return COPULA_SEEM
    if any(c.dep_ == "cop" for c in root.children):
        return COPULA_BE
    return COPULA_NONE


def is_existential(doc: Doc) -> bool:
    """*There is/are X* -- an expletive subject on a copular root."""
    return any(t.dep_ == "expl" and t.lower_ == "there" for t in doc)


def is_comparative(doc: Doc) -> bool:
    """A comparative degree: *faster than*, *more robust than*, *as fast as*."""
    if any(t.tag_ in ("JJR", "RBR") for t in doc):
        return True
    return any(t.lower_ == "than" and t.dep_ in ("prep", "mark", "pcomp") for t in doc)


def is_conditional(doc: Doc) -> bool:
    """An *if* / *unless* clause is present."""
    return any(t.lower_ in _CONDITIONAL_MARKERS and t.dep_ == "mark" for t in doc)


def has_subordinate(doc: Doc) -> bool:
    return any(t.dep_ in _SUBORDINATE_DEPS for t in doc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if not Doc.has_extension("input_profile"):
    Doc.set_extension("input_profile", default=None)


def classify_input(doc: Doc) -> InputProfile:
    """Describe ``doc`` for the generators.  Computed once per Doc, then cached.

    The driver calls every generator's ``applies`` on every sentence, and nearly
    all of them consult the profile, so the cache is what keeps classification
    off the hot path.
    """
    cached = doc._.input_profile
    if cached is not None:
        return cached

    cues = existing_cues(doc)
    profile = InputProfile(
        clause_type=clause_type_of(doc),
        voice=voice_of(doc),
        polarity=POL_NEGATED if cues else POL_AFFIRMATIVE,
        existing_cues=cues,
        existing_count=len(cues),
        modality=modality_of(doc),
        clause_count=len(clause_heads(doc)),
        has_subordinate=has_subordinate(doc),
        is_existential=is_existential(doc),
        is_comparative=is_comparative(doc),
        is_conditional=is_conditional(doc),
        copula_type=copula_type_of(doc),
    )
    doc._.input_profile = profile
    return profile
