"""Map a corpus's gold cue onto one of our 13 families.

The corpora label cues, not families: CD-SCO says *this token is a cue*, SFU
says *this token is a negation cue or a speculation cue*, and neither commits to
how the negation is realised.  Our taxonomy does, so mapping is what makes a
cue-family distribution over external data comparable with one over our
generated corpus.

The mapping reuses :mod:`negation.lexicons` and follows the same priority order
as :func:`negation.classify._cue_family`, for the same reasons: *never* is a
negative adverb before it is a quantifier, *no* is a quantifier even where a
parser calls it ``neg``, an implicit trigger outranks an affixal reading.  What
it adds is the two cases a running parser never sees.

First, a cue that is a *part of* its token.  CD-SCO annotates the affix alone
(cue ``in`` on *infrequent*), so a cue shorter than the token it sits on is
affixal by construction, and that check comes first.

Second, multiword cues, which are common and which the corpora tokenise in ways
a lexicon lookup on the first word will miss: SFU splits contractions (*do n't*,
*ca n't*), BioScope writes *can not* apart, and CD-SCO has phrases like *by no
means*.  So a phrase that names a family outright is matched whole, and
otherwise every word of the cue is scanned rather than only its head.

Families with no lexical signature stay unreachable here, and that is correct.
E_antonym is a relation between a variant and its base rather than a property of
a cue; K1/K2 are properties of a *pair* of cues, computed in the statistics
script; L_scope_position is a property of where a cue sits relative to a
quantifier, which :func:`scope_position_family` handles separately.
"""

from __future__ import annotations

from typing import Optional

from ..lexicons import (
    CONTRASTIVE_FRAMES,
    IMPLICIT_TRIGGER_LEMMAS,
    IMPLICIT_TRIGGER_PREDICATES,
    INTENSIFIERS,
    LEXICALLY_NEGATIVE,
    NEGATIVE_ADVERBS,
    NEGATIVE_AUX_BASES,
    NEGATIVE_PREFIXES,
    NEGATIVE_QUANTIFIERS,
    PRIVATIVE_HEADS,
    SCOPE_BEARING_QUANTIFIERS,
)
from ..schema import (
    FAM_AFFIXAL,
    FAM_CONTRASTIVE,
    FAM_HEDGED,
    FAM_IMPLICIT,
    FAM_INTENSIFIED,
    FAM_NEG_ADVERB,
    FAM_PREPOSITIONAL,
    FAM_QUANTIFIER,
    FAM_SCOPE_POSITION,
    FAM_SYNTACTIC,
)
from .schema import ANN_SPECULATION, UNMAPPED

#: Plain clausal negators.  ``n't`` appears split, joined and apostrophe-typed
#: across the corpora, so every spelling seen in the data is listed.
_SYNTACTIC_CUES: frozenset[str] = frozenset(
    {"not", "n't", "nt", "cannot", "cant", "dont", "doesnt", "didnt", "isnt",
     "wasnt", "arent", "werent", "wont", "cant", "couldnt", "wouldnt",
     "shouldnt", "havent", "hasnt", "hadnt"}
) | frozenset(NEGATIVE_AUX_BASES)

#: Suffix that negates without a prefix.
_NEGATIVE_SUFFIX = "less"

#: Multiword cues that name a whole family on their own.  Checked before the
#: word-by-word scan, because *by no means* would otherwise be read as the
#: quantifier *no* rather than as the emphatic phrase it is.
_PHRASE_FAMILIES: dict[str, str] = {
    **{m.text: FAM_INTENSIFIED for m in INTENSIFIERS if " " in m.text},
    **{phrase: FAM_CONTRASTIVE for phrase, _ in CONTRASTIVE_FRAMES},
}

#: Characters the corpora use for an apostrophe.  The SFU XML carries raw
#: Windows-1252 bytes for a curly apostrophe, which survive parsing as U+0092
#: and would otherwise leave that spelling of *don't* unmatched against *n't*.
_APOSTROPHES = "’‘´`"


def normalise_cue(cue_text: str) -> str:
    """Lowercase, collapse whitespace, and regularise apostrophes."""
    text = cue_text.lower().strip()
    for char in _APOSTROPHES:
        text = text.replace(char, "'")
    return " ".join(text.split())


def _is_affixal(cue_text: str, token_text: str) -> bool:
    """True when the cue is an affix of its token rather than a whole word.

    This is how CD-SCO records *unhappy*: the cue is ``un`` and the token is
    ``unhappy``.  A cue equal to its token is a free morpheme and not affixal,
    whatever it spells.
    """
    cue, token = normalise_cue(cue_text), normalise_cue(token_text)
    if not cue or not token or cue == token:
        return False
    if len(cue) >= len(token):
        return False
    if not token.startswith(cue) and not token.endswith(cue):
        return False
    return cue in NEGATIVE_PREFIXES or cue == _NEGATIVE_SUFFIX


def _implicit(word: str) -> bool:
    """An implicit trigger, in any inflection the corpora show."""
    if word in IMPLICIT_TRIGGER_LEMMAS or word in IMPLICIT_TRIGGER_PREDICATES:
        return True
    return any(
        word.startswith(lemma)
        and word[len(lemma) :] in ("s", "d", "ed", "es", "ing", "ure")
        for lemma in IMPLICIT_TRIGGER_LEMMAS
    )


def _looks_affixally_negative(word: str) -> bool:
    """A lexical negative whose negativity is carried by a prefix or suffix.

    Used only for cues the corpora give as a whole word (*impossible*), where
    there is no separate affix annotation to key off.  Deliberately shape-based
    rather than WordNet-gated: the word is already known to be a negation cue
    because the corpus annotated it as one, so the only open question is which
    family realises it.
    """
    if word.endswith(_NEGATIVE_SUFFIX) and len(word) > len(_NEGATIVE_SUFFIX) + 2:
        return True
    return any(
        word.startswith(prefix) and len(word) - len(prefix) >= 3
        for prefix in NEGATIVE_PREFIXES
    )


def cue_family(
    cue_text: str,
    token_text: Optional[str] = None,
    *,
    annotation_type: str = "negation",
) -> str:
    """The family that would have produced ``cue_text``, or ``"unmapped"``.

    ``token_text`` is the token the cue sits on, needed only to recognise an
    affixal cue.  ``annotation_type`` routes a speculation cue to ``H_hedged``:
    that family is exactly "negation with its force weakened", which is what a
    hedge cue marks, and it is the reason the hedging corpora are worth having.
    """
    surface = normalise_cue(cue_text)
    if not surface:
        return UNMAPPED

    if annotation_type == ANN_SPECULATION:
        return FAM_HEDGED

    if token_text is not None and _is_affixal(surface, token_text):
        return FAM_AFFIXAL

    if surface in _PHRASE_FAMILIES:
        return _PHRASE_FAMILIES[surface]
    if surface in ("no one", "no-one"):
        return FAM_QUANTIFIER

    words = surface.split()

    # Priority mirrors negation.classify._cue_family, applied across every word
    # of the cue rather than only its first.
    for test, family in (
        (lambda w: w in NEGATIVE_ADVERBS, FAM_NEG_ADVERB),
        (lambda w: w in NEGATIVE_QUANTIFIERS, FAM_QUANTIFIER),
        (lambda w: w in _SYNTACTIC_CUES, FAM_SYNTACTIC),
        (lambda w: w in PRIVATIVE_HEADS, FAM_PREPOSITIONAL),
        (_implicit, FAM_IMPLICIT),
    ):
        if any(test(word) for word in words):
            return family

    # A single-word lexical negative none of the lists above covers: decide
    # between an affixal realisation and a plain negative content word.
    if len(words) == 1 and words[0] in LEXICALLY_NEGATIVE:
        return FAM_AFFIXAL if _looks_affixally_negative(words[0]) else FAM_IMPLICIT

    return UNMAPPED


def scope_position_family(
    cue_family_label: str, tokens: list[str], cue_indices: list[int]
) -> str:
    """Relabel a cue as ``L_scope_position`` when it preposes a quantifier.

    *"Not everyone passed"* is a syntactic cue by its lexicon entry, but what
    makes it interesting is that it sits in front of a scope-bearing quantifier
    and so has a distinct reading from *"Everyone did not pass"*.  Only applied
    where the corpus's own cue is syntactic, so nothing else is relabelled.
    """
    if cue_family_label != FAM_SYNTACTIC or not cue_indices:
        return cue_family_label
    nxt = max(cue_indices) + 1
    if nxt < len(tokens) and tokens[nxt].lower() in SCOPE_BEARING_QUANTIFIERS:
        return FAM_SCOPE_POSITION
    return cue_family_label
