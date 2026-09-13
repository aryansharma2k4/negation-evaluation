"""Affixal negation morphology, in both directions.

``derived_forms`` builds a negative derivative from a positive stem
(*important* -> *unimportant*); ``negative_base_forms`` runs the same machinery
backwards to recover the stem from a derivative (*unimportant* -> *important*).
Generation needs the first, input classification and affirmation need the
second, and both need the same WordNet gating -- so the logic lives here, below
the generator layer, rather than inside ``D_affixal``.

English prefix selection is lexically arbitrary (*unhappy* but *impossible* but
*irregular*), so candidacy is never assumed.  A pairing is admitted only if it
clears three gates:

1. a cheap onset constraint (``im-`` only before labials, ``ir-`` before ``r``,
   ``il-`` before ``l``), which prunes impossible allomorphs without a lookup;
2. both forms exist in WordNet -- this kills ``*unpossible`` / ``*inhappy``;
3. WordNet records an **antonymy** link between stem and derivative, in either
   direction.  Attestation alone is not enough: ``dis`` + ``array`` and ``in`` +
   ``tense`` are both real words that are not negations of their stems, and only
   the antonymy gate rejects them.

Plain ``noun + -less`` is deliberately not attempted: the derivative is an
adjective, and splicing it into the noun's slot would produce an ungrammatical
sentence.  Only the ``-ful``/``-less`` alternation keeps the part of speech
intact.
"""

from __future__ import annotations

from spacy.tokens import Token

from .lexicons import (
    LEXICALLY_NEGATIVE,
    NEGATIVE_PREFIXES,
    PREFIX_ONSET_CONSTRAINTS,
)
from .nlp_core import antonyms, in_wordnet, wn_pos

_ELIGIBLE_POS = frozenset({"ADJ", "NOUN"})
#: Comparatives/superlatives do not take negative prefixes cleanly.
_BLOCKED_TAGS = frozenset({"JJR", "JJS"})

#: Shortest stem a negative prefix may be stripped back to.  Guards against
#: reading *nonce* as ``non`` + ``ce``.
MIN_STEM = 3


def eligible(token: Token) -> bool:
    """Cheap check: a base-form adjective or noun that is not already negative."""
    return (
        token.pos_ in _ELIGIBLE_POS
        and token.tag_ not in _BLOCKED_TAGS
        and token.is_alpha
        and len(token.text) >= 4
        and token.lower_ not in LEXICALLY_NEGATIVE
        and token.lemma_.lower() not in LEXICALLY_NEGATIVE
    )


def is_negative_derivative(stem: str, candidate: str, pos: str | None) -> bool:
    """True when WordNet links ``stem`` and ``candidate`` as antonyms.

    Checked in both directions because WordNet's antonym edges are not always
    symmetric across the pair (*linear* has no antonym, but *nonlinear* points
    back at it).
    """
    if pos is None:
        return False
    if not in_wordnet(candidate, pos):
        return False
    lowered = {a.lower() for a in antonyms(stem, pos)}
    if candidate in lowered:
        return True
    return stem in {a.lower() for a in antonyms(candidate, pos)}


def derived_forms(token: Token) -> list[tuple[str, str]]:
    """WordNet-validated negative derivations of ``token`` as ``(form, subtype)``."""
    stem = token.lemma_.lower() if token.pos_ == "NOUN" else token.lower_
    pos = wn_pos(token.pos_)
    out: list[tuple[str, str]] = []

    for prefix in NEGATIVE_PREFIXES:
        onset = PREFIX_ONSET_CONSTRAINTS.get(prefix)
        if onset is not None and stem[:1] not in onset:
            continue
        if stem.startswith(prefix):
            continue  # already carries this prefix
        candidate = prefix + stem
        if is_negative_derivative(stem, candidate, pos):
            out.append((candidate, f"prefix_{prefix}"))

    if token.pos_ == "ADJ" and stem.endswith("ful") and len(stem) > 4:
        candidate = stem[:-3] + "less"
        if is_negative_derivative(stem, candidate, pos):
            out.append((candidate, "suffix_less"))

    return out


def negative_base_forms(token: Token) -> list[tuple[str, str]]:
    """Positive stems ``token`` negates, as ``(stem, subtype)``.

    The inverse of :func:`derived_forms`: strip each negative prefix (or undo
    the ``-ful``/``-less`` alternation) and keep only the stems WordNet both
    attests and links antonymously back to the token.  That gate is what keeps
    *income* from being read as ``in`` + *come* and *nonce* from being read as
    ``non`` + *ce*.
    """
    surface = token.lemma_.lower() if token.pos_ == "NOUN" else token.lower_
    pos = wn_pos(token.pos_)
    if pos is None or not token.is_alpha:
        return []
    out: list[tuple[str, str]] = []

    for prefix in NEGATIVE_PREFIXES:
        if not surface.startswith(prefix):
            continue
        stem = surface[len(prefix) :]
        if len(stem) < MIN_STEM:
            continue
        onset = PREFIX_ONSET_CONSTRAINTS.get(prefix)
        if onset is not None and stem[:1] not in onset:
            continue
        if is_negative_derivative(stem, surface, pos):
            out.append((stem, f"prefix_{prefix}"))

    if token.pos_ == "ADJ" and surface.endswith("less") and len(surface) > 5:
        stem = surface[:-4] + "ful"
        if is_negative_derivative(stem, surface, pos):
            out.append((stem, "suffix_less"))

    return out


def is_affixal_negative(token: Token) -> bool:
    """True when ``token`` is a WordNet-attested affixal negation of some stem."""
    return bool(negative_base_forms(token))
