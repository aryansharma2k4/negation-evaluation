"""D_affixal -- negation morphologised onto a content word.

Two mechanisms:

*prefixation*  ``un-``, ``in-``, ``im-``, ``ir-``, ``il-``, ``non-``, ``dis-``
               on an adjective or noun (*important* -> *unimportant*).
*suffixation*  ``-ful`` -> ``-less`` on an adjective (*useful* -> *useless*).

English prefix selection is lexically arbitrary (*unhappy* but *impossible* but
*irregular*), so candidacy is never assumed.  A derived form is emitted only if
it clears three gates:

1. a cheap onset constraint (``im-`` only before labials, ``ir-`` before ``r``,
   ``il-`` before ``l``), which prunes impossible allomorphs without a lookup;
2. the form exists in WordNet -- this kills ``*unpossible`` / ``*inhappy``;
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

from spacy.tokens import Doc, Token

from ..base import Generator, register, scope_target_for
from ..lexicons import (
    LEXICALLY_NEGATIVE,
    NEGATIVE_PREFIXES,
    PREFIX_ONSET_CONSTRAINTS,
)
from ..nlp_core import antonyms, in_wordnet, matches_case, wn_pos
from ..schema import FAM_AFFIXAL, NegationVariant
from ..splice import Edit

_ELIGIBLE_POS = frozenset({"ADJ", "NOUN"})
#: Comparatives/superlatives do not take negative prefixes cleanly.
_BLOCKED_TAGS = frozenset({"JJR", "JJS"})


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


def _is_negative_derivative(stem: str, candidate: str, pos: str | None) -> bool:
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
        if _is_negative_derivative(stem, candidate, pos):
            out.append((candidate, f"prefix_{prefix}"))

    if token.pos_ == "ADJ" and stem.endswith("ful") and len(stem) > 4:
        candidate = stem[:-3] + "less"
        if _is_negative_derivative(stem, candidate, pos):
            out.append((candidate, "suffix_less"))

    return out


@register
class AffixalNegation(Generator):
    """Replace a content word with its WordNet-attested negative derivative.

    *"The result is important."* -> *"The result is unimportant."*
    *"The check is useful."*     -> *"The check is useless."*
    """

    family = FAM_AFFIXAL
    name = "affix_derive_v1"

    def applies(self, doc: Doc) -> bool:
        return any(eligible(t) for t in doc)

    def generate(self, doc: Doc) -> list[NegationVariant]:
        out: list[NegationVariant] = []
        for token in doc:
            if not eligible(token):
                continue
            for form, subtype in derived_forms(token):
                edit = Edit.replace(
                    token.idx,
                    token.idx + len(token.text),
                    matches_case(form, token),
                    cue=True,
                )
                record = self.build(
                    doc,
                    [edit],
                    subtype=subtype,
                    net_negation=1,
                    scope_target=scope_target_for(token),
                )
                if record:
                    out.append(record)
        return out
