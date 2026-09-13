"""D_affixal -- negation morphologised onto a content word.

Two mechanisms:

*prefixation*  ``un-``, ``in-``, ``im-``, ``ir-``, ``il-``, ``non-``, ``dis-``
               on an adjective or noun (*important* -> *unimportant*).
*suffixation*  ``-ful`` -> ``-less`` on an adjective (*useful* -> *useless*).

The derivation itself -- and the WordNet gating that keeps ``*unpossible`` and
``dis`` + ``array`` out -- lives in :mod:`negation.affixes`, because input
classification and the affirmation family need to run the same machinery
backwards.  This module is only the generator wrapper around it.
"""

from __future__ import annotations

from spacy.tokens import Doc

from ..affixes import derived_forms, eligible  # noqa: F401  (re-exported)
from ..base import Generator, register, scope_target_for
from ..nlp_core import matches_case
from ..schema import FAM_AFFIXAL, NegationVariant
from ..splice import Edit


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
