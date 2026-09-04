"""H_hedged -- clausal negation with its force weakened.

*maybe / possibly / probably not* and *might / may not* assert the negation
under an epistemic operator, so the variant is negative but non-committal.
Records carry ``intensity_hint="hedged"`` and ``depth=2``: the hedge composes
over an A_syntactic negation and nothing may stack on top of it.
"""

from __future__ import annotations

from ..base import register
from ..lexicons import HEDGES
from ..schema import FAM_HEDGED, INT_HEDGED
from ._modified import ModifiedNegation


@register
class HedgedNegation(ModifiedNegation):
    """Weaken a clausal negation with an epistemic hedge.

    *"The function sorts the array."*
      -> *"The function probably does not sort the array."*  (adverb slot)
      -> *"The function might not sort the array."*           (modal slot)
    """

    family = FAM_HEDGED
    name = "hedge_wrap_v1"
    modifiers = HEDGES
    intensity = INT_HEDGED
