"""I_intensified -- clausal negation with its force strengthened.

The mirror image of H: *surely / definitely / certainly / absolutely not*,
*not at all* and *by no means* keep the same truth conditions as plain *not*
but raise the speaker's commitment.  Records carry
``intensity_hint="intensified"`` and ``depth=2``.

*not at all* and *by no means* are post-auxiliary phrases, so they are declared
for the copular (and, for *by no means*, auxiliary) frames only -- with
do-support they would have to be postposed past the object, which this family
does not attempt.
"""

from __future__ import annotations

from ..base import register
from ..lexicons import INTENSIFIERS
from ..schema import FAM_INTENSIFIED, INT_INTENSIFIED
from ._modified import ModifiedNegation


@register
class IntensifiedNegation(ModifiedNegation):
    """Strengthen a clausal negation with an emphatic modifier.

    *"The task is simple."* -> *"The task is definitely not simple."*
                            -> *"The task is by no means simple."*
    """

    family = FAM_INTENSIFIED
    name = "intensify_wrap_v1"
    modifiers = INTENSIFIERS
    intensity = INT_INTENSIFIED
