"""B_quantifier -- swap a positive quantifier for its negative counterpart.

Quantifier negation is lexical, not syntactic: the negation rides on the
determiner or indefinite pronoun itself (*some* -> *no*, *always* -> *never*),
so no auxiliary has to be introduced.  The swap is only ever applied to a token
whose dependency label shows it really is functioning as a quantifier
(determiner, subject/object pronoun, or adverbial modifier), never to a
homograph in some other role.
"""

from __future__ import annotations

from spacy.tokens import Doc, Token

from ..base import Generator, register, scope_target_for
from ..lexicons import QUANTIFIER_KEYS, QUANTIFIER_SWAPS
from ..schema import FAM_QUANTIFIER, NegationVariant
from ..splice import Edit, lower_initial

#: Dependency labels in which a quantifier word is genuinely quantifying.
_QUANT_DEPS = frozenset(
    {"det", "predet", "nsubj", "nsubjpass", "dobj", "pobj", "advmod", "attr", "npadvmod", "nummod"}
)


def _is_quantifier(token: Token) -> bool:
    return token.lower_ in QUANTIFIER_KEYS and token.dep_ in _QUANT_DEPS


@register
class QuantifierSwap(Generator):
    """Replace a positive quantifier with its negative counterpart.

    *"Some students solved the problem."* -> *"No students solved the problem."*
    *"He always arrives early."*          -> *"He never arrives early."*
    *"Both tests passed."*                -> *"Neither test passed."* (surface
    number is left to the parse-validity filter; only the determiner is touched).

    ``all`` maps to the partitive *none of the*, which subsumes any determiner
    that already follows it so *"all the students"* does not become
    *"none of the the students"*.
    """

    family = FAM_QUANTIFIER
    name = "quant_swap_v1"

    def applies(self, doc: Doc) -> bool:
        return any(_is_quantifier(t) for t in doc)

    def generate(self, doc: Doc) -> list[NegationVariant]:
        out: list[NegationVariant] = []
        for token in doc:
            if not _is_quantifier(token):
                continue
            swap = QUANTIFIER_SWAPS[token.lower_]
            end = token.idx + len(token.text)
            if swap.absorbs_following_det:
                nxt = doc[token.i + 1] if token.i + 1 < len(doc) else None
                if nxt is not None and nxt.dep_ == "det":
                    # A determiner is already there; emit the bare partitive so
                    # "all the students" -> "none of the students".
                    replacement = swap.replacement[: swap.replacement.rfind(" ")]
                else:
                    replacement = swap.replacement
            else:
                replacement = swap.replacement
            edit = Edit.replace(token.idx, end, lower_initial(replacement), cue=True)
            record = self.build(
                doc,
                [edit],
                subtype=swap.subtype,
                net_negation=1,
                scope_target=scope_target_for(token),
            )
            if record:
                out.append(record)
        return out
