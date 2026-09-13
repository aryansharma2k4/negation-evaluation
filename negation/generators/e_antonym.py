"""E_antonym -- polarity reversal by WordNet antonym substitution.

Antonymy is not negation in the logical sense, but for a similarity model it is
the closest lexical neighbour that flips the sentence's truth conditions, so it
belongs in the graded battery.

Only direct ``Lemma.antonyms()`` links are followed -- no similar-to or
derivationally-related hops, which drift semantically fast.  The retrieved
antonym is a citation form, so it is re-inflected to fill exactly the
morphological slot the original token vacated (*"increases"* -> *"decreases"*,
*"increased"* -> *"decreased"*).

Where WordNet has no antonym at all, :mod:`negation.antonym_vec` can be asked
instead.  It is a **fallback, not a supplement**: it is consulted only for
tokens WordNet draws a blank on, never to add alternatives to a token WordNet
already answered, so it cannot displace or dilute a gold link.  Its records are
tagged ``generator="antonym_vec_v1"`` and carry a ``confidence``, because unlike
everything else in this package it comes from a model whose output is graded
rather than a rule that is either right or absent -- and, on the evidence in
``docs/antonym_vectors.md``, a model whose precision is low.  It stays off
unless ``NEGATION_ANTONYM_VEC=1`` is set.
"""

from __future__ import annotations

from spacy.tokens import Doc, Token

from ..base import Generator, register, scope_target_for
from ..lexicons import (
    ANTONYM_POS,
    ANTONYM_VERB_STOPLIST,
    ANTONYM_WORD_STOPLIST,
    LEXICALLY_NEGATIVE,
)
from ..nlp_core import antonyms, match_inflection, matches_case, wn_pos
from ..schema import FAM_ANTONYM, NegationVariant
from ..splice import Edit

#: Comparative/superlative adjectives and adverbs: the WordNet antonym is a
#: base form and re-inflecting it reliably is not worth the error rate.
_BLOCKED_TAGS = frozenset({"JJR", "JJS", "RBR", "RBS"})

#: Cap per token so a highly polysemous word cannot flood the output.
MAX_ANTONYMS_PER_TOKEN = 2

#: Name every vector-derived record carries, so they can be filtered downstream
#: without re-deriving which generator produced them.
VECTOR_GENERATOR = "antonym_vec_v1"


def eligible(token: Token) -> bool:
    if token.pos_ not in ANTONYM_POS or token.tag_ in _BLOCKED_TAGS:
        return False
    if not token.is_alpha or token.is_stop:
        return False
    if token.lower_ in ANTONYM_WORD_STOPLIST:
        return False
    # The antonym of an already-negative word is positive, so substituting it
    # would produce an affirmation rather than a negation.
    if token.lower_ in LEXICALLY_NEGATIVE or token.lemma_.lower() in LEXICALLY_NEGATIVE:
        return False
    if token.pos_ == "VERB" and token.lemma_.lower() in ANTONYM_VERB_STOPLIST:
        return False
    return True


@register
class AntonymSubstitution(Generator):
    """Swap a content word for a direct WordNet antonym, matching inflection.

    *"The score increased."* -> *"The score decreased."*
    *"The interface is simple."* -> *"The interface is complex."*
    """

    family = FAM_ANTONYM
    name = "antonym_wordnet_v1"

    def applies(self, doc: Doc) -> bool:
        return any(eligible(t) for t in doc)

    def generate(self, doc: Doc) -> list[NegationVariant]:
        out: list[NegationVariant] = []
        for token in doc:
            if not eligible(token):
                continue
            pos = wn_pos(token.pos_)
            if pos is None:
                continue
            found = antonyms(token.lemma_.lower(), pos)[:MAX_ANTONYMS_PER_TOKEN]
            for antonym in found:
                record = self._substitute(doc, token, antonym)
                if record:
                    out.append(record)
            if not found:
                record = self._vector_fallback(doc, token, pos)
                if record:
                    out.append(record)
        return out

    def _substitute(
        self,
        doc: Doc,
        token: Token,
        antonym: str,
        *,
        generator: str | None = None,
        confidence: float | None = None,
    ):
        """Splice ``antonym`` into ``token``'s slot, matching its inflection."""
        surface = matches_case(match_inflection(antonym, token), token)
        if surface.lower() == token.lower_:
            return None
        edit = Edit.replace(token.idx, token.idx + len(token.text), surface, cue=True)
        return self.build(
            doc,
            [edit],
            subtype=f"antonym_{token.pos_.lower()}",
            net_negation=1,
            scope_target=scope_target_for(token),
            generator=generator,
            confidence=confidence,
        )

    def _vector_fallback(self, doc: Doc, token: Token, pos: str):
        """Ask the embedding model, but only where WordNet said nothing.

        Imported lazily and failing soft: the model needs a vector file that is
        not vendored, and a generator in this pipeline must keep working without
        it.  ``generate_antonym`` returns ``None`` rather than a guess whenever
        it is unavailable, out of vocabulary, or below its confidence floor.
        """
        try:
            from ..antonym_vec.api import generate_antonym
        except ImportError:  # pragma: no cover - optional dependency
            return None
        suggestion = generate_antonym(token.lemma_.lower(), pos)
        if suggestion is None:
            return None
        word, confidence = suggestion
        return self._substitute(
            doc, token, word, generator=VECTOR_GENERATOR, confidence=confidence
        )
