"""Public entry point: ``generate_antonym(word, pos, context=None)``.

Wired into ``E_antonym`` as a **fallback only**, for words where WordNet returns
nothing. It is off unless switched on, for three separate reasons, and all three
are about not degrading what already works:

* the model needs a ~1 GB vector file that is not vendored, so an unguarded
  import would break a clean checkout;
* loading it costs seconds and hundreds of megabytes, which no unit test should
  pay for;
* its precision is low (see ``docs/antonym_vectors.md``), so records it produces
  must be opt-in and separable downstream rather than silently mixed into the
  corpus.

Every record it does produce is tagged ``generator="antonym_vec_v1"`` and
carries a confidence, so it can be filtered or weighted without re-deriving
which records came from where.

The confidence is deliberately pessimistic. It starts from the cosine margin
between the top candidate and the runner-up, then applies a hard veto: a
candidate that WordNet lists as a **synonym** of the query -- or that shares a
synset with it -- is discarded outright rather than downweighted. Synonym return
is the characteristic failure mode of distributional antonymy, and a
near-synonym spliced in as a negation produces a variant that asserts the
opposite of what the record claims. WordNet having no *antonym* for a word does
not mean it has no *synonyms* for it, so this filter is available exactly when
the fallback fires.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from .embeddings import EmbeddingsNotFetched, load_static
from .methods import Candidates, LinearMap, Reflection

#: Set to "1" to allow the fallback to load. Off by default.
ENABLE_ENV = "NEGATION_ANTONYM_VEC"

#: Minimum confidence before a suggestion is returned at all.
#:
#: Set by judgement, not by calibration, and deliberately on the high side. No
#: threshold on this score separates right answers from wrong ones well enough
#: to be worth fitting -- see the reliability figures in
#: ``docs/antonym_vectors.md`` -- so this is a floor that suppresses the
#: obviously arbitrary suggestions rather than a tuned operating point. Treat a
#: suggestion that clears it as a candidate for review, not as a result.
MIN_CONFIDENCE = 0.15

#: Candidate pool size. Large enough to be a realistic retrieval problem, small
#: enough to keep the fallback's latency tolerable inside a generator.
POOL_SIZE = 50_000


@dataclass
class Prediction:
    """A suggested antonym and how much it should be trusted."""

    word: str
    confidence: float
    method: str

    def as_tuple(self) -> tuple[str, float]:
        return (self.word, self.confidence)


class AntonymVecModel:
    """A fitted map plus the candidate pool it retrieves from."""

    def __init__(self, source, method, candidates: Candidates) -> None:
        self.source = source
        self.method = method
        self.candidates = candidates

    def suggest(
        self, word: str, pos: str, context: Optional[str] = None, top: int = 10
    ) -> list[tuple[str, float]]:
        return self.method.suggest(word.lower(), pos, self.source, self.candidates, top=top)


def _wordnet_related(word: str, pos: str) -> set[str]:
    """Synonyms and same-synset lemmas of ``word`` -- the veto list.

    Not an approximation of "words close in meaning": it is exactly the set
    WordNet would call interchangeable, which is the set that must never be
    offered as an opposite.
    """
    from nltk.corpus import wordnet as wn

    tag = {"a": wn.ADJ, "v": wn.VERB, "r": wn.ADV, "n": wn.NOUN}.get(pos)
    related: set[str] = {word.lower()}
    try:
        synsets = wn.synsets(word, pos=tag)
    except Exception:  # pragma: no cover - corpus edge cases
        return related
    for synset in synsets:
        for lemma in synset.lemmas():
            related.add(lemma.name().lower())
        for similar in synset.similar_tos():
            for lemma in similar.lemmas():
                related.add(lemma.name().lower())
    return related


def _confidence(top_score: float, runner_up: float) -> float:
    """Cosine margin, squashed to ``[0, 1]``.

    The margin, not the raw cosine: in a space where everything sits in a narrow
    cone, an absolute cosine of 0.6 is unremarkable, while being 0.1 clear of the
    next candidate is a genuine signal that the prediction landed somewhere
    specific.
    """
    margin = max(0.0, top_score - runner_up)
    return float(min(1.0, margin * 4.0) * max(0.0, min(1.0, top_score)))


@lru_cache(maxsize=1)
def _load_model() -> Optional[AntonymVecModel]:
    """Build the model once, or return ``None`` if it cannot be built.

    Returning ``None`` rather than raising is deliberate: the caller is a
    generator in a pipeline that must keep working when the vectors are absent.
    """
    if os.environ.get(ENABLE_ENV) != "1":
        return None
    try:
        from .data import build_dataset

        dataset = build_dataset()
        lemmas = dataset.train_lemmas() | dataset.test_lemmas()
        source = load_static(restrict_to=lemmas, max_vocab=POOL_SIZE)

        words = [w for w in list(source.vocabulary())[:POOL_SIZE]]
        matrix = np.stack([source.vector(w, "a") for w in words])
        candidates = Candidates(tuple(words), matrix)

        # Fitted on training pairs only, so the deployed model is the one the
        # report measured rather than a stronger one fitted on everything.
        method = LinearMap(alpha=100.0)
        method.fit(dataset.train, source)
        if method.W is None:
            return None
        return AntonymVecModel(source, method, candidates)
    except (EmbeddingsNotFetched, ImportError, ValueError):
        return None


def is_available() -> bool:
    """Whether the fallback can currently produce anything."""
    return _load_model() is not None


def generate_antonym(
    word: str, pos: str, context: Optional[str] = None
) -> Optional[tuple[str, float]]:
    """Suggest an opposite for ``word``, or ``None``.

    ``pos`` is a WordNet tag (``a``, ``v``, ``r``). ``context`` is accepted for
    the contextual embedding source and ignored by the static one.

    Returns ``None`` -- not a guess -- whenever the model is unavailable, the
    word is out of vocabulary, every candidate is vetoed as a synonym, or the
    confidence falls below :data:`MIN_CONFIDENCE`. A generator that cannot tell
    a real suggestion from a filler one would put unverifiable records into the
    corpus, which is worse than producing nothing.
    """
    model = _load_model()
    if model is None:
        return None

    ranked = model.suggest(word, pos, context, top=10)
    if not ranked:
        return None

    vetoed = _wordnet_related(word, pos)
    surviving = [(w, s) for w, s in ranked if w not in vetoed and w.isalpha()]
    if not surviving:
        return None

    best_word, best_score = surviving[0]
    runner_up = surviving[1][1] if len(surviving) > 1 else 0.0
    confidence = _confidence(best_score, runner_up)
    if confidence < MIN_CONFIDENCE:
        return None
    return Prediction(best_word, round(confidence, 4), "antonym_vec_v1").as_tuple()


def reset_cache() -> None:
    """Drop the loaded model. For tests that toggle the enable flag."""
    _load_model.cache_clear()
