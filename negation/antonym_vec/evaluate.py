"""Metrics, with synonym contamination treated as a headline number.

Precision@k answers "did the right word come back". It does not answer the
question that actually decides whether this module is usable, which is *what
came back instead* -- because the characteristic failure of distributional
antonymy is returning a **synonym**. *hot* and *warm* share contexts, so a
model asked for the opposite of *hot* will happily return *warm*, and under
precision alone that miss is indistinguishable from returning an unrelated word.
One is a near-useless answer; the other is an actively wrong one that would
generate a false negation. :class:`Result` therefore carries
``synonym_at_1`` and ``synonym_at_10`` next to the precisions, and nothing in
this module reports one without the other.

Everything is broken out by POS and by whether the gold antonym is morphological
or suppletive. That split is not cosmetic: *happy/unhappy* is recoverable by
string edit, so a method that only learns the affixal cases can post a
respectable pooled average while being useless for exactly the cases WordNet
already covers badly. A single headline average hides it, so no single headline
average is printed.

Two coverage numbers are kept apart, because conflating them is how this kind of
module gets oversold:

*test coverage*   fraction of test items where the method returns anything. For
                  WordNet this is ~1.0 by construction -- the test set was
                  *drawn from* WordNet -- and means nothing.
*vocabulary coverage*  fraction of a general word sample where the method fires.
                  This is the real question for the integration, since the
                  fallback exists precisely for words WordNet does not cover.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from .data import MORPHOLOGICAL, POS_NAMES, SUPPLETIVE, Dataset, Pair
from .methods import Candidates

#: Ranks beyond this are recorded as "not found" rather than as a large number,
#: so a handful of catastrophic misses cannot dominate a mean rank.
RANK_CEILING = 10_000


@dataclass
class Result:
    """Scores for one method on one slice of the test set."""

    method: str
    slice_name: str
    n: int = 0
    hits_at_1: int = 0
    hits_at_5: int = 0
    hits_at_10: int = 0
    #: Items where a *synonym* of the query was returned instead.
    synonym_hits_at_1: int = 0
    synonym_hits_at_10: int = 0
    #: Items where the method returned anything at all.
    fired: int = 0
    ranks: list[int] = field(default_factory=list)
    examples: list[tuple[str, str, list[str]]] = field(default_factory=list)

    def _rate(self, count: int) -> float:
        return count / self.n if self.n else 0.0

    @property
    def p_at_1(self) -> float:
        return self._rate(self.hits_at_1)

    @property
    def p_at_5(self) -> float:
        return self._rate(self.hits_at_5)

    @property
    def p_at_10(self) -> float:
        return self._rate(self.hits_at_10)

    @property
    def synonym_at_1(self) -> float:
        return self._rate(self.synonym_hits_at_1)

    @property
    def synonym_at_10(self) -> float:
        return self._rate(self.synonym_hits_at_10)

    @property
    def coverage(self) -> float:
        return self._rate(self.fired)

    @property
    def mean_rank(self) -> Optional[float]:
        """Mean rank of the gold antonym over items where it was rankable."""
        return float(np.mean(self.ranks)) if self.ranks else None

    @property
    def median_rank(self) -> Optional[float]:
        return float(np.median(self.ranks)) if self.ranks else None

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "slice": self.slice_name,
            "n": self.n,
            "p@1": round(self.p_at_1, 4),
            "p@5": round(self.p_at_5, 4),
            "p@10": round(self.p_at_10, 4),
            "synonym@1": round(self.synonym_at_1, 4),
            "synonym@10": round(self.synonym_at_10, 4),
            "coverage": round(self.coverage, 4),
            "mean_rank": None if self.mean_rank is None else round(self.mean_rank, 1),
            "median_rank": None if self.median_rank is None else round(self.median_rank, 1),
        }


def gold_antonyms(pairs: Iterable[Pair]) -> dict[tuple[str, str], set[str]]:
    """``(word, pos) -> every antonym recorded for it``.

    Credit is given for *any* gold antonym, not only the one that happened to be
    in the evaluated row: *good* is antonymous with *bad*, *evil* and *ill*, and
    scoring *evil* as wrong because the row said *bad* would measure the test
    set's arbitrary ordering rather than the method.
    """
    out: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pair in pairs:
        out[(pair.word, pair.pos)].add(pair.antonym)
    return out


def synonym_index(dataset: Dataset) -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pair in dataset.synonyms:
        out[(pair.word, pair.pos)].add(pair.antonym)
    return out


def evaluate_method(
    method,
    test_pairs: Sequence[Pair],
    source,
    candidates: Candidates,
    *,
    all_gold: dict[tuple[str, str], set[str]],
    synonyms: dict[tuple[str, str], set[str]],
    slice_name: str = "all",
    keep_examples: int = 8,
) -> Result:
    """Score ``method`` over ``test_pairs``.

    One evaluation row per *query word*, not per pair, so a word with three
    antonyms is not counted three times and does not dominate the average.
    """
    result = Result(method=method.name, slice_name=slice_name)
    seen_queries: set[tuple[str, str]] = set()

    for pair in test_pairs:
        query = (pair.word, pair.pos)
        if query in seen_queries:
            continue
        seen_queries.add(query)
        result.n += 1

        gold = all_gold.get(query, {pair.antonym})
        near_synonyms = synonyms.get(query, set())
        ranked = method.suggest(pair.word, pair.pos, source, candidates, top=10)
        if not ranked:
            continue
        result.fired += 1

        words = [w for w, _ in ranked]
        if words and words[0] in gold:
            result.hits_at_1 += 1
        if any(w in gold for w in words[:5]):
            result.hits_at_5 += 1
        if any(w in gold for w in words[:10]):
            result.hits_at_10 += 1

        # Contamination: a synonym returned where the gold antonym was not.
        if words and words[0] in near_synonyms and words[0] not in gold:
            result.synonym_hits_at_1 += 1
        if any(w in near_synonyms and w not in gold for w in words[:10]):
            result.synonym_hits_at_10 += 1

        rank = _best_rank(method, pair, source, candidates, gold)
        if rank is not None:
            result.ranks.append(min(rank, RANK_CEILING))

        if len(result.examples) < keep_examples:
            result.examples.append((pair.word, sorted(gold)[0], words[:5]))
    return result


def _best_rank(method, pair, source, candidates, gold) -> Optional[int]:
    """Rank of the best-placed gold antonym, or ``None`` if unrankable."""
    predicted = getattr(method, "predict", lambda *a: None)(pair.word, pair.pos, source)
    if predicted is None:
        # Word-returning methods (WordNet) have no vector to rank against; their
        # rank is 1 when they produce a gold answer at all.
        ranked = method.suggest(pair.word, pair.pos, source, candidates, top=10)
        for position, (word, _) in enumerate(ranked, start=1):
            if word in gold:
                return position
        return None
    ranks = [
        r
        for r in (candidates.rank_of(predicted, g, exclude=(pair.word,)) for g in gold)
        if r is not None
    ]
    return min(ranks) if ranks else None


def slice_pairs(pairs: Sequence[Pair]) -> dict[str, list[Pair]]:
    """The reporting slices: overall, per POS, and per formation type."""
    out: dict[str, list[Pair]] = {"all": list(pairs)}
    for pos in sorted({p.pos for p in pairs}):
        out[f"pos={POS_NAMES.get(pos, pos)}"] = [p for p in pairs if p.pos == pos]
    for relation in (MORPHOLOGICAL, SUPPLETIVE):
        subset = [p for p in pairs if p.relation == relation]
        if subset:
            out[f"form={relation}"] = subset
    return out


def vocabulary_coverage(
    method, words: Sequence[str], pos: str, source, candidates: Candidates
) -> float:
    """Fraction of ``words`` for which ``method`` returns anything.

    The number that matters for the integration: the fallback exists for words
    WordNet does not cover, so what counts is how often each method fires on
    general vocabulary rather than on a test set drawn from WordNet itself.
    """
    if not words:
        return 0.0
    fired = sum(
        1 for word in words if method.suggest(word, pos, source, candidates, top=1)
    )
    return fired / len(words)


def format_table(results: Sequence[Result]) -> str:
    """A fixed-width comparison table, contamination beside precision."""
    header = (
        f"{'method':<20} {'slice':<22} {'n':>5} {'P@1':>7} {'P@5':>7} {'P@10':>7} "
        f"{'syn@1':>7} {'syn@10':>7} {'cover':>7} {'medRank':>8}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        median = result.median_rank
        lines.append(
            f"{result.method:<20} {result.slice_name:<22} {result.n:>5} "
            f"{result.p_at_1:>7.3f} {result.p_at_5:>7.3f} {result.p_at_10:>7.3f} "
            f"{result.synonym_at_1:>7.3f} {result.synonym_at_10:>7.3f} "
            f"{result.coverage:>7.3f} "
            f"{'-' if median is None else f'{median:>8.0f}'}"
        )
    return "\n".join(lines)
