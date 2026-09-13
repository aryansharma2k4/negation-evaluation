"""Driver: parse base sentences once, run the licensed generators, filter."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Collection, Iterable, Optional

from spacy.tokens import Doc

from . import generators  # noqa: F401  (import registers every generator)
from .base import registry
from .filters import FilterReport, run_filters
from .nlp_core import parse_batch, tree_depth
from .schema import NegationVariant


def make_base_id(sentence: str, index: int) -> str:
    """Stable, content-derived id so reruns produce the same ids."""
    digest = hashlib.sha1(sentence.strip().encode("utf-8")).hexdigest()[:10]
    return f"s{index:05d}_{digest}"


def parse_bases(sentences: Iterable[str]) -> list[Doc]:
    """Parse each base sentence exactly once and stamp it with its ``base_id``.

    Every generator reads this same :class:`~spacy.tokens.Doc`; nothing in the
    pipeline re-parses a base sentence.
    """
    cleaned = [s.strip() for s in sentences if s and s.strip()]
    docs = parse_batch(cleaned)
    for index, doc in enumerate(docs):
        doc._.base_id = make_base_id(doc.text, index)
    return docs


def generate_all(
    sentences: list[str],
    *,
    per_family_dedup: bool = True,
    report: Optional[FilterReport] = None,
    stages: Optional[Collection[int]] = None,
) -> list[NegationVariant]:
    """Generate every licensed, valid variant for ``sentences``.

    Each sentence is parsed once; the registry is consulted with the cheap
    ``applies`` predicate and only the licensed generators do any string work;
    the pooled candidates then go through :func:`negation.filters.run_filters`
    in one batched re-parse.

    ``stages`` restricts the registry to generators from the given layers:
    ``1`` is the original affirmative-declarative battery, ``2`` the
    arbitrary-input layer (affirmation, rescoping, and the non-declarative
    sentence types).  The default runs both.  Passing ``stages=(1,)`` reproduces
    stage 1's output exactly, which is what the regression test pins.
    """
    docs = parse_bases(sentences)
    base_depths = {doc._.base_id: tree_depth(doc) for doc in docs}
    licensed = [g for g in registry() if stages is None or g.stage in stages]

    candidates: list[NegationVariant] = []
    for doc in docs:
        for generator in licensed:
            if generator.applies(doc):
                candidates.extend(generator.generate(doc))

    kept, _ = run_filters(
        candidates, base_depths, per_family_dedup=per_family_dedup, report=report
    )
    return kept


def summarize(records: Iterable[NegationVariant]) -> dict[str, Counter]:
    """Counts by family, by family/subtype, and by net_negation."""
    by_family: Counter[str] = Counter()
    by_subtype: Counter[str] = Counter()
    by_net: Counter[int] = Counter()
    for record in records:
        by_family[record.family] += 1
        by_subtype[f"{record.family}/{record.subtype}"] += 1
        by_net[record.net_negation] += 1
    return {"family": by_family, "subtype": by_subtype, "net_negation": by_net}
