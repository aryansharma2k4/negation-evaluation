"""Validity filters, ordered cheapest-first.

Every generated record runs this gauntlet before it reaches the output:

1. **dedup** -- a normalised hash (lowercase, whitespace collapsed, punctuation
   stripped).  Pure string work, so it runs first and removes the bulk.
2. **blacklist** -- precompiled regexes for surface patterns that are known to
   be malformed (*"not not"*, stacked auxiliaries, doubled determiners).  Still
   no parsing.
3. **parse validity** -- the survivors are re-parsed *in one batched pipe pass*
   and rejected if the parse has more than one ROOT, contains an unreachable
   head (a cycle), or its tree depth differs from the base's by more than
   :data:`MAX_DEPTH_DELTA`.

The ordering matters: step 3 is the only expensive one, and it only ever sees
records that already passed 1 and 2.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

from spacy.tokens import Doc

from .lexicons import BLACKLIST_PATTERNS
from .nlp_core import parse_batch, roots, tree_depth
from .schema import NegationVariant

#: Maximum allowed difference in parse-tree depth between variant and base.
MAX_DEPTH_DELTA = 3

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace -- the dedup key."""
    folded = unicodedata.normalize("NFKC", text).lower()
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", folded)).strip()


def blacklist_hit(text: str) -> Optional[str]:
    """Name of the first blacklist pattern ``text`` matches, if any."""
    for name, pattern in BLACKLIST_PATTERNS:
        if pattern.search(text):
            return name
    return None


@dataclass
class FilterReport:
    """Why records were dropped, for the CLI summary."""

    kept: int = 0
    dropped: Counter[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.dropped is None:
            self.dropped = Counter()

    def drop(self, reason: str) -> None:
        self.dropped[reason] += 1


def dedup(
    records: Iterable[NegationVariant],
    report: FilterReport,
    *,
    per_family: bool = True,
) -> list[NegationVariant]:
    """Drop repeated variants.

    ``per_family=True`` keys on ``(base_id, normalised text, family)``: when two
    families derive the same surface string they are two different linguistic
    analyses of it and both labels are worth keeping (D_affixal and E_antonym
    both reach *"is unimportant"*, by different routes).  ``per_family=False``
    keys on ``(base_id, normalised text)`` alone for a strictly unique corpus.
    """
    seen: set[tuple[str, ...]] = set()
    out: list[NegationVariant] = []
    for record in records:
        norm = normalize(record.variant)
        if not norm or norm == normalize(record.base_sentence):
            report.drop("identical_to_base")
            continue
        key = (record.base_id, norm, record.family) if per_family else (record.base_id, norm)
        if key in seen:
            report.drop("duplicate")
            continue
        seen.add(key)
        out.append(record)
    return out


def apply_blacklist(
    records: Iterable[NegationVariant], report: FilterReport
) -> list[NegationVariant]:
    """Drop records matching a known-bad surface pattern."""
    out: list[NegationVariant] = []
    for record in records:
        hit = blacklist_hit(record.variant)
        if hit is not None:
            report.drop(f"blacklist:{hit}")
            continue
        out.append(record)
    return out


def parse_is_valid(variant_doc: Doc, base_depth: int) -> Optional[str]:
    """Reason the variant's parse is unacceptable, or ``None`` if it is fine."""
    if len(roots(variant_doc)) != 1:
        return "multiple_roots"
    depth = tree_depth(variant_doc)
    if depth < 0:
        return "unreachable_head"
    if abs(depth - base_depth) > MAX_DEPTH_DELTA:
        return "depth_delta"
    return None


def apply_parse_check(
    records: list[NegationVariant],
    base_depths: dict[str, int],
    report: FilterReport,
) -> list[NegationVariant]:
    """Re-parse every candidate in one batched pass and keep the well-formed ones."""
    if not records:
        return []
    docs = parse_batch(r.variant for r in records)
    out: list[NegationVariant] = []
    for record, doc in zip(records, docs):
        reason = parse_is_valid(doc, base_depths.get(record.base_id, 0))
        if reason is not None:
            report.drop(f"parse:{reason}")
            continue
        out.append(record)
    return out


def run_filters(
    records: list[NegationVariant],
    base_depths: dict[str, int],
    *,
    per_family_dedup: bool = True,
    report: Optional[FilterReport] = None,
) -> tuple[list[NegationVariant], FilterReport]:
    """Run all three filter stages in cost order."""
    report = report or FilterReport()
    records = dedup(records, report, per_family=per_family_dedup)
    records = apply_blacklist(records, report)
    records = apply_parse_check(records, base_depths, report)
    report.kept = len(records)
    return records, report
