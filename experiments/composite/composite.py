"""Composite negation detector: content gate -> cancellation veto -> NLI.

Nothing here is recomputed. Rows, splits, v2 cue analyses, cos / c_resid and
NLI P(contradiction) all come from experiments/polarity's ``run.build()``,
which reads its own caches (and through them negation_detection's NLI cache
and negation_direction's embedding cache). The splits are polarity's:

    DEV        our OS/CN dataset + CANNOT phase 1
    VALIDATION 30% of CANNOT phase 2
    TEST       the other 70% of phase 2 + xNot360
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
POLARITY = REPO / "experiments" / "polarity"
sys.path.insert(0, str(POLARITY))

import data as pdata  # noqa: E402  (polarity's splits)
import polarity as P  # noqa: E402
import run as prun  # noqa: E402  (polarity's build: rows + caches)

SEED = 0


def kind(r: dict) -> str:
    """The pair category every breakdown in the report is keyed on."""
    if r["dataset"] == "ours":
        return {"K1_cancellation": "K1", "ANCHOR_paraphrase": "ours_paraphrase",
                "ANCHOR_unrelated": "ours_unrelated"}.get(r["family"], "ours_negation")
    if r["dataset"] == "xnot360":
        return f"xnot360_{r['subtype']}"
    if r["subtype"] == "paraphrase" and r["source"] == "antonym-substitution":
        return "SemAntoNeg"   # human-written: "It's a good idea" / "It's not an evil idea"
    return r["subtype"]       # paraphrase, sts_low, snli_neutral, random_pair, negation


def load() -> tuple[dict[str, list[dict]], dict]:
    """Polarity's splits, each row carrying split, kind, cos, c_resid, nli."""
    splits, analyses, _ = prun.build()
    pdata.assert_disjoint(splits)
    for rows in splits.values():
        for r in rows:
            r["kind"] = kind(r)
    return splits, analyses
