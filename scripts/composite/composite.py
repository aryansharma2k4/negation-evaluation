"""Composite negation detector: content gate -> cancellation veto -> NLI.

Nothing here is recomputed. Rows, splits, v2 cue analyses, cos / c_resid and
NLI P(contradiction) all come from scripts/polarity's ``run.build()``,
which reads its own caches (and through them negation_detection's NLI cache
and negation_direction's embedding cache). The splits are polarity's:

    DEV        our OS/CN dataset + CANNOT phase 1
    VALIDATION 30% of CANNOT phase 2
    TEST       the other 70% of phase 2 + xNot360
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
POLARITY = REPO / "scripts" / "polarity"
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


# ---------------------------------------------------------------------------
# Stage 1: content gate
# ---------------------------------------------------------------------------

SIMILARITIES = ("cos", "c_resid")   # raw cosine; cosine after removing the negation direction
MIN_GATE_RECALL = 0.95
T1_GRID = np.round(np.arange(0.0, 1.0, 0.01), 2)


def labels(rows) -> np.ndarray:
    return np.array([r["label"] for r in rows])


def dev_only(rows) -> None:
    """Every fit goes through this: a threshold fit on anything but DEV raises."""
    bad = {r["split"] for r in rows} - {"dev"}
    assert not bad, f"fit attempted on {bad}"


def gate_pass(rows, sim: str, t1: float) -> np.ndarray:
    """True = similar enough to go on; False = predict NOT a negation and stop."""
    return np.array([r[sim] for r in rows]) >= t1


def youden(y: np.ndarray, pred: np.ndarray) -> float:
    return float(pred[y == 1].mean() - pred[y == 0].mean())


def fit_gate(dev, sim: str, nli_threshold: float) -> dict:
    """t1 maximising DEV Youden's J of gate -> NLI, subject to the gate keeping
    >= MIN_GATE_RECALL of DEV positives. Ties go to the lower (gentler) t1.

    Not "as high as the recall floor allows": the floor is a guard, not the
    target. At the floor (cos 0.586) validation recall of gate -> NLI falls to
    0.83, because true negations like "Some meetings ..." / "None of the
    meetings ..." sit at cosine 0.45.
    """
    dev_only(dev)
    y = labels(dev)
    nli = np.array([r["nli"] for r in dev]) >= nli_threshold
    trace = []
    for t1 in T1_GRID:
        passed = gate_pass(dev, sim, t1)
        recall = float(passed[y == 1].mean())
        if recall < MIN_GATE_RECALL:
            break
        trace.append({"t1": float(t1), "gate_recall": round(recall, 4),
                      "dev_j": round(youden(y, passed & nli), 4)})
    best = max(trace, key=lambda t: (t["dev_j"], -t["t1"]))
    return {"sim": sim, **best, "trace": trace}


# ---------------------------------------------------------------------------
# Stage 2: cancellation veto
# ---------------------------------------------------------------------------

def cancellation_detected(a: "P.Analysis", b: "P.Analysis") -> bool:
    """B carries >= 2 cues inside one clause_scope that cancel by parity (an
    even count), AND A's net v2 polarity equals B's.

    An explicit signal, deliberately NOT "|delta_p| is small": delta_p is also
    0 when neither sentence has any cue ("smaller" / "larger"), which is 97%
    of CANNOT's lexical `other` negations, the type where NLI's recall is
    97.8% and polarity's 2.8%. Vetoing those would throw NLI's strength away.
    """
    cancels = any(len(cs) >= 2 and len(cs) % 2 == 0 for cs in P.by_clause(b.cues).values())
    return cancels and P.v2(a) == P.v2(b)


def veto(rows, analyses) -> np.ndarray:
    return np.array([cancellation_detected(analyses[r["a"]], analyses[r["b"]]) for r in rows])


# ---------------------------------------------------------------------------
# Stage 3 and the cascade
# ---------------------------------------------------------------------------

STOPPED = -1.0   # score of a pair stopped before NLI: ranks below every P(contradiction)


def fit_threshold(dev, score: np.ndarray) -> float:
    """Youden's J on DEV, as in scripts/polarity (DEV is 66% positive, so
    max-F1 would pick "everything is a negation")."""
    dev_only(dev)
    return prun.youden_threshold(labels(dev), score)


def cascade(rows, analyses, sim: str | None = None, t1: float | None = None,
            use_veto: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Score = NLI P(contradiction) for pairs that reach stage 3, STOPPED otherwise.

    Returns (score, stage) where stage is 1 (stopped by the gate), 2 (vetoed)
    or 3 (reached NLI). The veto is only evaluated on pairs the gate passed.
    """
    stage = np.full(len(rows), 3)
    if sim is not None:
        stage[~gate_pass(rows, sim, t1)] = 1
    if use_veto:
        live = np.flatnonzero(stage == 3)
        stage[live[veto([rows[i] for i in live], analyses)]] = 2
    score = np.where(stage == 3, [r["nli"] for r in rows], STOPPED)
    return score, stage
