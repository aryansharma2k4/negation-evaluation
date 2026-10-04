"""Shared plumbing for experiments/nli2: rows, metrics, CIs, permutation tests, test-read log.

Rows come from experiments/composite (which reuses experiments/polarity's
splits and caches), so every nli2 number is on the same pairs as Report_NLI.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
SEED = 0
sys.path.insert(0, str(REPO / "experiments" / "composite"))
sys.path.insert(0, str(REPO))

import composite as C  # noqa: E402

TEST_LOG = RESULTS / "test_reads.json"


def log_test_read(step: str, what: str) -> None:
    """Every evaluation that touches TEST is recorded, for the caveats."""
    log = json.loads(TEST_LOG.read_text()) if TEST_LOG.exists() else []
    log.append({"step": step, "what": what, "time": time.strftime("%Y-%m-%d %H:%M")})
    TEST_LOG.write_text(json.dumps(log, indent=2))


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = score >= threshold
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(y), "accuracy": round((tp + tn) / len(y), 4),
            "auc": round(float(roc_auc_score(y, score)), 4) if len(set(y)) > 1 else None,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0}


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> list[float]:
    """95% percentile CI of accuracy from per-pair correctness."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), (n_boot, len(correct)))
    means = correct[idx].mean(1)
    return [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)]


def paired_permutation_p(correct_a: np.ndarray, correct_b: np.ndarray, n_perm: int = 10000,
                         seed: int = SEED) -> float:
    """Two-sided paired permutation test on accuracy: swap each pair's two outcomes at random."""
    rng = np.random.default_rng(seed)
    d = correct_b.astype(float) - correct_a.astype(float)
    observed = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], (n_perm, len(d)))
    null = np.abs((signs * d).mean(1))
    return round(float((1 + (null >= observed - 1e-12).sum()) / (1 + n_perm)), 5)


def save(name: str, obj) -> None:
    (RESULTS / name).write_text(json.dumps(obj, indent=2, default=float))
