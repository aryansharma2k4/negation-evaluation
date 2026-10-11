"""Shared plumbing for scripts/nli2: rows, metrics, CIs, permutation tests, test-read log.

Rows come from scripts/composite (which reuses scripts/polarity's
splits and caches), so every nli2 number is on the same pairs as reports/06-composite-nli.md.
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
RESULTS = REPO / "reports" / "results" / "nli2"
SEED = 0
sys.path.insert(0, str(REPO / "scripts" / "composite"))
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


# ---------------------------------------------------------------------------
# Rows for the clean protocol: DEV = datasets/splits/dev_clean.csv, VALIDATION / TEST as before
# ---------------------------------------------------------------------------

CACHE = REPO / ".cache" / "nli2"


def _nli_probs(pairs: list[tuple[str, str]]) -> dict:
    """All three NLI probabilities per pair. Reads negation_detection's and polarity's
    caches; computes only what neither has, into nli2's own cache."""
    import pickle
    detect = C.prun.detect
    nli = detect.NLI()                                  # loads .cache/negation_detection/nli.pkl
    for extra in (REPO / ".cache/polarity/nli_with_ours.pkl", CACHE / "nli.pkl"):
        if extra.exists():
            nli.cache.update(pickle.loads(extra.read_bytes()))
    nli.path = CACHE / "nli.pkl"                        # never write into another experiment's cache
    CACHE.mkdir(exist_ok=True)
    nli.score(pairs)
    return {p: nli.cache[p] for p in pairs}


def _embeddings(sentences: list[str]) -> dict:
    import pickle
    from experiment import embed  # negation_direction (on sys.path via composite -> polarity run)
    shared = {}
    for path in (REPO / ".cache/negation_direction/mpnet.pkl",
                 REPO / ".cache/polarity/mpnet_ours.pkl"):
        if path.exists():
            shared.update(pickle.loads(path.read_bytes()))
    missing = [s for s in dict.fromkeys(sentences) if s not in shared]
    if missing:
        shared.update(embed(missing, "mpnet", cache_path=CACHE / "mpnet.pkl"))
    return {s: shared[s] for s in sentences}


def read_dev_clean() -> list[dict]:
    import csv
    rows = []
    with (REPO / "datasets" / "splits" / "dev_clean.csv").open() as f:
        for r in csv.DictReader(f):
            r["label"] = int(r["label"])
            r["neg_type"] = r["neg_type"] or None
            r["split"] = "dev"
            r["kind"] = "hard_negative" if r["subtype"] == "hard_negative" else C.kind(r)
            rows.append(r)
    return rows


def load_clean():
    """{"dev": dev_clean rows, "validation": ..., "test": ...}, every row carrying
    nli_c / nli_e / nli_n, cos, and c_resid against a negation axis fit on clean DEV only.
    Also returns polarity analyses for every sentence."""
    import pickle
    path = CACHE / "clean_rows.pkl"
    digest = json.loads((RESULTS / "1_2_dev_clean.json").read_text())["sha256"]
    if path.exists():
        blob = pickle.loads(path.read_bytes())
        if blob["sha256"] == digest:
            return blob["splits"], blob["analyses"]
    import hashlib
    assert hashlib.sha256((REPO / "datasets/splits/dev_clean.csv").read_bytes()).hexdigest() == digest, \
        "datasets/splits/dev_clean.csv differs from the committed one; rebuild it with build_dev_clean.py --hard"
    old, analyses = C.load()
    splits = {"dev": read_dev_clean(), "validation": old["validation"], "test": old["test"]}
    C.pdata.assert_disjoint(splits)
    rows = [r for s in splits.values() for r in s]
    probs = _nli_probs([(r["a"], r["b"]) for r in rows])
    emb = _embeddings([s for r in rows for s in (r["a"], r["b"])])
    axis, _ = C.prun.detect.fit_axis(splits["dev"], emb)          # clean DEV only
    for r in rows:
        p = probs[(r["a"], r["b"])]
        r["nli_c"], r["nli_e"], r["nli_n"] = p["contradiction"], p["entailment"], p["neutral"]
        a, b = emb[r["a"]], emb[r["b"]]
        r["cos"] = float(a @ b)
        ra, rb = C.prun.detect.residual(np.stack([a, b]), axis)
        r["c_resid"] = float(ra @ rb)
        r["nli"] = r["nli_c"]
    missing = [s for s in dict.fromkeys(s for r in splits["dev"] for s in (r["a"], r["b"])) if s not in analyses]
    analyses = dict(analyses)
    analyses.update(C.P.analyse_many(missing))
    path.write_bytes(pickle.dumps({"sha256": digest, "splits": splits, "analyses": analyses}))
    return splits, analyses


# ---------------------------------------------------------------------------
# Evaluation with every breakdown the report uses
# ---------------------------------------------------------------------------

FPR_KINDS = ["paraphrase", "SemAntoNeg", "sts_low", "snli_neutral", "random_pair", "hard_negative",
             "xnot360_not_negation", "K1"]
RECALL_TYPES = ["verbal", "antonym", "other", "xnot360_negation"]


def youden_threshold(rows, score) -> float:
    C.dev_only(rows)
    return C.prun.youden_threshold(C.labels(rows), np.asarray(score))


def evaluate(rows, score, threshold, parts=("all", "cannot_phase2", "xnot360")) -> dict:
    score = np.asarray(score, float)
    y = C.labels(rows)
    pred = score >= threshold
    out = {"threshold": float(threshold)}
    for part in parts:
        idx = np.array([i for i, r in enumerate(rows) if part == "all" or r["dataset"] == part])
        if len(idx) == 0:
            continue
        m = metrics(y[idx], score[idx], threshold)
        m["ci95"] = bootstrap_ci((pred[idx] == y[idx]).astype(float))
        out[part] = m
    out["fpr_by_kind"] = {}
    for k in FPR_KINDS:
        m = np.array([r["kind"] == k and r["label"] == 0 for r in rows])
        if m.any():
            out["fpr_by_kind"][k] = {"n": int(m.sum()), "fpr": round(float(pred[m].mean()), 4)}
    out["recall_by_type"] = {}
    for t in RECALL_TYPES:
        m = np.array([r["label"] == 1 and (r.get("neg_type") == t or r["kind"] == t) for r in rows])
        if m.any():
            out["recall_by_type"][t] = {"n": int(m.sum()), "recall": round(float(pred[m].mean()), 4)}
    return out


def correct(rows, score, threshold) -> np.ndarray:
    return ((np.asarray(score) >= threshold) == C.labels(rows)).astype(float)


def compare(rows, base: tuple, new: tuple, part: str = "all") -> dict:
    """Accuracy gain of new over base on one part, with a paired permutation p-value.
    base / new are (score, threshold)."""
    idx = [i for i, r in enumerate(rows) if part == "all" or r["dataset"] == part]
    sub = [rows[i] for i in idx]
    a = correct(sub, np.asarray(base[0])[idx], base[1])
    b = correct(sub, np.asarray(new[0])[idx], new[1])
    return {"part": part, "n": len(idx), "gain": round(float(b.mean() - a.mean()), 4),
            "p_permutation": paired_permutation_p(a, b)}
