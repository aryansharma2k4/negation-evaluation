"""Phase 4: a learned combiner over the signals the pipeline already computes.

    ./venv/bin/python experiments/nli2/phase4.py --step 4.1   # feature table
    ./venv/bin/python experiments/nli2/phase4.py --step 4.2   # train LR + GBM on clean DEV
    ./venv/bin/python experiments/nli2/phase4.py --step 4.3   # cascade vs LR vs GBM on TEST
    ./venv/bin/python experiments/nli2/phase4.py --step 4.4   # per-family NLI thresholds

Features are exactly the brief's list. Training data: clean DEV only. Cross-validation folds
are grouped by connected component of shared sentences (experiments/polarity data.components):
no sentence, as A or B, is in two folds. That is stricter than negation_detection's
group_folds, which groups by sentence A only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np

import conditionals as K
import pipeline as PL
import rules as R
from common import C, REPO, RESULTS, SEED, compare, evaluate, log_test_read, save, youden_threshold

sys.path.insert(0, str(REPO / "experiments" / "negation_direction"))
from experiment import cue_count  # noqa: E402

import build_dev_clean as B  # noqa: E402

FEATURES = ["cos_sim", "p_contradiction", "p_entailment", "p_neutral", "cue_count_a", "cue_count_b",
            "net_negation", "cancellation_a", "cancellation_b", "antonym_aligned", "has_conditional_a",
            "has_conditional_b", "antecedent_only_flip", "len_a", "len_b", "len_diff", "token_overlap"]
FEATURE_TEXT = {
    "cos_sim": "mpnet cosine(A, B)", "p_contradiction": "NLI P(contradiction)", "p_entailment": "NLI P(entailment)",
    "p_neutral": "NLI P(neutral)", "cue_count_a": "lexicon negation cues in A", "cue_count_b": "lexicon negation cues in B",
    "net_negation": "v2 polarity(B) - v2 polarity(A)", "cancellation_a": "A has >= 2 cues cancelling in one clause",
    "cancellation_b": "B has >= 2 cues cancelling in one clause", "antonym_aligned": "2.3 aligned antonym swap",
    "has_conditional_a": "A is a conditional", "has_conditional_b": "B is a conditional",
    "antecedent_only_flip": "3.2: only the antecedent's polarity flips", "len_a": "words in A", "len_b": "words in B",
    "len_diff": "|len A - len B|", "token_overlap": "word-set Jaccard overlap"}
FEATURE_CSV = RESULTS / "4_1_features.csv"


def self_cancels(analysis) -> int:
    return int(any(len(cs) >= 2 and len(cs) % 2 == 0 for cs in C.P.by_clause(analysis.cues).values()))


def featurise(rows, analyses) -> np.ndarray:
    sents = [s for r in rows for s in (r["a"], r["b"])]
    parsed, conds = R.parse(sents), K.conditionals(sents)
    X = []
    for r in rows:
        aa, ab = analyses[r["a"]], analyses[r["b"]]
        ta, tb = parsed[r["a"]], parsed[r["b"]]
        ca, cb = K.primary(conds[r["a"]]), K.primary(conds[r["b"]])
        la, lb = PL.n_words(ta), PL.n_words(tb)
        X.append([r["cos"], r["nli_c"], r["nli_e"], r["nli_n"], cue_count(r["a"]), cue_count(r["b"]),
                  C.P.v2(ab) - C.P.v2(aa), self_cancels(aa), self_cancels(ab), int(R.antonym_aligned(ta, tb)),
                  int(ca is not None), int(cb is not None), int(K.decide(ca, cb, aa, ab) == "not_negation"),
                  la, lb, abs(la - lb), B.overlap(r["a"], r["b"])])
    return np.array(X, float)


def k1_rows():
    """Our 91 K1 pairs with all three NLI probabilities (read from the existing caches)."""
    from common import _nli_probs
    old, _ = C.load()
    rows = [dict(r) for r in old["dev"] if r["kind"] == "K1"]
    probs = _nli_probs([(r["a"], r["b"]) for r in rows])
    for r in rows:
        p = probs[(r["a"], r["b"])]
        r["nli_c"], r["nli_e"], r["nli_n"] = p["contradiction"], p["entailment"], p["neutral"]
    return rows


def table():
    splits, analyses, sl = PL.load()
    sets = {"dev_clean": sl["dev_clean"], "validation": sl["validation"], "test": sl["test"], "k1": k1_rows()}
    return sets, analyses, {k: featurise(v, analyses) for k, v in sets.items()}


def step_4_1() -> None:
    sets, _, X = table()
    with FEATURE_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "dataset", "kind", "neg_type", "label"] + FEATURES)
        for name, rows in sets.items():
            for r, x in zip(rows, X[name]):
                w.writerow([name, r["dataset"], r["kind"], r.get("neg_type") or "", r["label"]]
                           + [round(float(v), 5) for v in x])
    summary = {name: {"n": len(rows), "feature_means": dict(zip(FEATURES, np.round(X[name].mean(0), 4).tolist()))}
               for name, rows in sets.items()}
    save("4_1_features.json", {"features": FEATURE_TEXT, "csv": str(FEATURE_CSV.relative_to(REPO)), "sets": summary})
    print("| feature | definition | clean DEV mean | TEST mean |\n| --- | --- | ---: | ---: |")
    for i, f in enumerate(FEATURES):
        print(f"| `{f}` | {FEATURE_TEXT[f]} | {X['dev_clean'][:, i].mean():.3f} | {X['test'][:, i].mean():.3f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["4.1", "4.2", "4.3", "4.4"])
    args = ap.parse_args()
    {"4.1": step_4_1}[args.step]()


if __name__ == "__main__":
    main()
