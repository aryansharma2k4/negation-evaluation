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


def grouped_folds(rows, k: int = 5) -> list[np.ndarray]:
    """k folds over connected components of shared sentences (seeded)."""
    groups = C.pdata.components(rows)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(groups))
    folds = [[] for _ in range(k)]
    for j, g in enumerate(order):
        folds[j % k].extend(groups[g])
    return [np.array(sorted(f)) for f in folds]


def models():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "logistic": {f"C={c}": (lambda c=c: make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)))
                     for c in (0.01, 0.1, 1.0, 10.0)},
        "gbm": {f"depth={d},lr={lr}": (lambda d=d, lr=lr: GradientBoostingClassifier(
                    n_estimators=300, max_depth=d, learning_rate=lr, subsample=0.8, random_state=SEED))
                for d in (2, 3) for lr in (0.03, 0.1)},
    }


def fit_combiners(sets, X) -> dict:
    """Pick hyperparameters by grouped-CV AUC on clean DEV; threshold = Youden on out-of-fold
    DEV predictions; final model refit on all of clean DEV."""
    from sklearn.metrics import roc_auc_score
    dev = sets["dev_clean"]
    y = C.labels(dev)
    folds = grouped_folds(dev)
    for f1 in range(len(folds)):
        for f2 in range(f1 + 1, len(folds)):
            a = C.pdata.sentences([dev[i] for i in folds[f1]])
            b = C.pdata.sentences([dev[i] for i in folds[f2]])
            assert not a & b, "a sentence crosses CV folds"
    out = {}
    for family, grid in models().items():
        cv = {}
        for name, make in grid.items():
            oof = np.zeros(len(dev))
            for k, test_idx in enumerate(folds):
                train_idx = np.setdiff1d(np.arange(len(dev)), test_idx)
                oof[test_idx] = make().fit(X["dev_clean"][train_idx], y[train_idx]).predict_proba(
                    X["dev_clean"][test_idx])[:, 1]
            cv[name] = {"cv_auc": round(float(roc_auc_score(y, oof)), 4), "_oof": oof}
        best = max(cv, key=lambda n: cv[n]["cv_auc"])
        thr = youden_threshold(dev, cv[best]["_oof"])
        model = grid[best]().fit(X["dev_clean"], y)
        out[family] = {"chosen": best, "cv": {n: v["cv_auc"] for n, v in cv.items()}, "threshold": thr,
                       "oof_dev_accuracy": round(float(((cv[best]["_oof"] >= thr) == y).mean()), 4),
                       "_model": model}
    return out


def explain(fits) -> dict:
    lr = fits["logistic"]["_model"]
    coef = lr[-1].coef_[0]
    gbm = fits["gbm"]["_model"]
    return {"logistic_standardised_coefficients": dict(sorted(zip(FEATURES, np.round(coef, 4).tolist()),
                                                              key=lambda kv: -abs(kv[1]))),
            "logistic_intercept": round(float(lr[-1].intercept_[0]), 4),
            "gbm_importances": dict(sorted(zip(FEATURES, np.round(gbm.feature_importances_, 4).tolist()),
                                           key=lambda kv: -kv[1]))}


def step_4_2() -> None:
    sets, _, X = table()
    fits = fit_combiners(sets, X)
    res = {"cv_grouping": "connected components of shared sentences, 5 folds, seed 0", "models": {}}
    for fam, f in fits.items():
        p = f["_model"].predict_proba(X["validation"])[:, 1]
        res["models"][fam] = {k: v for k, v in f.items() if not k.startswith("_")}
        res["models"][fam]["validation"] = evaluate(sets["validation"], p, f["threshold"], parts=("all",))["all"]
    res.update(explain(fits))
    save("4_2_combiners.json", res)
    print(json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in res["models"].items()}, indent=1))
    print("| feature | logistic coef (standardised) | GBM importance |\n| --- | ---: | ---: |")
    for f in res["logistic_standardised_coefficients"]:
        print(f"| `{f}` | {res['logistic_standardised_coefficients'][f]:+.3f} | {res['gbm_importances'][f]:.3f} |")


def adopted_cascade() -> PL.Config:
    p2 = json.loads((RESULTS / "2_all.json").read_text())["configs"][1]["config"]
    return PL.Config(**{**p2, "name": "cascade (Phase 2 adopted)",
                        "length_buckets": tuple(tuple(b) for b in p2["length_buckets"])})


def step_4_3() -> None:
    sets, analyses, X = table()
    fits = fit_combiners(sets, X)
    splits, _, sl = PL.load()
    casc = PL.run(adopted_cascade(), splits, analyses, sl)
    nli_thr = youden_threshold(sets["dev_clean"], [r["nli_c"] for r in sets["dev_clean"]])
    log_test_read("4.3", "NLI alone, cascade, logistic, GBM on TEST")
    scored = {"NLI alone (honest threshold)": ({n: np.array([r["nli_c"] for r in sets[n]]) for n in sets}, nli_thr),
              "cascade (Phase 2 adopted)": ({"test": casc["_scores"]["test"], "validation": casc["_scores"]["validation"],
                                             "k1": casc["_scores"]["k1"]}, casc["stage3_threshold"])}
    for fam, label in (("logistic", "logistic regression"), ("gbm", "gradient-boosted trees")):
        m = fits[fam]["_model"]
        scored[label] = ({n: m.predict_proba(X[n])[:, 1] for n in ("test", "validation", "k1")}, fits[fam]["threshold"])
    test = sets["test"]
    phase2 = sl["validation"] + [r for r in test if r["dataset"] == "cannot_phase2"]
    sem_idx_v = [i for i, r in enumerate(sets["validation"]) if r["kind"] == "SemAntoNeg"]
    sem_idx_t = [i for i, r in enumerate(test) if r["kind"] == "SemAntoNeg"]
    res = {"rows": {}, "gains": {}}
    for name, (sc, thr) in scored.items():
        ev = evaluate(test, sc["test"], thr)
        ev["validation_accuracy"] = evaluate(sets["validation"], sc["validation"], thr, parts=("all",))["all"]["accuracy"]
        ev["k1_fpr"] = round(float((sc["k1"] >= thr).mean()), 4)
        sem = np.concatenate([sc["validation"][sem_idx_v], sc["test"][sem_idx_t]])
        ev["semantoneg_phase2_fpr"] = round(float((sem >= thr).mean()), 4)
        res["rows"][name] = ev
        for base in ("NLI alone (honest threshold)", "cascade (Phase 2 adopted)"):
            if name != base:
                for part in ("all", "cannot_phase2", "xnot360"):
                    res["gains"][f"{name} vs {base}|{part}"] = compare(test, (scored[base][0]["test"], scored[base][1]),
                                                                       (sc["test"], thr), part)
    save("4_3_compare.json", res)
    pc = lambda x: f"{100 * x:.1f}%"
    print("| model | phase-2 acc (95% CI) | xNot360 acc (95% CI) | all-TEST acc (95% CI) | AUC | recall | FPR paraphrase | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | SemAntoNeg FPR (n=41) | VAL acc |")
    print("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, t in res["rows"].items():
        f = t["fpr_by_kind"]
        ci = lambda part: f"{pc(t[part]['accuracy'])} ({pc(t[part]['ci95'][0])}–{pc(t[part]['ci95'][1])})"
        print(f"| {name} | {ci('cannot_phase2')} | {ci('xnot360')} | {ci('all')} | {t['all']['auc']:.3f} | {pc(t['all']['recall'])} | "
              + " | ".join(pc(f[k]["fpr"]) for k in ("paraphrase", "sts_low", "snli_neutral", "random_pair", "xnot360_not_negation"))
              + f" | {pc(t['recall_by_type']['antonym']['recall'])} | {pc(t['k1_fpr'])} | {pc(t['semantoneg_phase2_fpr'])} | {pc(t['validation_accuracy'])} |")
    for k, v in res["gains"].items():
        print(k, v["gain"], v["p_permutation"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["4.1", "4.2", "4.3", "4.4"])
    args = ap.parse_args()
    {"4.1": step_4_1, "4.2": step_4_2, "4.3": step_4_3}[args.step]()


if __name__ == "__main__":
    main()
