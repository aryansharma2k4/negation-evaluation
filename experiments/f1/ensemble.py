"""Phase 2: an ensemble of NLI cross-encoders.

    ./venv/bin/python experiments/f1/ensemble.py --step 2.2   # each model standalone
    ./venv/bin/python experiments/f1/ensemble.py --step 2.3   # disagreement + correlation
    ./venv/bin/python experiments/f1/ensemble.py --step 2.4   # ensembling methods, selection
    ./venv/bin/python experiments/f1/ensemble.py --step 2.5   # overfitting check + importances

Every configuration keeps the Phase 0 cancellation override in front (it is part of the baseline).
Model probabilities: premise A, hypothesis B, from nli_models.py caches; the reference model's
are the exact values the baseline features use (row fields nli_c / nli_e / nli_n).

Selection rules, stated before any run:
* Model sets. "all" = every scored model. "pruned" = drop models whose standalone VALIDATION
  accuracy is more than 3 pts below the reference's; then, for any pair whose binary decisions
  agree on > 95% of clean-DEV pairs, drop the one with lower standalone VALIDATION accuracy.
  (The brief asks for the disagreement matrix on TEST; it is reported, but choosing models on TEST
  disagreement would be selection on TEST, so pruning uses clean DEV.)
* Configuration: highest VALIDATION accuracy among the Phase 0 baseline and every (method, set);
  tie -> higher out-of-fold clean-DEV accuracy (threshold-only methods: clean-DEV accuracy at the
  DEV-fit threshold).
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from core import C, N, RESULTS, ABLATION_HEADER, ablation_line, gain, load, log_test_read, save, system_row

import phase0 as P0
import phase4 as P4  # noqa: E402  (nli2)
from nli_models import MODELS, Scorer

REF = "deberta_base"
SETS = P0.SETS


def scored_models(sets) -> list[str]:
    """Models whose cache covers every row of every set (reference first)."""
    out = [REF]
    pairs = list(dict.fromkeys((r["a"], r["b"]) for k in SETS for r in sets[k]))
    for key in MODELS:
        if key != REF:
            cache = Scorer(key).cache
            if all(p in cache for p in pairs):
                out.append(key)
    return out


def probs(key, rows) -> np.ndarray:
    """(n, 3): P(contradiction), P(entailment), P(neutral)."""
    if key == REF:
        return np.array([[r["nli_c"], r["nli_e"], r["nli_n"]] for r in rows])
    cache = Scorer(key).cache
    return np.array([[cache[(r["a"], r["b"])][l] for l in ("contradiction", "entailment", "neutral")] for r in rows])


def all_probs(sets, keys) -> dict:
    return {key: {k: probs(key, sets[k]) for k in SETS} for key in keys}


def youden(sets, dev_score) -> float:
    return N.youden_threshold(sets["dev_clean"], dev_score)


def standalone(sets, P, key, masks) -> tuple[dict, float, dict]:
    sc = {k: P[key][k][:, 0] for k in SETS}
    thr = youden(sets, sc["dev_clean"])
    return sc, thr, system_row(sets, sc, thr)


def fit_gbm(sets, Xdev) -> dict:
    """nli2 step 4.2's GBM procedure on an arbitrary feature matrix: grouped 5-fold CV over
    connected components, grid by CV AUC, Youden on out-of-fold predictions, refit on all DEV."""
    dev = sets["dev_clean"]
    y = C.labels(dev)
    folds = P4.grouped_folds(dev)
    grid = P4.models()["gbm"]
    cv = {}
    for name, make in grid.items():
        oof = np.zeros(len(dev))
        for test_idx in folds:
            train_idx = np.setdiff1d(np.arange(len(dev)), test_idx)
            oof[test_idx] = make().fit(Xdev[train_idx], y[train_idx]).predict_proba(Xdev[test_idx])[:, 1]
        cv[name] = (round(float(roc_auc_score(y, oof)), 4), oof)
    best = max(cv, key=lambda n: cv[n][0])
    thr = N.youden_threshold(dev, cv[best][1])
    return {"model": grid[best]().fit(Xdev, y), "chosen": best, "cv_auc": {n: v[0] for n, v in cv.items()},
            "threshold": thr, "oof_dev_accuracy": round(float(((cv[best][1] >= thr) == y).mean()), 4), "oof": cv[best][1]}


def fit_logistic_oof(sets, Xdev) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    dev = sets["dev_clean"]
    y = C.labels(dev)
    make = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    oof = np.zeros(len(dev))
    for test_idx in P4.grouped_folds(dev):
        train_idx = np.setdiff1d(np.arange(len(dev)), test_idx)
        oof[test_idx] = make().fit(Xdev[train_idx], y[train_idx]).predict_proba(Xdev[test_idx])[:, 1]
    thr = N.youden_threshold(dev, oof)
    return {"model": make().fit(Xdev, y), "threshold": thr, "oof_dev_accuracy": round(float(((oof >= thr) == y).mean()), 4)}


def platt(sets, p_dev: np.ndarray):
    """Per-model calibration of P(contradiction) on clean DEV: logistic on logit(p)."""
    from sklearn.linear_model import LogisticRegression
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    m = LogisticRegression(C=1e6, max_iter=5000).fit(lg(p_dev)[:, None], C.labels(sets["dev_clean"]))
    return lambda p: m.predict_proba(lg(p)[:, None])[:, 1]


def ensembles(sets, X, P, keys, masks) -> dict:
    """Every method on one model set. Returns {method: (scores per set, thr, extra)}."""
    out = {}
    others = [k for k in keys if k != REF]
    # (a) mean P(contradiction), one Youden threshold
    sc = {k: np.mean([P[m][k][:, 0] for m in keys], 0) for k in SETS}
    out["(a) mean P(contradiction)"] = (sc, youden(sets, sc["dev_clean"]), {})
    # median, as a robustness variant of (a)
    sc = {k: np.median([P[m][k][:, 0] for m in keys], 0) for k in SETS}
    out["(a') median P(contradiction)"] = (sc, youden(sets, sc["dev_clean"]), {})
    # (c) per-model Platt calibration on clean DEV, then mean
    cal = {m: platt(sets, P[m]["dev_clean"][:, 0]) for m in keys}
    sc = {k: np.mean([cal[m](P[m][k][:, 0]) for m in keys], 0) for k in SETS}
    out["(c) calibrated mean P(contradiction)"] = (sc, youden(sets, sc["dev_clean"]), {})
    # (d) logistic stacker over all models' three probabilities (no other features)
    Xs = {k: np.hstack([P[m][k] for m in keys]) for k in SETS}
    lr = fit_logistic_oof(sets, Xs["dev_clean"])
    out["(d) logistic stacker over model probabilities"] = (
        {k: lr["model"].predict_proba(Xs[k])[:, 1] for k in SETS}, lr["threshold"], {"oof_dev_accuracy": lr["oof_dev_accuracy"]})
    # (b) GBM: 17 features + 3 probabilities per extra model
    Xb = {k: np.hstack([X[k]] + [P[m][k] for m in others]) for k in SETS}
    g = fit_gbm(sets, Xb["dev_clean"])
    names = P4.FEATURES + [f"{m}:{l}" for m in others for l in ("p_c", "p_e", "p_n")]
    out["(b) GBM, 17 features + every model's probabilities"] = (
        {k: g["model"].predict_proba(Xb[k])[:, 1] for k in SETS}, g["threshold"],
        {"oof_dev_accuracy": g["oof_dev_accuracy"], "chosen": g["chosen"], "cv_auc": g["cv_auc"], "n_features": len(names),
         "importances": dict(sorted(zip(names, np.round(g["model"].feature_importances_, 4).tolist()), key=lambda kv: -kv[1]))})
    # (e) GBM on the same 17 features, the three NLI features replaced by the across-model mean
    Xe = {k: X[k].copy() for k in SETS}
    for k in SETS:
        Xe[k][:, 1:4] = np.mean([P[m][k] for m in keys], 0)
    g = fit_gbm(sets, Xe["dev_clean"])
    out["(e) GBM, 17 features with NLI probabilities averaged across models"] = (
        {k: g["model"].predict_proba(Xe[k])[:, 1] for k in SETS}, g["threshold"],
        {"oof_dev_accuracy": g["oof_dev_accuracy"], "chosen": g["chosen"], "cv_auc": g["cv_auc"], "n_features": 17,
         "importances": dict(sorted(zip(P4.FEATURES, np.round(g["model"].feature_importances_, 4).tolist()), key=lambda kv: -kv[1]))})
    return {name: (P0.with_override(sc, masks), thr, extra) for name, (sc, thr, extra) in out.items()}


def context():
    b = P0.baseline()
    keys = scored_models(b["sets"])
    return b, keys, all_probs(b["sets"], keys)


def step_2_2() -> None:
    b, keys, P = context()
    sets = b["sets"]
    log_test_read("2.2", f"each NLI model standalone: {keys}")
    timing = load("2_1_timing.json")["models"] if (RESULTS / "2_1_timing.json").exists() else {}
    res = {"models": {}}
    for key in keys:
        sc, thr, ev = standalone(sets, P, key, b["masks"])
        res["models"][key] = {"hub_id": MODELS[key][0], "revision": MODELS[key][1], "parameters": MODELS[key][2],
                              "threshold": thr, "row": ev, "ms_per_pair": timing.get(key, {}).get("ms_per_pair")}
    save("2_2_standalone.json", res)
    print(ABLATION_HEADER)
    for key, v in res["models"].items():
        print(ablation_line(f"{key} ({v['parameters']}) thr {v['threshold']:.3f}", v["row"]), "xnot", v["row"]["xnot360"]["accuracy"],
              "DEV", v["row"]["dev_clean"]["accuracy"])


def decisions(sets, P, key, split) -> np.ndarray:
    thr = youden(sets, P[key]["dev_clean"][:, 0])
    return P[key][split][:, 0] >= thr


def step_2_3() -> None:
    b, keys, P = context()
    sets = b["sets"]
    log_test_read("2.3", "pairwise disagreement of NLI models on TEST")
    res = {}
    for split in ("test", "dev_clean"):
        d = {k: decisions(sets, P, k, split) for k in keys}
        res[split] = {"disagreement": {f"{x}|{y}": round(float((d[x] != d[y]).mean()), 4) for x, y in itertools.combinations(keys, 2)},
                      "pearson_p_contradiction": {f"{x}|{y}": round(float(np.corrcoef(P[x][split][:, 0], P[y][split][:, 0])[0, 1]), 4)
                                                  for x, y in itertools.combinations(keys, 2)}}
        res[split]["mean_pairwise_correlation"] = round(float(np.mean(list(res[split]["pearson_p_contradiction"].values()))), 4)
        res[split]["pairs_agreeing_over_95pct"] = [k for k, v in res[split]["disagreement"].items() if v < 0.05]
    # the pruning rule, on clean DEV and VALIDATION only
    sa = load("2_2_standalone.json")["models"]
    val = {k: sa[k]["row"]["validation"]["accuracy"] for k in keys}
    keep = [k for k in keys if val[k] >= val[REF] - 0.03]
    for pair in res["dev_clean"]["pairs_agreeing_over_95pct"]:
        x, y = pair.split("|")
        if x in keep and y in keep:
            keep.remove(min((x, y), key=lambda k: (val[k], k != REF)))
    res["pruned_set"] = keep
    res["pruning_rule"] = ("drop standalone VALIDATION accuracy < reference - 3 pts; then, of any pair agreeing on > 95% of "
                           "clean-DEV decisions, drop the lower-VALIDATION one")
    save("2_3_disagreement.json", res)
    for split in ("test", "dev_clean"):
        print(split, "mean corr", res[split]["mean_pairwise_correlation"], ">95% agree:", res[split]["pairs_agreeing_over_95pct"])
        print("| | " + " | ".join(keys) + " |")
        for x in keys:
            cells = []
            for y in keys:
                k = f"{x}|{y}" if f"{x}|{y}" in res[split]["disagreement"] else f"{y}|{x}"
                cells.append("—" if x == y else f"{100 * res[split]['disagreement'][k]:.1f}%")
            print(f"| {x} | " + " | ".join(cells) + " |")
    print("pruned:", keep)


def step_2_4() -> None:
    b, keys, P = context()
    sets, X = b["sets"], b["X"]
    pruned = load("2_3_disagreement.json")["pruned_set"]
    model_sets = {"all": keys, "pruned": pruned}
    log_test_read("2.4", "NLI ensembles on TEST")
    base = (b["scores"], b["thr"])
    res = {"model_sets": model_sets, "rows": {"Phase 0 baseline": {"row": system_row(sets, b["scores"], b["thr"]),
                                                                    "oof_dev_accuracy": b["fit"]["oof_dev_accuracy"]}},
           "rule": __doc__.split("Selection rules, stated before any run:")[1].strip()}
    scores = {"Phase 0 baseline": base}
    for set_name, ks in model_sets.items():
        if len(ks) < 2:
            continue
        for method, (sc, thr, extra) in ensembles(sets, X, P, ks, b["masks"]).items():
            name = f"{method} [{set_name}: {len(ks)} models]"
            ev = system_row(sets, sc, thr)
            res["rows"][name] = {"row": ev, "threshold": thr, **extra,
                                 "oof_dev_accuracy": extra.get("oof_dev_accuracy", ev["dev_clean"]["accuracy"]),
                                 "gain_vs_phase0": gain(sets, (b["scores"]["test"], b["thr"]), (sc["test"], thr))}
            scores[name] = (sc, thr)
    best = max(res["rows"], key=lambda n: (res["rows"][n]["row"]["validation"]["accuracy"], res["rows"][n]["oof_dev_accuracy"]))
    res["selected"] = best
    save("2_4_ensembles.json", res)
    print(ABLATION_HEADER)
    for name, v in res["rows"].items():
        g = v.get("gain_vs_phase0", {}).get("cannot_phase2")
        print(ablation_line(name, v["row"]), "| oofDEV", v["oof_dev_accuracy"], "| p2 gain", g and (g["gain"], g["p_permutation"]))
    print("SELECTED:", best)


def step_2_5() -> None:
    r = load("2_4_ensembles.json")
    out = {}
    for name, v in r["rows"].items():
        if "GBM" in name or name == "Phase 0 baseline":
            out[name] = {"oof_dev_accuracy": v["oof_dev_accuracy"], "validation_accuracy": v["row"]["validation"]["accuracy"],
                         "test_accuracy": v["row"]["all"]["accuracy"], "phase2_accuracy": v["row"]["cannot_phase2"]["accuracy"],
                         "oof_minus_test_pts": round(100 * (v["oof_dev_accuracy"] - v["row"]["all"]["accuracy"]), 2),
                         "n_features": v.get("n_features", 17), "importances": v.get("importances")}
    save("2_5_overfit.json", out)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "importances"} for k, v in out.items()}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["2.2", "2.3", "2.4", "2.5"])
    {"2.2": step_2_2, "2.3": step_2_3, "2.4": step_2_4, "2.5": step_2_5}[ap.parse_args().step]()


if __name__ == "__main__":
    main()


METHOD_TO_DETECTOR = {"Phase 0 baseline": "phase0", "(b)": "gbm_all_probs", "(e)": "gbm_mean_probs"}


def selected_config():
    """The Phase 2 selection as a detector.Config (combiner methods only)."""
    from detector import Config
    r = load("2_4_ensembles.json")
    name = r["selected"]
    if name == "Phase 0 baseline":
        return name, Config()
    key = name.split()[0]
    if key not in METHOD_TO_DETECTOR:
        raise NotImplementedError(f"selected method {name!r} has no detector wiring")
    keys = r["model_sets"]["all" if "[all:" in name else "pruned"]
    return name, Config(tuple(keys), METHOD_TO_DETECTOR[key])


def selected_system() -> dict:
    """Recompute the Phase 2-selected system on every set (deterministic), with its out-of-fold DEV
    predictions (needed to fit the guardrail on the pairs it actually sees)."""
    from detector import extend
    name, cfg = selected_config()
    b = P0.baseline()
    P = {k: {s: probs(k, b["sets"][s]) for s in SETS} for k in cfg.nli_keys}
    X = {s: extend(cfg, b["X"][s], {k: P[k][s] for k in cfg.nli_keys}) for s in SETS}
    g = fit_gbm(b["sets"], X["dev_clean"])
    scores = P0.with_override({s: g["model"].predict_proba(X[s])[:, 1] for s in SETS}, b["masks"])
    return {"name": name, "cfg": cfg, "sets": b["sets"], "X": X, "scores": scores, "thr": g["threshold"],
            "masks": b["masks"], "oof_dev": g["oof"], "oof_thr": g["threshold"], "model": g["model"]}
