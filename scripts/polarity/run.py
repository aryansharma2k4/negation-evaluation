"""Iterate the polarity scorer on DEV, check on VALIDATION, score TEST once.

    ./venv/bin/python scripts/polarity/run.py --versions v1 v2    # dev + validation
    ./venv/bin/python scripts/polarity/run.py --final             # test, once

Everything lands in results_polarity.json. The test set is scored only under
--final; every --final run is counted in cache/test_scorings.json and the
count is printed in the report.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "negation_direction"))
sys.path.insert(0, str(REPO / "scripts" / "negation_detection"))
sys.path.insert(0, str(HERE))

import detect  # noqa: E402  (negation_detection: metrics, thresholds, arch 2, NLI)
from experiment import cue_count, embed  # noqa: E402  (negation_direction)

import data  # noqa: E402
import polarity as P  # noqa: E402

SEED = 0
CACHE = REPO / ".cache" / "polarity"
RESULTS = REPO / "reports" / "results" / "polarity" / "results_polarity.json"
FORMULAS = ["f1", "f2", "f3", "f4", "f5"]
FORMULA_TEXT = {
    "f1": "|dp| * cos(A,B)",
    "f2": "|dp| * c_resid (cosine after removing the negation direction)",
    "f3": "|dp|",
    "f4": "|dp| if cos(A,B) >= t else 0  (t fit on dev)",
    "f5": "dp = p(B) - p(A), signed",
}
BASELINES = ["cue_delta", "raw_cosine", "nli_contradiction", "arch2_margin"]


# ---------------------------------------------------------------------------
# Inputs computed once
# ---------------------------------------------------------------------------

def cached(name, fn):
    path = CACHE / name
    if path.exists():
        return pickle.loads(path.read_bytes())
    value = fn()
    CACHE.mkdir(exist_ok=True)
    path.write_bytes(pickle.dumps(value))
    return value


def build():
    splits, notes = data.load_splits()
    rows = [r for s in splits.values() for r in s]
    for name, split in splits.items():
        for r in split:
            r["split"] = name
    sents = list(dict.fromkeys(s for r in rows for s in (r["a"], r["b"])))

    analyses = cached("analysis.pkl", lambda: P.analyse_many(sents))
    missing = [s for s in sents if s not in analyses]
    if missing:
        analyses.update(P.analyse_many(missing))
        (CACHE / "analysis.pkl").write_bytes(pickle.dumps(analyses))

    ours = data.sentences([r for r in rows if r["dataset"] == "ours"])
    emb = embed([s for s in sents if s not in ours], "mpnet")                  # shared cache, read
    emb = {**emb, **embed(sorted(ours), "mpnet", cache_path=CACHE / "mpnet_ours.pkl")}

    axis, _ = detect.fit_axis(splits["dev"], emb)                             # DEV only
    A = np.stack([emb[r["a"]] for r in rows]); B = np.stack([emb[r["b"]] for r in rows])
    cos = np.einsum("ij,ij->i", A, B)
    cres = np.einsum("ij,ij->i", detect.residual(A, axis), detect.residual(B, axis))
    for r, c, cr in zip(rows, cos, cres):
        r["cos"], r["c_resid"] = float(c), float(cr)

    def baselines():
        nli = detect.NLI()
        nli.path = CACHE / "nli_with_ours.pkl"   # never write into negation_detection's cache
        before = len(nli.cache)
        pairs = [(r["a"], r["b"]) for r in rows]
        new = sum(p not in nli.cache for p in dict.fromkeys(pairs))
        probs = nli.score(pairs)[:, 0]
        vemb = detect.variant_embeddings(
            [v["text"] for x in detect.load_refsets().values() for v in x["neg"] + x["same"]], "mpnet")
        refsets = detect.load_refsets()
        cannot = [i for i, r in enumerate(rows) if r["dataset"] != "ours"]
        a2 = detect.arch2_scores([rows[i] for i in cannot], emb, vemb, axis, refsets)
        margin = np.full(len(rows), np.nan)
        margin[cannot] = a2["margin_cos"]
        return {"key": [(r["a"], r["b"]) for r in rows], "nli": probs, "arch2": margin,
                "nli_cached_before": before, "nli_new_pairs": new}

    base = cached("baselines.pkl", baselines)
    assert base["key"] == [(r["a"], r["b"]) for r in rows], "row order changed; delete cache/baselines.pkl"
    for r, n, m in zip(rows, base["nli"], base["arch2"]):
        r["nli"], r["arch2"] = float(n), float(m)
        r["cue_delta"] = float(cue_count(r["b"]) - cue_count(r["a"]))
    info = {"nli_pairs_reused_from_cache": len(rows) - base["nli_new_pairs"],
            "nli_pairs_computed_new": base["nli_new_pairs"], **notes}
    return splits, analyses, info


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def pols(analyses, version, weights=None) -> dict[str, float]:
    return {s: P.score(a, version, weights) for s, a in analyses.items()}


def formula(rows, pol, name, gate=None) -> np.ndarray:
    pa = np.array([pol[r["a"]] for r in rows]); pb = np.array([pol[r["b"]] for r in rows])
    dp = pb - pa
    cos = np.array([r["cos"] for r in rows]); cres = np.array([r["c_resid"] for r in rows])
    if name == "f1":
        return np.abs(dp) * cos
    if name == "f2":
        return np.abs(dp) * cres
    if name == "f3":
        return np.abs(dp)
    if name == "f4":
        return np.where(cos >= gate, np.abs(dp), 0.0)
    if name == "f5":
        return dp
    raise KeyError(name)


def fit_gate(dev, pol) -> float:
    y = np.array([r["label"] for r in dev])
    cos = np.array([r["cos"] for r in dev])
    best, best_auc = 0.0, -1.0
    for t in np.quantile(cos, np.linspace(0.0, 0.95, 20)):
        auc = roc_auc_score(y, formula(dev, pol, "f4", t))
        if auc > best_auc:
            best, best_auc = float(t), auc
    return best


def auc(y, s):
    s = np.nan_to_num(s, nan=-1e9)
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else None


def evaluate(rows, score, threshold) -> dict:
    """Headline metrics plus the two numbers that matter, and every breakdown."""
    keep = ~np.isnan(score)
    rows = [r for r, k in zip(rows, keep) if k]; score = score[keep]
    y = np.array([r["label"] for r in rows])
    out = detect.metrics(y, score, threshold)
    out["n"] = len(rows)
    pred = score >= threshold

    def fpr(mask):
        m = np.array(mask)
        return {"n": int(m.sum()), "fpr": round(float(pred[m].mean()), 4) if m.any() else None}

    out["paraphrase_fpr"] = fpr([r.get("subtype") == "paraphrase" or r.get("family") == "ANCHOR_paraphrase"
                                 for r in rows])
    out["k1_fpr"] = fpr([r.get("family") == "K1_cancellation" for r in rows])
    by = {}
    for ds in sorted({r["dataset"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["dataset"] == ds]
        sub = [rows[i] for i in idx]
        block = detect.metrics(y[idx], score[idx], threshold) if len(set(y[idx])) > 1 else {}
        block["n"] = len(idx)
        key = "family" if ds == "ours" else "subtype"
        block[f"by_{key}"] = detect.breakdown(sub, y[idx], score[idx], threshold, key)
        if ds != "ours":
            block["by_neg_type"] = detect.breakdown(sub, y[idx], score[idx], threshold, "neg_type")
        by[ds] = block
    out["by_dataset"] = by
    return out


def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising TPR - FPR on the data given (DEV only).

    Not max-F1: DEV is 66% positive, and on it max-F1 picks "everything is a
    negation" (every paraphrase and K1 pair a false positive). Youden's J does
    not move with class balance.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, score)
    return float(thr[int(np.argmax(tpr - fpr))])


def run_scheme(splits, score_fn, include_test=False) -> dict:
    """score_fn(rows) -> scores. Threshold fit on DEV only (Youden's J)."""
    dev = splits["dev"]
    y_dev = np.array([r["label"] for r in dev])
    s_dev = score_fn(dev)
    finite = ~np.isnan(s_dev)
    threshold = youden_threshold(y_dev[finite], s_dev[finite])
    out = {"threshold": threshold, "dev": evaluate(dev, s_dev, threshold),
           "validation": evaluate(splits["validation"], score_fn(splits["validation"]), threshold)}
    if include_test:
        out["test"] = evaluate(splits["test"], score_fn(splits["test"]), threshold)
    return out


def summary(block: dict) -> dict:
    d, v = block["dev"], block["validation"]
    return {"dev_auc": d["auc"], "dev_f1": d["f1"], "val_auc": v["auc"], "val_f1": v["f1"],
            "gap_auc": round(d["auc"] - v["auc"], 4),
            "dev_paraphrase_fpr": d["paraphrase_fpr"]["fpr"], "val_paraphrase_fpr": v["paraphrase_fpr"]["fpr"],
            "dev_k1_fpr": d["k1_fpr"]["fpr"]}


# ---------------------------------------------------------------------------
# v6: weights fit on DEV
# ---------------------------------------------------------------------------

def fit_weights(splits, analyses, version: str, start: dict) -> dict:
    """Coordinate ascent on DEV AUC of f1, each weight over {0.1, ..., 1.0}.

    Only DEV sentences are scored during the search.
    """
    dev = splits["dev"]
    dev_sents = {s for r in dev for s in (r["a"], r["b"])}
    sub = {s: analyses[s] for s in dev_sents}
    y = np.array([r["label"] for r in dev])
    weights = json.loads(json.dumps(start))
    grid = [round(0.1 * k, 1) for k in range(1, 11)]

    def objective(w):
        return roc_auc_score(y, formula(dev, pols(sub, version, w), "f1"))

    best = objective(weights)
    trace = [round(best, 4)]
    for _ in range(2):
        for group in sorted(weights):
            for key in sorted(weights[group]):
                for value in grid:
                    trial = json.loads(json.dumps(weights)); trial[group][key] = value
                    score = objective(trial)
                    if score > best + 1e-6:
                        best, weights = score, trial
        trace.append(round(best, 4))
    return {"weights": weights, "dev_auc_trace": trace}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_results() -> dict:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def errors(rows, score, threshold, pol, k, rng, kind) -> list[dict]:
    y = np.array([r["label"] for r in rows])
    pred = score >= threshold
    idx = [i for i in range(len(rows)) if (pred[i] and y[i] == 0) == (kind == "fp") and pred[i] != y[i]]
    pick = rng.sample(idx, min(k, len(idx)))
    return [{"dataset": rows[i]["dataset"], "type": rows[i].get("family") or rows[i].get("subtype"),
             "neg_type": rows[i].get("neg_type"), "a": rows[i]["a"], "b": rows[i]["b"],
             "p_a": round(pol[rows[i]["a"]], 3), "p_b": round(pol[rows[i]["b"]], 3),
             "cos": round(rows[i]["cos"], 3), "score": round(float(score[i]), 3), "label": int(y[i])}
            for i in pick]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="*", default=[])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--show-errors", default=None, help="version:formula, print dev errors")
    args = ap.parse_args()

    splits, analyses, info = build()
    res = load_results()
    res["seed"] = SEED
    res["data"] = {
        "info": info,
        "splits": {k: {"n": len(v), "labels": dict(Counter(str(r["label"]) for r in v)),
                       "datasets": dict(Counter(r["dataset"] for r in v))} for k, v in splits.items()},
        "ours_by_family": dict(sorted(Counter(f"{r['family']}={r['label']}" for r in splits["dev"]
                                              if r["dataset"] == "ours").items())),
        "sentence_overlap_between_splits": 0,   # load_splits asserts it
        "formulas": FORMULA_TEXT,
    }

    if "baselines" not in res:
        res["baselines"] = {}
        for name, key, sign in [("cue_delta", "cue_delta", 1), ("raw_cosine", "cos", -1),
                                ("nli_contradiction", "nli", 1), ("arch2_margin", "arch2", 1)]:
            fn = lambda rows, key=key, sign=sign: sign * np.array([r[key] for r in rows], float)
            res["baselines"][name] = run_scheme(splits, fn)

    res.setdefault("iterations", {})
    weights_store = res.setdefault("fitted_weights", {})
    for version in args.versions:
        weights = None
        if version in P.FITTED:
            fit = fit_weights(splits, analyses, P.FITTED[version]["base"], P.FITTED[version]["start"])
            weights_store[version] = fit
            weights = fit["weights"]
        base_version = P.FITTED[version]["base"] if version in P.FITTED else version
        pol = pols(analyses, base_version, weights)
        gate = fit_gate(splits["dev"], pol)
        grid = {}
        for f in FORMULAS:
            grid[f] = run_scheme(splits, lambda rows, f=f: formula(rows, pol, f, gate))
        grid["f4"]["gate"] = gate
        res["iterations"][version] = {"grid": grid, "summary": {f: summary(g) for f, g in grid.items()}}
        print(version, json.dumps({f: summary(g) for f, g in grid.items()}["f1"]))

    if args.show_errors:
        version, f = args.show_errors.split(":")
        weights = weights_store.get(version, {}).get("weights")
        base_version = P.FITTED[version]["base"] if version in P.FITTED else version
        pol = pols(analyses, base_version, weights)
        grid = res["iterations"][version]["grid"][f]
        dev = splits["dev"]
        s = formula(dev, pol, f, grid.get("gate"))
        rng = random.Random(SEED)
        for kind in ("fp", "fn"):
            errs = errors(dev, s, grid["threshold"], pol, 25, rng, kind)
            print(f"\n=== DEV {kind} ({version}:{f}) ===")
            for e in errs:
                print(f"{e['dataset'][:8]:8s} {str(e['type'])[:16]:16s} {str(e['neg_type'])[:7]:7s} "
                      f"pA={e['p_a']:.2f} pB={e['p_b']:.2f} cos={e['cos']:.2f} | {e['a'][:80]} || {e['b'][:80]}")

    if args.final:
        log_path = CACHE / "test_scorings.json"
        log = json.loads(log_path.read_text()) if log_path.exists() else {"count": 0, "runs": []}
        cand = [(v, f, it["summary"][f]["val_auc"]) for v, it in res["iterations"].items() for f in it["summary"]]
        version, f, val_auc = max(cand, key=lambda t: t[2])
        weights = weights_store.get(version, {}).get("weights")
        base_version = P.FITTED[version]["base"] if version in P.FITTED else version
        pol = pols(analyses, base_version, weights)
        gate = res["iterations"][version]["grid"]["f4"]["gate"]
        block = run_scheme(splits, lambda rows: formula(rows, pol, f, gate), include_test=True)
        base_test = {name: run_scheme(splits, lambda rows, key=key, sign=sign:
                                      sign * np.array([r[key] for r in rows], float), include_test=True)["test"]
                     for name, key, sign in [("cue_delta", "cue_delta", 1), ("raw_cosine", "cos", -1),
                                             ("nli_contradiction", "nli", 1), ("arch2_margin", "arch2", 1)]}
        log["count"] += 1
        log["runs"].append({"scheme": f"{version}:{f}", "test_auc": block["test"]["auc"]})
        log_path.write_text(json.dumps(log, indent=2))
        test_rows = splits["test"]
        s_test = formula(test_rows, pol, f, gate)
        rng = random.Random(SEED)
        nli_thr = res["baselines"]["nli_contradiction"]["threshold"]
        res["final"] = {
            "scheme": {"version": version, "formula": f, "formula_text": FORMULA_TEXT[f],
                       "validation_auc": val_auc, "weights": weights},
            "result": block, "baselines_test": base_test, "test_scorings": log,
            "errors": {"false_positives": errors(test_rows, s_test, block["threshold"], pol, 15, rng, "fp"),
                       "false_negatives": errors(test_rows, s_test, block["threshold"], pol, 15, rng, "fn")},
            "nli_threshold": nli_thr,
        }
        res["worked_examples"] = {s: {v: round(P.polarity(s, P.FITTED[v]["base"] if v in P.FITTED else v,
                                                            weights_store.get(v, {}).get("weights")), 3)
                                      for v in res["iterations"]}
                                  for s in P.WORKED_EXAMPLES}
        print("TEST (scored", log["count"], "time(s)):", json.dumps({k: block["test"][k] for k in
                                                                    ("auc", "f1", "accuracy")}))
    RESULTS.write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
