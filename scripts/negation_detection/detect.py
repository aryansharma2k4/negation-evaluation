"""Detecting whether sentence B is the negation of sentence A.

Three architectures and three baselines, on CANNOT (phase 1 / phase 2), on a
mixed held-out set, and once on xNot360.

    ./venv/bin/python scripts/negation_detection/detect.py
    ./venv/bin/python scripts/negation_detection/detect.py --model mpnet --no-nli

Reuses the embedding cache, the loader and the axis machinery from
``scripts/negation_direction``; nothing in ``negation/`` is touched.

Architecture 1  feature scorer: logistic regression over
                [p(A), p(B), delta_p, c_raw, c_resid, cue_delta]
Architecture 2  reference-set comparison: generate variants of A with the
                stage-1 generator, split them into NEG / SAME by net polarity,
                and score B by max-similarity margin under three similarity
                functions (cosine, residual-cosine + polarity agreement, NLI)
Architecture 3  hybrid: architecture 1's features plus architecture 2's margin
Baselines       raw cosine, lexical cue_delta, NLI contradiction probability
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DIRECTION = REPO / "scripts" / "negation_direction"
sys.path.insert(0, str(DIRECTION))
sys.path.insert(0, str(REPO))

# `experiment` here is the negation_direction module: same embedder, same cache,
# same cue lexicon. This file is named detect.py so the two never collide.
from experiment import cue_count, embed, unit  # noqa: E402

SEED = 0
FOLDS = 5
#: architecture 2 under the NLI similarity is the expensive one: it needs a
#: cross-encoder pass per (B, reference sentence). Capped to keep the whole run
#: on CPU in the tens of minutes; the caps are reported with the results.
NLI_REF_CAP = 8          # top-N of each reference set, ranked by cosine
NLI_SUBSAMPLE = 300      # pairs per evaluation set
FEATURES = ["p_a", "p_b", "delta_p", "c_raw", "c_resid", "cue_delta"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(name: str) -> list[dict]:
    folder = ("generated", "negation_detection") if name == "refsets" else ("splits",)
    path = REPO.joinpath("datasets", *folder, f"{name}.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_refsets() -> dict[str, dict]:
    return {r["a"]: r for r in load("refsets")}


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def variant_embeddings(texts: list[str], model_key: str) -> dict[str, np.ndarray]:
    """Embeddings for the ~100k generated reference sentences.

    Same embedder as everything else, but a cache file of its own: these
    sentences are specific to this experiment and would swamp the shared cache
    that negation_direction keeps.
    """
    return embed(texts, model_key,
                 cache_path=REPO / ".cache" / "negation_detection" / f"variants_{model_key}.pkl")


# ---------------------------------------------------------------------------
# Negation axis and pair features
# ---------------------------------------------------------------------------

def fit_axis(rows: list[dict], emb: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    """Mean difference vector over the negated pairs of the training rows.

    Sign convention as in negation_direction: the axis points A -> negated B, so
    a higher projection means more negated. (That experiment found the mean
    difference and a logistic probe give axes at cosine 0.975-0.988, so only the
    cheap one is fit here.)
    """
    diffs = np.stack([emb[r["b"]] - emb[r["a"]] for r in rows if r["label"] == 1])
    axis = unit(diffs.mean(axis=0))
    projections = [float(emb[r["a"]] @ axis) for r in rows if r["label"] == 1]
    projections += [float(emb[r["b"]] @ axis) for r in rows if r["label"] == 1]
    return axis, float(np.mean(projections))


def residual(vectors: np.ndarray, axis: np.ndarray) -> np.ndarray:
    out = vectors - np.outer(vectors @ axis, axis)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def pair_features(rows, emb, axis) -> np.ndarray:
    A = np.stack([emb[r["a"]] for r in rows])
    B = np.stack([emb[r["b"]] for r in rows])
    p_a, p_b = A @ axis, B @ axis
    c_raw = np.einsum("ij,ij->i", A, B)
    c_resid = np.einsum("ij,ij->i", residual(A, axis), residual(B, axis))
    cue_delta = np.array([cue_count(r["b"]) - cue_count(r["a"]) for r in rows], float)
    return np.column_stack([p_a, p_b, p_b - p_a, c_raw, c_resid, cue_delta])


# ---------------------------------------------------------------------------
# Architecture 2: reference-set comparison
# ---------------------------------------------------------------------------

def arch2_scores(rows, emb, vemb, axis, refsets) -> dict[str, np.ndarray]:
    """max-similarity against the NEG and SAME reference sets, two similarities.

    (i)  raw cosine
    (ii) residual cosine minus polarity distance: cos of the parts of the two
         embeddings orthogonal to the negation axis, penalised by how far apart
         they sit *along* it. Content agreement, polarity disagreement.
    """
    out = {k: np.zeros(len(rows)) for k in
           ("s_neg_cos", "s_same_cos", "s_neg_res", "s_same_res")}
    coverage_fail = np.zeros(len(rows), dtype=bool)
    axis_u = axis

    for i, row in enumerate(rows):
        ref = refsets.get(row["a"])
        b = emb[row["b"]]
        p_b = float(b @ axis_u)
        b_res = residual(b[None, :], axis_u)[0]
        neg_texts = [v["text"] for v in ref["neg"]] if ref else []
        same_texts = ([v["text"] for v in ref["same"]] if ref else []) + [row["a"]]
        coverage_fail[i] = not neg_texts

        for name, texts in (("neg", neg_texts), ("same", same_texts)):
            if not texts:
                out[f"s_{name}_cos"][i] = -1.0
                out[f"s_{name}_res"][i] = -2.0
                continue
            V = np.stack([vemb.get(t, emb.get(t)) for t in texts])
            out[f"s_{name}_cos"][i] = float(np.max(V @ b))
            sims = residual(V, axis_u) @ b_res - np.abs(V @ axis_u - p_b)
            out[f"s_{name}_res"][i] = float(np.max(sims))

    out["margin_cos"] = out["s_neg_cos"] - out["s_same_cos"]
    out["margin_res"] = out["s_neg_res"] - out["s_same_res"]
    out["coverage_fail"] = coverage_fail
    out["max_sim_cos"] = np.maximum(out["s_neg_cos"], out["s_same_cos"])
    return out


# ---------------------------------------------------------------------------
# NLI cross-encoder
# ---------------------------------------------------------------------------

class NLI:
    """cross-encoder/nli-deberta-v3-base, cached on disk by (premise, hypothesis)."""

    NAME = "cross-encoder/nli-deberta-v3-base"

    def __init__(self) -> None:
        self.path = REPO / ".cache" / "negation_detection" / "nli.pkl"
        self.cache = pickle.loads(self.path.read_bytes()) if self.path.exists() else {}
        self.model = None

    def _load(self):
        if self.model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.NAME).eval()
            self.labels = [self.model.config.id2label[i].lower()
                           for i in range(self.model.config.num_labels)]
        return self.model

    def score(self, pairs: list[tuple[str, str]], batch: int = 16) -> np.ndarray:
        """P(contradiction), P(entailment) per (premise, hypothesis) pair."""
        todo = [p for p in dict.fromkeys(pairs) if p not in self.cache]
        if todo:
            self._load()
            start = time.time()
            for i in range(0, len(todo), batch):
                chunk = todo[i : i + batch]
                enc = self.tokenizer([p for p, _ in chunk], [h for _, h in chunk],
                                     padding=True, truncation=True, max_length=256,
                                     return_tensors="pt")
                with self.torch.no_grad():
                    logits = self.model(**enc).logits
                probs = self.torch.softmax(logits, dim=1).numpy()
                for (premise, hypothesis), row in zip(chunk, probs):
                    self.cache[(premise, hypothesis)] = {
                        label: float(p) for label, p in zip(self.labels, row)}
                if i % (batch * 20) == 0:
                    rate = (i + len(chunk)) / max(time.time() - start, 1e-9)
                    print(f"    nli {i + len(chunk)}/{len(todo)} ({rate:.1f}/s)", flush=True)
            self.path.write_bytes(pickle.dumps(self.cache))
        return np.array([[self.cache[p].get("contradiction", 0.0),
                          self.cache[p].get("entailment", 0.0)] for p in pairs])


def arch2_nli(rows, emb, vemb, refsets, nli: NLI, rng) -> dict:
    """Architecture 2 under the NLI similarity, on a subsample (cost)."""
    idx = rng.choice(len(rows), size=min(NLI_SUBSAMPLE, len(rows)), replace=False)
    pairs, meta = [], []
    for i in idx:
        row = rows[int(i)]
        ref = refsets.get(row["a"])
        b = emb[row["b"]]
        for name in ("neg", "same"):
            texts = [v["text"] for v in ref[name]] if ref else []
            if name == "same":
                texts = texts + [row["a"]]
            if texts:
                V = np.stack([vemb.get(t, emb.get(t)) for t in texts])
                order = np.argsort(-(V @ b))[:NLI_REF_CAP]   # closest references first
                texts = [texts[j] for j in order]
            meta.append((int(i), name, len(pairs), len(texts)))
            pairs.extend((t, row["b"]) for t in texts)
    scores = nli.score(pairs)
    s = {"neg": {}, "same": {}}
    for i, name, start, count in meta:
        s[name][i] = float(np.max(scores[start : start + count, 1])) if count else 0.0
    order = [int(i) for i in idx]
    return {"index": order,
            "s_neg": np.array([s["neg"][i] for i in order]),
            "s_same": np.array([s["same"][i] for i in order]),
            "margin": np.array([s["neg"][i] - s["same"][i] for i in order]),
            "y": np.array([rows[i]["label"] for i in order])}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = score >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"auc": round(float(roc_auc_score(y, score)), 4) if len(set(y)) > 1 else None,
            "accuracy": round((tp + tn) / len(y), 4),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "threshold": round(float(threshold), 4)}


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising F1 on the data it is given (train folds only)."""
    candidates = np.unique(score)
    if len(candidates) > 400:
        candidates = np.quantile(score, np.linspace(0, 1, 400))
    best, best_f1 = candidates[0], -1.0
    for t in candidates:
        m = metrics(y, score, t)
        if m["f1"] > best_f1:
            best, best_f1 = t, m["f1"]
    return float(best)


def breakdown(rows, y, score, threshold, key) -> dict:
    out = {}
    for value in sorted({r.get(key) for r in rows if r.get(key) is not None}):
        idx = [i for i, r in enumerate(rows) if r.get(key) == value]
        yy, ss = y[idx], score[idx]
        pred = ss >= threshold
        block = {"n": len(idx), "predicted_negation_rate": round(float(pred.mean()), 4)}
        if len(set(yy)) > 1:
            block |= metrics(yy, ss, threshold)
        elif yy[0] == 0:
            block["false_positive_rate"] = round(float(pred.mean()), 4)
        else:
            block["recall"] = round(float(pred.mean()), 4)
        out[str(value)] = block
    return out


# ---------------------------------------------------------------------------
# Scorers: one split in, scores out
# ---------------------------------------------------------------------------

def labels(rows) -> np.ndarray:
    return np.array([r["label"] for r in rows])


def run_split(train_rows, test_rows, emb, vemb, refsets, nli_scores=None) -> dict:
    """Fit every architecture on `train_rows`, score `test_rows`.

    The axis, the logistic regressions, the feature scaling and every threshold
    come from the training rows only.
    """
    axis, _ = fit_axis(train_rows, emb)
    Xtr, Xte = pair_features(train_rows, emb, axis), pair_features(test_rows, emb, axis)
    ytr, yte = labels(train_rows), labels(test_rows)
    a2tr = arch2_scores(train_rows, emb, vemb, axis, refsets)
    a2te = arch2_scores(test_rows, emb, vemb, axis, refsets)

    def logistic(train_X, test_X, names):
        mu, sd = train_X.mean(axis=0), train_X.std(axis=0) + 1e-9
        model = LogisticRegression(max_iter=5000).fit((train_X - mu) / sd, ytr)
        return (model.predict_proba((train_X - mu) / sd)[:, 1],
                model.predict_proba((test_X - mu) / sd)[:, 1],
                dict(zip(names, np.round(model.coef_[0], 4).tolist())))

    out: dict[str, dict] = {}
    tr, te, coef = logistic(Xtr, Xte, FEATURES)
    out["arch1_features"] = {"train": tr, "test": te, "coef": coef}

    out["arch2_cosine_margin"] = {"train": a2tr["margin_cos"], "test": a2te["margin_cos"]}
    out["arch2_resid_margin"] = {"train": a2tr["margin_res"], "test": a2te["margin_res"]}

    hyb_tr = np.column_stack([Xtr, a2tr["margin_cos"], a2tr["margin_res"]])
    hyb_te = np.column_stack([Xte, a2te["margin_cos"], a2te["margin_res"]])
    tr, te, coef = logistic(hyb_tr, hyb_te, FEATURES + ["margin_cos", "margin_res"])
    out["arch3_hybrid"] = {"train": tr, "test": te, "coef": coef}

    out["baseline_cosine"] = {"train": -Xtr[:, 3], "test": -Xte[:, 3]}
    out["baseline_cue_delta"] = {"train": Xtr[:, 5], "test": Xte[:, 5]}
    if nli_scores is not None:
        out["baseline_nli_contradiction"] = {
            "train": nli_scores["train"], "test": nli_scores["test"]}

    for name, block in out.items():
        block["threshold"] = best_threshold(ytr, block["train"])
    out["_axis"] = axis
    out["_a2_test"] = a2te
    out["_a2_train"] = a2tr
    out["_y"] = yte
    return out


def three_way(a2, margin, thresholds) -> list[str]:
    """negation / agrees / unrelated, with 'unrelated' decided by max similarity."""
    labels_out = []
    for i in range(len(margin)):
        if a2["max_sim_cos"][i] < thresholds["unrelated"]:
            labels_out.append("unrelated")
        elif margin[i] >= thresholds["negation"]:
            labels_out.append("negation")
        else:
            labels_out.append("agrees")
    return labels_out


def gold_three_way(rows) -> list[str]:
    mapping = {"negation": "negation", "paraphrase": "agrees", "not_negation": "agrees",
               "sts_low": "unrelated", "snli_neutral": "unrelated",
               "random_pair": "unrelated"}
    return [mapping[r["subtype"]] for r in rows]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(model_key, phase2_rows, scores, a2, features, y) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = REPO / "reports" / "figures" / "negation_detection"
    plots.mkdir(exist_ok=True)
    subtypes = sorted({r["subtype"] for r in phase2_rows})

    # 1. score histograms per class, for each architecture
    names = [n for n in scores if not n.startswith("_")]
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 2.1 * len(names)))
    for ax, name in zip(np.atleast_1d(axes), names):
        values = scores[name]["test"]
        bins = np.linspace(float(np.min(values)), float(np.max(values)), 40)
        for subtype in subtypes:
            idx = [i for i, r in enumerate(phase2_rows) if r["subtype"] == subtype]
            ax.hist(values[idx], bins=bins, alpha=0.5, label=subtype, density=True)
        ax.axvline(scores[name]["threshold"], color="k", ls="--", lw=1)
        ax.set_ylabel(name, fontsize=8)
        ax.legend(fontsize=6)
    np.atleast_1d(axes)[0].set_title(f"{model_key}: phase-2 scores by subtype "
                                     f"(dashed = threshold fit on phase 1)")
    fig.tight_layout()
    fig.savefig(plots / f"scores_{model_key}.png", dpi=130)
    plt.close(fig)

    # 2. c_resid vs delta_p
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for subtype in subtypes:
        idx = [i for i, r in enumerate(phase2_rows) if r["subtype"] == subtype]
        ax.scatter(features[idx, 2], features[idx, 4], s=9, alpha=0.5, label=subtype)
    ax.set_xlabel("delta_p  (projection shift along the negation axis)")
    ax.set_ylabel("c_resid  (cosine with the axis projected out)")
    ax.set_title(f"{model_key}: phase 2, the two features that carry the signal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / f"resid_vs_deltap_{model_key}.png", dpi=130)
    plt.close(fig)

    # 3. margin distributions
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    for ax, key, title in ((axes[0], "margin_cos", "cosine margin  (s_neg - s_same)"),
                           (axes[1], "margin_res", "residual+polarity margin")):
        values = a2[key]
        bins = np.linspace(float(np.min(values)), float(np.max(values)), 40)
        for subtype in subtypes:
            idx = [i for i, r in enumerate(phase2_rows) if r["subtype"] == subtype]
            ax.hist(values[idx], bins=bins, alpha=0.5, label=subtype, density=True)
        ax.axvline(0, color="k", lw=1)
        ax.set_title(f"{model_key}: {title}", fontsize=10)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(plots / f"margins_{model_key}.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def group_folds(rows, n_folds: int, rng) -> list[np.ndarray]:
    """5 folds, grouped by sentence A: no A appears in two folds."""
    groups = sorted({r["a"] for r in rows})
    rng.shuffle(groups)
    assignment = {g: i % n_folds for i, g in enumerate(groups)}
    return [np.array([i for i, r in enumerate(rows) if assignment[r["a"]] == k])
            for k in range(n_folds)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="minilm", choices=["minilm", "mpnet", "bert"])
    ap.add_argument("--no-nli", action="store_true", help="skip the cross-encoder")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="results.json",
                    help="results file, relative to this directory")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    phase1, phase2, external = load("phase1"), load("phase2"), load("xnot360")
    refsets = load_refsets()
    all_rows = phase1 + phase2 + external

    print("embedding pair sentences ...", flush=True)
    sentences = [s for r in all_rows for s in (r["a"], r["b"])]
    emb = embed(sentences, args.model)
    print("embedding generated reference sentences ...", flush=True)
    variants = [v["text"] for r in refsets.values() for v in r["neg"] + r["same"]]
    vemb = variant_embeddings(variants, args.model)
    print(f"  {len(set(sentences))} pair sentences, {len(set(variants))} variants",
          flush=True)

    nli = None if args.no_nli else NLI()
    nli_all = None
    if nli is not None:
        print("NLI baseline over every pair ...", flush=True)
        probs = nli.score([(r["a"], r["b"]) for r in all_rows])
        nli_all = {id(r): float(p) for r, p in zip(all_rows, probs[:, 0])}

    def nli_for(rows):
        return None if nli_all is None else np.array([nli_all[id(r)] for r in rows])

    results = {
        "config": {"model": args.model, "seed": args.seed, "folds": FOLDS,
                   "features": FEATURES, "nli_model": NLI.NAME if nli else None,
                   "nli_ref_cap": NLI_REF_CAP, "nli_subsample": NLI_SUBSAMPLE,
                   "threshold_rule": "maximises F1 on the training rows"},
        "data": {name: {"n": len(rows),
                        "by_subtype": dict(Counter(r["subtype"] for r in rows)),
                        "by_source": dict(Counter(r["source"] for r in rows)),
                        "by_neg_type": dict(Counter(r["neg_type"] for r in rows
                                                    if r["neg_type"]))}
                 for name, rows in (("phase1", phase1), ("phase2", phase2),
                                    ("xnot360", external))},
        "coverage": {}, "cv": {}, "phase2": {}, "xnot360": {}, "three_way": {},
        "arch2_nli": {},
    }

    # generator coverage
    for name, rows in (("phase1", phase1), ("phase2", phase2), ("xnot360", external)):
        fails = [r for r in rows if refsets[r["a"]]["coverage_fail"]]
        empty = [r for r in rows if refsets[r["a"]]["n_variants"] == 0]
        results["coverage"][name] = {
            "pairs": len(rows),
            "unique_a": len({r["a"] for r in rows}),
            "pairs_with_no_NEG_variant": len(fails),
            "pairs_with_no_variants_at_all": len(empty),
            "rate_no_NEG": round(len(fails) / len(rows), 4),
            "mean_neg_set_size": round(float(np.mean(
                [len(refsets[r["a"]]["neg"]) for r in rows])), 2),
            "mean_same_set_size": round(float(np.mean(
                [len(refsets[r["a"]]["same"]) for r in rows])), 2),
        }

    # ---- 5-fold CV on phase 1, grouped by A
    print("5-fold CV on phase 1 ...", flush=True)
    folds = group_folds(phase1, FOLDS, np.random.default_rng(args.seed))
    fold_metrics: dict[str, list] = defaultdict(list)
    pooled: dict[str, list] = defaultdict(list)
    pooled_y: list = []
    coefs: dict[str, list] = defaultdict(list)
    for k, test_idx in enumerate(folds):
        train_idx = np.array([i for i in range(len(phase1)) if i not in set(test_idx.tolist())])
        train_rows = [phase1[i] for i in train_idx]
        test_rows = [phase1[i] for i in test_idx]
        nli_scores = None
        if nli_all is not None:
            nli_scores = {"train": nli_for(train_rows), "test": nli_for(test_rows)}
        split = run_split(train_rows, test_rows, emb, vemb, refsets, nli_scores)
        y = labels(test_rows)
        pooled_y.extend(y.tolist())
        for name, block in split.items():
            if name.startswith("_"):
                continue
            fold_metrics[name].append(metrics(y, block["test"], block["threshold"]))
            pooled[name].extend(block["test"].tolist())
            if "coef" in block:
                coefs[name].append(block["coef"])
        print(f"  fold {k + 1}/{FOLDS}", flush=True)

    pooled_y = np.array(pooled_y)
    for name, values in pooled.items():
        values = np.array(values)
        results["cv"][name] = {
            "pooled_auc": round(float(roc_auc_score(pooled_y, values)), 4),
            "fold_mean": {k: round(float(np.mean([m[k] for m in fold_metrics[name]])), 4)
                          for k in ("auc", "accuracy", "precision", "recall", "f1")},
            "fold_std": {k: round(float(np.std([m[k] for m in fold_metrics[name]])), 4)
                         for k in ("auc", "accuracy", "precision", "recall", "f1")},
            "folds": fold_metrics[name],
        }
        if name in coefs:
            results["cv"][name]["coefficients_mean"] = {
                f: round(float(np.mean([c[f] for c in coefs[name]])), 4)
                for f in coefs[name][0]}

    # ---- train on all of phase 1, evaluate once on phase 2, once on xNot360
    print("final fit on phase 1 -> phase 2, xNot360 ...", flush=True)
    for name, rows in (("phase2", phase2), ("xnot360", external)):
        nli_scores = None
        if nli_all is not None:
            nli_scores = {"train": nli_for(phase1), "test": nli_for(rows)}
        split = run_split(phase1, rows, emb, vemb, refsets, nli_scores)
        y = labels(rows)
        block = {}
        for scorer, sc in split.items():
            if scorer.startswith("_"):
                continue
            entry = metrics(y, sc["test"], sc["threshold"])
            entry["by_subtype"] = breakdown(rows, y, sc["test"], sc["threshold"], "subtype")
            entry["by_neg_type"] = breakdown(rows, y, sc["test"], sc["threshold"], "neg_type")
            entry["by_source"] = breakdown(rows, y, sc["test"], sc["threshold"], "source")
            if "coef" in sc:
                entry["coefficients"] = sc["coef"]
            block[scorer] = entry
        results[name] = block

        if name == "phase2":
            a2 = split["_a2_test"]
            a2tr = split["_a2_train"]
            thresholds = {
                # "unrelated" cannot be fit on phase 1 (it has no unrelated pairs),
                # so the threshold is the 5th percentile of what *related* training
                # pairs look like: below that, B matches nothing the generator made.
                "unrelated": float(np.percentile(a2tr["max_sim_cos"], 5)),
                "negation": split["arch2_cosine_margin"]["threshold"],
            }
            predicted = three_way(a2, a2["margin_cos"], thresholds)
            gold = gold_three_way(rows)
            matrix = Counter(zip(gold, predicted))
            # Where the margins actually fall, by subtype: separates "the margin
            # has no signal" from "the threshold is in the wrong place".
            results["margin_signs"] = {}
            scorer_of = {"margin_cos": "arch2_cosine_margin",
                         "margin_res": "arch2_resid_margin"}
            for sub in sorted({r["subtype"] for r in rows}):
                sel = np.array([i for i, r in enumerate(rows) if r["subtype"] == sub])
                for key, scorer in scorer_of.items():
                    v = a2[key][sel]
                    results["margin_signs"].setdefault(sub, {})[key] = {
                        "n": len(sel),
                        "frac_positive": round(float(np.mean(v > 1e-9)), 4),
                        "frac_zero": round(float(np.mean(np.abs(v) <= 1e-9)), 4),
                        "frac_negative": round(float(np.mean(v < -1e-9)), 4),
                        "rate_above_fitted_threshold": round(float(np.mean(
                            v >= split[scorer]["threshold"])), 4),
                        "fitted_threshold": round(split[scorer]["threshold"], 4),
                    }
            results["three_way"] = {
                "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
                "confusion": {f"{g}->{p}": n for (g, p), n in sorted(matrix.items())},
                "accuracy": round(sum(n for (g, p), n in matrix.items() if g == p)
                                  / len(rows), 4),
            }
            features = pair_features(rows, emb, split["_axis"])
            make_plots(args.model, rows, {k: v for k, v in split.items()
                                          if not k.startswith("_")}, a2, features, y)

    # ---- architecture 2 with the NLI similarity, on a subsample
    if nli is not None:
        print("architecture 2 under the NLI similarity (subsample) ...", flush=True)
        for name, rows in (("phase1", phase1), ("phase2", phase2)):
            block = arch2_nli(rows, emb, vemb, refsets, nli,
                              np.random.default_rng(args.seed))
            y = block["y"]
            threshold = best_threshold(y, block["margin"]) if name == "phase1" else \
                results["arch2_nli"]["phase1"]["metrics"]["threshold"]
            # The plain NLI(A, B) baseline on exactly the same subsample, so the
            # two are comparable (the headline baseline runs on every pair).
            base = np.array([nli_all[id(rows[i])] for i in block["index"]])
            base_threshold = best_threshold(y, base) if name == "phase1" else \
                results["arch2_nli"]["phase1"]["nli_baseline_same_subsample"]["threshold"]
            results["arch2_nli"][name] = {
                "n": len(y),
                "metrics": metrics(y, block["margin"], threshold),
                "nli_baseline_same_subsample": metrics(y, base, base_threshold),
                "mean_s_neg": round(float(block["s_neg"].mean()), 4),
                "mean_s_same": round(float(block["s_same"].mean()), 4),
                "note": f"subsample of {len(y)} pairs, reference sets capped at "
                        f"{NLI_REF_CAP} closest by cosine",
            }

    # ---- what each feature is worth on its own (axis fit on all of phase 1)
    axis, _ = fit_axis(phase1, emb)
    results["features"] = {}
    for name, rows in (("phase1", phase1), ("phase2", phase2), ("xnot360", external)):
        X, y = pair_features(rows, emb, axis), labels(rows)
        results["features"][name] = {
            "per_feature_auc": {f: round(float(roc_auc_score(y, X[:, i])), 4)
                                for i, f in enumerate(FEATURES)},
            "mean_by_class": {str(c): {f: round(float(X[y == c, i].mean()), 4)
                                       for i, f in enumerate(FEATURES)}
                              for c in (0, 1)},
            "note": "phase-1 numbers are in-sample for the axis, which is fit on "
                    "phase-1 negated pairs; phase-2 and xNot360 are not.",
        }

    (REPO / "reports" / "results" / "negation_detection" / args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {REPO / 'reports' / 'results' / 'negation_detection' / args.out}")
    print("\nphase 2 (trained on phase 1):")
    for scorer, entry in results["phase2"].items():
        para = entry["by_subtype"].get("paraphrase", {})
        print(f"  {scorer:<28} auc={entry['auc']:<7} f1={entry['f1']:<7} "
              f"acc={entry['accuracy']:<7} paraphrase FPR="
              f"{para.get('false_positive_rate')}")


if __name__ == "__main__":
    main()
