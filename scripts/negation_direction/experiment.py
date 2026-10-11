"""Is there a consistent 'negation direction' in sentence-embedding space?

Exploratory measurement on the stage-1 variant corpus.  Nothing here modifies
the generator; it only reads its output.

    ./venv/bin/python scripts/negation_direction/experiment.py
    ./venv/bin/python scripts/negation_direction/experiment.py --models minilm

Writes results.json and plots/*.png next to this file.  Embeddings are cached
under cache/ keyed by (model, sentence), so reruns are seconds.

Embeddings come from transformers directly: mean pooling over the last hidden
state with the attention mask, then L2 normalisation.  That is exactly what
sentence-transformers does for all-MiniLM-L6-v2 and all-mpnet-base-v2 (both are
Transformer -> mean pool -> normalize, no dense layer), so the extra dependency
buys nothing.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from negation.lexicons import (  # noqa: E402
    IMPLICIT_TRIGGER_LEMMAS,
    IMPLICIT_TRIGGER_PREDICATES,
    LEXICALLY_NEGATIVE,
    NEGATIVE_ADVERBS,
    NEGATIVE_QUANTIFIER_FLIPS,
    NEGATIVE_QUANTIFIERS,
    PRIVATIVE_HEADS,
)

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bert": "bert-base-uncased",
}

#: Spec groups.  L_scope_position and M_modal are not in any of them; they are
#: reported under "other" and excluded from the four headline groups.
GROUP_OF_FAMILY = {
    "A_syntactic": "explicit", "B_quantifier": "explicit", "C_neg_adverb": "explicit",
    "H_hedged": "explicit", "I_intensified": "explicit",
    "D_affixal": "implicit", "E_antonym": "implicit", "F_implicit": "implicit",
    "G_prepositional": "implicit", "J_contrastive": "implicit",
    "K1_cancellation": "cancellation", "K2_compound": "compound",
    "L_scope_position": "other", "M_modal": "other",
    "CONTROL_paraphrase": "control",
}
GROUPS = ["explicit", "implicit", "cancellation", "compound", "other", "control"]
TRAIN_FRAC = 0.7
SEED = 0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_pairs() -> list[dict]:
    """(base, variant) pairs from the stage-1 corpus plus the paraphrase controls."""
    keep = ("base_sentence", "variant", "family", "subtype", "net_negation",
            "intensity_hint", "operation")
    rows = []
    for path in (REPO / "datasets" / "generated" / "negation_direction" / "variants.jsonl", REPO / "datasets" / "generated" / "negation_direction" / "control_pairs.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            pair = {k: record[k] for k in keep}
            pair["group"] = GROUP_OF_FAMILY[record["family"]]
            rows.append(pair)
    # One record de-negates its input ("The task is impossible." -> "... possible.").
    # It is the only operation != negate/paraphrase and would enter every group
    # statistic with the opposite expected sign, so it is held out and reported
    # on its own.
    affirms = [r for r in rows if r["operation"] == "affirm"]
    rows = [r for r in rows if r["operation"] != "affirm"]
    return rows, affirms


def split_bases(bases: list[str], seed: int = SEED) -> tuple[set[str], set[str]]:
    """70/30 split BY BASE SENTENCE, so no base sentence appears in both sides."""
    order = sorted(bases)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    cut = round(TRAIN_FRAC * len(order))
    return set(order[:cut]), set(order[cut:])


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def embed(sentences: list[str], model_key: str, batch: int = 32,
          cache_path: Path | None = None) -> dict[str, np.ndarray]:
    """L2-normalised mean-pooled embeddings, disk-cached by (model, sentence).

    ``cache_path`` lets another experiment keep its own cache file (the sibling
    negation_detection run embeds ~100k generated sentences, which have no
    business in this experiment's cache).
    """
    cache_path = cache_path or REPO / ".cache" / "negation_direction" / f"{model_key}.pkl"
    cache_path.parent.mkdir(exist_ok=True)
    cache = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    todo = [s for s in dict.fromkeys(sentences) if s not in cache]
    if todo:
        name = MODELS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).eval()
        for i in range(0, len(todo), batch):
            chunk = todo[i : i + batch]
            enc = tokenizer(chunk, padding=True, truncation=True, max_length=256,
                            return_tensors="pt")
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            for sentence, vector in zip(chunk, pooled):
                cache[sentence] = vector.numpy().astype(np.float32)
        cache_path.write_bytes(pickle.dumps(cache))
    return cache


# ---------------------------------------------------------------------------
# Small stats helpers
# ---------------------------------------------------------------------------

def stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"n": 0}
    q25, med, q75 = np.percentile(a, [25, 50, 75])
    return {"n": int(a.size), "mean": round(float(a.mean()), 4),
            "std": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 4),
            "min": round(float(a.min()), 4), "q25": round(float(q25), 4),
            "median": round(float(med), 4), "q75": round(float(q75), 4),
            "max": round(float(a.max()), 4),
            "values": [round(float(x), 4) for x in a]}


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def by_group(pairs, values, groups=GROUPS) -> dict:
    return {g: stats([v for p, v in zip(pairs, values) if p["group"] == g]) for g in groups}


def by_family(pairs, values) -> dict:
    families = sorted({p["family"] for p in pairs})
    return {f: stats([v for p, v in zip(pairs, values) if p["family"] == f]) for f in families}


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

def fit_axes(train_pairs, emb) -> dict[str, dict]:
    """Two negation axes fit on explicit pairs only.

    Sign convention: both axes point base -> negated variant, so a *higher*
    projection means *more negated*.  (The brief phrases this as the negated
    sentence scoring lower; that is the same statement with the axis negated.)
    """
    explicit = [p for p in train_pairs if p["group"] == "explicit"]
    diffs = np.stack([emb[p["variant"]] - emb[p["base_sentence"]] for p in explicit])
    axes = {"mean_diff": unit(diffs.mean(axis=0))}

    labelled: dict[str, int] = {}
    for p in explicit:
        labelled.setdefault(p["base_sentence"], 0)
        labelled[p["variant"]] = 1
    X = np.stack([emb[s] for s in labelled])
    y = np.fromiter(labelled.values(), dtype=int)
    probe = LogisticRegression(max_iter=5000, class_weight="balanced").fit(X, y)
    axes["probe"] = unit(probe.coef_[0])

    out = {}
    for name, axis in axes.items():
        base_scores = [float(emb[p["base_sentence"]] @ axis) for p in explicit]
        var_scores = [float(emb[p["variant"]] @ axis) for p in explicit]
        # Midpoint of the training affirmative and negated means: the "sum to
        # zero" reference point.
        out[name] = {"axis": axis,
                     "midpoint": (float(np.mean(base_scores)) + float(np.mean(var_scores))) / 2}
    return out


def score_pairs(pairs, emb, axis, midpoint) -> dict:
    """Per-group projection behaviour for one axis."""
    deltas, base_side, var_side, opposite = [], [], [], []
    for p in pairs:
        b = float(emb[p["base_sentence"]] @ axis)
        v = float(emb[p["variant"]] @ axis)
        deltas.append(v - b)
        base_side.append(b > midpoint)
        var_side.append(v > midpoint)
        opposite.append(b <= midpoint < v)
    out = {}
    for g in GROUPS:
        idx = [i for i, p in enumerate(pairs) if p["group"] == g]
        if not idx:
            out[g] = {"n": 0}
            continue
        d = [deltas[i] for i in idx]
        out[g] = {
            "delta": stats(d),
            "pairwise_acc_variant_more_negated": round(float(np.mean([x > 0 for x in d])), 4),
            "frac_variant_on_negated_side": round(float(np.mean([var_side[i] for i in idx])), 4),
            "frac_base_on_negated_side": round(float(np.mean([base_side[i] for i in idx])), 4),
            "frac_opposite_sides_of_midpoint": round(float(np.mean([opposite[i] for i in idx])), 4),
        }
    return out


def loocv(pairs, emb) -> dict:
    """Leave-one-base-sentence-out: every pair scored by an axis that never saw its base."""
    bases = sorted({p["base_sentence"] for p in pairs})
    held: dict[str, list] = {"mean_diff": [], "probe": []}
    held_pairs: list[dict] = []
    for base in bases:
        train = [p for p in pairs if p["base_sentence"] != base]
        test = [p for p in pairs if p["base_sentence"] == base]
        if not test or not any(p["group"] == "explicit" for p in train):
            continue
        axes = fit_axes(train, emb)
        held_pairs.extend(test)
        for name, fit in axes.items():
            for p in test:
                b = float(emb[p["base_sentence"]] @ fit["axis"])
                v = float(emb[p["variant"]] @ fit["axis"])
                held[name].append((v - b, b, v, fit["midpoint"]))
    out = {}
    for name, rows in held.items():
        deltas = [r[0] for r in rows]
        out[name] = {
            "by_group": {
                g: {"delta": stats([d for p, d in zip(held_pairs, deltas) if p["group"] == g]),
                    "pairwise_acc_variant_more_negated": round(float(np.mean(
                        [d > 0 for p, d in zip(held_pairs, deltas) if p["group"] == g])), 4)
                    if any(p["group"] == g for p in held_pairs) else None,
                    "frac_variant_on_negated_side": round(float(np.mean(
                        [r[2] > r[3] for p, r in zip(held_pairs, rows) if p["group"] == g])), 4)
                    if any(p["group"] == g for p in held_pairs) else None,
                    "frac_opposite_sides_of_midpoint": round(float(np.mean(
                        [r[1] <= r[3] < r[2] for p, r in zip(held_pairs, rows) if p["group"] == g])), 4)
                    if any(p["group"] == g for p in held_pairs) else None}
                for g in GROUPS},
            "by_family": {
                f: {"delta": stats([d for p, d in zip(held_pairs, deltas) if p["family"] == f]),
                    "pairwise_acc_variant_more_negated": round(float(np.mean(
                        [d > 0 for p, d in zip(held_pairs, deltas) if p["family"] == f])), 4)}
                for f in sorted({p["family"] for p in held_pairs})},
            # The decisive question for a scoring mechanism: at pair level, does
            # the projection shift tell a negation apart from a paraphrase?
            "negation_vs_control_delta_auc": round(float(roc_auc_score(
                [p["group"] != "control" for p in held_pairs], deltas)), 4),
            "cancellation_vs_explicit_delta_auc": round(float(roc_auc_score(
                [p["group"] == "explicit" for p, d in zip(held_pairs, deltas)
                 if p["group"] in ("explicit", "cancellation")],
                [d for p, d in zip(held_pairs, deltas)
                 if p["group"] in ("explicit", "cancellation")])), 4),
        }
    return out, held_pairs, held


# ---------------------------------------------------------------------------
# Lexical baseline
# ---------------------------------------------------------------------------

_BASE_CUES = (
    {"not", "cannot", "no", "nt"}
    | set(NEGATIVE_ADVERBS) | set(NEGATIVE_QUANTIFIERS) | set(NEGATIVE_QUANTIFIER_FLIPS)
    | set(PRIVATIVE_HEADS) | set(LEXICALLY_NEGATIVE)
    | set(IMPLICIT_TRIGGER_LEMMAS) | set(IMPLICIT_TRIGGER_PREDICATES)
)
#: The lexicons store lemmas; the corpus contains inflected forms.
CUES = _BASE_CUES | {w + s for w in _BASE_CUES for s in ("s", "es", "ed", "d", "ing")}
#: Affixal negation the word lists do not enumerate ("unimportant", "useless").
#: Deliberately excludes in-/dis-, which strip into false positives
#: ("increased" -> "creased").
CUE_PREFIXES = ("un", "non", "im", "ir", "il")


def cue_count(sentence: str) -> int:
    """How many negation cues a pure word-lookup finds.  The trivial baseline."""
    words = [w.strip(".,;:!?'\"").replace("n't", " nt").lower() for w in sentence.split()]
    words = [w for chunk in words for w in chunk.split()]
    n = 0
    for w in words:
        if w in CUES:
            n += 1
        elif w.endswith("less") and len(w) > 6:
            n += 1
        elif any(w.startswith(p) and len(w) > len(p) + 4 for p in CUE_PREFIXES):
            n += 1
    return n


def lexical_baseline(pairs) -> dict:
    deltas = [cue_count(p["variant"]) - cue_count(p["base_sentence"]) for p in pairs]
    out = {}
    for g in GROUPS:
        d = [x for p, x in zip(pairs, deltas) if p["group"] == g]
        if not d:
            out[g] = {"n": 0}
            continue
        out[g] = {"n": len(d),
                  "pairwise_acc_variant_more_cues": round(float(np.mean([x > 0 for x in d])), 4),
                  "frac_tied": round(float(np.mean([x == 0 for x in d])), 4),
                  "mean_cue_delta": round(float(np.mean(d)), 4)}
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(model_key, pairs, emb, cosines, held_pairs, held) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = REPO / "reports" / "figures" / "negation_direction"
    plots.mkdir(exist_ok=True)
    present = [g for g in GROUPS if any(p["group"] == g for p in pairs)]

    # 1. cosine(base, variant) distribution per group
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [[c for p, c in zip(pairs, cosines) if p["group"] == g] for g in present]
    ax.boxplot(data, tick_labels=present, showfliers=False)
    rng = np.random.default_rng(SEED)
    for i, values in enumerate(data, start=1):
        ax.scatter(rng.normal(i, 0.06, len(values)), values, s=8, alpha=0.4)
    ax.set_ylabel("cosine(base, variant)")
    ax.set_title(f"{model_key}: how far the embedding moves, by group")
    fig.tight_layout()
    fig.savefig(plots / f"cosine_{model_key}.png", dpi=130)
    plt.close(fig)

    # 2. held-out projections onto the mean-difference axis, base vs variant
    rows = held["mean_diff"]
    fig, axes = plt.subplots(len(present), 1, figsize=(8, 2.1 * len(present)), sharex=True)
    for ax, g in zip(np.atleast_1d(axes), present):
        sel = [r for p, r in zip(held_pairs, rows) if p["group"] == g]
        if not sel:
            continue
        bins = np.linspace(min(min(r[1], r[2]) for r in rows),
                           max(max(r[1], r[2]) for r in rows), 30)
        ax.hist([r[1] for r in sel], bins=bins, alpha=0.6, label="base")
        ax.hist([r[2] for r in sel], bins=bins, alpha=0.6, label="variant")
        ax.axvline(np.mean([r[3] for r in sel]), color="k", ls="--", lw=1, label="train midpoint")
        ax.set_ylabel(g, fontsize=9)
        ax.legend(fontsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("projection onto negation axis (held-out fit)")
    np.atleast_1d(axes)[0].set_title(f"{model_key}: projection onto the negation axis")
    fig.tight_layout()
    fig.savefig(plots / f"projection_{model_key}.png", dpi=130)
    plt.close(fig)

    # 3. PCA of the embeddings, base -> variant arrows for a sample
    sentences = list(dict.fromkeys([p["base_sentence"] for p in pairs]
                                   + [p["variant"] for p in pairs]))
    coords = PCA(n_components=2, random_state=SEED).fit_transform(
        np.stack([emb[s] for s in sentences]))
    index = {s: i for i, s in enumerate(sentences)}
    fig, ax = plt.subplots(figsize=(9, 7))
    for g in present:
        sel = [p for p in pairs if p["group"] == g]
        pts = np.array([coords[index[p["variant"]]] for p in sel])
        ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.6, label=g)
        for p in sel[:: max(1, len(sel) // 8)]:
            b, v = coords[index[p["base_sentence"]]], coords[index[p["variant"]]]
            ax.annotate("", xy=v, xytext=b,
                        arrowprops=dict(arrowstyle="->", alpha=0.35, lw=0.7))
    bases = np.array([coords[index[p["base_sentence"]]] for p in pairs])
    ax.scatter(bases[:, 0], bases[:, 1], s=26, c="k", marker="x", label="base")
    ax.legend(fontsize=8)
    ax.set_title(f"{model_key}: PCA (2D) with base -> variant arrows")
    fig.tight_layout()
    fig.savefig(plots / f"pca_{model_key}.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def analyse(model_key, pairs, train_bases, test_bases, affirms) -> dict:
    sentences = [p["base_sentence"] for p in pairs] + [p["variant"] for p in pairs]
    sentences += [s for p in affirms for s in (p["base_sentence"], p["variant"])]
    emb = embed(sentences, model_key)
    norms = np.linalg.norm(np.stack([emb[s] for s in dict.fromkeys(sentences)]), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "embeddings must be L2-normalised"

    # 1. raw cosine
    cosines = [float(emb[p["base_sentence"]] @ emb[p["variant"]]) for p in pairs]
    negation_pairs = [i for i, p in enumerate(pairs) if p["group"] != "control"]
    control_pairs = [i for i, p in enumerate(pairs) if p["group"] == "control"]
    y = np.r_[np.ones(len(negation_pairs)), np.zeros(len(control_pairs))]
    # Lower cosine is supposed to mean "negated", so -cosine is the score.
    scores = np.array([-cosines[i] for i in negation_pairs + control_pairs])
    thresholds = np.unique(scores)
    accs = [float(((scores > t).astype(float) == y).mean()) for t in thresholds]
    # The classes are wildly imbalanced (454 negation pairs vs 21 controls), so
    # plain accuracy is dominated by the majority class; balanced accuracy is
    # the number that means anything.
    bal = [float((((scores > t)[y == 1]).mean() + ((scores <= t)[y == 0]).mean()) / 2)
           for t in thresholds]
    cosine_block = {
        "by_group": by_group(pairs, cosines),
        "by_family": by_family(pairs, cosines),
        "negation_vs_control_auc": round(float(roc_auc_score(y, scores)), 4),
        "negation_vs_control_best_threshold_acc": round(float(max(accs)), 4),
        "negation_vs_control_best_threshold_balanced_acc": round(float(max(bal)), 4),
        "majority_class_acc": round(float(max(y.mean(), 1 - y.mean())), 4),
        "note": "thresholds are fit on the same data they score: an upper bound.",
    }

    # 2. difference vectors
    diff = {p_i: emb[p["variant"]] - emb[p["base_sentence"]] for p_i, p in enumerate(pairs)}
    diff_block = {"within_group_pairwise_cosine": {}, "group_mean_vector": {},
                  "group_mean_cosine_matrix": {}}
    means = {}
    for g in GROUPS:
        idx = [i for i, p in enumerate(pairs) if p["group"] == g]
        if len(idx) < 2:
            continue
        D = np.stack([unit(diff[i]) for i in idx])
        sim = D @ D.T
        off = sim[np.triu_indices(len(idx), k=1)]
        diff_block["within_group_pairwise_cosine"][g] = stats(off) | {"values": "omitted"}
        means[g] = np.stack([diff[i] for i in idx]).mean(axis=0)
        diff_block["group_mean_vector"][g] = {
            "norm": round(float(np.linalg.norm(means[g])), 4),
            "mean_norm_of_members": round(
                float(np.mean([np.linalg.norm(diff[i]) for i in idx])), 4)}
    for g in means:
        diff_block["group_mean_cosine_matrix"][g] = {
            h: round(float(unit(means[g]) @ unit(means[h])), 4) for h in means}
    for f in sorted({p["family"] for p in pairs}):
        idx = [i for i, p in enumerate(pairs) if p["family"] == f]
        if len(idx) < 2:
            continue
        D = np.stack([unit(diff[i]) for i in idx])
        off = (D @ D.T)[np.triu_indices(len(idx), k=1)]
        diff_block["within_group_pairwise_cosine"][f"family:{f}"] = (
            stats(off) | {"values": "omitted"})

    # 3. axis, fit on TRAIN explicit only, evaluated on TEST
    train_pairs = [p for p in pairs if p["base_sentence"] in train_bases]
    test_pairs = [p for p in pairs if p["base_sentence"] in test_bases]
    fits = fit_axes(train_pairs, emb)
    axis_block = {name: {"midpoint": round(fit["midpoint"], 4),
                         "test": score_pairs(test_pairs, emb, fit["axis"], fit["midpoint"]),
                         "train": score_pairs(train_pairs, emb, fit["axis"], fit["midpoint"])}
                  for name, fit in fits.items()}
    axis_block["mean_diff_vs_probe_cosine"] = round(
        float(fits["mean_diff"]["axis"] @ fits["probe"]["axis"]), 4)

    # LOOCV: every pair scored by an axis fit without its base sentence
    cv, held_pairs, held = loocv(pairs, emb)

    # 4. intensity: hedged vs intensified, held-out projections
    intensity = {}
    for name in ("mean_diff", "probe"):
        rows = held[name]
        intensity[name] = {
            hint: stats([r[0] for p, r in zip(held_pairs, rows) if p["intensity_hint"] == hint])
            for hint in sorted({p["intensity_hint"] for p in held_pairs})}
        intensity[name] |= {
            f"family:{f}": stats([r[0] for p, r in zip(held_pairs, rows) if p["family"] == f])
            for f in ("H_hedged", "I_intensified")}
        hi = [(p["family"] == "I_intensified", r[0]) for p, r in zip(held_pairs, rows)
              if p["family"] in ("H_hedged", "I_intensified")]
        # 0.5 = the axis cannot tell "probably not" from "certainly not" at all.
        intensity[name]["hedged_vs_intensified_auc"] = round(
            float(roc_auc_score([a for a, _ in hi], [d for _, d in hi])), 4)

    # the single de-negation record, reported separately
    affirm_block = [{"base": p["base_sentence"], "variant": p["variant"],
                     "cosine": round(float(emb[p["base_sentence"]] @ emb[p["variant"]]), 4),
                     "delta_mean_diff_axis": round(
                         float((emb[p["variant"]] - emb[p["base_sentence"]])
                               @ fits["mean_diff"]["axis"]), 4)} for p in affirms]

    make_plots(model_key, pairs, emb, cosines, held_pairs, held)
    return {"model": MODELS[model_key], "dim": int(len(next(iter(emb.values())))),
            "cosine": cosine_block, "difference_vectors": diff_block,
            "axis": axis_block, "loocv": cv, "intensity": intensity,
            "denegation_records": affirm_block}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    pairs, affirms = load_pairs()
    bases = sorted({p["base_sentence"] for p in pairs})
    train_bases, test_bases = split_bases(bases, args.seed)

    results = {
        "config": {"seed": args.seed, "train_frac": TRAIN_FRAC,
                   "models": {k: MODELS[k] for k in args.models},
                   "axis_sign": "higher projection = more negated"},
        "data": {
            "n_pairs": len(pairs), "n_base_sentences": len(bases),
            "pairs_per_group": {g: sum(p["group"] == g for p in pairs) for g in GROUPS},
            "pairs_per_family": {f: sum(p["family"] == f for p in pairs)
                                 for f in sorted({p["family"] for p in pairs})},
            "train_bases": sorted(train_bases), "test_bases": sorted(test_bases),
            "train_pairs": sum(p["base_sentence"] in train_bases for p in pairs),
            "test_pairs": sum(p["base_sentence"] in test_bases for p in pairs),
            "held_out_denegation_pairs": len(affirms),
        },
        "lexical_baseline": lexical_baseline(pairs),
        "models": {},
    }
    for key in args.models:
        print(f"[{key}] embedding + analysing ...", flush=True)
        results["models"][key] = analyse(key, pairs, train_bases, test_bases, affirms)

    (REPO / "reports" / "results" / "negation_direction" / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {REPO / 'reports' / 'results' / 'negation_direction' / 'results.json'}")
    for key in args.models:
        cv = results["models"][key]["loocv"]["mean_diff"]["by_group"]
        print(f"\n{key}: LOOCV pairwise accuracy (mean-difference axis)")
        for g in GROUPS:
            row = cv[g]
            if row["delta"]["n"]:
                print(f"  {g:<13} n={row['delta']['n']:<4} "
                      f"acc={row['pairwise_acc_variant_more_negated']:<7} "
                      f"mean delta={row['delta']['mean']}")


if __name__ == "__main__":
    main()
