"""Steps 5-7: validate the labels, search for a grading formula, write PATTERN.md.

    ./venv/bin/python experiments/pattern/analyse.py

Reads   data/to_label.jsonl, data/metadata.jsonl, data/labels_pass{1,2,3}.jsonl,
        data/build_report.json
Writes  results.json, plots/*.png, data/human_seed.csv, ../../PATTERN.md

Every number in PATTERN.md is read from the results dict this script builds;
none is typed. If the step-5 gate fails, only the gate is reported and no
formula is fitted.

Decision rule, fixed before any label existed (PRE_REGISTERED below): a graded
alpha is justified iff the families differ (Kruskal-Wallis p < 0.05) AND the
per-family lookup -- the simplest graded formula, chosen in advance so the
verdict does not rest on the best of five models -- beats the best flat
constant on 5-fold grouped CV MAE, with the 95% base-sentence cluster-bootstrap
CI of the MAE difference excluding zero.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"
PLOTS = HERE / "plots"
SEED = 0
FOLDS = 5
N_BOOT = 2000
HSS_ALPHA = 0.7
RIDGE = 1.0

GATE_KRIPPENDORFF = 0.6
#: anchors "separate" iff paraphrases average in the top band, unrelated pairs
#: in the bottom band, and the two distributions do not overlap in their bulk
GATE_PARA_MEAN_MIN, GATE_UNREL_MEAN_MAX = 4.0, 1.0

PRE_REGISTERED = {
    "omnibus": "Kruskal-Wallis across negation families, p < 0.05",
    "model": "per_family",
    "baseline": "best_flat",
    "criterion": "CV MAE(per_family) < CV MAE(best_flat) and the 95% cluster-bootstrap "
                 "CI of MAE(per_family) - MAE(best_flat) lies entirely below 0",
}

PLANNED = [
    ("H_hedged vs I_intensified", "is intensity real to a rater?",
     lambda m: m["family"] == "H_hedged", lambda m: m["family"] == "I_intensified"),
    ("K1_cancellation vs A_syntactic", "does cancellation behave like no negation?",
     lambda m: m["family"] == "K1_cancellation", lambda m: m["family"] == "A_syntactic"),
    ("K1_cancellation vs paraphrase anchors", "... and is it as close as a paraphrase?",
     lambda m: m["family"] == "K1_cancellation", lambda m: m["family"] == "ANCHOR_paraphrase"),
    ("F_implicit vs A_syntactic", "is implicit negation weaker or equal?",
     lambda m: m["family"] == "F_implicit", lambda m: m["family"] == "A_syntactic"),
    ("C_neg_adverb partial vs absolute", "is 'rarely' weaker than 'never'?",
     lambda m: m.get("subtype") == "partial", lambda m: m.get("subtype") == "absolute"),
]

#: families whose pairs a cue-count detector cannot represent (brief, section 6)
HSS_BLIND = ["F_implicit", "D_affixal", "G_prepositional", "J_contrastive", "K1_cancellation"]
ANCHORS = ["ANCHOR_paraphrase", "ANCHOR_unrelated"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load() -> list[dict]:
    items = {r["id"]: r for r in jsonl(DATA / "to_label.jsonl")}
    meta = {r["id"]: r for r in jsonl(DATA / "metadata.jsonl")}
    passes = {k: {r["id"]: r["score"] for r in jsonl(DATA / f"labels_pass{k}.jsonl")}
              for k in (1, 2, 3)}
    rows = []
    for id_, item in items.items():
        scores = [passes[k].get(id_) for k in (1, 2, 3)]
        present = [s for s in scores if s is not None]
        rows.append({**item, **meta[id_], "scores": scores,
                     "label": float(np.median(present)) if present else None})
    return rows


# ---------------------------------------------------------------------------
# Step 5: label validation
# ---------------------------------------------------------------------------

def krippendorff_ordinal(units: list[list[float | None]]) -> float:
    """Krippendorff's alpha, ordinal metric, missing values allowed.

    Coincidence-matrix form (Krippendorff 2011, "Computing Krippendorff's
    alpha-reliability"): units with fewer than two values are not pairable and
    drop out.
    """
    units = [[v for v in u if v is not None] for u in units]
    units = [u for u in units if len(u) >= 2]
    values = sorted({v for u in units for v in u})
    index = {v: i for i, v in enumerate(values)}
    o = np.zeros((len(values), len(values)))
    for u in units:
        m = len(u)
        for i, j in ((i, j) for i in range(m) for j in range(m) if i != j):
            o[index[u[i]], index[u[j]]] += 1 / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    cum = np.cumsum(n_c)
    delta = np.zeros_like(o)
    for c in range(len(values)):
        for k in range(len(values)):
            lo, hi = min(c, k), max(c, k)
            between = cum[hi] - (cum[lo - 1] if lo > 0 else 0)
            delta[c, k] = (between - (n_c[c] + n_c[k]) / 2) ** 2
    observed = (o * delta).sum()
    expected = (np.outer(n_c, n_c) * delta).sum() / (n - 1)
    return float(1 - observed / expected) if expected else float("nan")


def validate(rows: list[dict]) -> dict:
    units = [r["scores"] for r in rows]
    alpha = krippendorff_ordinal(units)
    spear = {}
    for i, j in combinations(range(3), 2):
        both = [(u[i], u[j]) for u in units if u[i] is not None and u[j] is not None]
        x, y = zip(*both)
        spear[f"pass{i + 1}_vs_pass{j + 1}"] = float(stats.spearmanr(x, y).statistic)

    def paired(i: int, j: int) -> dict:
        both = [(u[i], u[j]) for u in units if u[i] is not None and u[j] is not None]
        x, y = map(np.array, zip(*both))
        t = stats.ttest_rel(x, y)
        return {"n": len(x), f"mean_pass{i + 1}": float(x.mean()), f"mean_pass{j + 1}": float(y.mean()),
                "mean_diff": float((x - y).mean()), "t": float(t.statistic), "p": float(t.pvalue),
                "exact_agreement": float((x == y).mean())}

    anchor = {}
    for fam in ANCHORS:
        labels = np.array([r["label"] for r in rows if r["family"] == fam and r["label"] is not None])
        anchor[fam] = {"n": len(labels), "mean": float(labels.mean()), "std": float(labels.std(ddof=1)),
                       "p10": float(np.percentile(labels, 10)), "p90": float(np.percentile(labels, 90)),
                       "counts": {str(k): int(v) for k, v in sorted(Counter(labels).items())}}
    para, unrel = anchor["ANCHOR_paraphrase"], anchor["ANCHOR_unrelated"]
    separated = bool(para["mean"] >= GATE_PARA_MEAN_MIN and unrel["mean"] <= GATE_UNREL_MEAN_MAX
                     and para["p10"] > unrel["p90"])
    unscored = [sum(r["scores"][k] is None for r in rows) for k in range(3)]

    # Diagnostics for a failed gate. Passes 1 and 2 share batch neighbours; pass
    # 3 does not. A pair that 1 and 2 agree on and 3 moves by >= 3 points is
    # scored by its neighbours, not by its own sentences.
    full = [r for r in rows if None not in r["scores"]]
    neighbour = [r for r in full if r["scores"][0] == r["scores"][1]
                 and abs(r["scores"][2] - r["scores"][0]) >= 3]
    unrel = [r for r in rows if r["family"] == "ANCHOR_unrelated"]
    neg = [r for r in rows if r["kind"] == "negation"]
    diagnostics = {
        "items_where_1_and_2_agree_but_3_moves_3plus": len(neighbour),
        "same_among_unrelated_anchors": sum(r["family"] == "ANCHOR_unrelated" for r in neighbour),
        "unrelated_anchors_median_ge_3": sum(r["label"] >= 3 for r in unrel),
        "unrelated_anchors_n": len(unrel),
        "negation_pairs_n": len(neg),
        "negation_pairs_median_0": sum(r["label"] == 0 for r in neg),
        "negation_pairs_mean_label": float(np.mean([r["label"] for r in neg])),
    }
    return {
        "diagnostics": diagnostics,
        "n_items": len(rows),
        "unscored_per_pass": unscored,
        "krippendorff_alpha_ordinal": alpha,
        "spearman_pairwise": spear,
        "spearman_mean": float(np.mean(list(spear.values()))),
        "order_effect_pass1_vs_pass2": paired(0, 1),
        "context_effect_pass1_vs_pass3": paired(0, 2),
        "anchors": anchor,
        "anchors_separate": separated,
        "gate": {"krippendorff_min": GATE_KRIPPENDORFF, "paraphrase_mean_min": GATE_PARA_MEAN_MIN,
                 "unrelated_mean_max": GATE_UNREL_MEAN_MAX,
                 "passed": bool(alpha >= GATE_KRIPPENDORFF and separated)},
    }


def write_human_seed(rows: list[dict], n: int = 60) -> dict:
    """60 items round-robin across every family and both anchor classes.

    The CSV carries id and the two sentences only -- human raters get the same
    blind view the LLM got.
    """
    rng = random.Random(SEED)
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["family"]].append(r)
    for b in buckets.values():
        rng.shuffle(b)
    picked = []
    while len(picked) < n:
        for fam in sorted(buckets):
            if buckets[fam] and len(picked) < n:
                picked.append(buckets[fam].pop())
    rng.shuffle(picked)
    with (DATA / "human_seed.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "sentence_a", "sentence_b", "rater1", "rater2", "rater3"])
        for r in picked:
            w.writerow([r["id"], r["a"], r["b"], "", "", ""])
    return {"n": len(picked), "per_family": dict(sorted(Counter(r["family"] for r in picked).items()))}


# ---------------------------------------------------------------------------
# Base similarity: STAND-IN for the HSS backbone
# ---------------------------------------------------------------------------

STAND_IN = "SBERT all-mpnet-base-v2 cosine (STAND-IN for the HSS backbone)"


def base_similarity(a: list[str], b: list[str]) -> np.ndarray:
    """sim_base for each pair. STAND-IN for the HSS backbone.

    Replace this one function to evaluate alpha on the real backbone; nothing
    else in the analysis knows where sim_base came from. Uses the
    negation_direction experiment's mean-pooled mpnet embedder (identical to
    sentence-transformers' all-mpnet-base-v2 pooling) with a cache of its own.
    """
    sys.path.insert(0, str(REPO / "experiments" / "negation_direction"))
    from experiment import embed  # noqa: E402

    emb = embed(a + b, "mpnet", cache_path=HERE / "cache" / "mpnet.pkl")
    return np.array([float(emb[x] @ emb[y]) for x, y in zip(a, b)])


# ---------------------------------------------------------------------------
# Step 6: family statistics
# ---------------------------------------------------------------------------

def cluster_boot(groups: np.ndarray, fn, rng: np.random.Generator, n: int = N_BOOT) -> tuple[float, float]:
    """95% percentile CI of ``fn(indices)``, resampling base sentences."""
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    draws = []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        draws.append(fn(idx))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def ls_alpha(y: np.ndarray, s: np.ndarray) -> float:
    """argmin_a sum (y - a s)^2  -- the alpha implied by labels y at sim_base s."""
    return float((y * s).sum() / (s * s).sum())


def family_stats(rows: list[dict], rng: np.random.Generator) -> dict:
    fams = sorted({r["family"] for r in rows})
    table = {}
    for fam in fams:
        sub = [r for r in rows if r["family"] == fam]
        lab = np.array([r["label"] for r in sub])
        y, s = lab / 5, np.array([r["sim"] for r in sub])
        g = np.array([r["base_line"] for r in sub])
        table[fam] = {
            "n": len(sub), "mean_label": float(lab.mean()), "std_label": float(lab.std(ddof=1)),
            "median_label": float(np.median(lab)), "mean_sim_base": float(s.mean()),
            "implied_alpha": ls_alpha(y, s),
            "implied_alpha_ci95": cluster_boot(g, lambda i: ls_alpha(y[i], s[i]), rng),
            "label_counts": {str(k): int(v) for k, v in sorted(Counter(lab).items())},
        }
    neg_fams = [f for f in fams if not f.startswith("ANCHOR")]
    groups = [[r["label"] for r in rows if r["family"] == f] for f in neg_fams]
    kw = stats.kruskal(*groups)
    an = stats.f_oneway(*groups)
    n_total = sum(map(len, groups))
    grand = np.mean([x for g in groups for x in g])
    ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    ss_total = sum((x - grand) ** 2 for g in groups for x in g)
    omnibus = {
        "families": neg_fams, "n": n_total,
        "kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue),
        "epsilon_squared": float(kw.statistic / (n_total - 1)),
        "anova_F": float(an.statistic), "anova_p": float(an.pvalue),
        "eta_squared": float(ss_between / ss_total) if ss_total else float("nan"),
    }

    planned = []
    for name, question, fa, fb in PLANNED:
        a = np.array([r["label"] for r in rows if fa(r)])
        b = np.array([r["label"] for r in rows if fb(r)])
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(N_BOOT)]
        planned.append({
            "comparison": name, "question": question, "n_a": len(a), "n_b": len(b),
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_diff": float(a.mean() - b.mean()),
            "mean_diff_ci95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
            "cohens_d": float((a.mean() - b.mean()) / pooled) if pooled else float("nan"),
            "cliffs_delta": float(2 * mw.statistic / (len(a) * len(b)) - 1),
            "mannwhitney_U": float(mw.statistic), "p": float(mw.pvalue),
        })
    # Holm over the planned family of tests
    order = sorted(range(len(planned)), key=lambda i: planned[i]["p"])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(planned) - rank) * planned[i]["p"]))
        planned[i]["p_holm"] = running
    return {"table": table, "omnibus": omnibus, "planned": planned}


# ---------------------------------------------------------------------------
# Step 6.2: alpha models. Every model predicts y_hat = alpha(x) * sim_base.
# ---------------------------------------------------------------------------

FEATURE_GROUPS = ["family", "net_negation", "intensity_hint", "cue_count",
                  "scope_span_ratio", "input_negated"]


def design(rows: list[dict], families: list[str], groups: list[str]) -> tuple[np.ndarray, list[str]]:
    """Feature matrix for alpha(x); reference levels are A_syntactic / neutral."""
    cols, names = [], []
    if "family" in groups:
        for f in families[1:]:
            cols.append([r["family"] == f for r in rows]); names.append(f"family={f}")
    if "net_negation" in groups:
        cols.append([r["net_negation"] for r in rows]); names.append("net_negation")
    if "intensity_hint" in groups:
        for h in ("hedged", "intensified", "partial"):
            cols.append([r["intensity_hint"] == h for r in rows]); names.append(f"intensity={h}")
    if "cue_count" in groups:
        cols.append([r["cue_count"] for r in rows]); names.append("cue_count")
    if "scope_span_ratio" in groups:
        cols.append([r["scope_span_ratio"] for r in rows]); names.append("scope_span_ratio")
    if "input_negated" in groups:
        cols.append([r["input_polarity"] == "negated" for r in rows]); names.append("A_already_negated")
    return np.array(cols, dtype=float).T.reshape(len(rows), len(cols)), names


class Flat:
    def __init__(self, alpha: float | None = None):
        self.fixed, self.alpha = alpha, alpha

    def fit(self, rows, y, s):
        if self.fixed is None:
            self.alpha = ls_alpha(y, s)
        return self

    def alpha_of(self, rows):
        return np.full(len(rows), self.alpha)


class PerFamily:
    def fit(self, rows, y, s):
        self.fallback = ls_alpha(y, s)
        fam = np.array([r["family"] for r in rows])
        self.table = {f: ls_alpha(y[fam == f], s[fam == f]) for f in np.unique(fam)}
        return self

    def alpha_of(self, rows):
        return np.array([self.table.get(r["family"], self.fallback) for r in rows])


class Linear:
    """alpha(x) = b0 + b.x, fitted as ridge on y ~ s * [1, x] (no free intercept)."""

    def __init__(self, families, groups):
        self.families, self.groups = families, groups

    def fit(self, rows, y, s):
        from sklearn.linear_model import Ridge
        x, self.names = design(rows, self.families, self.groups)
        z = np.column_stack([np.ones(len(rows)), x]) * s[:, None]
        self.model = Ridge(alpha=RIDGE, fit_intercept=False).fit(z, y)
        return self

    def alpha_of(self, rows):
        x, _ = design(rows, self.families, self.groups)
        return np.column_stack([np.ones(len(rows)), x]) @ self.model.coef_

    def coefficients(self):
        return dict(zip(["intercept"] + self.names, map(float, self.model.coef_)))


class Tree:
    """Depth<=3 tree on per-item alpha y/s, weighted s^2 == least squares on y."""

    def __init__(self, families):
        self.families = families

    def fit(self, rows, y, s):
        from sklearn.tree import DecisionTreeRegressor
        x, self.names = design(rows, self.families, FEATURE_GROUPS)
        self.model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=SEED)
        self.model.fit(x, y / s, sample_weight=s ** 2)
        return self

    def alpha_of(self, rows):
        return self.model.predict(design(rows, self.families, FEATURE_GROUPS)[0])

    def rules(self):
        from sklearn.tree import export_text
        return export_text(self.model, feature_names=self.names, decimals=3)


def folds(rows: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold
    groups = np.array([r["base_line"] for r in rows])
    return list(GroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
                .split(np.zeros(len(rows)), groups=groups))


def oof(make, rows, y, s, splits) -> np.ndarray:
    pred = np.empty(len(rows))
    for train, test in splits:
        tr = [rows[i] for i in train]
        model = make().fit(tr, y[train], s[train])
        pred[test] = model.alpha_of([rows[i] for i in test]) * s[test]
    return pred


def score(y, pred) -> dict:
    r = stats.pearsonr(y, pred).statistic if np.std(pred) > 0 and np.std(y) > 0 else float("nan")
    return {"mae": float(np.abs(y - pred).mean()), "pearson_r": float(r), "n": len(y)}


def model_search(rows: list[dict], rng: np.random.Generator) -> dict:
    families = sorted({r["family"] for r in rows})
    families.remove("A_syntactic")
    families = ["A_syntactic"] + families
    y = np.array([r["label"] / 5 for r in rows])
    s = np.array([r["sim"] for r in rows])
    groups = np.array([r["base_line"] for r in rows])
    fam = np.array([r["family"] for r in rows])
    splits = folds(rows)

    makers = {
        "flat_0.7": lambda: Flat(HSS_ALPHA),
        "best_flat": lambda: Flat(),
        "per_family": PerFamily,
        "linear": lambda: Linear(families, FEATURE_GROUPS),
        "tree_depth3": lambda: Tree(families),
        "no_penalty_1.0 (reference)": lambda: Flat(1.0),
    }
    preds = {name: oof(make, rows, y, s, splits) for name, make in makers.items()}
    cv = {}
    for name, pred in preds.items():
        per_fam = {f: score(y[fam == f], pred[fam == f]) for f in families}
        cv[name] = {"overall": score(y, pred), "per_family": per_fam}

    abs_err = {k: np.abs(y - p) for k, p in preds.items()}
    diff = lambda a, b: (lambda i: float(abs_err[a][i].mean() - abs_err[b][i].mean()))
    contrasts = {}
    for a, b in [("per_family", "best_flat"), ("per_family", "flat_0.7"), ("best_flat", "flat_0.7"),
                 ("linear", "best_flat"), ("tree_depth3", "best_flat"), ("linear", "per_family"),
                 ("flat_0.7", "no_penalty_1.0 (reference)")]:
        contrasts[f"{a} - {b}"] = {"mae_diff": diff(a, b)(np.arange(len(y))),
                                   "ci95": cluster_boot(groups, diff(a, b), rng)}

    # full-data fits, for the tables a reader applies by hand
    full = {name: make().fit(rows, y, s) for name, make in makers.items()}
    alpha_table = {}
    for f in families:
        m = fam == f
        alpha_table[f] = {"alpha": full["per_family"].table[f], "n": int(m.sum()),
                          "ci95": cluster_boot(groups[m], lambda i, yy=y[m], ss=s[m]: ls_alpha(yy[i], ss[i]), rng)}

    # 6.3 ablation: alpha from one feature group, then family + one more
    ablation = {}
    specs = [[g] for g in FEATURE_GROUPS] + [["family", g] for g in FEATURE_GROUPS[1:]] + [FEATURE_GROUPS]
    for spec in specs:
        pred = oof(lambda spec=spec: Linear(families, spec), rows, y, s, splits)
        ablation[" + ".join(spec)] = score(y, pred)
    fam_only = oof(lambda: Linear(families, ["family"]), rows, y, s, splits)
    ablation_vs_family = {}
    for g in FEATURE_GROUPS[1:]:
        pred = oof(lambda g=g: Linear(families, ["family", g]), rows, y, s, splits)
        e1, e0 = np.abs(y - pred), np.abs(y - fam_only)
        ablation_vs_family[f"family + {g}"] = {
            "mae_diff": float((e1 - e0).mean()),
            "ci95": cluster_boot(groups, lambda i: float((e1[i] - e0[i]).mean()), rng)}

    no_leak = all(not (set(groups[tr]) & set(groups[te])) for tr, te in splits)
    return {
        "n": len(rows), "folds": FOLDS, "fold_sizes": [len(te) for _, te in splits],
        "no_base_sentence_spans_folds": no_leak,
        "cv": cv, "contrasts": contrasts,
        "best_flat_alpha_full": full["best_flat"].alpha,
        "per_family_alpha": alpha_table,
        "linear_coefficients": full["linear"].coefficients(),
        "linear_ridge_lambda": RIDGE,
        "tree_rules": full["tree_depth3"].rules(),
        "ablation": ablation, "ablation_vs_family": ablation_vs_family,
        "_preds": preds, "_y": y, "_fam": fam,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plots(rows, fam_stats, search, prefix: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS.mkdir(exist_ok=True)
    fams = sorted(fam_stats["table"], key=lambda f: fam_stats["table"][f]["mean_label"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot([[r["label"] for r in rows if r["family"] == f] for f in fams], showmeans=True)
    ax.set_xticks(range(1, len(fams) + 1),
                  [f"{f}\n(n={fam_stats['table'][f]['n']})" for f in fams], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("median of three LLM passes (0-5)")
    ax.set_title("Similarity label by family (machine labels; triangles = means)")
    fig.tight_layout(); fig.savefig(PLOTS / f"{prefix}labels_by_family.png", dpi=130); plt.close(fig)

    preds, y = search["_preds"], search["_y"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (name, pred) in zip(axes.flat, preds.items()):
        ax.scatter(pred, y + np.random.default_rng(SEED).uniform(-.02, .02, len(y)), s=8, alpha=.5)
        ax.plot([0, 1], [0, 1], "k--", lw=.8)
        cv = search["cv"][name]["overall"]
        ax.set_title(f"{name}\nMAE {cv['mae']:.3f}  r {cv['pearson_r']:.3f}", fontsize=9)
        ax.set_xlabel("predicted  alpha x sim_base (stand-in)"); ax.set_ylabel("label / 5")
    fig.suptitle("Out-of-fold predictions, 5-fold CV grouped by base sentence")
    fig.tight_layout(); fig.savefig(PLOTS / f"{prefix}predicted_vs_actual.png", dpi=130); plt.close(fig)

    tab = search["per_family_alpha"]
    order = sorted(tab, key=lambda f: tab[f]["alpha"])
    fig, ax = plt.subplots(figsize=(10, 5))
    a = np.array([tab[f]["alpha"] for f in order])
    lo = np.array([tab[f]["ci95"][0] for f in order]); hi = np.array([tab[f]["ci95"][1] for f in order])
    ax.errorbar(range(len(order)), a, yerr=[a - lo, hi - a], fmt="o", capsize=3)
    ax.axhline(HSS_ALPHA, color="r", ls="--", lw=1, label="HSS flat 0.7")
    ax.axhline(search["best_flat_alpha_full"], color="g", ls=":", lw=1, label="best flat constant")
    ax.set_xticks(range(len(order)), [f"{f}\n(n={tab[f]['n']})" for f in order], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("implied alpha = LS fit of label/5 on sim_base")
    ax.set_title("Alpha by family, 95% cluster-bootstrap CI (sim_base = SBERT stand-in)")
    ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / f"{prefix}alpha_by_family.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = load()
    results = {"seed": SEED, "pre_registered": PRE_REGISTERED,
               "labeller": json.loads((DATA / "labels_config.json").read_text()),
               "build": json.loads((DATA / "build_report.json").read_text())}
    results["validation"] = validate(rows)
    results["human_seed"] = write_human_seed(rows)
    labelled = [r for r in rows if r["label"] is not None]

    if results["validation"]["gate"]["passed"]:
        sims = base_similarity([r["a"] for r in labelled], [r["b"] for r in labelled])
        for r, sim in zip(labelled, sims):
            r["sim"] = float(sim)
        results["sim_base"] = STAND_IN
        results["family"] = family_stats(labelled, rng)
        neg = [r for r in labelled if r["kind"] == "negation"]
        search = model_search(neg, rng)
        results["models"] = search
        plots(labelled, results["family"], search)
        fam = results["family"]
        c = search["contrasts"]["per_family - best_flat"]
        results["verdict"] = {
            "families_differ": fam["omnibus"]["kruskal_p"] < 0.05,
            "graded_beats_flat": c["mae_diff"] < 0 and c["ci95"][1] < 0,
            "deciding_number": c,
        }
        results["verdict"]["graded_justified"] = (results["verdict"]["families_differ"]
                                                  and results["verdict"]["graded_beats_flat"])
    (HERE / "results.json").write_text(json.dumps(jsonable(results), indent=2))

    from report import render
    (REPO / "PATTERN.md").write_text(render(jsonable(results)))
    print("gate passed:", results["validation"]["gate"]["passed"])
    if "verdict" in results:
        print(json.dumps(jsonable(results["verdict"]), indent=2))


if __name__ == "__main__":
    main()
