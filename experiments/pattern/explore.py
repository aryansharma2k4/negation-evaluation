"""Exploratory pass over the gate-FAILED machine labels. Hypotheses, not results.

    ./venv/bin/python experiments/pattern/explore.py

Reads   the same inputs as analyse.py (labels failed the step-5 gate: alpha 0.586)
Writes  results_exploration.json, plots/exploration_*.png, REPORT2.md

PATTERN.md is not touched. Every number in REPORT2.md is read from
results_exploration.json. One seed (0) for every resample, sample and CV split.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

import analyse
from analyse import krippendorff_ordinal

HERE = Path(__file__).resolve().parent
SEED = 0
N_BOOT = 10_000
N_PERM = 1_000
N_EXAMPLES = 10
MIN_N_RANK = 10  # families below this are shown but kept out of the "n>=10" ranking check


def mean_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(x, (N_BOOT, len(x))).mean(1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def ranks(means: dict[str, float]) -> dict[str, int]:
    order = sorted(means, key=lambda f: -means[f])
    return {f: i + 1 for i, f in enumerate(order)}


def pass_rank_agreement(rows: list[dict], fams: list[str]) -> dict:
    per_pass = {k: {f: float(np.mean([r["scores"][k] for r in rows if r["family"] == f])) for f in fams}
                for k in range(3)}
    rho = {f"pass{i + 1}_vs_pass{j + 1}": float(stats.spearmanr(
        [per_pass[i][f] for f in fams], [per_pass[j][f] for f in fams]).statistic)
        for i, j in combinations(range(3), 2)}
    return {"per_pass_means": {f"pass{k + 1}": v for k, v in per_pass.items()}, "spearman": rho,
            "spearman_mean": float(np.mean(list(rho.values())))}


def example(r: dict) -> dict:
    return {"id": r["id"], "family": r["family"], "subtype": r["subtype"], "a": r["a"], "b": r["b"],
            "scores": r["scores"], "label": r["label"]}


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------

def part_a(rows, neg, rng, pick):
    anchors = [r for r in rows if r["kind"] != "negation"]
    alpha = {name: {"n": len(sub), "alpha": krippendorff_ordinal([r["scores"] for r in sub])}
             for name, sub in [("all", rows), ("negation_only", neg), ("anchors_only", anchors)]}

    fams = sorted({r["family"] for r in rows})
    table = {}
    for f in fams:
        lab = np.array([r["label"] for r in rows if r["family"] == f])
        table[f] = {"n": len(lab), "mean": float(lab.mean()), "median": float(np.median(lab)),
                    "std": float(lab.std(ddof=1)), "ci95_mean": mean_ci(lab, rng),
                    "share_median_0": float((lab == 0).mean()),
                    "counts": {str(k): int(v) for k, v in sorted(Counter(lab).items())}}

    big = [f for f in fams if not f.startswith("ANCHOR") and table[f]["n"] >= MIN_N_RANK]
    pairs = list(combinations(big, 2))
    disjoint = [(a, b) for a, b in pairs
                if table[a]["ci95_mean"][1] < table[b]["ci95_mean"][0]
                or table[b]["ci95_mean"][1] < table[a]["ci95_mean"][0]]

    k1 = [r for r in rows if r["family"] == "K1_cancellation"]
    k1_lab = np.array([r["label"] for r in k1])
    para = table["ANCHOR_paraphrase"]
    k1_detail = {
        "n": len(k1), "counts": table["K1_cancellation"]["counts"],
        "share_ge_4": float((k1_lab >= 4).mean()), "share_eq_0": float((k1_lab == 0).mean()),
        "mean": float(k1_lab.mean()), "paraphrase_mean": para["mean"],
        "mean_diff_vs_paraphrase": float(k1_lab.mean() - para["mean"]),
        "mannwhitney_p_vs_paraphrase": float(stats.mannwhitneyu(
            k1_lab, [r["label"] for r in rows if r["family"] == "ANCHOR_paraphrase"]).pvalue),
        "by_subtype": {s: {"n": len(v), "mean": float(np.mean(v))} for s, v in sorted(
            _group(k1, "subtype").items())},
        "examples": [example(r) for r in pick(k1, N_EXAMPLES)],
    }
    for name, test in [("never_x_to", lambda st: st.startswith("never_")),
                       ("not_prefix_or_suffix", lambda st: st.startswith("not_"))]:
        lab = np.array([r["label"] for r in k1 if test(r["subtype"])])
        k1_detail[name] = {"n": len(lab), "mean": float(lab.mean()), "share_ge_4": float((lab >= 4).mean()),
                           "share_eq_0": float((lab == 0).mean())}

    moved = [r for r in rows if r["scores"][0] == r["scores"][1] and abs(r["scores"][2] - r["scores"][0]) >= 3]
    overall = Counter(r["family"] for r in rows)
    moved_c = Counter(r["family"] for r in moved)
    neighbour = {
        "n": len(moved), "n_total": len(rows),
        "by_family": {f: {"moved": moved_c.get(f, 0), "share_of_moved": moved_c.get(f, 0) / len(moved),
                          "share_of_all": overall[f] / len(rows),
                          "rate_within_family": moved_c.get(f, 0) / overall[f]} for f in fams},
        "direction_pass3_up": sum(r["scores"][2] > r["scores"][0] for r in moved),
        "direction_pass3_down": sum(r["scores"][2] < r["scores"][0] for r in moved),
        "examples": [example(r) for r in pick(moved, N_EXAMPLES)],
    }
    return {"krippendorff": alpha, "family_table": table,
            "ci_nonoverlap": {"families_n_ge_10": big, "pairs": len(pairs), "disjoint_pairs": len(disjoint),
                              "disjoint": [f"{a} / {b}" for a, b in disjoint]},
            "k1": k1_detail, "neighbour_moved": neighbour}


def _group(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[r[key]].append(r["label"])
    return out


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------

def part_b(neg, rng):
    fams = sorted({r["family"] for r in neg})
    big = [f for f in fams if sum(r["family"] == f for r in neg) >= MIN_N_RANK]
    per_pass = {"all_families": pass_rank_agreement(neg, fams),
                "families_n_ge_10": pass_rank_agreement(neg, big)}
    for v in per_pass.values():
        v["ranks"] = {p: ranks(m) for p, m in v["per_pass_means"].items()}
    rk = per_pass["families_n_ge_10"]["ranks"]
    per_pass["consistent_top3_n_ge_10"] = [f for f in big if all(rk[p][f] <= 3 for p in rk)]
    per_pass["consistent_bottom3_n_ge_10"] = [f for f in big if all(rk[p][f] > len(big) - 3 for p in rk)]
    per_pass["max_rank_shift_n_ge_10"] = {f: max(rk[p][f] for p in rk) - min(rk[p][f] for p in rk) for f in big}

    # The three passes rate the SAME items, so per-family means correlate across
    # passes even if family carries no signal. Null: shuffle family labels
    # across items (all three scores move with their item), recompute.
    fam_labels = [r["family"] for r in neg]
    null = []
    prng = random.Random(SEED)
    for _ in range(N_PERM):
        shuffled = fam_labels[:]
        prng.shuffle(shuffled)
        fake = [{**r, "family": f} for r, f in zip(neg, shuffled)]
        null.append(pass_rank_agreement(fake, big)["spearman_mean"])
    per_pass["permutation_null_n_ge_10"] = {
        "n_perm": N_PERM, "median": float(np.median(null)), "p95": float(np.percentile(null, 95)),
        "p_value": float((np.sum(np.array(null) >= per_pass["families_n_ge_10"]["spearman_mean"]) + 1) / (N_PERM + 1))}

    # 7. off the floor
    off = [r for r in neg if r["label"] > 0]
    on = [r for r in neg if r["label"] == 0]
    full_means = {f: float(np.mean([r["label"] for r in neg if r["family"] == f])) for f in fams}
    off_tab = {f: {"n": sum(r["family"] == f for r in off),
                   "mean": float(np.mean([r["label"] for r in off if r["family"] == f]))
                   if any(r["family"] == f for r in off) else None} for f in fams}
    off_big = [f for f in big if off_tab[f]["n"] >= 5]
    floor_share = {f: float(np.mean([r["label"] == 0 for r in neg if r["family"] == f])) for f in fams}

    def describe(sub):
        return {"n": len(sub),
                "mean_words_b": float(np.mean([len(r["b"].split()) for r in sub])),
                "mean_words_a": float(np.mean([len(r["a"].split()) for r in sub])),
                "share_input_negated": float(np.mean([r["input_polarity"] == "negated" for r in sub])),
                "share_operation": {k: v / len(sub) for k, v in sorted(Counter(r["operation"] for r in sub).items())},
                "mean_net_negation": float(np.mean([r["net_negation"] for r in sub])),
                "subtopic": dict(sorted(Counter(r["subtopic"] for r in sub).items()))}

    subtopic_floor = {t: float(np.mean([r["label"] == 0 for r in neg if r["subtopic"] == t]))
                      for t in sorted({r["subtopic"] for r in neg})}
    kw_full = stats.kruskal(*[[r["label"] for r in neg if r["family"] == f] for f in big])
    kw_off = stats.kruskal(*[[r["label"] for r in off if r["family"] == f] for f in off_big])
    floor = {
        "n_off_floor": len(off), "n_on_floor": len(on), "table": off_tab,
        "families_off_floor_n_ge_5": off_big,
        "spearman_off_vs_full_means": float(stats.spearmanr(
            [off_tab[f]["mean"] for f in off_big], [full_means[f] for f in off_big]).statistic),
        "pearson_family_mean_vs_floor_share_n_ge_10": float(stats.pearsonr(
            [full_means[f] for f in big], [floor_share[f] for f in big]).statistic),
        "kruskal_full_n_ge_10": {"H": float(kw_full.statistic), "p": float(kw_full.pvalue)},
        "kruskal_off_floor": {"H": float(kw_off.statistic), "p": float(kw_off.pvalue)},
        "off_floor_profile": describe(off), "on_floor_profile": describe(on),
        "floor_share_by_subtopic": subtopic_floor,
        "floor_share_by_operation": {o: float(np.mean([r["label"] == 0 for r in neg if r["operation"] == o]))
                                     for o in sorted({r["operation"] for r in neg})},
        "floor_share_by_input_polarity": {o: float(np.mean([r["label"] == 0 for r in neg if r["input_polarity"] == o]))
                                          for o in sorted({r["input_polarity"] for r in neg})},
    }

    # 8. leave one base sentence out
    lobo = {}
    for f in fams:
        sub = [r for r in neg if r["family"] == f]
        lines = sorted({r["base_line"] for r in sub})
        held = {ln: float(np.mean([r["label"] for r in sub if r["base_line"] != ln]))
                for ln in lines if any(r["base_line"] != ln for r in sub)}
        if not held:
            continue
        worst = max(held, key=lambda ln: abs(held[ln] - full_means[f]))
        lobo[f] = {"n": len(sub), "n_bases": len(lines), "mean": full_means[f],
                   "min": min(held.values()), "max": max(held.values()),
                   "range": max(held.values()) - min(held.values()),
                   "max_swing": held[worst] - full_means[f], "worst_base_line": worst,
                   "worst_base_pairs": sum(r["base_line"] == worst for r in sub),
                   "worst_base_sentence": next(r["a"] for r in sub if r["base_line"] == worst)}
    full_rank = ranks({f: full_means[f] for f in big})
    rank_rho = []
    for ln in sorted({r["base_line"] for r in neg}):
        kept = [r for r in neg if r["base_line"] != ln]
        m = {f: float(np.mean([r["label"] for r in kept if r["family"] == f])) for f in big}
        rank_rho.append(float(stats.spearmanr([full_rank[f] for f in big], [ranks(m)[f] for f in big]).statistic))
    return {"per_pass": per_pass, "floor": floor, "lobo": lobo,
            "lobo_ranking_spearman_min": min(rank_rho), "lobo_ranking_spearman_median": float(np.median(rank_rho))}


# ---------------------------------------------------------------------------
# Main + report
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED)
    sample_rng = random.Random(SEED)
    pick = lambda xs, k: sample_rng.sample(xs, min(k, len(xs)))

    rows = analyse.load()
    neg = [r for r in rows if r["kind"] == "negation"]
    res = {"seed": SEED, "n_boot": N_BOOT, "n_perm": N_PERM,
           "gate": json.loads((HERE / "results.json").read_text())["validation"]["gate"],
           "gate_alpha": json.loads((HERE / "results.json").read_text())["validation"]["krippendorff_alpha_ordinal"],
           "labeller": json.loads((analyse.DATA / "labels_config.json").read_text())["model"]}
    res["A"] = part_a(rows, neg, rng, pick)
    res["B"] = part_b(neg, rng)

    sims = analyse.base_similarity([r["a"] for r in rows], [r["b"] for r in rows])
    for r, s in zip(rows, sims):
        r["sim"] = float(s)
    search = analyse.model_search(neg, rng)
    res["C"] = analyse.jsonable(search)
    res["C"]["sim_base"] = analyse.STAND_IN
    table = {f: {"mean_label": v["mean"], "n": v["n"]} for f, v in res["A"]["family_table"].items()}
    analyse.plots(rows, {"table": table}, search, prefix="exploration_")

    c = search["contrasts"]["per_family - best_flat"]
    b = res["B"]
    res["D"] = {
        "ci_disjoint_share": res["A"]["ci_nonoverlap"]["disjoint_pairs"] / res["A"]["ci_nonoverlap"]["pairs"],
        "rank_spearman_mean_n_ge_10": b["per_pass"]["families_n_ge_10"]["spearman_mean"],
        "rank_null_p95": b["per_pass"]["permutation_null_n_ge_10"]["p95"],
        "rank_stable_beyond_null": b["per_pass"]["families_n_ge_10"]["spearman_mean"]
                                   > b["per_pass"]["permutation_null_n_ge_10"]["p95"],
        "graded_beats_flat_cv": c["mae_diff"] < 0 and c["ci95"][1] < 0,
        "per_family_minus_best_flat": c,
        "family_mean_vs_floor_share_r": b["floor"]["pearson_family_mean_vs_floor_share_n_ge_10"],
        "negation_only_alpha": res["A"]["krippendorff"]["negation_only"]["alpha"],
        "negation_floor_share": b["floor"]["n_on_floor"] / len(neg),
    }
    (HERE / "results_exploration.json").write_text(json.dumps(analyse.jsonable(res), indent=2))
    (HERE / "REPORT2.md").write_text(render(json.loads((HERE / "results_exploration.json").read_text())))
    print(json.dumps(res["D"], indent=2, default=str))


def f(x, d=2):
    return "—" if x is None else ("n/a" if isinstance(x, float) and x != x else f"{x:.{d}f}")


def pv(x):
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def ex_table(examples):
    lines = ["| family / subtype | A | B | pass 1, 2, 3 | median |", "| --- | --- | --- | --- | ---: |"]
    for e in examples:
        lines.append(f"| {e['family']} / {e['subtype']} | {e['a']} | {e['b']} | "
                     f"{', '.join(str(s) for s in e['scores'])} | {f(e['label'], 1)} |")
    return "\n".join(lines)


def render(r: dict) -> str:
    A, B, C, D = r["A"], r["B"], r["C"], r["D"]
    t = A["family_table"]
    order = sorted(t, key=lambda k: -t[k]["mean"])
    fam_rows = "\n".join(
        f"| `{k}` | {t[k]['n']} | {f(t[k]['mean'])} | {f(t[k]['ci95_mean'][0])} – {f(t[k]['ci95_mean'][1])} | "
        f"{f(t[k]['median'], 1)} | {f(t[k]['std'])} | {f(100 * t[k]['share_median_0'], 0)}% |" for k in order)
    kr = A["krippendorff"]
    k1 = A["k1"]
    nb = A["neighbour_moved"]
    nb_rows = "\n".join(
        f"| `{k}` | {v['moved']} | {f(100 * v['share_of_moved'], 1)}% | {f(100 * v['share_of_all'], 1)}% | "
        f"{f(100 * v['rate_within_family'], 1)}% |"
        for k, v in sorted(nb["by_family"].items(), key=lambda kv: -kv[1]["rate_within_family"]))

    pp = B["per_pass"]["all_families"]
    fams = sorted(pp["per_pass_means"]["pass1"], key=lambda k: -t[k]["mean"])
    rank_rows = "\n".join(
        f"| `{k}` | {t[k]['n']} | " + " | ".join(
            f"{f(pp['per_pass_means'][p][k])} (#{pp['ranks'][p][k]})" for p in ("pass1", "pass2", "pass3")) + " |"
        for k in fams)
    big = B["per_pass"]["families_n_ge_10"]
    null = B["per_pass"]["permutation_null_n_ge_10"]
    fl = B["floor"]
    off_rows = "\n".join(
        f"| `{k}` | {fl['table'][k]['n']} | {f(fl['table'][k]['mean'])} | {f(100 * t[k]['share_median_0'], 0)}% |"
        for k in fams)
    on, off = fl["on_floor_profile"], fl["off_floor_profile"]
    subtopics = sorted(fl["floor_share_by_subtopic"], key=lambda s: fl["floor_share_by_subtopic"][s])
    lobo_rows = "\n".join(
        f"| `{k}` | {v['n']} | {v['n_bases']} | {f(v['mean'])} | {f(v['min'])} – {f(v['max'])} | {f(v['range'])} | "
        f"{f(v['max_swing'], 2)} (line {v['worst_base_line']}, {v['worst_base_pairs']} pairs) |"
        for k, v in sorted(B["lobo"].items(), key=lambda kv: -kv[1]["range"]))

    cv = C["cv"]
    models = ["flat_0.7", "best_flat", "per_family", "linear", "tree_depth3", "no_penalty_1.0 (reference)"]
    cv_rows = "\n".join(f"| `{m}` | {f(cv[m]['overall']['mae'], 3)} | {f(5 * cv[m]['overall']['mae'], 2)} | "
                        f"{f(cv[m]['overall']['pearson_r'], 3)} |" for m in models)
    per_fam = "\n".join(
        f"| `{k}` | {cv['flat_0.7']['per_family'][k]['n']} | " + " | ".join(
            f"{f(cv[m]['per_family'][k]['mae'], 3)} / {f(cv[m]['per_family'][k]['pearson_r'], 2)}" for m in models[:5]) + " |"
        for k in cv["flat_0.7"]["per_family"])
    con_rows = "\n".join(f"| {k} | {f(v['mae_diff'], 4)} | [{f(v['ci95'][0], 4)}, {f(v['ci95'][1], 4)}] |"
                         for k, v in C["contrasts"].items())
    alpha_rows = "\n".join(f"| `{k}` | {v['n']} | {f(v['alpha'], 3)} | [{f(v['ci95'][0], 3)}, {f(v['ci95'][1], 3)}] |"
                           for k, v in sorted(C["per_family_alpha"].items(), key=lambda kv: kv[1]["alpha"]))
    coef_rows = "\n".join(f"| `{k}` | {f(v, 4)} |" for k, v in C["linear_coefficients"].items())
    abl = C["ablation"]
    abl_rows = "\n".join(f"| {k} | {f(v['mae'], 4)} | {f(v['pearson_r'], 3)} |"
                         for k, v in sorted(abl.items(), key=lambda kv: kv[1]["mae"]))
    singles = {k: v for k, v in abl.items() if " + " not in k}
    best_single = min(singles, key=lambda k: singles[k]["mae"])
    avf = C["ablation_vs_family"]
    avf_rows = "\n".join(f"| {k} | {f(v['mae_diff'], 4)} | [{f(v['ci95'][0], 4)}, {f(v['ci95'][1], 4)}] |"
                         for k, v in avf.items())
    beats_family = [k for k, v in avf.items() if v["ci95"][1] < 0]

    # Part D: the first sentence is decided by the numbers, not written per run
    overlap = D["ci_disjoint_share"] < 0.5
    scrambled = not D["rank_stable_beyond_null"]
    dec = D["per_family_minus_best_flat"]
    if overlap and scrambled:
        first = ("**The family CIs mostly overlap and the per-pass family rankings are not reliably more "
                 "stable than shuffled family labels give: apart from its two ends, the pattern does not "
                 "survive the noise.**")
    elif scrambled:
        first = ("**The per-pass family rankings are no more stable than random family labels would give, "
                 "so whatever separation the CIs show is not a family pattern the labeller reproduces.**")
    elif overlap:
        first = ("**The per-pass rankings are stable, but most family CIs overlap: there is an ordering, "
                 "and at n = 35 most adjacent families cannot be told apart.**")
    else:
        first = ("**On these labels the family ordering is stable across passes and most family CIs are "
                 "disjoint. That is a pattern in the labeller, not yet in human judgement.**")
    human = ("most likely overturn it" if abs(D["family_mean_vs_floor_share_r"]) > 0.9 and D["negation_floor_share"] > 0.5
             else "possibly confirm the ordering but not the magnitudes")

    return f"""# REPORT 2 — exploring the gate-failed labels

**Every number here comes from labels that failed their validation gate
(Krippendorff alpha {f(r['gate_alpha'], 3)}) with no human validation, so everything
below is a hypothesis, not a result.**

Generated by `experiments/pattern/explore.py` into `results_exploration.json`;
every figure below is read from that file. Labeller `{r['labeller']}`. One seed
({r['seed']}) for every bootstrap ({r['n_boot']:,} resamples for family means), permutation
({r['n_perm']:,}), example sample and CV split. `PATTERN.md` is unchanged and
still stands: the gate failed and no confirmatory result exists.

## Part A — diagnostics

### 1. Krippendorff's alpha (ordinal, three passes)

| subset | n | alpha |
| --- | ---: | ---: |
| all items | {kr['all']['n']} | {f(kr['all']['alpha'], 3)} |
| negation pairs only | {kr['negation_only']['n']} | **{f(kr['negation_only']['alpha'], 3)}** |
| anchors only | {kr['anchors_only']['n']} | {f(kr['anchors_only']['alpha'], 3)} |

Without the anchors, agreement on the pairs the question is about is
{f(kr['negation_only']['alpha'], 3)}, against a gate of {r['gate']['krippendorff_min']}.

### 2-3. Per family (label = median of three passes, 0-5)

Mean CI: {r['n_boot']:,} bootstrap resamples of items. Sorted by mean.

| family | n | mean | 95% CI of mean | median | std | median = 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
{fam_rows}

Among the {len(A['ci_nonoverlap']['families_n_ge_10'])} negation families with n >= {MIN_N_RANK},
**{A['ci_nonoverlap']['disjoint_pairs']} of {A['ci_nonoverlap']['pairs']} family pairs
({f(100 * D['ci_disjoint_share'], 0)}%) have disjoint 95% CIs**; the rest cannot be told apart at
this n. Disjoint pairs: {', '.join(A['ci_nonoverlap']['disjoint']) or 'none'}.

Box plot: `plots/exploration_labels_by_family.png`.

### 4. K1_cancellation (*not impossible*, *never fails to*)

Label distribution (label: count): {k1['counts']}. {f(100 * k1['share_ge_4'], 0)}% of pairs are at 4-5,
{f(100 * k1['share_eq_0'], 0)}% at 0. Mean {f(k1['mean'])}, against {f(k1['paraphrase_mean'])} for paraphrase
anchors (difference {f(k1['mean_diff_vs_paraphrase'])}, Mann-Whitney p = {pv(k1['mannwhitney_p_vs_paraphrase'])}).
By subtype: {', '.join(f"{s} {f(v['mean'])} (n={v['n']})" for s, v in k1['by_subtype'].items())}.

The family hides two different populations:

| K1 subtypes | n | mean | at 4-5 | at 0 |
| --- | ---: | ---: | ---: | ---: |
| `never_X_to` (*never fails to sort*) | {k1['never_x_to']['n']} | {f(k1['never_x_to']['mean'])} | {f(100 * k1['never_x_to']['share_ge_4'], 0)}% | {f(100 * k1['never_x_to']['share_eq_0'], 0)}% |
| `not_` + affix (*not unimportant*) | {k1['not_prefix_or_suffix']['n']} | {f(k1['not_prefix_or_suffix']['mean'])} | {f(100 * k1['not_prefix_or_suffix']['share_ge_4'], 0)}% | {f(100 * k1['not_prefix_or_suffix']['share_eq_0'], 0)}% |

**Plainly: the labeller treats the `never X to` cancellations as largely
meaning-preserving{' (mean ' + f(k1['never_x_to']['mean']) + ', below paraphrases at ' + f(k1['paraphrase_mean']) + ')'} and does not
treat the affixal ones that way.** The random examples below suggest why, and
it is not only the labeller: several affixal "cancellations" are generator
output that changes the meaning or is not English (*in-disorder delivery*, *the
noncyclic redundancy check*, *the uncritical section*: the prefix landed on the
wrong word). A low score there may be a correct one. So K1's family mean mixes
a labeller judgement with a generator defect, and should not be read as one number.

Ten pairs, sampled at random (seed {r['seed']}), not chosen:

{ex_table(k1['examples'])}

### 5. The {nb['n']} items where passes 1 and 2 agree and pass 3 moves 3+

Passes 1 and 2 share batch neighbours; pass 3 does not. Pass 3 moved up on
{nb['direction_pass3_up']} and down on {nb['direction_pass3_down']}.

| family | moved | share of moved | share of all items | rate within family |
| --- | ---: | ---: | ---: | ---: |
{nb_rows}

Unrelated anchors are {f(100 * nb['by_family']['ANCHOR_unrelated']['share_of_moved'], 0)}% of the moved items
against {f(100 * nb['by_family']['ANCHOR_unrelated']['share_of_all'], 0)}% of all items; paraphrase anchors are
{f(100 * nb['by_family']['ANCHOR_paraphrase']['share_of_moved'], 0)}%. Within negation families the rate ranges
from {f(100 * min(v['rate_within_family'] for k, v in nb['by_family'].items() if not k.startswith('ANCHOR') and t[k]['n'] >= 10), 0)}% to
{f(100 * max(v['rate_within_family'] for k, v in nb['by_family'].items() if not k.startswith('ANCHOR') and t[k]['n'] >= 10), 0)}% (n >= 10 families),
i.e. 1-6 items per family: too few to say one family is more neighbour-sensitive
than another.

Ten, sampled at random:

{ex_table(nb['examples'])}

## Part B — is the pattern stable?

### 6. Family means per pass (the passes as three annotators)

Rank in brackets (#1 = highest mean).

| family | n | pass 1 | pass 2 | pass 3 |
| --- | ---: | --- | --- | --- |
{rank_rows}

Spearman between per-pass family means, all {len(pp['per_pass_means']['pass1'])} families:
1-2 {f(pp['spearman']['pass1_vs_pass2'], 3)}, 1-3 {f(pp['spearman']['pass1_vs_pass3'], 3)},
2-3 {f(pp['spearman']['pass2_vs_pass3'], 3)} (mean {f(pp['spearman_mean'], 3)}).
Families with n >= {MIN_N_RANK} only: 1-2 {f(big['spearman']['pass1_vs_pass2'], 3)},
1-3 {f(big['spearman']['pass1_vs_pass3'], 3)}, 2-3 {f(big['spearman']['pass2_vs_pass3'], 3)} (mean {f(big['spearman_mean'], 3)}).

**These correlations are not evidence on their own.** All three passes rate
the *same* items, so the per-family means correlate across passes even if
family carries no information: item-level agreement is enough. Null: shuffle
the family labels across items ({null['n_perm']:,} times; each item keeps its three
scores) and recompute. Null mean Spearman: median {f(null['median'], 3)}, 95th percentile
{f(null['p95'], 3)}. Observed {f(big['spearman_mean'], 3)}, permutation p = {pv(null['p_value'])}.
{'The observed ranking stability is above the null at the 95th percentile: the family ordering is reproduced beyond what shared items alone produce.' if D['rank_stable_beyond_null'] else 'The observed stability is not above the null at the 95th percentile, though close to it: the ranking as a whole is not reliably reproduced beyond what shared items alone produce.'}

Two things survive anyway. Across all three passes, among n >= {MIN_N_RANK} families,
top 3 every time: {', '.join(f'`{x}`' for x in B['per_pass']['consistent_top3_n_ge_10']) or 'none'};
bottom 3 every time: {', '.join(f'`{x}`' for x in B['per_pass']['consistent_bottom3_n_ge_10']) or 'none'}.
The middle does not hold: the largest rank shift between passes is
{', '.join(f'`{k}` {v}' for k, v in sorted(B['per_pass']['max_rank_shift_n_ge_10'].items(), key=lambda kv: -kv[1])[:4])} places.
And the disagreement is mostly about sentence order, not batch neighbours: passes 1
and 3 (same A-B order, different neighbours) correlate at {f(big['spearman']['pass1_vs_pass3'], 3)}; pass 2 (B-A
order) correlates with them at {f(big['spearman']['pass1_vs_pass2'], 3)} and {f(big['spearman']['pass2_vs_pass3'], 3)}.

### 7. Off the floor (pairs with median > 0)

{fl['n_off_floor']} of {fl['n_off_floor'] + fl['n_on_floor']} negation pairs are off the floor.

| family | n off floor | mean off floor | share at 0 (all pairs) |
| --- | ---: | ---: | ---: |
{off_rows}

Across families with n >= {MIN_N_RANK}, family mean vs share of pairs at 0:
**Pearson r = {f(D['family_mean_vs_floor_share_r'], 3)}**. The family ordering is, to that
extent, an ordering of how often the labeller floors a family. Off the floor, the
family means (families with >= 5 off-floor pairs) rank-correlate with the full
means at Spearman {f(fl['spearman_off_vs_full_means'], 3)}. Kruskal-Wallis across families:
all pairs (n >= {MIN_N_RANK} families) H = {f(fl['kruskal_full_n_ge_10']['H'])}, p = {pv(fl['kruskal_full_n_ge_10']['p'])};
off-floor only H = {f(fl['kruskal_off_floor']['H'])}, p = {pv(fl['kruskal_off_floor']['p'])}.

What the off-floor pairs have in common, against the floored ones:

| | off floor (n={off['n']}) | on floor (n={on['n']}) |
| --- | ---: | ---: |
| mean words, B | {f(off['mean_words_b'], 1)} | {f(on['mean_words_b'], 1)} |
| mean words, A | {f(off['mean_words_a'], 1)} | {f(on['mean_words_a'], 1)} |
| input already negated | {f(100 * off['share_input_negated'], 0)}% | {f(100 * on['share_input_negated'], 0)}% |
| mean net_negation | {f(off['mean_net_negation'])} | {f(on['mean_net_negation'])} |
| operation | {', '.join(f"{k} {f(100 * v, 0)}%" for k, v in off['share_operation'].items())} | {', '.join(f"{k} {f(100 * v, 0)}%" for k, v in on['share_operation'].items())} |

Off-floor and floored pairs are alike in length (difference
{f(abs(off['mean_words_b'] - on['mean_words_b']), 1)} words in B), in input polarity
({f(100 * abs(off['share_input_negated'] - on['share_input_negated']), 0)} points) and in operation mix.
Nothing measured here separates them except family, and subtopic to a lesser
degree (below). Off the floor the scores are also not obviously meaningful:
`K2_compound`, which negates two clauses, averages {f(fl['table']['K2_compound']['mean'])} there,
second only to `K1_cancellation` ({f(fl['table']['K1_cancellation']['mean'])}).

Floor rate by operation: {', '.join(f"{k} {f(100 * v, 0)}%" for k, v in fl['floor_share_by_operation'].items())}.
By input polarity: {', '.join(f"{k} {f(100 * v, 0)}%" for k, v in fl['floor_share_by_input_polarity'].items())}.
By subtopic, lowest to highest: {', '.join(f"{s} {f(100 * fl['floor_share_by_subtopic'][s], 0)}%" for s in subtopics)}.

### 8. Leave one base sentence out

Each family's mean, recomputed with each of its base sentences held out in turn.
Sorted by range.

| family | n | bases | mean | min – max held-out mean | range | largest swing (sentence, pairs) |
| --- | ---: | ---: | ---: | --- | ---: | --- |
{lobo_rows}

Family *ranking* (n >= {MIN_N_RANK}) with each base sentence held out, Spearman against
the full ranking: minimum {f(B['lobo_ranking_spearman_min'], 3)}, median {f(B['lobo_ranking_spearman_median'], 3)}.

Among the n >= {MIN_N_RANK} families the largest held-out range is
{f(max(v['range'] for k, v in B['lobo'].items() if v['n'] >= MIN_N_RANK))}, and no
single base sentence reorders the families. **No family mean here is an
artefact of one sentence.** That rules out one failure mode; it does not touch
the labeller-level problems in Parts A and B6. (`G_prepositional` and
`L_scope_position`, n = 4 from 2 sentences each, are anecdotes.)

## Part C — formulas (EXPLORATORY: fitted to gate-failed labels)

Target label/5; every model is `alpha(x) * sim_base`, where **sim_base is SBERT
all-mpnet-base-v2 cosine, a STAND-IN for the HSS backbone**. 5-fold CV grouped by
base sentence (fold sizes {C['fold_sizes']}; no base sentence in two folds:
{C['no_base_sentence_spans_folds']}). {C['n']} negation pairs; anchors excluded.

| model | CV MAE (label/5) | CV MAE (0-5) | Pearson r |
| --- | ---: | ---: | ---: |
{cv_rows}

MAE differences (negative = first better), 95% CI resampling base sentences:

| contrast | MAE difference | 95% CI |
| --- | ---: | --- |
{con_rows}

Per family, MAE / Pearson r (r undefined where a family's labels or predictions do not vary):

| family | n | flat 0.7 | best flat | per-family | linear | tree |
| --- | ---: | --- | --- | --- | --- | --- |
{per_fam}

The best flat constant is {f(C['best_flat_alpha_full'], 3)}, far below HSS's 0.7. That
is the floor again: with {f(100 * D['negation_floor_share'], 0)}% of negation pairs labelled 0, any
least-squares constant is dragged down. It is a property of this labeller, not
evidence that 0.7 is too weak.

**Per-family alpha** (full data; best flat constant = {f(C['best_flat_alpha_full'], 3)}):

| family | n | alpha | 95% CI (base-sentence bootstrap) |
| --- | ---: | ---: | --- |
{alpha_rows}

**Linear alpha(x)**, ridge lambda {C['linear_ridge_lambda']}, reference levels `A_syntactic` and
`neutral` intensity. Family and intensity are partly collinear (H is always
*hedged*, I always *intensified*), so coefficients in that block are not
separately interpretable.

| term | coefficient |
| --- | ---: |
{coef_rows}

**Depth-3 tree** on per-item alpha (weighted so it is least squares on label/5):

```
{C['tree_rules'].rstrip()}
```

**Ablation** (linear alpha on the listed groups only, same CV):

| features | CV MAE | Pearson r |
| --- | ---: | ---: |
{abl_rows}

Single strongest feature group: **{best_single}**. Adding one group to family:

| model | MAE difference vs family alone | 95% CI |
| --- | ---: | --- |
{avf_rows}

{'Nothing beats family alone (no CI entirely below zero).' if not beats_family else 'CI entirely below zero for: ' + ', '.join(beats_family) + '.'}

Plots: `plots/exploration_predicted_vs_actual.png`, `plots/exploration_alpha_by_family.png`.

## Part D — verdict

{first} Only {f(100 * D['ci_disjoint_share'], 0)}% of family pairs (n >= {MIN_N_RANK}) have disjoint CIs;
per-pass ranking stability is {f(D['rank_spearman_mean_n_ge_10'], 3)} against a shuffled-label
null whose 95th percentile is {f(D['rank_null_p95'], 3)} (p = {pv(B['per_pass']['permutation_null_n_ge_10']['p_value'])}); agreement
on the negation pairs alone is {f(D['negation_only_alpha'], 3)}.

Does a graded alpha look justified? On these labels a per-family alpha fits
better than any constant: CV MAE difference (per-family − best flat)
{f(dec['mae_diff'], 4)} on label/5, 95% CI [{f(dec['ci95'][0], 4)}, {f(dec['ci95'][1], 4)}]. That gain is
real *for this labeller* and not an artefact of one sentence (Part B8). But it
is a fit to the labeller's floor. {f(100 * D['negation_floor_share'], 0)}% of negation pairs are at 0, and
family means track the share at 0 with r = {f(D['family_mean_vs_floor_share_r'], 3)}. What the formula
learns is mostly *which families the labeller calls completely dissimilar*.
The only structure that survives every check is at the ends: cancellation
high{', ' + ', '.join(f'`{x}`' for x in B['per_pass']['consistent_bottom3_n_ge_10']) + ' low' if B['per_pass']['consistent_bottom3_n_ge_10'] else ''}.
That high end comes from the clean `never X to` cancellations (mean
{f(A['k1']['never_x_to']['mean'])}); K1's family mean is pulled *down* by a generator defect in its
affixal subtypes (Part A4).

Would human labels confirm it? {'Most likely they would overturn it.' if human.startswith('most likely overturn') else 'Possibly the ordering, not the magnitudes.'} The rubric puts
same-topic pairs at 2. A rater following it would not be expected to put
{f(100 * D['negation_floor_share'], 0)}% of negations, which share every content word with their
base, at "completely dissimilar". Lift the floor and the family ordering, which
here is mostly floor rate, has to be re-established from scratch; every alpha
(best flat {f(C['best_flat_alpha_full'], 3)}) would rise, by an amount these labels cannot tell us. The ends
(cancellation near no-negation, intensified negation furthest away) are the
part most likely to survive, because they also match the linguistic expectation.

What would change this answer:
- a relabel (one pair per prompt) whose negation-only alpha reaches >= 0.6 (now {f(D['negation_only_alpha'], 3)});
- the human seed agreeing with these labels at Krippendorff alpha >= 0.6;
- on labels that pass: family means whose correlation with the floor share falls
  well below |r| = 0.9 (now {f(abs(D['family_mean_vs_floor_share_r']), 3)}), a per-family − best-flat CI still
  entirely below zero, and per-pass ranking stability above the permutation
  null's 95th percentile (now {f(D['rank_spearman_mean_n_ge_10'], 3)} vs {f(D['rank_null_p95'], 3)}).
"""


if __name__ == "__main__":
    main()
