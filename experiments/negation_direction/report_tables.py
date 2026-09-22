"""Print the markdown tables used in REPORT.md straight from results.json.

Keeps the report's numbers transcription-free:

    ./venv/bin/python experiments/negation_direction/report_tables.py > tables.md
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
GROUPS = ["explicit", "implicit", "cancellation", "compound", "other", "control"]
MODELS = list(R["models"])


def row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def header(cells) -> str:
    return row(cells) + "\n" + row(["---"] * len(cells))


print("### 1. Raw cosine(base, variant)\n")
print(header(["model", "group", "n", "mean", "std", "min", "median", "max"]))
for m in MODELS:
    for g in GROUPS:
        s = R["models"][m]["cosine"]["by_group"][g]
        if s.get("n"):
            print(row([m, g, s["n"], s["mean"], s["std"], s["min"], s["median"], s["max"]]))

print("\n**Negation pairs vs paraphrase controls, using cosine alone:**\n")
print(header(["model", "ROC-AUC", "best-threshold acc", "best-threshold balanced acc",
              "majority-class acc"]))
for m in MODELS:
    c = R["models"][m]["cosine"]
    print(row([m, c["negation_vs_control_auc"], c["negation_vs_control_best_threshold_acc"],
               c.get("negation_vs_control_best_threshold_balanced_acc", "-"),
               c.get("majority_class_acc", "-")]))

print("\n### 1b. Cosine by family\n")
print(header(["family"] + MODELS + ["n"]))
fams = sorted(R["data"]["pairs_per_family"])
for f in fams:
    cells = [f]
    n = 0
    for m in MODELS:
        s = R["models"][m]["cosine"]["by_family"].get(f, {})
        cells.append(s.get("mean", "-"))
        n = s.get("n", 0)
    print(row(cells + [n]))

print("\n### 2. Difference vectors\n")
print("Mean pairwise cosine between difference vectors within a group "
      "(1.0 = one single direction, 0.0 = unrelated):\n")
print(header(["model", "group/family", "n pairs of pairs", "mean", "std", "median"]))
for m in MODELS:
    for k, s in R["models"][m]["difference_vectors"]["within_group_pairwise_cosine"].items():
        print(row([m, k, s["n"], s["mean"], s["std"], s["median"]]))

print("\nCosine between group **mean** difference vectors:\n")
for m in MODELS:
    mat = R["models"][m]["difference_vectors"]["group_mean_cosine_matrix"]
    keys = list(mat)
    print(f"\n*{m}*\n")
    print(header([""] + keys))
    for a in keys:
        print(row([a] + [mat[a][b] for b in keys]))

print("\nNorm of the group mean difference vector vs the mean norm of its members "
      "(equal norms = perfectly aligned; a much smaller mean = cancellation):\n")
print(header(["model", "group", "||mean d||", "mean ||d||", "ratio"]))
for m in MODELS:
    for g, s in R["models"][m]["difference_vectors"]["group_mean_vector"].items():
        print(row([m, g, s["norm"], s["mean_norm_of_members"],
                   round(s["norm"] / s["mean_norm_of_members"], 3)]))

print("\n### 3. Negation axis — held-out test split (fit on TRAIN explicit only)\n")
for axis in ("mean_diff", "probe"):
    print(f"\n*axis: {axis}*\n")
    print(header(["model", "group", "n", "pairwise acc", "mean delta", "median delta",
                  "frac variant on negated side", "frac opposite sides of midpoint"]))
    for m in MODELS:
        block = R["models"][m]["axis"][axis]["test"]
        for g in GROUPS:
            s = block.get(g, {})
            if not s.get("delta", {}).get("n"):
                continue
            print(row([m, g, s["delta"]["n"], s["pairwise_acc_variant_more_negated"],
                       s["delta"]["mean"], s["delta"]["median"],
                       s["frac_variant_on_negated_side"], s["frac_opposite_sides_of_midpoint"]]))

print("\n### 3b. Leave-one-base-sentence-out cross-validation (all 20 bases)\n")
for axis in ("mean_diff", "probe"):
    print(f"\n*axis: {axis}*\n")
    print(header(["model", "group", "n", "pairwise acc", "mean delta", "std",
                  "frac variant on negated side", "frac opposite sides"]))
    for m in MODELS:
        block = R["models"][m]["loocv"][axis]["by_group"]
        for g in GROUPS:
            s = block[g]
            if not s["delta"].get("n"):
                continue
            print(row([m, g, s["delta"]["n"], s["pairwise_acc_variant_more_negated"],
                       s["delta"]["mean"], s["delta"]["std"],
                       s["frac_variant_on_negated_side"], s["frac_opposite_sides_of_midpoint"]]))

print("\n**Can the projection shift tell a negation from a paraphrase, pair by pair?** "
      "(ROC-AUC of delta; 0.5 = chance)\n")
print(header(["model", "axis", "negation vs control AUC", "explicit vs cancellation AUC"]))
for m in MODELS:
    for axis in ("mean_diff", "probe"):
        b = R["models"][m]["loocv"][axis]
        print(row([m, axis, b.get("negation_vs_control_delta_auc", "-"),
                   b.get("cancellation_vs_explicit_delta_auc", "-")]))

print("\nPer family (LOOCV, mean-difference axis):\n")
print(header(["family"] + [f"{m} acc" for m in MODELS] + [f"{m} mean delta" for m in MODELS]))
for f in fams:
    accs, deltas = [], []
    for m in MODELS:
        s = R["models"][m]["loocv"]["mean_diff"]["by_family"].get(f)
        accs.append(s["pairwise_acc_variant_more_negated"] if s else "-")
        deltas.append(s["delta"]["mean"] if s else "-")
    print(row([f] + accs + deltas))

print("\n### 4. Intensity: hedged vs intensified (LOOCV projections)\n")
print(header(["model", "axis", "bucket", "n", "mean delta", "std", "median"]))
for m in MODELS:
    for axis in ("mean_diff", "probe"):
        for bucket, s in R["models"][m]["intensity"][axis].items():
            if isinstance(s, dict) and s.get("n"):
                print(row([m, axis, bucket, s["n"], s["mean"], s["std"], s["median"]]))

print("\nIntensified vs hedged, ROC-AUC of the projection shift (0.5 = indistinguishable):\n")
print(header(["model", "axis", "AUC"]))
for m in MODELS:
    for axis in ("mean_diff", "probe"):
        print(row([m, axis, R["models"][m]["intensity"][axis].get("hedged_vs_intensified_auc", "-")]))

print("\n### 5. Lexical cue-lookup baseline (no embeddings at all)\n")
print(header(["group", "n", "pairwise acc (more cues in variant)", "frac tied", "mean cue delta"]))
for g in GROUPS:
    s = R["lexical_baseline"][g]
    if s.get("n"):
        print(row([g, s["n"], s["pairwise_acc_variant_more_cues"], s["frac_tied"],
                   s["mean_cue_delta"]]))

print("\nEmbedding axis vs word lookup, same pairwise task (LOOCV for the axis):\n")
print(header(["group", "lexical"] + [f"{m} mean_diff" for m in MODELS]
             + [f"{m} probe" for m in MODELS]))
for g in GROUPS:
    lex = R["lexical_baseline"][g]
    if not lex.get("n"):
        continue
    cells = [g, lex["pairwise_acc_variant_more_cues"]]
    for axis in ("mean_diff", "probe"):
        for m in MODELS:
            cells.append(R["models"][m]["loocv"][axis]["by_group"][g]
                         ["pairwise_acc_variant_more_negated"])
    print(row(cells))

print("\n### Held-out de-negation records (operation = affirm, n = "
      f"{R['data']['held_out_denegation_pairs']})\n")
print(header(["model", "base", "variant", "cosine", "delta on negation axis"]))
for m in MODELS:
    for rec in R["models"][m]["denegation_records"]:
        print(row([m, rec["base"], rec["variant"], rec["cosine"], rec["delta_mean_diff_axis"]]))
