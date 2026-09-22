"""Print REPORT.md's tables straight from results.json (no hand-typed numbers).

    ./venv/bin/python experiments/negation_detection/report_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
SCORERS = [k for k in R["phase2"]]
NICE = {"arch1_features": "arch 1 — feature scorer",
        "arch2_cosine_margin": "arch 2 — refset margin (cosine)",
        "arch2_resid_margin": "arch 2 — refset margin (resid+polarity)",
        "arch3_hybrid": "arch 3 — hybrid",
        "baseline_cosine": "baseline — raw cosine",
        "baseline_cue_delta": "baseline — lexical cue delta",
        "baseline_nli_contradiction": "baseline — NLI P(contradiction)"}


def row(cells): return "| " + " | ".join(str(c) for c in cells) + " |"
def header(cells): return row(cells) + "\n" + row(["---"] * len(cells))


print("### Data\n")
print(header(["set", "n", "composition"]))
for name, block in R["data"].items():
    print(row([name, block["n"], ", ".join(f"{k}: {v}" for k, v in
                                           sorted(block["by_subtype"].items()))]))
print("\nPhase-1 negated pairs by derived negation type: "
      + ", ".join(f"{k}: {v}" for k, v in sorted(R["data"]["phase1"]["by_neg_type"].items()))
      + ".\n")

print("### Generator coverage (architecture 2's dependency)\n")
print(header(["set", "pairs", "unique A", "A with no NEG variant", "rate",
              "no variants at all", "mean NEG set", "mean SAME set"]))
for name, c in R["coverage"].items():
    print(row([name, c["pairs"], c["unique_a"], c["pairs_with_no_NEG_variant"],
               c["rate_no_NEG"], c["pairs_with_no_variants_at_all"],
               c["mean_neg_set_size"], c["mean_same_set_size"]]))

print("\n### Phase 1: 5-fold CV, grouped by sentence A\n")
print(header(["scorer", "AUC (fold mean ± sd)", "accuracy", "precision", "recall", "F1"]))
for s in SCORERS:
    c = R["cv"][s]
    print(row([NICE[s], f"{c['fold_mean']['auc']} ± {c['fold_std']['auc']}",
               c["fold_mean"]["accuracy"], c["fold_mean"]["precision"],
               c["fold_mean"]["recall"], c["fold_mean"]["f1"]]))

print("\nLogistic-regression coefficients (standardised features, mean over folds):\n")
for s in ("arch1_features", "arch3_hybrid"):
    coef = R["cv"][s].get("coefficients_mean", {})
    print(f"\n*{NICE[s]}*\n")
    print(header(list(coef)))
    print(row(list(coef.values())))

print("\n### Phase 2: trained on all of phase 1, evaluated once\n")
print(header(["scorer", "AUC", "accuracy", "precision", "recall", "F1",
              "**paraphrase FPR**"]))
for s in SCORERS:
    e = R["phase2"][s]
    print(row([NICE[s], e["auc"], e["accuracy"], e["precision"], e["recall"], e["f1"],
               f"**{e['by_subtype']['paraphrase']['false_positive_rate']}**"]))

print("\nFalse-positive rate by negative subtype (lower is better; "
      "paraphrase is the one that matters for grading):\n")
subtypes = ["paraphrase", "sts_low", "snli_neutral", "random_pair"]
print(header(["scorer"] + subtypes + ["recall on negations"]))
for s in SCORERS:
    e = R["phase2"][s]
    print(row([NICE[s]]
              + [e["by_subtype"][t]["false_positive_rate"] for t in subtypes]
              + [e["by_subtype"]["negation"]["recall"]]))

print("\nRecall on negated pairs by derived negation type:\n")
types = ["verbal", "antonym", "other"]
print(header(["scorer"] + types))
for s in SCORERS:
    e = R["phase2"][s]
    print(row([NICE[s]] + [e["by_neg_type"].get(t, {}).get("recall", "-") for t in types]))

print("\n### xNot360 (external, evaluated once at the end)\n")
print(header(["scorer", "AUC", "accuracy", "precision", "recall", "F1"]))
for s in SCORERS:
    e = R["xnot360"][s]
    print(row([NICE[s], e["auc"], e["accuracy"], e["precision"], e["recall"], e["f1"]]))

print("\n### Architecture 2 with an NLI similarity function (subsample)\n")
print(header(["set", "n", "arch 2 + NLI: AUC", "F1",
              "plain NLI(A,B) on the same pairs: AUC", "F1"]))
for name, b in R["arch2_nli"].items():
    print(row([name, b["n"], b["metrics"]["auc"], b["metrics"]["f1"],
               b["nli_baseline_same_subsample"]["auc"],
               b["nli_baseline_same_subsample"]["f1"]]))
print(f"\nMean max-similarity to the NEG set vs the SAME set: "
      + "; ".join(f"{k}: {v['mean_s_neg']} vs {v['mean_s_same']}"
                  for k, v in R["arch2_nli"].items()) + ".\n")

print("### Three-way decision (architecture 2, cosine margin)\n")
tw = R["three_way"]
print(f"Thresholds: unrelated below max-similarity {tw['thresholds']['unrelated']} "
      f"(5th percentile of training pairs), negation above margin "
      f"{tw['thresholds']['negation']}. Overall accuracy **{tw['accuracy']}**.\n")
golds = ["negation", "agrees", "unrelated"]
print(header(["gold \\ predicted"] + golds))
for g in golds:
    print(row([g] + [tw["confusion"].get(f"{g}->{p}", 0) for p in golds]))

print("\n### Where the architecture-2 margins fall (phase 2, cosine margin)\n")
ms = R["margin_signs"]
print(header(["subtype", "n", "margin > 0", "margin = 0", "margin < 0",
              "above the fitted threshold"]))
for sub, b in ms.items():
    c = b["margin_cos"]
    print(row([sub, c["n"], c["frac_positive"], c["frac_zero"], c["frac_negative"],
               f'{c["rate_above_fitted_threshold"]} (thr {c["fitted_threshold"]})']))

print("\n### What each feature is worth on its own (ROC-AUC)\n")
feats = R["config"]["features"]
print(header(["set"] + feats))
for name, b in R["features"].items():
    print(row([name] + [b["per_feature_auc"][f] for f in feats]))
print("\nMean feature value by class:\n")
print(header(["set", "class"] + feats))
for name, b in R["features"].items():
    for c in ("0", "1"):
        print(row([name, "not negation" if c == "0" else "negation"]
                  + [b["mean_by_class"][c][f] for f in feats]))
