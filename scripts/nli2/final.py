"""Assemble the report's summary sections from the committed results, plus the final error analysis.

    ./venv/bin/python scripts/nli2/final.py

1. picks the final configuration by VALIDATION accuracy among every adopted / compared system
   (never by TEST), reading the results files of steps 1.4, 2_all, 4.3 and 5.3;
2. the full ablation table (every configuration evaluated anywhere in Phases 1-5);
3. per-family results of the final configuration on TEST;
4. error analysis: 20 false positives and 20 false negatives of the final configuration on TEST,
   sampled at random (seed 0), assigned to buckets by the rules in bucket() -- not by hand;
5. dataset composition for train (clean DEV, and the fine-tune's generator data) / VALIDATION / TEST.
Writes results/final.json and prints the markdown.
"""

from __future__ import annotations

import json
import random
from collections import Counter

import numpy as np

import conditionals as K
import pipeline as PL
import rules as R
from common import C, RESULTS, SEED, log_test_read, save

load = lambda name: json.loads((RESULTS / name).read_text())


def ablation_rows() -> list[dict]:
    """(label, test block, K1 FPR, SemAntoNeg FPR, VAL acc, source) for every configuration."""
    rows = []
    r13 = load("1_3_refit_nli.json")
    for key, label in (("nli|old", "NLI alone, old threshold 0.00125 (Report_NLI.md)"),
                       ("nli_gate|old", "NLI + gate, old thresholds (Report_NLI.md)")):
        t = r13["rows"][key]["test"]
        rows.append({"label": label, "test": t, "k1": None, "sem": None,
                     "val": r13["rows"][key]["validation"]["all"]["accuracy"], "src": "1_3"})
    r14 = load("1_4_refit_gate.json")
    for key, label in (("nli", "NLI alone, honest threshold (1.3)"), ("nli_gate", "NLI + gate, clean thresholds (1.4)")):
        t = r14["rows"][key]["test"]
        rows.append({"label": label, "test": t, "k1": None, "sem": None,
                     "val": r14["rows"][key]["validation"]["all"]["accuracy"], "src": "1_4"})
    for name, label in (("2.1", None), ("2.2", None), ("2.3", None), ("2.4", None)):
        d = load(f"2_{name}.json")
        if name == "2.1":
            b = d["baseline"]
            rows.append({"label": "+ veto = Phase 1 baseline (full cascade, 1.4)", "test": b["slices"]["test"],
                         "k1": b["slices"]["k1"]["fpr"], "sem": b["slices"]["semantoneg_phase2"]["fpr"],
                         "val": b["slices"]["validation"]["all"]["accuracy"], "src": "2_2.1"})
        c = d["configs"][0]
        rows.append({"label": f"Phase 1 baseline + {c['config']['name']}", "test": c["slices"]["test"],
                     "k1": c["slices"]["k1"]["fpr"], "sem": c["slices"]["semantoneg_phase2"]["fpr"],
                     "val": c["slices"]["validation"]["all"]["accuracy"], "src": f"2_{name}"})
    d = load("2_all.json")
    for c in d["configs"]:
        rows.append({"label": f"Phase 2: {c['config']['name']}", "test": c["slices"]["test"],
                     "k1": c["slices"]["k1"]["fpr"], "sem": c["slices"]["semantoneg_phase2"]["fpr"],
                     "val": c["slices"]["validation"]["all"]["accuracy"], "src": "2_all"})
    d = load("3_3_conditionals_posthoc.json")
    for v in d["variants"]:
        c = v["result"]
        tag = " (post-hoc, test-informed)" if "post-hoc" in c["config"]["name"] else ""
        rows.append({"label": f"Phase 2 adopted + conditional rule: {c['config']['name']}{tag}",
                     "test": c["slices"]["test"], "k1": c["slices"]["k1"]["fpr"],
                     "sem": c["slices"]["semantoneg_phase2"]["fpr"],
                     "val": c["slices"]["validation"]["all"]["accuracy"], "src": "3_3"})
    d = load("4_3_compare.json")
    for name in ("logistic regression", "gradient-boosted trees"):
        t = d["rows"][name]
        rows.append({"label": f"learned combiner: {name}", "test": t, "k1": t["k1_fpr"], "sem": t["semantoneg_phase2_fpr"],
                     "val": t["validation_accuracy"], "src": "4_3"})
    if (RESULTS / "5_3_ft_eval.json").exists():
        d = load("5_3_ft_eval.json")
        for name in ("fine-tuned MiniLM alone", "fine-tuned MiniLM in cascade"):
            t = d["rows"][name]
            rows.append({"label": name, "test": t, "k1": t["k1_fpr"], "sem": t["semantoneg_phase2_fpr"],
                         "val": t["validation_accuracy"], "src": "5_3"})
    return rows


def md_ablation(rows) -> str:
    pc = lambda x: "—" if x is None else f"{100 * x:.1f}%"
    out = ["| configuration | phase-2 acc (95% CI) | all-TEST acc | AUC | FPR paraphrase | SemAntoNeg (n=41) | sts_low | "
           "snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | VAL acc | source |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        t, f = r["test"], r["test"]["fpr_by_kind"]
        p2 = t["cannot_phase2"]
        out.append(f"| {r['label']} | {pc(p2['accuracy'])} ({pc(p2['ci95'][0])}–{pc(p2['ci95'][1])}) | "
                   f"{pc(t['all']['accuracy'])} | {t['all']['auc']:.3f} | {pc(f['paraphrase']['fpr'])} | {pc(r['sem'])} | "
                   + " | ".join(pc(f[k]["fpr"]) for k in ("sts_low", "snli_neutral", "random_pair", "xnot360_not_negation"))
                   + f" | {pc(t['recall_by_type']['antonym']['recall'])} | {pc(r['k1'])} | {pc(r['val'])} | `{r['src']}` |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Final configuration: the GBM combiner or whatever wins on VALIDATION
# ---------------------------------------------------------------------------

def final_scores(choice: str):
    """Recompute the chosen system's TEST scores (deterministic, seed 0)."""
    import phase4 as P4
    sets, analyses, X = P4.table()
    if choice in ("learned combiner: gradient-boosted trees", "learned combiner: logistic regression"):
        fam = "gbm" if "gradient" in choice else "logistic"
        fit = P4.fit_combiners(sets, X)[fam]
        return sets, analyses, X, fit["_model"].predict_proba(X["test"])[:, 1], fit["threshold"]
    raise NotImplementedError(f"final scorer for {choice!r} not wired; add it here")


def bucket(r, x: dict, pred: bool) -> str:
    """Rule-based error buckets (x = the row's features)."""
    if r["dataset"] == "xnot360":
        if x["has_conditional_a"] and x["has_conditional_b"]:
            return "xNot360: both sides conditional (inverse / antecedent flip read as negation)" if pred \
                else "xNot360: conditional negation missed"
        if x["has_conditional_a"]:
            return "xNot360: conditional restated as a fact (P and not Q) — misjudged"
        return "xNot360: non-conditional legal / policy pair"
    if pred:   # false positive on CANNOT phase 2
        if r["kind"] in ("sts_low", "snli_neutral", "random_pair"):
            return "topically related, different claim (STS-low / SNLI neutral)"
        if r["kind"] == "SemAntoNeg" or x["cue_count_a"] != x["cue_count_b"]:
            return "paraphrase that swaps a cue for an antonym (polarity changes, meaning kept)"
        return "paraphrase with a content change (lexical rewording read as contradiction)"
    # false negative on CANNOT phase 2
    if x["antonym_aligned"] or r.get("neg_type") == "antonym":
        return "antonym negation missed"
    if x["cue_count_a"] >= 2 or x["cue_count_b"] >= 2 or (x["cue_count_a"] and x["cue_count_b"]):
        return "cue on both sides / double negation"
    if r.get("neg_type") == "verbal":
        return "verbal negation missed (single cue)"
    return "lexical substitution, no cue on either side (CANNOT 'other')"


def errors(sets, X, scores, thr, k=20) -> dict:
    import phase4 as P4
    test = sets["test"]
    y = C.labels(test)
    pred = scores >= thr
    rng = random.Random(SEED)
    out = {}
    for kind, mask in (("false_positives", pred & (y == 0)), ("false_negatives", ~pred & (y == 1))):
        idx = rng.sample(list(np.flatnonzero(mask)), min(k, int(mask.sum())))
        items = []
        for i in idx:
            x = dict(zip(P4.FEATURES, X["test"][i]))
            items.append({"bucket": bucket(test[i], x, bool(pred[i])), "dataset": test[i]["dataset"], "kind": test[i]["kind"],
                          "neg_type": test[i].get("neg_type"), "a": test[i]["a"], "b": test[i]["b"],
                          "score": round(float(scores[i]), 4), "p_contradiction": round(x["p_contradiction"], 4),
                          "cos": round(x["cos_sim"], 3)})
        counts = Counter(e["bucket"] for e in items)
        out[kind] = {"total_errors": int(mask.sum()), "sampled": len(items), "bucket_counts": dict(counts.most_common()),
                     "items": items}
        # bucket counts over ALL errors, not just the sample
        allb = Counter(bucket(test[i], dict(zip(P4.FEATURES, X["test"][i])), bool(pred[i])) for i in np.flatnonzero(mask))
        out[kind]["bucket_counts_all_errors"] = dict(allb.most_common())
    return out


def per_family(sets, scores, thr) -> list[dict]:
    test = sets["test"]
    y = C.labels(test)
    pred = scores >= thr
    out = []
    groups = [("negation / verbal", lambda r: r["label"] == 1 and r.get("neg_type") == "verbal"),
              ("negation / antonym", lambda r: r["label"] == 1 and r.get("neg_type") == "antonym"),
              ("negation / other", lambda r: r["label"] == 1 and r.get("neg_type") == "other"),
              ("xNot360 negation", lambda r: r["kind"] == "xnot360_negation"),
              ("paraphrase", lambda r: r["kind"] == "paraphrase"),
              ("SemAntoNeg paraphrase", lambda r: r["kind"] == "SemAntoNeg"),
              ("sts_low", lambda r: r["kind"] == "sts_low"),
              ("snli_neutral", lambda r: r["kind"] == "snli_neutral"),
              ("random_pair", lambda r: r["kind"] == "random_pair"),
              ("xNot360 not_negation", lambda r: r["kind"] == "xnot360_not_negation")]
    for name, fn in groups:
        idx = [i for i, r in enumerate(test) if fn(r)]
        lab = int(y[idx[0]])
        rate = float(pred[idx].mean())
        out.append({"family": name, "n": len(idx), "label": lab,
                    "metric": "recall" if lab == 1 else "FPR", "value": round(rate, 4),
                    "accuracy": round(float((pred[idx] == y[idx]).mean()), 4)})
    return out


def composition(sets) -> dict:
    out = {}
    for name in ("dev_clean", "validation", "test"):
        out[name] = dict(sorted(Counter(f"{r['dataset']}|{r['kind']}|{r.get('neg_type')}|{r['label']}"
                                        for r in sets[name]).items()))
    return out


def main() -> None:
    rows = ablation_rows()
    candidates = [r for r in rows if "post-hoc" not in r["label"] and "old" not in r["label"]]
    best = max(candidates, key=lambda r: r["val"])
    log_test_read("final", f"error analysis and per-family table of {best['label']}")
    sets, analyses, X, scores, thr = final_scores(best["label"])
    res = {"final_choice": {"label": best["label"], "validation_accuracy": best["val"],
                            "rule": "highest VALIDATION accuracy among non-post-hoc configurations"},
           "candidates_validation": {r["label"]: r["val"] for r in candidates},
           "ablation": [{k: v for k, v in r.items()} for r in rows],
           "per_family": per_family(sets, scores, thr), "errors": errors(sets, X, scores, thr),
           "composition": composition(sets)}
    save("final.json", res)
    print("FINAL:", best["label"], best["val"])
    print(md_ablation(rows))
    print(json.dumps(res["per_family"], indent=0))
    for kind in ("false_positives", "false_negatives"):
        print(kind, res["errors"][kind]["total_errors"], res["errors"][kind]["bucket_counts"],
              res["errors"][kind]["bucket_counts_all_errors"])


if __name__ == "__main__":
    main()
