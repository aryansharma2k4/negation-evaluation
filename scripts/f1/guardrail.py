"""Phase 3: generate-and-match guardrail.

    ./venv/bin/python scripts/f1/guardrail.py --step 3.3   # generator coverage on clean DEV (the precondition)
    ./venv/bin/python scripts/f1/guardrail.py --step 3.4   # guardrail variants on TEST, selection by VALIDATION

The rule (3.1, margin comparison, mandatory):
    N      = negations of T (variants.py: generator variants with flipped parity + Phase 1 affirmations)
    s_orig = sim(S, T);  s_neg = agg over n in N of sim(S, n);  margin = s_neg - s_orig
    margin > delta -> keep "negation";  otherwise downgrade to "not a negation"
It runs after the Phase 2-selected system, only on pairs that system calls a negation, and can only
downgrade (3.2). When the generator produced nothing for T, the guardrail abstains (no downgrade).

Variants tried (3.1 choices), each with delta fit by Youden's J on clean DEV:
    sim  : mpnet cosine | word-set Jaccard | mean of the two
    agg  : max | mean of the top 3
    fit  : delta on all clean-DEV pairs (the brief) | delta on the clean-DEV pairs the system's
           out-of-fold predictions call negations (the pairs the guardrail actually sees)
    plus : the margin as an extra combiner feature instead of a veto (an architecture variant;
           it can upgrade as well as downgrade, so it is reported against 3.2's constraint)
Selection rule, stated before the run: highest VALIDATION accuracy among "no guardrail" and every
variant; tie -> higher clean-DEV accuracy.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from core import C, N, ABLATION_HEADER, ablation_line, gain, load, log_test_read, save, system_row

import build_dev_clean as B  # noqa: E402  (nli2: word-set Jaccard)
import phase0 as P0
import variants as V

SETS = P0.SETS
NO_VARIANTS = np.nan


def sims(sim: str, s: str, others: list[str], emb) -> np.ndarray:
    if sim == "cos":
        return np.array([float(emb[s] @ emb[o]) for o in others])
    if sim == "jaccard":
        return np.array([B.overlap(s, o) for o in others])
    return (sims("cos", s, others, emb) + sims("jaccard", s, others, emb)) / 2


def margins(rows, neg, emb, sim="cos", agg="max") -> np.ndarray:
    out = np.full(len(rows), NO_VARIANTS)
    for i, r in enumerate(rows):
        vs = [v for v, _ in neg.get(r["a"], [])]
        if not vs:
            continue
        s_orig = sims(sim, r["b"], [r["a"]], emb)[0]
        sn = np.sort(sims(sim, r["b"], vs, emb))[::-1]
        s_neg = sn[0] if agg == "max" else sn[:3].mean()
        out[i] = s_neg - s_orig
    return out


def context(sets):
    neg = V.negations([r["a"] for k in SETS for r in sets[k]])
    emb = V.embed([s for k in SETS for r in sets[k] for s in (r["a"], r["b"])] + [v for vs in neg.values() for v, _ in vs])
    return neg, emb


# ---------------------------------------------------------------------------
# 3.3 coverage
# ---------------------------------------------------------------------------

def step_3_3() -> None:
    sets = P0.P4.table()[0]
    neg, emb = context(sets)
    dev = sets["dev_clean"]
    res = {"by_split": {}}
    for split in ("dev_clean",):
        rows = [r for r in sets[split] if r["label"] == 1]
        best, best_fam, per_fam, exact = [], [], defaultdict(list), 0
        no_var = 0
        for r in rows:
            vs = neg.get(r["a"], [])
            if not vs:
                no_var += 1
                best.append(np.nan); best_fam.append("(no variants)")
                continue
            s = sims("cos", r["b"], [v for v, _ in vs], emb)
            j = int(np.argmax(s))
            best.append(float(s[j])); best_fam.append(vs[j][1])
            exact += any(V_norm(v) == V_norm(r["b"]) for v, _ in vs)
            fams = defaultdict(float)
            for (v, f), x in zip(vs, s):
                fams[f] = max(fams[f], float(x))
            for f, x in fams.items():
                per_fam[f].append(x)
        best = np.array(best)
        valid = best[~np.isnan(best)]
        cover = lambda t, idx=None: round(float(np.nansum((best >= t) if idx is None else (best[idx] >= t)) /
                                                (len(best) if idx is None else len(idx))), 4)
        by_type = {}
        for t in ("verbal", "antonym", "other"):
            idx = np.array([i for i, r in enumerate(rows) if r.get("neg_type") == t])
            by_type[t] = {"n": len(idx), ">=0.9": cover(0.9, idx), ">=0.8": cover(0.8, idx), ">=0.7": cover(0.7, idx),
                          "median_best": round(float(np.nanmedian(best[idx])), 4)}
        res["by_split"][split] = {
            "gold_negations": len(rows), "with_no_variant": no_var, "exact_match_with_a_variant": exact,
            "coverage": {">=0.9": cover(0.9), ">=0.8": cover(0.8), ">=0.7": cover(0.7)},
            "best_similarity_quantiles": {q: round(float(np.quantile(valid, q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "sim_to_original_quantiles": {q: round(float(np.quantile([float(emb[r["a"]] @ emb[r["b"]]) for r in rows], q)), 4)
                                          for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "by_gold_type": by_type,
            "argmax_family": dict(Counter(best_fam).most_common()),
            "per_family_best_ge_0.9": {f: {"bases_with_family": len(x), "share_ge_0.9": round(float(np.mean(np.array(x) >= 0.9)), 4)}
                                       for f, x in sorted(per_fam.items(), key=lambda kv: -len(kv[1]))},
            "variants_per_teacher_median": float(np.median([len(neg.get(r["a"], [])) for r in rows])),
        }
    # the discriminative precondition: does the margin separate negations from non-negations on DEV?
    from sklearn.metrics import roc_auc_score
    for sim in ("cos", "jaccard", "mix"):
        m = margins(dev, neg, emb, sim)
        ok = ~np.isnan(m)
        y = C.labels(dev)
        res.setdefault("margin_auc_dev", {})[sim] = {
            "auc": round(float(roc_auc_score(y[ok], m[ok])), 4), "pairs_with_variants": int(ok.sum()), "n": len(dev),
            "share_margin_gt_0_negations": round(float((m[ok & (y == 1)] > 0).mean()), 4),
            "share_margin_gt_0_non_negations": round(float((m[ok & (y == 0)] > 0).mean()), 4)}
        for kind in ("paraphrase", "SemAntoNeg", "hard_negative", "sts_low", "snli_neutral", "random_pair"):
            idx = np.array([i for i, r in enumerate(dev) if r["kind"] == kind and ok[i]])
            if len(idx):
                res["margin_auc_dev"][sim].setdefault("share_margin_gt_0_by_kind", {})[kind] = {
                    "n": len(idx), "share": round(float((m[idx] > 0).mean()), 4)}
    examples = []
    rows = [r for r in dev if r["label"] == 1]
    rng = np.random.default_rng(0)
    for i in rng.choice(len(rows), 12, replace=False):
        r = rows[int(i)]
        vs = neg.get(r["a"], [])
        if vs:
            s = sims("cos", r["b"], [v for v, _ in vs], emb)
            j = int(np.argmax(s))
            examples.append({"a": r["a"], "b": r["b"], "neg_type": r.get("neg_type"), "best_variant": vs[j][0],
                             "family": vs[j][1], "sim": round(float(s[j]), 3)})
    res["examples"] = examples
    save("3_3_coverage.json", res)
    import json
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1)[:7000])
    for e in examples:
        print(e)


def V_norm(s: str) -> str:
    import re
    return " ".join(re.findall(r"[a-z0-9']+", s.lower()))


# ---------------------------------------------------------------------------
# 3.4 guardrail
# ---------------------------------------------------------------------------

def apply(scores, thr, m, delta) -> dict:
    """Downgrade pairs called negations whose margin <= delta; abstain where no variants."""
    out = {}
    for k in SETS:
        s = np.asarray(scores[k], float).copy()
        down = (s >= thr) & ~np.isnan(m[k]) & (m[k] <= delta)
        s[down] = C.STOPPED
        out[k] = s
    return out


def step_3_4() -> None:
    import ensemble as P2
    sys_ = P2.selected_system()
    sets, scores, thr = sys_["sets"], sys_["scores"], sys_["thr"]
    neg, emb = context(sets)
    dev = sets["dev_clean"]
    res = {"base_system": sys_["name"], "rule": __doc__.split("Selection rule, stated before the run:")[1].strip(),
           "rows": {"no guardrail": {"row": system_row(sets, scores, thr)}}}
    all_scores = {"no guardrail": (scores, thr)}
    oof_called = sys_["oof_dev"] >= sys_["oof_thr"]
    for sim in ("cos", "jaccard", "mix"):
        for agg in ("max", "top3"):
            m = {k: margins(sets[k], neg, emb, sim, agg) for k in SETS}
            ok = ~np.isnan(m["dev_clean"])
            for fit in ("all DEV", "DEV pairs called negation (out-of-fold)"):
                idx = ok if fit == "all DEV" else ok & oof_called
                rows_fit = [r for r, keep in zip(dev, idx) if keep]
                delta = N.youden_threshold(rows_fit, m["dev_clean"][idx]) - 1e-12   # margin > delta keeps
                g = apply(scores, thr, m, delta)
                name = f"guardrail: sim={sim}, agg={agg}, delta fit on {fit}"
                ev = system_row(sets, g, thr)
                y = C.labels(sets["test"])
                called = np.asarray(scores["test"]) >= thr
                down = called & (g["test"] < thr)
                ev["downgrades"] = {"test": int(down.sum()), "test_wrong (were real negations)": int((down & (y == 1)).sum()),
                                    "validation": int(((np.asarray(scores["validation"]) >= thr) & (g["validation"] < thr)).sum()),
                                    "dev_clean": int(((np.asarray(scores["dev_clean"]) >= thr) & (g["dev_clean"] < thr)).sum())}
                ev["delta"] = float(delta)
                ev["gain_vs_no_guardrail"] = gain(sets, (scores["test"], thr), (g["test"], thr))
                res["rows"][name] = {"row": ev}
                all_scores[name] = (g, thr)
    # margin as an extra combiner feature (architecture variant; may upgrade too)
    m = {k: np.nan_to_num(margins(sets[k], neg, emb, "cos", "max"), nan=0.0) for k in SETS}
    Xm = {k: np.hstack([sys_["X"][k], m[k][:, None]]) for k in SETS}
    g = P2.fit_gbm(sets, Xm["dev_clean"])
    sc = P0.with_override({k: g["model"].predict_proba(Xm[k])[:, 1] for k in SETS}, sys_["masks"])
    ev = system_row(sets, sc, g["threshold"])
    ev["gain_vs_no_guardrail"] = gain(sets, (scores["test"], thr), (sc["test"], g["threshold"]))
    ev["importance_of_margin"] = round(float(g["model"].feature_importances_[-1]), 4)
    res["rows"]["margin (cos, max) as an extra combiner feature"] = {"row": ev, "oof_dev_accuracy": g["oof_dev_accuracy"]}
    all_scores["margin (cos, max) as an extra combiner feature"] = (sc, g["threshold"])
    log_test_read("3.4", "guardrail variants on TEST")
    best = max(res["rows"], key=lambda n: (res["rows"][n]["row"]["validation"]["accuracy"], res["rows"][n]["row"]["dev_clean"]["accuracy"]))
    res["selected"] = best
    res["selected_params"] = next(({"sim": sim, "agg": agg, "delta": res["rows"][best]["row"]["delta"]}
                                   for sim in ("cos", "jaccard", "mix") for agg in ("max", "top3")
                                   if best.startswith(f"guardrail: sim={sim}, agg={agg},")), None)
    # the same guardrail variant on the Phase 0 baseline, for the ablation's "this phase alone" row
    if res["selected_params"]:
        b = P0.baseline()
        gp = res["selected_params"]
        mb = {k: margins(b["sets"][k], neg, emb, gp["sim"], gp["agg"]) for k in SETS}
        gb = apply(b["scores"], b["thr"], mb, gp["delta"])
        res["phase0_plus_selected_guardrail"] = {"row": system_row(b["sets"], gb, b["thr"]),
                                                 "gain_vs_phase0": gain(b["sets"], (b["scores"]["test"], b["thr"]), (gb["test"], b["thr"]))}
    # xNot360 conditional slice (both sides conditional) for every row
    import conditionals as K
    test = sets["test"]
    conds = K.conditionals([s for r in test for s in (r["a"], r["b"])])
    idx = np.array([i for i, r in enumerate(test) if r["dataset"] == "xnot360" and K.primary(conds[r["a"]]) and K.primary(conds[r["b"]])])
    y = C.labels(test)
    for name, (sc, t) in all_scores.items():
        pred = np.asarray(sc["test"])[idx] >= t
        res["rows"][name]["xnot360_both_conditional"] = {"n": len(idx), "accuracy": round(float((pred == y[idx]).mean()), 4),
                                                         "fpr": round(float(pred[y[idx] == 0].mean()), 4)}
    save("3_4_guardrail.json", res)
    print(ABLATION_HEADER)
    for name, v in res["rows"].items():
        r = v["row"]
        print(ablation_line(name, r), "| DEV", r["dev_clean"]["accuracy"], "| down", r.get("downgrades"),
              "| xcond", v["xnot360_both_conditional"]["accuracy"])
    print("SELECTED:", best)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["3.3", "3.4"])
    {"3.3": step_3_3, "3.4": step_3_4}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
