"""Summary for report_f2.md: the adopted configuration against the Phase 0 baseline on the final val_ood
(MNLI + conditionals), paired tests including the low-overlap slice, and the error analysis.

    ./venv/bin/python experiments/f2/final_f2.py    -> results/final_f2.json

Adopted = step 2.3's choice: pruned 4-model GBM (26 features) + cancellation override + low-overlap NLI
override (theta 0.5, rho 0.9, kappa 0.6). Phase 4 changes affirmation coverage only; Phase 5's variant B
was not adopted; Phase 6 is descriptive. Error analysis: 20 FP + 20 FN on TEST, random (seed 0), bucketed
by nli2's final.bucket rule (unchanged).
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from core2 import C, N, SEED, STOPPED, log_test_read, save

import deberta as D
import ensemble as E  # noqa: E402  (f1)
import eval2 as E2
import final as NF  # noqa: E402  (nli2)
import phase4 as P4  # noqa: E402


def main() -> None:
    ctx = D.context()
    X_c = {k: np.hstack([ctx["X"][k]] + [ctx["P"][m][k] for m in D.PRUNED if m != "deberta_base"]) for k in E2.SETS}
    g = E.fit_gbm(ctx["sets"], X_c["dev_clean"])
    sc_c = {k: np.where(ctx["masks"][k], STOPPED, g["model"].predict_proba(X_c[k])[:, 1]) for k in E2.SETS}
    adopted = D.with_rule(ctx, sc_c, "deberta_base")
    p0 = ctx["scores"]
    p0_rule = D.with_rule(ctx, p0, "deberta_base")
    nli = {k: ctx["pc"][k] for k in E2.SETS}
    log_test_read("final", "adopted vs Phase 0 on TEST, VALIDATION, val_ood (v2); error analysis")
    rows = {"Phase 0 baseline": (p0, ctx["thr"]), "Phase 0 + override (step 1.3)": (p0_rule, ctx["thr"]),
            "adopted: C + override (step 2.3)": (adopted, g["threshold"]), "NLI alone": (nli, ctx["nli_thr"])}
    res = {"rows": {n: E2.row(ctx["sets"], s, t, ctx["ood"]) for n, (s, t) in rows.items()}, "adopted_threshold": g["threshold"]}
    t = ctx["sets"]["test"]
    y = C.labels(t)
    ov = ctx["X"]["test"][:, 16]
    a, b = (np.asarray(p0["test"]) >= ctx["thr"]), (np.asarray(adopted["test"]) >= g["threshold"])
    res["gain_adopted_vs_phase0"] = {p: N.compare(t, (p0["test"], ctx["thr"]), (adopted["test"], g["threshold"]), p)
                                     for p in ("all", "cannot_phase2", "xnot360")}
    for tag, m in (("lowov_negations_lt_0.6", (y == 1) & (ov < 0.6)), ("lowov_non_negations_lt_0.4", (y == 0) & (ov < 0.4))):
        ca, cb = (a[m] == y[m]).astype(float), (b[m] == y[m]).astype(float)
        res["gain_adopted_vs_phase0"][tag] = {"n": int(m.sum()), "gain": round(float(cb.mean() - ca.mean()), 4),
                                              "p_permutation": N.paired_permutation_p(ca, cb)}
    # error analysis of the adopted configuration
    rng = random.Random(SEED)
    errs = {}
    for kind, mask in (("false_positives", b & (y == 0)), ("false_negatives", ~b & (y == 1))):
        idx = list(np.flatnonzero(mask))
        samp = rng.sample(idx, min(20, len(idx)))
        bk = lambda i: NF.bucket(t[i], dict(zip(P4.FEATURES, ctx["X"]["test"][i])), bool(b[i]))
        errs[kind] = {"total": len(idx), "sample_buckets": dict(Counter(bk(i) for i in samp).most_common()),
                      "all_buckets": dict(Counter(bk(i) for i in idx).most_common()),
                      "examples": {bucket: next(({"a": t[i]["a"], "b": t[i]["b"], "overlap": round(float(ov[i]), 3)} for i in samp if bk(i) == bucket), None)
                                   for bucket in set(bk(i) for i in samp)},
                      "overlap_of_errors": {"median": round(float(np.median(ov[idx])), 3), "share_below_0.6": round(float((ov[idx] < 0.6).mean()), 3)}}
    res["errors"] = errs
    res["per_family"] = NF.per_family(ctx["sets"], np.asarray(adopted["test"]), g["threshold"])
    save("final_f2.json", res)
    print(E2.HEADER)
    for n, ev in res["rows"].items():
        print(E2.line(n, ev), ev["val_ood_by_dataset"])
    print({k: (v["gain"], v["p_permutation"]) for k, v in res["gain_adopted_vs_phase0"].items()})
    for k, v in errs.items():
        print(k, v["total"], v["sample_buckets"], v["all_buckets"], v["overlap_of_errors"])
        for bk_, ex in v["examples"].items():
            print("   ", bk_, "|", ex)


if __name__ == "__main__":
    main()
