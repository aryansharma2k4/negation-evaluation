"""Phase 2: settle the DeBERTa question.

    ./venv/bin/python scripts/f2/deberta.py --step 2.1    # why the f1 post-hoc swap collapsed xNot360
    ./venv/bin/python scripts/f2/deberta.py --step 2.2    # configs A / B / C, retrained, +/- the override

Every combiner is retrained from scratch with nli2 step 4.2's protocol (f1 ensemble.fit_gbm: grid by
grouped-CV AUC, folds by connected component of shared sentences, Youden on out-of-fold predictions,
refit on all clean DEV). The cancellation override stays in front. "+ override" applies step 1.3's rule
(theta 0.5, rho 0.9, kappa 0.6) with the configuration's primary NLI model's P(contradiction).
Selection: step 0.3 rule against the currently adopted configuration (Phase 0 baseline + override), plus
the 2.5 overfitting check from reports/08-guardrail-ensemble.md (out-of-fold DEV exceeding TEST by > 2 pts, reported).
"""

from __future__ import annotations

import argparse

import numpy as np

from core2 import C, STOPPED, baseline, log_test_read, pc, save

import eval2 as E2
import ensemble as E  # noqa: E402  (f1)
import override as O
import phase4 as P4  # noqa: E402
import val_ood as VO
from nli_models import MODELS

ANLI = "ml_deberta_base_anli"
PRUNED = ("deberta_base", "roberta_base", ANLI, "bart_large")
RULE = (0.5, 0.9, 0.6)


def context():
    ctx = O.context()
    import ensemble as Ens
    keys = list(PRUNED)
    P = {k: {s: Ens.probs(k, ctx["sets"][s]) for s in ("dev_clean", "validation", "test", "k1")} for k in keys}
    _, extra = VO.featurise(ctx["ood"], keys=tuple(keys))
    for k in keys:
        P[k]["val_ood"] = extra["probs"][k]
    ctx["P"] = P
    return ctx


def configs(ctx) -> dict:
    X, P = ctx["X"], ctx["P"]
    S = E2.SETS
    out = {}
    XA = {k: X[k].copy() for k in S}
    for k in S:
        XA[k][:, 1:4] = P[ANLI][k]
    out["A — replace: DeBERTa-anli's probabilities only (17 features)"] = (XA, ANLI)
    out["B — alongside: reference + DeBERTa-anli (20 features)"] = ({k: np.hstack([X[k], P[ANLI][k]]) for k in S}, "deberta_base")
    out["C — pruned ensemble: 4 models (26 features)"] = (
        {k: np.hstack([X[k]] + [P[m][k] for m in PRUNED if m != "deberta_base"]) for k in S}, "deberta_base")
    return out


def with_rule(ctx, scores, primary) -> dict:
    theta, rho, kappa = RULE
    out = {}
    for k in E2.SETS:
        Xk = ctx["X"][k]
        fire = (Xk[:, O.J_OV] < theta) & (ctx["P"][primary][k][:, 0] > rho) & (Xk[:, O.J_COS] >= kappa) & ~ctx["masks"][k]
        out[k] = np.where(fire, 1.0, scores[k])
    return out


def step_2_1() -> None:
    """Diagnosis without TEST metrics: distributions of each model's P(contradiction) on clean DEV by
    label, and on xNot360 inverse pairs (both sides conditional, label 0) -- the slice that collapsed."""
    ctx = context()
    import conditionals as K
    t = ctx["sets"]["test"]
    log_test_read("2.1", "P(contradiction) distributions on xNot360 conditional pairs (diagnosis)")
    conds = K.conditionals([s for r in t for s in (r["a"], r["b"])])
    inv = np.array([r["dataset"] == "xnot360" and r["label"] == 0 and bool(K.primary(conds[r["a"]])) and bool(K.primary(conds[r["b"]])) for r in t])
    d = ctx["sets"]["dev_clean"]
    yd = C.labels(d)
    ovd = ctx["X"]["dev_clean"][:, O.J_OV]
    res = {"retrained": True, "evidence": ("experiments/f1/ensemble.py lines 274-278: Xf['dev_clean'] carries the ANLI "
                                           "probabilities and fit_gbm(sets, Xf['dev_clean']) refits the full grid; "
                                           "the swap was retrained, so the 57.2% is a real result, not a bug"), "models": {}}
    q = lambda v: {f"q{int(100*x)}": round(float(np.quantile(v, x)), 3) for x in (0.1, 0.25, 0.5, 0.75)}
    for k in ("deberta_base", ANLI):
        pd_, pt = ctx["P"][k]["dev_clean"][:, 0], ctx["P"][k]["test"][:, 0]
        res["models"][k] = {
            "dev negations, overlap >= 0.6": q(pd_[(yd == 1) & (ovd >= 0.6)]),
            "dev non-negations, overlap >= 0.6 (paraphrase-like)": q(pd_[(yd == 0) & (ovd >= 0.6)]),
            "dev non-negations with P_c > 0.5, overlap >= 0.6": int(((yd == 0) & (ovd >= 0.6) & (pd_ > 0.5)).sum()),
            "xNot360 conditional inverse pairs (label 0)": q(pt[inv]), "n_inverse": int(inv.sum()),
            "share of inverse pairs with P_c > 0.5": round(float((pt[inv] > 0.5).mean()), 3),
            "share of dev high-overlap negations with P_c > 0.5": round(float((pd_[(yd == 1) & (ovd >= 0.6)] > 0.5).mean()), 3)}
    # what the two retrained GBMs do on the inverse slice
    XA = {k: ctx["X"][k].copy() for k in ("dev_clean", "test")}
    for k in XA:
        XA[k][:, 1:4] = ctx["P"][ANLI][k]
    gA = E.fit_gbm(ctx["sets"], XA["dev_clean"])
    g0 = baseline()["fit"]
    res["inverse_slice_called_negation"] = {
        "Phase 0 GBM (reference model)": round(float((g0["_model"].predict_proba(ctx["X"]["test"][inv])[:, 1] >= g0["threshold"]).mean()), 3),
        "GBM retrained on DeBERTa-anli": round(float((gA["model"].predict_proba(XA["test"][inv])[:, 1] >= gA["threshold"]).mean()), 3),
        "DeBERTa-anli standalone (thr 0.931)": round(float((ctx["P"][ANLI]["test"][inv][:, 0] >= 0.9305).mean()), 3),
        "reference standalone (thr 0.975)": round(float((ctx["P"]["deberta_base"]["test"][inv][:, 0] >= 0.9753).mean()), 3)}
    res["gbm_threshold_anli"] = gA["threshold"]
    res["importances_anli_gbm"] = dict(sorted(zip(P4.FEATURES, np.round(gA["model"].feature_importances_, 4).tolist()), key=lambda kv: -kv[1])[:5])
    save("2_1_swap_diagnosis.json", res)
    import json
    print(json.dumps(res, indent=1))


def step_2_2() -> None:
    ctx = context()
    log_test_read("2.2", "DeBERTa configs A/B/C retrained, with and without the override")
    base = E2.row(ctx["sets"], with_rule(ctx, ctx["scores"], "deberta_base"), ctx["thr"], ctx["ood"])
    res = {"current_adopted": base, "rows": {}}
    cost = {k: __import__("core2").F1.load("2_1_timing.json")["models"][k]["ms_per_pair"] for k in PRUNED}
    for name, (X, primary) in configs(ctx).items():
        g = E.fit_gbm(ctx["sets"], X["dev_clean"])
        sc = {k: np.where(ctx["masks"][k], STOPPED, g["model"].predict_proba(X[k])[:, 1]) for k in E2.SETS}
        models = [ANLI] if name.startswith("A") else ["deberta_base", ANLI] if name.startswith("B") else list(PRUNED)
        for tag, s in (("", sc), (" + override", with_rule(ctx, sc, primary))):
            ev = E2.row(ctx["sets"], s, g["threshold"], ctx["ood"])
            ev["oof_dev_accuracy"] = g["oof_dev_accuracy"]
            ev["oof_minus_test_pts"] = round(100 * (g["oof_dev_accuracy"] - ev["all"]["accuracy"]), 2)
            ev["overfit_flag"] = ev["oof_minus_test_pts"] > 2
            ev["passes_rule"] = E2.adopt(base, ev)
            ev["nli_ms_per_pair"] = round(sum(cost[m] for m in models), 1)
            ev["gain_vs_phase0"] = O.F1gain(ctx, s, g["threshold"])
            ev["chosen"] = g["chosen"]
            res["rows"][name + tag] = ev
    passing = [n for n, ev in res["rows"].items() if ev["passes_rule"]]
    res["selected"] = max(passing, key=lambda n: E2.gain_score(base, res["rows"][n])) if passing else "keep current (Phase 0 + override)"
    save("2_2_deberta.json", res)
    print(E2.HEADER)
    print(E2.line("current: Phase 0 + override", base))
    for n, ev in res["rows"].items():
        print(E2.line(n, ev), "| oof", ev["oof_dev_accuracy"], "gap", ev["oof_minus_test_pts"], "pass" if ev["passes_rule"] else "", ev["nli_ms_per_pair"])
    print("SELECTED:", res["selected"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["2.1", "2.2"])
    {"2.1": step_2_1, "2.2": step_2_2}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
