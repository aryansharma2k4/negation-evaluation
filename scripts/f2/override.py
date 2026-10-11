"""Phase 1: the low-overlap NLI override and its variants.

    ./venv/bin/python scripts/f2/override.py --step 1.1   # hard rule: grid, surface
    ./venv/bin/python scripts/f2/override.py --step 1.3   # every variant, selection by the 0.3 rule

Variants (all keep the cancellation override first; it is never undone):
  rule (brief):       overlap < theta and P_c > rho                      -> negation
  rule + topic floor: overlap < theta and P_c > rho and cos >= kappa      -> negation   (added: step 0.1 shows
                      NLI calls ~50% of unrelated low-overlap pairs contradictions)
  soft feature:       the rule's boolean, and NLI margin P_c - P_e, as extra combiner features (retrained)
  interaction:        P_c * (1 - overlap) as an extra feature (retrained)
  drop shortcut:      combiner retrained without token_overlap and len_diff (15 features)
  routing:            overlap < s -> NLI alone (clean-DEV threshold), else combiner; s fit on DEV

Fitting (stated before the run): rule parameters maximise accuracy on clean DEV (combiner out-of-fold) pooled
with the val_ood *fit* half; routing switch likewise. Retrained variants: nli2 step 4.2 protocol on clean DEV.
Selection: the step-0.3 rule (VALIDATION and val_ood select-half accuracy both non-decreasing vs Phase 0;
among passing candidates the largest summed gain; tie -> clean DEV).
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from core2 import C, STOPPED, baseline, log_test_read, pc, save

import eval2 as E2
import ensemble as E  # noqa: E402  (f1: fit_gbm)
import phase0 as P0  # noqa: E402
import phase4 as P4  # noqa: E402

THETAS = (0.3, 0.4, 0.5, 0.6, 0.7)
RHOS = (0.9, 0.95, 0.975, 0.99, 0.995, 0.999)
KAPPAS = (None, 0.5, 0.6, 0.7)
J_OV, J_LD, J_COS = (P4.FEATURES.index(f) for f in ("token_overlap", "len_diff", "cos_sim"))


def context():
    sets, X, scores, masks, ood = E2.baseline_scores()
    b = baseline()
    pcon = {k: np.array([r["nli_c"] for r in sets[k]]) for k in E2.SETS}
    pent = {k: np.array([r["nli_e"] for r in sets[k]]) for k in E2.SETS}
    return {"sets": sets, "X": X, "scores": scores, "masks": masks, "ood": ood, "thr": b["thr"], "nli_thr": b["nli_thr"],
            "pc": pcon, "pe": pent, "oof_dev": b["oof_dev"]}


def fires(ctx, k, theta, rho, kappa) -> np.ndarray:
    X = ctx["X"][k]
    m = (X[:, J_OV] < theta) & (ctx["pc"][k] > rho) & ~ctx["masks"][k]
    if kappa is not None:
        m &= X[:, J_COS] >= kappa
    return m


def apply_rule(ctx, scores, theta, rho, kappa) -> dict:
    return {k: np.where(fires(ctx, k, theta, rho, kappa), 1.0, scores[k]) for k in E2.SETS}


def fit_pool(ctx):
    """clean DEV (out-of-fold combiner) + val_ood fit half: (labels, index masks)."""
    fit = np.array([r["part"] == "fit" for r in ctx["ood"]])
    return fit


def pooled_accuracy(ctx, sc_dev, sc_ood, thr) -> float:
    fit = fit_pool(ctx)
    yd, yo = C.labels(ctx["sets"]["dev_clean"]), C.labels(ctx["ood"])
    right = ((np.asarray(sc_dev) >= thr) == yd).sum() + ((np.asarray(sc_ood)[fit] >= thr) == yo[fit]).sum()
    return float(right / (len(yd) + fit.sum()))


def grid(ctx) -> dict:
    thr = ctx["thr"]
    base_dev = ctx["oof_dev"]
    out = {}
    for kappa in KAPPAS:
        surf = {}
        for theta, rho in itertools.product(THETAS, RHOS):
            d = np.where(fires(ctx, "dev_clean", theta, rho, kappa), 1.0, base_dev)
            o = np.where(fires(ctx, "val_ood", theta, rho, kappa), 1.0, ctx["scores"]["val_ood"])
            surf[f"{theta}|{rho}"] = round(pooled_accuracy(ctx, d, o, thr), 4)
        best = max(surf, key=surf.get)
        out[str(kappa)] = {"surface": surf, "best": best, "best_pooled_accuracy": surf[best]}
    out["no_rule_pooled_accuracy"] = round(pooled_accuracy(ctx, base_dev, ctx["scores"]["val_ood"], thr), 4)
    return out


def retrained(ctx, X: dict, name: str) -> tuple[dict, float, dict]:
    g = E.fit_gbm(ctx["sets"], X["dev_clean"])
    sc = {k: np.where(ctx["masks"][k], STOPPED, g["model"].predict_proba(X[k])[:, 1]) for k in E2.SETS}
    return sc, g["threshold"], {"oof_dev_accuracy": g["oof_dev_accuracy"], "chosen": g["chosen"], "model": g["model"]}


def variants(ctx, fitted: dict) -> dict:
    out = {"Phase 0 baseline": (ctx["scores"], ctx["thr"], {})}
    for kappa in KAPPAS:
        theta, rho = map(float, fitted[str(kappa)]["best"].split("|"))
        tag = f"rule: overlap < {theta}, P_c > {rho}" + ("" if kappa is None else f", cos >= {kappa}")
        out[tag] = (apply_rule(ctx, ctx["scores"], theta, rho, kappa), ctx["thr"], {"theta": theta, "rho": rho, "kappa": kappa})
    # soft: the best topic-floor rule's boolean + NLI margin as features
    kbest = max((k for k in KAPPAS if k is not None), key=lambda k: fitted[str(k)]["best_pooled_accuracy"])
    theta, rho = map(float, fitted[str(kbest)]["best"].split("|"))
    Xs = {k: np.hstack([ctx["X"][k], fires(ctx, k, theta, rho, kbest)[:, None].astype(float),
                        (ctx["pc"][k] - ctx["pe"][k])[:, None]]) for k in E2.SETS}
    sc, thr, ex = retrained(ctx, Xs, "soft")
    ex["importances_new"] = {"nli_overrides_combiner": round(float(ex["model"].feature_importances_[-2]), 4),
                             "nli_margin": round(float(ex["model"].feature_importances_[-1]), 4)}
    out["soft: + nli_overrides_combiner, + NLI margin (features, retrained)"] = (sc, thr, ex)
    Xi = {k: np.hstack([ctx["X"][k], (ctx["pc"][k] * (1 - ctx["X"][k][:, J_OV]))[:, None]]) for k in E2.SETS}
    sc, thr, ex = retrained(ctx, Xi, "interaction")
    ex["importances_new"] = {"pc_x_1_minus_overlap": round(float(ex["model"].feature_importances_[-1]), 4)}
    out["interaction: + P_c x (1 - overlap) (retrained)"] = (sc, thr, ex)
    keep = [j for j in range(17) if j not in (J_OV, J_LD)]
    Xd = {k: ctx["X"][k][:, keep] for k in E2.SETS}
    sc, thr, ex = retrained(ctx, Xd, "drop")
    ex["importances"] = dict(sorted(zip([P4.FEATURES[j] for j in keep], np.round(ex["model"].feature_importances_, 4).tolist()),
                                    key=lambda kv: -kv[1]))
    out["drop shortcut: no token_overlap, no len_diff (15 features, retrained)"] = (sc, thr, ex)
    # routing: overlap < s -> NLI alone (thr 0.975, cancellation override kept), else combiner
    best_s, best_acc = None, -1
    for s in (0.3, 0.4, 0.5, 0.6, 0.7):
        d = np.where(ctx["X"]["dev_clean"][:, J_OV] < s, np.where(ctx["pc"]["dev_clean"] >= ctx["nli_thr"], 1.0, STOPPED), ctx["oof_dev"])
        o = np.where(ctx["X"]["val_ood"][:, J_OV] < s, np.where(ctx["pc"]["val_ood"] >= ctx["nli_thr"], 1.0, STOPPED), ctx["scores"]["val_ood"])
        acc = pooled_accuracy(ctx, np.where(ctx["masks"]["dev_clean"], STOPPED, d), np.where(ctx["masks"]["val_ood"], STOPPED, o), ctx["thr"])
        if acc > best_acc:
            best_s, best_acc = s, acc
    sc = {k: np.where(ctx["masks"][k], STOPPED,
                      np.where(ctx["X"][k][:, J_OV] < best_s, np.where(ctx["pc"][k] >= ctx["nli_thr"], 1.0, STOPPED), ctx["scores"][k]))
          for k in E2.SETS}
    out[f"routing: overlap < {best_s} -> NLI alone, else combiner"] = (sc, ctx["thr"], {"switch": best_s, "pooled_fit_accuracy": round(best_acc, 4)})
    return out


def step_1_1() -> None:
    ctx = context()
    g = grid(ctx)
    save("1_1_rule_grid.json", g)
    print("no rule pooled", g["no_rule_pooled_accuracy"])
    for kappa in KAPPAS:
        s = g[str(kappa)]["surface"]
        print(f"== kappa {kappa}: best {g[str(kappa)]['best']} {g[str(kappa)]['best_pooled_accuracy']}")
        print("theta\\rho " + " ".join(f"{r:>6}" for r in RHOS))
        for t in THETAS:
            print(f"{t:<9} " + " ".join(f"{s[f'{t}|{r}']:.4f}" for r in RHOS))


def step_1_3() -> None:
    ctx = context()
    fitted = load_grid()
    V = variants(ctx, fitted)
    log_test_read("1.3", "low-overlap override variants on TEST")
    base_ev = None
    res = {"rows": {}, "rule": "step 0.3: VALIDATION and val_ood (select) both non-decreasing vs Phase 0"}
    for name, (sc, thr, ex) in V.items():
        ev = E2.row(ctx["sets"], sc, thr, ctx["ood"])
        if base_ev is None:
            base_ev = ev
        ev["gain_vs_phase0"] = F1gain(ctx, sc, thr)
        ev["passes_rule"] = E2.adopt(base_ev, ev) and name != "Phase 0 baseline"
        ev["extra"] = {k: v for k, v in ex.items() if k != "model"}
        ev["threshold"] = thr
        for part in ("dev_clean", "validation", "test", "val_ood"):
            ev.setdefault("fires", {})
        res["rows"][name] = ev
    passing = [n for n, ev in res["rows"].items() if ev["passes_rule"]]
    res["selected"] = max(passing, key=lambda n: (E2.gain_score(base_ev, res["rows"][n]), res["rows"][n]["dev_clean"]["accuracy"])) \
        if passing else "Phase 0 baseline"
    save("1_3_override.json", res)
    print(E2.HEADER)
    for name, ev in res["rows"].items():
        print(E2.line(name, ev), "pass" if ev["passes_rule"] else "", ev["dev_clean"]["accuracy"])
    print("SELECTED:", res["selected"])


def F1gain(ctx, sc, thr) -> dict:
    from core import gain
    return gain(ctx["sets"], (ctx["scores"]["test"], ctx["thr"]), (sc["test"], thr))


def load_grid():
    from core2 import load
    return load("1_1_rule_grid.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["1.1", "1.3"])
    {"1.1": step_1_1, "1.3": step_1_3}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
