"""Phase 0.1 / 0.2: does the combiner's advantage over NLI disappear as token overlap falls?

    ./venv/bin/python scripts/f2/overlap.py --step 0.1   -> results/0_1_overlap.json
    ./venv/bin/python scripts/f2/overlap.py --step 0.2   -> results/0_2_mechanism.json, reports/figures/f2/*.png

Combiner = reports/08-guardrail-ensemble.md's adopted configuration (GBM + cancellation override, threshold 0.428).
NLI alone = cross-encoder/nli-deberta-v3-base P(contradiction) >= its clean-DEV Youden threshold (0.975).
token_overlap = word-set Jaccard (the combiner's own feature). On clean DEV the combiner's predictions are
out-of-fold (the final model has seen every DEV pair). Bins: quintiles of the slice's own overlap
distribution, and fixed width 0-0.2, ..., 0.8-1.0. Paired permutation p-values are uncorrected.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from core2 import C, N, REPO, baseline, load, log_test_read, pc, save

import build_dev_clean as B  # noqa: E402  (nli2: the overlap function)

FIXED = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]


def compare(rows, comb_pred, nli_pred, idx) -> dict:
    idx = np.asarray(idx, int)
    if len(idx) == 0:
        return {"n": 0}
    y = C.labels(rows)[idx]
    c, n = comb_pred[idx], nli_pred[idx]
    lab = int(y[0])
    rate = lambda p: round(float(p.mean()), 4)
    out = {"n": len(idx), "label": lab,
           "combiner": rate(c), "nli_alone": rate(n), "combiner_minus_nli": round(float(c.mean() - n.mean()), 4),
           "metric": "recall" if lab == 1 else "FPR",
           "p_permutation": N.paired_permutation_p((n == y).astype(float), (c == y).astype(float)),
           "datasets": dict(Counter(rows[i]["dataset"] for i in idx))}
    return out


def binned(rows, ov, comb, nli, mask, edges=None) -> list[dict]:
    idx = np.flatnonzero(mask)
    if edges is None:
        edges = list(np.quantile(ov[idx], [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        edges[-1] += 1e-9
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = idx[(ov[idx] >= lo) & (ov[idx] < hi)]
        out.append({"bin": f"{lo:.3f}-{min(hi, 1.0):.3f}", **compare(rows, comb, nli, sub)})
    return out


def dist(values) -> dict:
    v = np.asarray(values, float)
    return {"n": len(v), "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
            "q10": round(float(np.quantile(v, 0.1)), 4), "q90": round(float(np.quantile(v, 0.9)), 4),
            "share_below_0.4": round(float((v < 0.4).mean()), 4), "share_below_0.6": round(float((v < 0.6).mean()), 4)}


def step_0_1() -> None:
    b = baseline()
    sets, thr, nthr = b["sets"], b["thr"], b["nli_thr"]
    log_test_read("0.1", "overlap bins: combiner vs NLI alone on TEST negations and non-negations")
    res = {"combiner_threshold": thr, "nli_threshold": nthr, "test": {}, "slices": {}, "overlap_distributions": {}}
    pred = lambda k, s: np.asarray(s) >= thr
    for name in ("test", "validation"):
        rows = sets[name]
        y = C.labels(rows)
        ov = b["overlap"][name]
        comb = np.asarray(b["scores"][name]) >= thr
        nli = np.array([r["nli_c"] for r in rows]) >= nthr
        block = {}
        for lab, tag in ((1, "negations"), (0, "non_negations")):
            m = y == lab
            block[tag] = {"quintile_bins": binned(rows, ov, comb, nli, m),
                          "fixed_bins": binned(rows, ov, comb, nli, m, FIXED),
                          "all": compare(rows, comb, nli, np.flatnonzero(m))}
            for ds in ("cannot_phase2", "xnot360"):
                mm = m & np.array([r["dataset"] == ds for r in rows])
                if mm.any():
                    block[tag][f"fixed_bins_{ds}"] = binned(rows, ov, comb, nli, mm, FIXED)
        res[name] = block
    # named human-written slices
    t, ov_t = sets["test"], b["overlap"]["test"]
    comb_t = np.asarray(b["scores"]["test"]) >= thr
    nli_t = np.array([r["nli_c"] for r in t]) >= nthr
    yt = C.labels(t)
    q = res["test"]["negations"]["quintile_bins"]
    lo2 = float(q[1]["bin"].split("-")[1])
    S = res["slices"]
    S["TEST negations, bottom two overlap quintiles (overlap < %.3f)" % lo2] = compare(t, comb_t, nli_t, np.flatnonzero((yt == 1) & (ov_t < lo2)))
    S["TEST negations, overlap < 0.4"] = compare(t, comb_t, nli_t, np.flatnonzero((yt == 1) & (ov_t < 0.4)))
    S["TEST CANNOT antonym-substitution negations"] = compare(t, comb_t, nli_t, [i for i, r in enumerate(t) if r.get("neg_type") == "antonym" and r["label"] == 1])
    v = sets["validation"]
    ph2 = v + [r for r in t if r["dataset"] == "cannot_phase2"]
    sc_ph2 = np.concatenate([np.asarray(b["scores"]["validation"]), np.asarray(b["scores"]["test"])[[i for i, r in enumerate(t) if r["dataset"] == "cannot_phase2"]]])
    S["SemAntoNeg paraphrases, phase 2 (validation + test)"] = compare(
        ph2, sc_ph2 >= thr, np.array([r["nli_c"] for r in ph2]) >= nthr, [i for i, r in enumerate(ph2) if r["kind"] == "SemAntoNeg"])
    d = sets["dev_clean"]
    ov_d = b["overlap"]["dev_clean"]
    comb_d = np.asarray(b["oof_dev"]) >= thr
    nli_d = np.array([r["nli_c"] for r in d]) >= nthr
    yd = C.labels(d)
    p1 = np.array([r["dataset"] == "cannot_phase1" for r in d])
    S["clean DEV CANNOT antonym-substitution negations (combiner out-of-fold)"] = compare(d, comb_d, nli_d, [i for i, r in enumerate(d) if r.get("neg_type") == "antonym" and r["label"] == 1])
    S["clean DEV CANNOT phase-1 negations, overlap < 0.4 (out-of-fold)"] = compare(d, comb_d, nli_d, np.flatnonzero(p1 & (yd == 1) & (ov_d < 0.4)))
    S["clean DEV CANNOT phase-1 non-negations, overlap < 0.4 (out-of-fold)"] = compare(d, comb_d, nli_d, np.flatnonzero(p1 & (yd == 0) & (ov_d < 0.4)))
    S["clean DEV CANNOT phase-1 negations, overlap >= 0.8 (out-of-fold)"] = compare(d, comb_d, nli_d, np.flatnonzero(p1 & (yd == 1) & (ov_d >= 0.8)))
    res["dev_clean_fixed_bins_negations"] = binned(d, ov_d, comb_d, nli_d, p1 & (yd == 1), FIXED)
    # overlap distributions: benchmarks vs application
    D = res["overlap_distributions"]
    D["CANNOT phase 2 negations (TEST)"] = dist(ov_t[[i for i, r in enumerate(t) if r["dataset"] == "cannot_phase2" and r["label"] == 1]])
    D["CANNOT phase 2 non-negations (TEST)"] = dist(ov_t[[i for i, r in enumerate(t) if r["dataset"] == "cannot_phase2" and r["label"] == 0]])
    D["xNot360 negations (TEST)"] = dist(ov_t[[i for i, r in enumerate(t) if r["dataset"] == "xnot360" and r["label"] == 1]])
    D["xNot360 non-negations (TEST)"] = dist(ov_t[[i for i, r in enumerate(t) if r["dataset"] == "xnot360" and r["label"] == 0]])
    D["clean DEV CANNOT phase-1 negations"] = dist(ov_d[p1 & (yd == 1)])
    its = [json.loads(l) for l in (REPO / "datasets" / "eval" / "paragraph_eval.jsonl").open()]
    for verdict in ("covered", "contradicted"):
        D[f"paragraph set gold {verdict} pairs (machine-authored)"] = dist(
            [B.overlap(it["teacher"][i], it["student"][g["student"]]) for it in its for i, g in enumerate(it["gold_teacher"]) if g["verdict"] == verdict])
    ans = [json.loads(l) for l in (REPO / "datasets" / "generated" / "student_answers.jsonl").open()]
    for cat in sorted({a["category"] for a in ans}):
        D[f"domain answers: {cat} (machine-generated)"] = dist([B.overlap(a["teacher"], a["student"]) for a in ans if a["category"] == cat])
    hum = [json.loads(l) for l in (REPO / "datasets" / "eval" / "student_answers_human.jsonl").open()]
    for cat in sorted({a["category"] for a in hum}):
        D[f"placeholder subset: {cat} (model-written, n=10)"] = dist([B.overlap(a["teacher"], a["student"]) for a in hum if a["category"] == cat])
    save("0_1_overlap.json", res)
    for tag in ("negations", "non_negations"):
        for kind in ("quintile_bins", "fixed_bins", "fixed_bins_cannot_phase2", "fixed_bins_xnot360"):
            print(f"== TEST {tag} {kind}")
            for r in res["test"][tag][kind]:
                if r["n"]:
                    print(f"  {r['bin']:13s} n={r['n']:4d} comb {pc(r['combiner'])} nli {pc(r['nli_alone'])} diff {100*r['combiner_minus_nli']:+.1f} p={r['p_permutation']} {r['datasets']}")
    print("== slices")
    for k, r in S.items():
        print(f"  {k}: n={r['n']} comb {pc(r['combiner'])} nli {pc(r['nli_alone'])} ({r['metric']}) p={r['p_permutation']}")
    print("== DEV fixed bins")
    for r in res["dev_clean_fixed_bins_negations"]:
        if r["n"]:
            print(f"  {r['bin']} n={r['n']} comb {pc(r['combiner'])} nli {pc(r['nli_alone'])} p={r['p_permutation']}")
    print("== distributions")
    for k, v in D.items():
        print(f"  {k}: {v}")


def step_0_2() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import phase4 as P4
    b = baseline()
    sets = b["sets"]
    log_test_read("0.2", "binned P(negation) vs overlap on TEST, label held fixed")
    imp = dict(sorted(zip(P4.FEATURES, np.round(b["fit"]["_model"].feature_importances_, 4).tolist()), key=lambda kv: -kv[1]))
    out = {"importances": imp, "curves": {}}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (name, sc) in zip(axes, (("clean DEV (out-of-fold)", b["oof_dev"]), ("TEST", b["scores"]["test"]))):
        key = "dev_clean" if "DEV" in name else "test"
        rows, ov = sets[key], b["overlap"][key]
        y = C.labels(rows)
        raw = np.clip(np.asarray(sc, float), 0, 1)
        for lab, color, tag in ((1, "#c0392b", "negations"), (0, "#2471a3", "non-negations")):
            pts = []
            for lo, hi in zip(FIXED[:-1], FIXED[1:]):
                m = (y == lab) & (ov >= lo) & (ov < hi)
                if m.sum() >= 5:
                    pts.append({"bin_mid": round((lo + min(hi, 1)) / 2, 2), "n": int(m.sum()), "mean_p": round(float(raw[m].mean()), 4),
                                "nli_mean_p": round(float(np.array([r["nli_c"] for r in rows])[m].mean()), 4)})
            out["curves"][f"{name} {tag}"] = pts
            ax.plot([p["bin_mid"] for p in pts], [p["mean_p"] for p in pts], "o-", color=color, label=f"{tag}: combiner")
            ax.plot([p["bin_mid"] for p in pts], [p["nli_mean_p"] for p in pts], "s--", color=color, alpha=0.5, label=f"{tag}: NLI alone")
            for p in pts:
                ax.annotate(str(p["n"]), (p["bin_mid"], p["mean_p"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(b["thr"], color="grey", lw=0.8, ls=":")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("token_overlap (word-set Jaccard), bin centre")
    axes[0].set_ylabel("mean output (combiner raw / NLI P(contradiction))")
    axes[0].legend(fontsize=7)
    fig.suptitle("Binned mean output vs token_overlap, label held fixed (n per point annotated)", fontsize=10)
    fig.tight_layout()
    path = REPO / "reports" / "figures" / "f2" / "overlap_mechanism.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    # partial dependence of the GBM on token_overlap (clean DEV feature distribution)
    from sklearn.inspection import partial_dependence
    j = P4.FEATURES.index("token_overlap")
    pd_ = partial_dependence(b["fit"]["_model"], b["X"]["dev_clean"], [j], grid_resolution=20, kind="average")
    out["partial_dependence_token_overlap"] = {"grid": np.round(pd_["grid_values"][0], 3).tolist(),
                                               "average": np.round(pd_["average"][0], 4).tolist()}
    save("0_2_mechanism.json", out)
    print(json.dumps(out["importances"], indent=0)[:600])
    for k, v in out["curves"].items():
        print(k, [(p["bin_mid"], p["n"], p["mean_p"], p["nli_mean_p"]) for p in v])
    print(out["partial_dependence_token_overlap"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["0.1", "0.2"])
    {"0.1": step_0_1, "0.2": step_0_2}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
