"""Summary for reports/08-guardrail-ensemble.md: the adopted configuration, the full ablation, per-family results and
the error analysis, from the committed step results.

    ./venv/bin/python scripts/f1/final_f1.py   -> results/final_f1.json, results/final_config.json

Adopted configuration = the Phase 2 selection, plus the Phase 3 guardrail if Phase 3 selected one
(both by VALIDATION accuracy, rules stated in ensemble.py / guardrail.py). Error analysis: 20 false
positives and 20 false negatives of the adopted configuration on TEST, sampled at random (seed 0),
bucketed by nli2's stated rule (final.bucket), with counts over all errors too.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from core import C, SEED, gain, load, log_test_read, save, system_row

import final as NF  # noqa: E402  (nli2: bucket rule)
import phase0 as P0
import phase4 as P4  # noqa: E402

pc = lambda x: "—" if x is None else f"{100 * x:.1f}%"


def adopted():
    import ensemble as E
    import guardrail as G
    sys_ = E.selected_system()
    g = load("3_4_guardrail.json")
    gp = g["selected_params"] if g["selected"].startswith("guardrail:") else None
    scores, thr = sys_["scores"], sys_["thr"]
    if gp:
        neg, emb = G.context(sys_["sets"])
        m = {k: G.margins(sys_["sets"][k], neg, emb, gp["sim"], gp["agg"]) for k in P0.SETS}
        scores = G.apply(scores, thr, m, gp["delta"])
    return sys_, gp, scores, thr


def ablation() -> list[dict]:
    rows = []
    o = load("0_1_override.json")["rows"]
    rows.append({"label": "Report_NLI2 final (GBM, no override)", "row": o["GBM (Report_NLI2 final)"], "src": "0_1"})
    rows.append({"label": "**Phase 0 baseline** (GBM + cancellation override, fixed conditionals)", "row": load("0_3_baseline.json")["row"], "src": "0_3"})
    rows.append({"label": "+ Phase 1 alone (affirmation generator): no change to the detector's decisions; it feeds the guardrail only",
                 "row": None, "src": "1_2"})
    e = load("2_4_ensembles.json")
    for name, v in e["rows"].items():
        if name != "Phase 0 baseline":
            rows.append({"label": f"+ Phase 2: {name}" + (" ← selected" if name == e["selected"] else ""), "row": v["row"], "src": "2_4"})
    g = load("3_4_guardrail.json")
    if "phase0_plus_selected_guardrail" in g:
        rows.append({"label": "+ Phase 3 alone: Phase 0 baseline + the selected guardrail", "row": g["phase0_plus_selected_guardrail"]["row"], "src": "3_4"})
    for name, v in g["rows"].items():
        if name != "no guardrail":
            rows.append({"label": f"Phase 2 selection + {name}" + (" ← selected" if name == g["selected"] else ""), "row": v["row"], "src": "3_4"})
    return rows


def table(rows) -> list[dict]:
    out = []
    for r in rows:
        t = r["row"]
        if t is None:
            out.append({"configuration": r["label"], "source": r["src"]})
            continue
        f = t["fpr_by_kind"]
        p2 = t["cannot_phase2"]
        out.append({"configuration": r["label"], "phase-2 acc (95% CI)": f"{pc(p2['accuracy'])} ({pc(p2['ci95'][0])}–{pc(p2['ci95'][1])})",
                    "all-TEST acc": pc(t["all"]["accuracy"]), "AUC": f"{t['all']['auc']:.3f}", "precision": pc(t["all"]["precision"]),
                    "recall": pc(t["all"]["recall"]), "F1": pc(t["all"]["f1"]),
                    **{k: pc(f[k]["fpr"]) for k in ("paraphrase", "sts_low", "snli_neutral", "random_pair", "xnot360_not_negation")},
                    "antonym recall (n=95)": pc(t["recall_by_type"]["antonym"]["recall"]), "K1 FPR (n=91)": pc(t["k1_fpr"]),
                    "VAL acc": pc(t["validation"]["accuracy"]), "source": r["src"]})
    return out


def errors(sets, X, scores, thr, k=20) -> dict:
    test = sets["test"]
    y = C.labels(test)
    pred = np.asarray(scores["test"]) >= thr
    rng = random.Random(SEED)
    out = {}
    for kind, mask in (("false_positives", pred & (y == 0)), ("false_negatives", ~pred & (y == 1))):
        allidx = list(np.flatnonzero(mask))
        idx = rng.sample(allidx, min(k, len(allidx)))
        b = lambda i: NF.bucket(test[i], dict(zip(P4.FEATURES, X["test"][i][:17])), bool(pred[i]))
        out[kind] = {"total": len(allidx), "sample_buckets": dict(Counter(b(i) for i in idx).most_common()),
                     "all_buckets": dict(Counter(b(i) for i in allidx).most_common()),
                     "sample": [{"bucket": b(i), "a": test[i]["a"], "b": test[i]["b"], "kind": test[i]["kind"],
                                 "neg_type": test[i].get("neg_type"), "score": round(float(scores["test"][i]), 4)} for i in idx]}
    return out


def main() -> None:
    sys_, gp, scores, thr = adopted()
    sets = sys_["sets"]
    log_test_read("final", "adopted configuration: ablation, per-family, error analysis")
    b = P0.baseline()
    ev = system_row(sets, scores, thr)
    res = {"adopted": {"phase2": sys_["name"], "guardrail": gp, "threshold": thr},
           "row": ev, "gain_vs_phase0": gain(sets, (b["scores"]["test"], b["thr"]), (scores["test"], thr)),
           "per_family": NF.per_family(sets, np.asarray(scores["test"]), thr),
           "errors": errors(sets, sys_["X"], scores, thr)}
    rows = ablation()
    res["ablation"] = rows
    res["ablation_table"] = table(rows)
    res["selection_note"] = ("Phase 2 and Phase 3 configurations selected by VALIDATION accuracy (tie: clean DEV), rules stated "
                             "before each run; TEST never used for selection.")
    save("final_f1.json", res)
    cfg = sys_["cfg"]
    from core import RESULTS
    import json
    (RESULTS / "final_config.json").write_text(json.dumps(
        {"nli_keys": list(cfg.nli_keys), "method": cfg.method, "guardrail": gp, "phase2_selection": sys_["name"],
         "guardrail_note": None if gp else "Phase 3 kept 'no guardrail' by the VALIDATION rule (step 3.5)"}, indent=1))
    print("ADOPTED:", res["adopted"])
    print(json.dumps(res["gain_vs_phase0"], indent=1))
    for r in res["ablation_table"]:
        print(r)
    for k in ("false_positives", "false_negatives"):
        print(k, res["errors"][k]["total"], res["errors"][k]["sample_buckets"], res["errors"][k]["all_buckets"])


if __name__ == "__main__":
    main()
