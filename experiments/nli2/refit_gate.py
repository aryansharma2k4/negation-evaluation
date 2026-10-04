"""1.4 Refit the content-gate threshold on the clean DEV set, same procedure as Report_NLI.md.

    ./venv/bin/python experiments/nli2/refit_gate.py

composite.fit_gate: for each similarity (cos, c_resid), t1 on a 0.01 grid maximising DEV
Youden's J of gate -> NLI, subject to the gate keeping >= 95% of DEV positives; higher J wins.
NLI threshold: step 1.3's. Stage-3 threshold then refit on DEV per cascade, as before.
The result is the Phase 1 baseline every Phase 2 change is measured against.
"""

from __future__ import annotations

import json

import numpy as np

from common import (C, REPO, compare, evaluate, load_clean, log_test_read, save, youden_threshold)

OLD = json.loads((REPO / "experiments/composite/results_composite.json").read_text())["config"]


def phase1_config(splits) -> dict:
    """The clean-protocol thresholds: NLI (1.3), gate (1.4), stage 3 per cascade."""
    dev = splits["dev"]
    nli_thr = youden_threshold(dev, [r["nli_c"] for r in dev])
    fits = {sim: C.fit_gate(dev, sim, nli_thr) for sim in C.SIMILARITIES}
    gate = max(fits.values(), key=lambda g: g["dev_j"])
    return {"nli_threshold": nli_thr, "gate_fits": fits, "sim": gate["sim"], "t1": gate["t1"]}


def main() -> None:
    splits, analyses = load_clean()
    dev, val, test = splits["dev"], splits["validation"], splits["test"]
    cfg = phase1_config(splits)
    sim, t1, nli_thr = cfg["sim"], cfg["t1"], cfg["nli_threshold"]
    nli = lambda rows: np.array([r["nli_c"] for r in rows])
    scorers = {
        "nli": lambda rows: nli(rows),
        "nli_gate_t1_old": lambda rows: C.cascade(rows, analyses, OLD["sim"], OLD["t1"])[0],
        "nli_gate": lambda rows: C.cascade(rows, analyses, sim, t1)[0],
        "full": lambda rows: C.cascade(rows, analyses, sim, t1, use_veto=True)[0],
    }
    thr = {k: (nli_thr if k == "nli" else youden_threshold(dev, fn(dev))) for k, fn in scorers.items()}
    log_test_read("1.4", "NLI, NLI+gate (old and new t1), full, all at clean thresholds")
    res = {"config": cfg, "old": {"sim": OLD["sim"], "t1": OLD["t1"]}, "stage3_thresholds": thr, "rows": {},
           "gate_recall_dev": {}}
    y = C.labels(dev)
    for t in (OLD["t1"], t1):
        res["gate_recall_dev"][str(t)] = round(float(C.gate_pass(dev, sim, t)[y == 1].mean()), 4)
    for k, fn in scorers.items():
        res["rows"][k] = {s: evaluate(rows, fn(rows), thr[k])
                          for s, rows in (("dev_clean", dev), ("validation", val), ("test", test))}
        res["rows"][k]["stages_test"] = {str(a): int(b) for a, b in
                                         zip(*np.unique(C.cascade(test, analyses, sim, t1, k == "full")[1],
                                                        return_counts=True))} if k in ("nli_gate", "full") else None
    res["gains"] = {f"{k}_over_nli|{p}": compare(test, (nli(test), thr["nli"]), (scorers[k](test), thr[k]), p)
                    for k in ("nli_gate_t1_old", "nli_gate", "full") for p in ("all", "cannot_phase2")}
    res["gains"]["nli_gate_new_t1_over_old_t1|cannot_phase2"] = compare(
        test, (scorers["nli_gate_t1_old"](test), thr["nli_gate_t1_old"]), (scorers["nli_gate"](test), thr["nli_gate"]),
        "cannot_phase2")
    save("1_4_refit_gate.json", res)
    print("fits:", {s: (f["t1"], f["dev_j"], f["gate_recall"]) for s, f in cfg["gate_fits"].items()}, "chosen", sim, t1)
    print("stage3 thr:", thr, "gate recall dev:", res["gate_recall_dev"])
    for k, v in res["rows"].items():
        t = v["test"]
        print(k, "p2", t["cannot_phase2"]["accuracy"], t["cannot_phase2"]["ci95"], "all", t["all"]["accuracy"],
              t["all"]["ci95"], "auc", t["all"]["auc"], "rec", t["all"]["recall"],
              {a: b["fpr"] for a, b in t["fpr_by_kind"].items()}, {a: b["recall"] for a, b in t["recall_by_type"].items()},
              "val", v["validation"]["all"]["accuracy"], "dev", v["dev_clean"]["all"]["accuracy"], v["stages_test"])
    print(json.dumps(res["gains"]))


if __name__ == "__main__":
    main()
