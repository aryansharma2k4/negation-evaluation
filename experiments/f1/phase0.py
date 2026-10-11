"""Phase 0: carry-over fixes on top of Report_NLI2.md's final configuration (GBM, threshold 0.428).

    ./venv/bin/python experiments/f1/phase0.py --step 0.1   # cancellation veto as a hard override before the GBM
    ./venv/bin/python experiments/f1/phase0.py --step 0.2   # conditional detection fix: coverage recount
    ./venv/bin/python experiments/f1/phase0.py --step 0.3   # the Phase 0 baseline every later phase is measured against

Override (0.1): a pair is "not a negation" without consulting the combiner when the cancellation
veto fires in either direction (composite.cancellation_detected(A, B) or (B, A)) or the
not + antonym shape of nli2 step 2.2 fires.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from core import C, STOPPED, ABLATION_HEADER, ablation_line, gain, load, log_test_read, save, system_row

import conditionals as K  # noqa: E402  (nli2)
import phase4 as P4  # noqa: E402
import rules as R  # noqa: E402

SETS = ("dev_clean", "validation", "test", "k1")


def override_mask(rows, analyses) -> np.ndarray:
    parsed = R.parse([s for r in rows for s in (r["a"], r["b"])])
    out = []
    for r in rows:
        aa, ab = analyses[r["a"]], analyses[r["b"]]
        out.append(C.cancellation_detected(aa, ab) or C.cancellation_detected(ab, aa)
                   or R.not_antonym_cancel(parsed[r["a"]], parsed[r["b"]]))
    return np.array(out, bool)


def gbm(sets, X):
    """nli2 step 4.2 exactly: grouped-CV grid, Youden on out-of-fold, refit on all clean DEV."""
    fit = P4.fit_combiners(sets, X)["gbm"]
    return {k: fit["_model"].predict_proba(X[k])[:, 1] for k in SETS}, fit


def with_override(scores: dict, masks: dict) -> dict:
    return {k: np.where(masks[k], STOPPED, scores[k]) for k in scores}


def baseline():
    """The Phase 0 baseline: GBM over the 17 features (conditional features from the current
    conditionals.py) with the cancellation override in front. Returns everything later phases need."""
    sets, analyses, X = P4.table()
    raw, fit = gbm(sets, X)
    masks = {k: override_mask(sets[k], analyses) for k in SETS}
    return {"sets": sets, "analyses": analyses, "X": X, "fit": fit, "raw": raw, "masks": masks,
            "scores": with_override(raw, masks), "thr": fit["threshold"]}


def fires(sets, masks) -> dict:
    out = {}
    for k in SETS:
        y = C.labels(sets[k])
        out[k] = {"n": len(y), "fires": int(masks[k].sum()), "on_true_negations": int((masks[k] & (y == 1)).sum())}
    out["test_fires_on_negations"] = [{"a": r["a"], "b": r["b"], "kind": r["kind"], "neg_type": r.get("neg_type")}
                                      for r, m in zip(sets["test"], masks["test"]) if m and r["label"] == 1]
    out["test_fires_examples"] = [{"a": r["a"], "b": r["b"], "kind": r["kind"], "label": r["label"]}
                                  for r, m in zip(sets["test"], masks["test"]) if m][:12]
    return out


def step_0_1() -> None:
    b = baseline()
    sets, thr = b["sets"], b["thr"]
    log_test_read("0.1", "GBM with and without the cancellation override")
    before, after = system_row(sets, b["raw"], thr), system_row(sets, b["scores"], thr)
    res = {"threshold": thr, "gbm_chosen": b["fit"]["chosen"],
           "rule": "the brief mandates the override; kept unless VALIDATION accuracy drops (then both rows are carried)",
           "rows": {"GBM (Report_NLI2 final)": before, "GBM + cancellation override": after},
           "override_fires": fires(sets, b["masks"]),
           "gain_vs_gbm": gain(sets, (b["raw"]["test"], thr), (b["scores"]["test"], thr))}
    res["adopted"] = after["validation"]["accuracy"] >= before["validation"]["accuracy"]
    save("0_1_override.json", res)
    print(ABLATION_HEADER)
    for name, ev in res["rows"].items():
        print(ablation_line(name, ev))
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows",)}, indent=1)[:5000])


def coverage_class(conds, r) -> str:
    ca, cb = K.primary(conds[r["a"]]), K.primary(conds[r["b"]])
    return "both" if ca and cb else "A only" if ca else "B only" if cb else "neither"


def step_0_2() -> None:
    """Recount nli2 step 3.1's coverage with the fixed detector, against the pre-fix cache."""
    import pickle
    sets, _, _ = P4.table()
    old = pickle.loads((K.CACHE / "conditionals.pkl").read_bytes())       # pre-fix detector output
    res = {}
    for name in ("dev_clean", "validation", "test"):
        rows = sets[name]
        new = K.conditionals([s for r in rows for s in (r["a"], r["b"])])
        changed, counts_old, counts_new = [], {}, {}
        for r in rows:
            co, cn = coverage_class(old, r), coverage_class(new, r)
            key = r["dataset"]
            counts_old[f"{key}|{co}"] = counts_old.get(f"{key}|{co}", 0) + 1
            counts_new[f"{key}|{cn}"] = counts_new.get(f"{key}|{cn}", 0) + 1
            mo = [c.marker for s in (r["a"], r["b"]) for c in old[s]]
            mn = [c.marker for s in (r["a"], r["b"]) for c in new[s]]
            if co != cn or mo != mn:
                changed.append({"a": r["a"], "b": r["b"], "old": co, "new": cn, "old_markers": mo, "new_markers": mn})
        res[name] = {"n": len(rows), "old": dict(sorted(counts_old.items())), "new": dict(sorted(counts_new.items())),
                     "pairs_changing_coverage_class": sum(c["old"] != c["new"] for c in changed),
                     "pairs_changing_marker_only": sum(c["old"] == c["new"] for c in changed), "changed": changed}
    save("0_2_conditional_fix.json", res)
    for name, v in res.items():
        print(name, v["n"], "class changes:", v["pairs_changing_coverage_class"], "marker-only:", v["pairs_changing_marker_only"])
        print("  old", v["old"]); print("  new", v["new"])
        for c in v["changed"][:8]:
            print("   ", c["old"], "->", c["new"], c["old_markers"], "->", c["new_markers"], "|", c["a"][:90], "|", c["b"][:90])


def step_0_3() -> None:
    b = baseline()
    sets, thr = b["sets"], b["thr"]
    log_test_read("0.3", "Phase 0 baseline: GBM (fixed conditionals) + cancellation override")
    ev = system_row(sets, b["scores"], thr)
    old = load("0_1_override.json")
    res = {"threshold": thr, "gbm_chosen": b["fit"]["chosen"], "cv": b["fit"]["cv"],
           "oof_dev_accuracy": b["fit"]["oof_dev_accuracy"], "row": ev,
           "override_fires": fires(sets, b["masks"]),
           "importances": dict(sorted(zip(P4.FEATURES, np.round(b["fit"]["_model"].feature_importances_, 4).tolist()),
                                      key=lambda kv: -kv[1])),
           "vs_0_1_row_phase2_accuracy": old["rows"]["GBM + cancellation override"]["cannot_phase2"]["accuracy"]}
    save("0_3_baseline.json", res)
    print(ABLATION_HEADER)
    print(ablation_line("Phase 0 baseline", ev))
    print(json.dumps({k: v for k, v in res.items() if k != "row"}, indent=1)[:3000])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["0.1", "0.2", "0.3"])
    {"0.1": step_0_1, "0.2": step_0_2, "0.3": step_0_3}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
