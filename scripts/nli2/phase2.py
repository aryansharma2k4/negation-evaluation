"""Phase 2: cheap rule fixes, each alone against the Phase 1 baseline, then together.

    ./venv/bin/python scripts/nli2/phase2.py --step 2.1     # symmetric veto
    ./venv/bin/python scripts/nli2/phase2.py --step 2.2     # + not + antonym cancellation
    ./venv/bin/python scripts/nli2/phase2.py --step 2.3     # antonym bypass
    ./venv/bin/python scripts/nli2/phase2.py --step 2.4     # length-aware gate
    ./venv/bin/python scripts/nli2/phase2.py --step all     # every fix together

Each writes results/2_<step>.json. Adoption is decided on VALIDATION accuracy (and clean
DEV), never on TEST; TEST is reported for every config.
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import numpy as np

import pipeline as PL
import rules as R
from common import C, RESULTS, log_test_read, save

BASE = PL.Config("Phase 1 baseline (full cascade, clean thresholds)")


def veto_fires(rows, analyses, cfg) -> dict:
    _, st = PL.stages(rows, analyses, cfg)
    fired = st == 2
    y = C.labels(rows)
    return {"fires": int(fired.sum()), "on_true_negations": int((fired & (y == 1)).sum()),
            "examples": [{"a": r["a"], "b": r["b"], "label": int(r["label"]), "kind": r["kind"]}
                         for r, f in zip(rows, fired) if f][:10]}


def fit_length_buckets(dev, nli_thr: float, edges=(6, 12)) -> dict:
    """2.4: t1 per min(len(A), len(B)) bucket, same rule as composite.fit_gate applied within
    each bucket: max DEV J of gate -> NLI subject to the gate keeping >= 95% of the bucket's
    positives. Edges are fixed in advance (<= 6, 7-12, >= 13 words), not tuned."""
    parsed = R.parse([s for r in dev for s in (r["a"], r["b"])])
    m = np.array([min(PL.n_words(parsed[r["a"]]), PL.n_words(parsed[r["b"]])) for r in dev])
    bounds = [(None, edges[0]), (edges[0], edges[1]), (edges[1], None)]
    out, buckets = {}, []
    for lo, hi in bounds:
        mask = (m > (lo or 0)) & (m <= (hi if hi is not None else 10 ** 9))
        rows = [r for r, k in zip(dev, mask) if k]
        fit = C.fit_gate(rows, "cos", nli_thr)
        key = f"{(lo or 0) + 1}-{hi if hi is not None else 'inf'}"
        out[key] = {"n": len(rows), "positives": int(sum(r["label"] for r in rows)), "t1": fit["t1"],
                    "dev_j": fit["dev_j"], "gate_recall": fit["gate_recall"]}
        buckets.append((hi, fit["t1"]))
    return {"buckets": tuple(buckets), "fits": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["2.1", "2.2", "2.3", "2.4", "all"])
    args = ap.parse_args()
    splits, analyses, sl = PL.load()
    base = PL.run(BASE, splits, analyses, sl)
    extra = {}

    if args.step == "2.1":
        configs = [dataclasses.replace(BASE, name="2.1 symmetric veto", veto="symmetric")]
        extra["veto_fires_test"] = {"b_only": veto_fires(sl["test"], analyses, BASE),
                                    "symmetric": veto_fires(sl["test"], analyses, configs[0])}
        extra["veto_fires_validation"] = {"b_only": veto_fires(sl["validation"], analyses, BASE)["fires"],
                                          "symmetric": veto_fires(sl["validation"], analyses, configs[0])["fires"]}
    elif args.step == "2.2":
        configs = [dataclasses.replace(BASE, name="2.2 + not-antonym cancellation", not_antonym=True)]
        parsed = R.parse([s for r in sl["semantoneg_phase2"] for s in (r["a"], r["b"])])
        extra["semantoneg_detector_hits"] = [
            {"a": r["a"], "b": r["b"], "hit": R.not_antonym_cancel(parsed[r["a"]], parsed[r["b"]])}
            for r in sl["semantoneg_phase2"]]
        extra["veto_fires_test"] = veto_fires(sl["test"], analyses, configs[0])
        extra["veto_fires_dev_clean"] = {k: veto_fires(sl["dev_clean"], analyses, c)["fires"]
                                         for k, c in (("baseline", BASE), ("not_antonym", configs[0]))}
    elif args.step == "2.3":
        configs = [dataclasses.replace(BASE, name="2.3 antonym bypass", bypass=True)]
        for name in ("dev_clean", "validation", "test"):
            _, st = PL.stages(sl[name], analyses, configs[0])
            y = C.labels(sl[name])
            extra[f"bypass_{name}"] = {"fires": int((st == 0).sum()), "on_negatives": int(((st == 0) & (y == 0)).sum()),
                                       "kinds": dict(zip(*np.unique([r["kind"] + "|" + str(r.get("neg_type"))
                                                                     for r, s in zip(sl[name], st) if s == 0],
                                                                    return_counts=True)))
                                       if (st == 0).any() else {}}
        extra["curated_antonyms"] = sorted(sorted(p) for p in R.CURATED_ANTONYMS)
    elif args.step == "2.4":
        lb = fit_length_buckets(sl["dev_clean"], json.loads((RESULTS / "1_4_refit_gate.json").read_text())
                                ["config"]["nli_threshold"])
        configs = [dataclasses.replace(BASE, name="2.4 length-aware gate", length_buckets=lb["buckets"])]
        extra["length_buckets"] = lb["fits"]
    else:
        lb = json.loads((RESULTS / "2_2.4.json").read_text())["length_buckets"]
        buckets = tuple((None if k.endswith("inf") else int(k.split("-")[1]), v["t1"]) for k, v in lb.items())
        single = {k: json.loads((RESULTS / f"2_{k}.json").read_text()) for k in ("2.1", "2.2", "2.3", "2.4")}
        adopt = {k: v["adopt_by_validation"] for k, v in single.items()}
        allc = PL.Config("all four fixes", veto="symmetric", not_antonym=True, bypass=True, length_buckets=buckets)
        adopted = PL.Config("fixes adopted by validation", veto="symmetric" if adopt["2.1"] else "b_only",
                            not_antonym=adopt["2.2"], bypass=adopt["2.3"],
                            length_buckets=buckets if adopt["2.4"] else ())
        configs = [allc, adopted]
        extra["adopted"] = adopt

    log_test_read(f"phase2 {args.step}", ", ".join(c.name for c in configs))
    results = [PL.run(c, splits, analyses, sl) for c in configs]
    out = {"baseline": PL.strip(base), "configs": [PL.strip(r) for r in results],
           "gains_vs_baseline": [PL.gains(base, r, sl) for r in results], **extra}
    if args.step != "all":
        bv, nv = base["slices"]["validation"]["all"]["accuracy"], results[0]["slices"]["validation"]["all"]["accuracy"]
        bd, nd = base["slices"]["dev_clean"]["all"]["accuracy"], results[0]["slices"]["dev_clean"]["all"]["accuracy"]
        out["adopt_by_validation"] = bool(nv > bv or (nv == bv and nd > bd))
        out["adoption_rule"] = "adopt if VALIDATION accuracy rises (clean DEV accuracy breaks a tie)"
    save(f"2_{args.step}.json", out)
    print(PL.HEADER)
    print(PL.row(BASE.name, base))
    for r, g in zip(results, out["gains_vs_baseline"]):
        print(PL.row(r["config"]["name"], r, g))
    print("all-TEST gains:", [g["all"] for g in out["gains_vs_baseline"]])
    print("stage counts test:", [r["stage_counts"]["test"] for r in [base] + results],
          "thr:", [r["stage3_threshold"] for r in [base] + results])
    print("dev_clean acc:", [r["slices"]["dev_clean"]["all"]["accuracy"] for r in [base] + results])
    if "adopt_by_validation" in out:
        print("adopt:", out["adopt_by_validation"])
    print(json.dumps({k: v for k, v in extra.items()}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
