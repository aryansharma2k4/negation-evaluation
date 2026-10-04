"""Phase 3: conditionals.

    ./venv/bin/python experiments/nli2/phase3.py --step 3.1   # how often each split has conditionals
    ./venv/bin/python experiments/nli2/phase3.py --step 3.2   # cascade + conditional rule, measured

3.1 reads no labels. 3.2/3.3 add the conditional decision in front of the Phase 2 config.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter

import numpy as np

import conditionals as K
import pipeline as PL
from common import C, RESULTS, compare, evaluate, log_test_read, save


def coverage(sl) -> dict:
    out = {}
    for name in ("dev_clean", "validation", "test"):
        rows = sl[name]
        conds = K.conditionals([s for r in rows for s in (r["a"], r["b"])])
        by = Counter()
        for r in rows:
            ca, cb = K.primary(conds[r["a"]]), K.primary(conds[r["b"]])
            key = "both" if ca and cb else "A only" if ca else "B only" if cb else "neither"
            by[(r["dataset"], key)] += 1
        out[name] = {f"{d}|{k}": v for (d, k), v in sorted(by.items())}
        out[name + "_markers"] = dict(Counter(c.marker for s in conds for c in conds[s]))
        out[name + "_nested_sentences"] = sum(any(c.nested for c in v) for v in conds.values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["3.1", "3.2"])
    args = ap.parse_args()
    splits, analyses, sl = PL.load()
    if args.step == "3.1":
        res = coverage(sl)
        save("3_1_conditional_coverage.json", res)
        print(json.dumps(res, indent=1))
        return

    p2 = json.loads((RESULTS / "2_all.json").read_text())["configs"][1]["config"]
    base_cfg = PL.Config(**{**p2, "name": "Phase 2 adopted config",
                            "length_buckets": tuple(tuple(b) for b in p2["length_buckets"])})
    cond_cfg = dataclasses.replace(base_cfg, name="+ conditional rule (3.2)", conditionals=True)
    log_test_read("3.2", "Phase 2 adopted config with and without the conditional rule")
    base, new = PL.run(base_cfg, splits, analyses, sl), PL.run(cond_cfg, splits, analyses, sl)
    test = sl["test"]
    _, st_new = PL.stages(test, analyses, cond_cfg)
    conds = K.conditionals([s for r in test for s in (r["a"], r["b"])])
    verdicts = [K.decide(K.primary(conds[r["a"]]), K.primary(conds[r["b"]]), analyses[r["a"]], analyses[r["b"]])
                for r in test]
    both = np.array([bool(K.primary(conds[r["a"]]) and K.primary(conds[r["b"]])) for r in test])
    slices = {"xnot360": [i for i, r in enumerate(test) if r["dataset"] == "xnot360"],
              "xnot360_both_conditional": [i for i, r in enumerate(test) if r["dataset"] == "xnot360" and both[i]],
              "xnot360_not_both_conditional": [i for i, r in enumerate(test) if r["dataset"] == "xnot360" and not both[i]],
              "all_conditional_pairs": [i for i in range(len(test)) if both[i]],
              "non_conditional_pairs": [i for i in range(len(test)) if not both[i]],
              "cannot_phase2": [i for i, r in enumerate(test) if r["dataset"] == "cannot_phase2"]}
    for v in ("negation", "contraposition", "not_negation"):
        slices[f"branch_{v}"] = [i for i in range(len(test)) if verdicts[i] == v]
    slices["branch_none_but_both_conditional"] = [i for i in range(len(test)) if verdicts[i] is None and both[i]]
    y = C.labels(test)
    res = {"base": PL.strip(base), "with_conditionals": PL.strip(new), "slices": {}}
    for name, idx in slices.items():
        idx = np.array(idx, int)
        block = {"n": len(idx), "labels": {str(k): int(v) for k, v in zip(*np.unique(y[idx], return_counts=True))}}
        for tag, r_ in (("before", base), ("after", new)):
            pred = r_["_scores"]["test"][idx] >= r_["stage3_threshold"]
            block[tag] = round(float((pred == y[idx]).mean()), 4) if len(idx) else None
        if len(idx):
            sub = [test[i] for i in idx]
            block["p_permutation"] = compare(sub, (base["_scores"]["test"][idx], base["stage3_threshold"]),
                                             (new["_scores"]["test"][idx], new["stage3_threshold"]))["p_permutation"]
        res["slices"][name] = block
    res["examples"] = {}
    for v in ("negation", "contraposition", "not_negation"):
        res["examples"][v] = [{"a": test[i]["a"], "b": test[i]["b"], "label": int(y[i]), "dataset": test[i]["dataset"]}
                              for i in slices[f"branch_{v}"][:3]]
    res["unless_examples"] = [{"a": r["a"], "b": r["b"], "label": int(r["label"]), "verdict": verdicts[i]}
                              for i, r in enumerate(test) if any(c.marker == "unless" for c in conds[r["a"]] + conds[r["b"]])]
    res["gains"] = PL.gains(base, new, sl)
    res["gain_xnot360"] = compare([test[i] for i in slices["xnot360"]],
                                  (base["_scores"]["test"][slices["xnot360"]], base["stage3_threshold"]),
                                  (new["_scores"]["test"][slices["xnot360"]], new["stage3_threshold"]))
    save("3_2_conditionals.json", res)
    print(PL.HEADER)
    print(PL.row(base_cfg.name, base))
    print(PL.row(cond_cfg.name, new, res["gains"]))
    print("thr", base["stage3_threshold"], new["stage3_threshold"], "stages", new["stage_counts"])
    print(json.dumps(res["slices"], indent=0))
    print("xnot gain", res["gain_xnot360"])
    print(json.dumps(res["examples"], indent=0)[:3000])
    print(json.dumps(res["unless_examples"], indent=0)[:1500])


if __name__ == "__main__":
    main()
