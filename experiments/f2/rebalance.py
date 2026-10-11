"""Phase 3: put the negation families to work against the overlap shortcut.

    ./venv/bin/python experiments/f2/rebalance.py --step 3.1   # token_overlap of (A, generated variant) by family
    ./venv/bin/python experiments/f2/rebalance.py --step 3.2   # augmented training sets
    ./venv/bin/python experiments/f2/rebalance.py --step 3.3   # retrain + measure

Bases are clean-DEV A sentences only (asserted disjoint from VALIDATION, TEST and val_ood).
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from core2 import baseline, save

import build_dev_clean as B  # noqa: E402  (nli2: overlap)
import variants as V  # noqa: E402  (f1)


def dev_bases() -> list[str]:
    return list(dict.fromkeys(r["a"] for r in baseline()["sets"]["dev_clean"]))


def step_3_1() -> None:
    neg = V.negations(dev_bases())
    by = defaultdict(list)
    for a, vs in neg.items():
        for v, fam in vs:
            by[fam].append(B.overlap(a, v))
    res = {}
    for fam, ov in sorted(by.items(), key=lambda kv: np.median(kv[1])):
        ov = np.array(ov)
        res[fam] = {"n": len(ov), "mean": round(float(ov.mean()), 3), "median": round(float(np.median(ov)), 3),
                    "share_below_0.4": round(float((ov < 0.4).mean()), 4), "share_below_0.6": round(float((ov < 0.6).mean()), 4)}
    allv = np.concatenate([np.array(v) for v in by.values()])
    res["_all"] = {"n": len(allv), "share_below_0.4": round(float((allv < 0.4).mean()), 4), "share_below_0.6": round(float((allv < 0.6).mean()), 4),
                   "n_below_0.6": int((allv < 0.6).sum()), "n_below_0.4": int((allv < 0.4).sum())}
    save("3_1_family_overlap.json", {"bases": len(neg), "families": res})
    for k, v in res.items():
        print(k, v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["3.1"])
    {"3.1": step_3_1}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
