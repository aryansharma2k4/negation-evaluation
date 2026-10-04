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


if __name__ == "__main__":
    main()
