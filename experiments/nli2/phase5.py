"""Phase 5: fine-tune a small cross-encoder on generator output ("is B the negation of A").

    ./venv/bin/python experiments/nli2/phase5.py --step data    # 5.1 regenerate training pairs (fixed generator)
    ./venv/bin/python experiments/nli2/phase5.py --step train   # 5.1/5.2 fine-tune, base-held-out check
    ./venv/bin/python experiments/nli2/phase5.py --step eval    # 5.3 alone / in cascade / off-the-shelf, TEST

Bases: the CANNOT phase-1 A sentences of the clean DEV set. None is in VALIDATION or TEST, so the
generator never sees a TEST base sentence. (negation_detection's refsets were generated from
phase-2 and xNot360 sentences too, and from the pre-0.1 affixal generator, so they are not used.)

Labels: the refsets rule (experiments/negation_detection/refsets.py): a variant is a negation
iff the parity of its net_negation differs from A's own cue parity. K1 cancellations and K2
compounds are label 0 by construction. Random re-pairings of two different DEV bases are added
as label 0, so "B differs from A" alone cannot be the learned rule.

Base sentences, not pairs, are held out: bases are split 90 / 10 by seed 0, and every pair of a
held-out base (variants and random pairs whose A is that base) is held out with it.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter

import numpy as np

from common import CACHE, REPO, RESULTS, SEED, save
from common import read_dev_clean

TRAIN_JSONL = CACHE / "ft_pairs.jsonl"
MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
MODEL_DIR = CACHE / "ft_model"
MAX_VARIANTS_PER_BASE = 8          # CPU budget: a few variants per base, families sampled evenly
BATCH = 200


def bases() -> list[str]:
    return sorted({r["a"] for r in read_dev_clean() if r["dataset"] == "cannot_phase1"})


def step_data() -> None:
    from negation import generate_all
    src = bases()
    rng = random.Random(SEED)
    pairs, stats = [], Counter()
    start = time.time()
    by_base: dict[str, list] = {}
    for i in range(0, len(src), BATCH):
        chunk = src[i:i + BATCH]
        try:
            records = generate_all(chunk)
        except Exception:
            records = []
            for s in chunk:
                try:
                    records.extend(generate_all([s]))
                except Exception:
                    stats["generator_error"] += 1
        for r in records:
            by_base.setdefault(r.base_sentence, []).append(r)
        print(f"  generated {min(i + BATCH, len(src))}/{len(src)} bases, {time.time() - start:.0f}s", flush=True)
    for a in src:
        rows = by_base.get(a, [])
        stats["variants_generated"] += len(rows)
        if not rows:
            stats["bases_without_variants"] += 1
            continue
        # sample evenly across families so C/H/I (most of the generator's output) do not dominate
        fams: dict[str, list] = {}
        for r in rows:
            fams.setdefault(r.family, []).append(r)
        for f in fams.values():
            rng.shuffle(f)
        picked = []
        while len(picked) < MAX_VARIANTS_PER_BASE and any(fams.values()):
            for f in sorted(fams):
                if fams[f] and len(picked) < MAX_VARIANTS_PER_BASE:
                    picked.append(fams[f].pop())
        for r in picked:
            label = int((r.net_negation % 2) != (r.input_cue_count % 2))
            pairs.append({"a": a, "b": r.variant, "label": label, "family": r.family, "subtype": r.subtype,
                          "source": "generator"})
    n_pos = sum(p["label"] for p in pairs)
    n_rand = max(0, n_pos - (len(pairs) - n_pos))            # top negatives up to a 50 / 50 balance
    for _ in range(n_rand):
        a, b = rng.sample(src, 2)
        pairs.append({"a": a, "b": b, "label": 0, "family": "random_pair", "subtype": "random_pair",
                      "source": "random"})
    rng.shuffle(pairs)
    CACHE.mkdir(exist_ok=True)
    TRAIN_JSONL.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
    comp = Counter(f"{p['family']}|{p['label']}" for p in pairs)
    res = {"bases": len(src), "pairs": len(pairs), "labels": dict(Counter(str(p["label"]) for p in pairs)),
           "by_family_label": dict(sorted(comp.items())), "stats": dict(stats),
           "max_variants_per_base": MAX_VARIANTS_PER_BASE, "generator_commit_note": "after fix 0.1"}
    save("5_1_ft_data.json", res)
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["data", "train", "eval"])
    args = ap.parse_args()
    {"data": step_data}[args.step]()


if __name__ == "__main__":
    main()
