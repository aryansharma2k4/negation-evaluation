"""1.1 / 1.2 Build datasets/splits/dev_clean.csv: a DEV set with no generator output.

    ./venv/bin/python scripts/nli2/build_dev_clean.py            # 1.1 only
    ./venv/bin/python scripts/nli2/build_dev_clean.py --hard     # 1.1 + 1.2 hard negatives

Natural pairs only, built the way CANNOT phase 2 was (scripts/negation_detection/data.py):

    positives  every CANNOT phase-1 negation (1,501)
    negatives  1,501, mirroring phase 2's negative mix:
               751 CANNOT phase-1 paraphrases (all 125 SemAntoNeg + 626 others, sampled)
               250 SNLI neutral (validation), 250 STS-B score < 2 (train + validation +
               unused test), 250 random re-pairings of STS-B / SNLI sentences
    --hard     + topic-matched hard negatives (step 1.2, see hard_negatives())

No sentence in VALIDATION or TEST may appear (normalised as in phase 2), and xNot360
is excluded entirely: all 360 pairs are in TEST. The CSV is gitignored (STS-B's licence
is unclear, so its text is not redistributed); it is rebuilt deterministically (seed 0)
and its sha256 is recorded in results/1_1_dev_clean.json.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from common import C, REPO, SEED, save

RAW = REPO / "datasets/raw/negation_detection"
OUT = REPO / "datasets" / "splits" / "dev_clean.csv"
FIELDS = ["a", "b", "label", "dataset", "source", "subtype", "neg_type", "note"]
N = {"paraphrase": 751, "snli_neutral": 250, "sts_low": 250, "random_pair": 250}


def norm(text: str) -> str:   # identical to negation_detection/data.py
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(" .")


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def overlap(a: str, b: str) -> float:
    """Jaccard overlap of lower-cased word sets."""
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / max(len(ta | tb), 1)


def load_sources():
    sts = [json.loads(l) for s in ("train", "validation", "test")
           for l in gzip.open(RAW / f"stsb_{s}.jsonl.gz", "rt")]
    snli = pq.read_table(RAW / "snli_validation.parquet").to_pylist()
    return sts, snli


def base_rows(rng, blocked: set[str]) -> list[dict]:
    phase1 = C.pdata.load_cannot("phase1")
    pos = [r for r in phase1 if r["label"] == 1]
    sem = [r for r in phase1 if r["label"] == 0 and r["source"] == "antonym-substitution"]
    para = [r for r in phase1 if r["label"] == 0 and r["source"] != "antonym-substitution"]
    rng.shuffle(para)
    rows = pos + sem + para[:N["paraphrase"] - len(sem)]
    for r in rows:
        blocked |= {norm(r["a"]), norm(r["b"])}
    rows = [{k: r.get(k) for k in FIELDS} | {"dataset": "cannot_phase1"} for r in rows]

    sts, snli = load_sources()

    def fresh(a, b):
        return norm(a) not in blocked and norm(b) not in blocked and norm(a) != norm(b)

    def take(cands, n, make):
        got = 0
        for c in cands:
            if got >= n:
                break
            r = make(c)
            if fresh(r["a"], r["b"]):
                blocked.update({norm(r["a"]), norm(r["b"])})
                rows.append(r); got += 1
        assert got == n, (n, got)

    low = [r for r in sts if r["score"] < 2.0]
    rng.shuffle(low)
    take(low, N["sts_low"], lambda r: {"a": r["sentence1"], "b": r["sentence2"], "label": 0, "dataset": "stsb",
                                       "source": f"stsb:{r['dataset']}", "subtype": "sts_low", "neg_type": None,
                                       "note": f"sts score {r['score']}"})
    neutral = [r for r in snli if r["label"] == 1]
    rng.shuffle(neutral)
    take(neutral, N["snli_neutral"], lambda r: {"a": r["premise"], "b": r["hypothesis"], "label": 0,
                                                "dataset": "snli", "source": "snli:neutral",
                                                "subtype": "snli_neutral", "neg_type": None, "note": None})
    pool_a = [r["sentence1"] for r in sts] + [r["premise"] for r in snli[:4000]]
    pool_b = [r["sentence2"] for r in sts] + [r["hypothesis"] for r in snli[:4000]]
    rng.shuffle(pool_a); rng.shuffle(pool_b)
    take(zip(pool_a, pool_b), N["random_pair"], lambda p: {"a": p[0], "b": p[1], "label": 0, "dataset": "random",
                                                           "source": "random_repair", "subtype": "random_pair",
                                                           "neg_type": None, "note": None})
    return rows


def write(rows: list[dict]) -> str:
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    return hashlib.sha256(OUT.read_bytes()).hexdigest()


def composition(rows) -> dict:
    return dict(sorted(Counter(f"{r['dataset']}|{r['source']}|{r['subtype']}|{r['label']}"
                               for r in rows).items()))


def blocked_sentences() -> set[str]:
    splits, _ = C.load()
    return {norm(s) for name in ("validation", "test") for r in splits[name] for s in (r["a"], r["b"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    blocked = blocked_sentences()
    held_out = set(blocked)
    rows = base_rows(rng, blocked)
    info = {}
    if args.hard:
        from hard_negatives import hard_negatives
        hard, info = hard_negatives(rng, blocked)
        rows += hard
    leaked = {norm(s) for r in rows for s in (r["a"], r["b"])} & held_out
    assert not leaked, f"{len(leaked)} DEV sentences are in VALIDATION/TEST"
    digest = write(rows)
    labels = Counter(r["label"] for r in rows)
    res = {"path": str(OUT.relative_to(REPO)), "sha256": digest, "n": len(rows),
           "labels": {str(k): v for k, v in sorted(labels.items())},
           "positive_share": round(labels[1] / len(rows), 4), "composition": composition(rows),
           "positives_by_neg_type": dict(Counter(r["neg_type"] for r in rows if r["label"] == 1)),
           "sentences_shared_with_validation_or_test": 0, "hard_negatives": info}
    save("1_2_dev_clean.json" if args.hard else "1_1_dev_clean.json", res)
    print(f"n={len(rows)} labels={dict(labels)} positive share={res['positive_share']} sha256={digest[:12]}")
    print("| dataset | source | pair type | label | count |\n| --- | --- | --- | ---: | ---: |")
    for k, n in res["composition"].items():
        print("| " + " | ".join(k.split("|")) + f" | {n} |")


if __name__ == "__main__":
    main()
