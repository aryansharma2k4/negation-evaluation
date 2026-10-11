"""Build the phase-1, phase-2 and external test sets for negation detection.

    ./venv/bin/python scripts/negation_detection/data.py

Sources, licences and download commands are in reports/02a-negation-detection-architectures.md; the raw files live in
``datasets/raw/negation_detection/`` (gitignored, none of it is vendored).

Phase 1 (pattern finding)  1,500 negated + 1,500 paraphrase pairs from CANNOT,
                           stratified by the source dataset each pair came from.
Phase 2 (held-out mixed)   1,000 negated + 500 paraphrase (CANNOT) + 500
                           dissimilar positives (STS-B score < 2, SNLI neutral,
                           random re-pairings), sentence-disjoint from phase 1.
External                   xNot360, all 360, touched once at the very end.

Splitting is by *group*: every CANNOT pair sharing a normalised sentence A goes
to the same phase, and a pair is dropped if either of its sentences was already
used by the other phase. That is stricter than the brief's "deduplicate on A"
and it is what the leakage test checks.
"""

from __future__ import annotations

import csv
import difflib
import glob
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RAW = REPO / "datasets" / "raw" / "negation_detection"
sys.path.insert(0, str(REPO / "scripts" / "negation_direction"))

from experiment import cue_count  # noqa: E402  (the lexicon cue counter)

sys.path.insert(0, str(REPO))
from nltk.corpus import wordnet as wn  # noqa: E402
from negation.nlp_core import _WN_POS, antonyms  # noqa: E402

SEED = 0
PHASE1 = {"negation": 1500, "paraphrase": 1500}
PHASE2 = {"negation": 1000, "paraphrase": 500}
DISSIMILAR = {"sts_low": 167, "snli_neutral": 167, "random_pair": 166}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(" .")


# ---------------------------------------------------------------------------
# CANNOT, with the source dataset recovered per pair
# ---------------------------------------------------------------------------

def source_map() -> dict[str, str]:
    """normalised sentence -> the dataset CANNOT merged it from.

    CANNOT's released .tsv has no provenance column, but the repository ships
    the per-source processed files it was built from, so the source is
    recoverable by exact text match (99.99% of rows; the rest are tagged
    ``unattributed``).
    """
    mapping: dict[str, str] = {}
    for path in sorted(glob.glob(str(RAW / "datasets" / "*" / "processed" / "*.tsv"))):
        name = Path(path).parents[1].name
        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                for value in list(row.values())[:2]:
                    if value:
                        mapping.setdefault(norm(value), name)
    return mapping


def cannot_rows() -> list[dict]:
    sources = source_map()
    rows = []
    with open(RAW / "cannot_v1.1.tsv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            a, b = row["premise"], row["hypothesis"]
            if not a or not b:
                continue
            source = sources.get(norm(a)) or sources.get(norm(b)) or "unattributed"
            label = int(row["label"])
            rows.append({
                "a": a, "b": b, "label": label, "dataset": "cannot", "source": source,
                "subtype": "negation" if label else "paraphrase",
                "neg_type": negation_type(a, b, source) if label else None,
            })
    return rows


def negation_type(a: str, b: str, source: str) -> str:
    """verbal / antonym / other.

    Derived, not annotated: CANNOT carries no such label. A pair whose cue count
    changes is verbal (*"is" -> "is not"*, and the reverse); a pair where a
    replaced word is a WordNet antonym of the one it replaced, or that comes
    from SemAntoNeg, is antonym; the rest are other (*"perpendicular" ->
    "adjacent"*, clause rewrites, world-knowledge flips).
    """
    if source == "antonym-substitution":
        return "antonym"          # SemAntoNeg is antonym substitution by construction
    if cue_count(b) != cue_count(a):
        return "verbal"
    if has_antonym_swap(a, b):
        return "antonym"
    return "other"


def has_antonym_swap(a: str, b: str) -> bool:
    """Do the words that differ between a and b include a WordNet antonym pair?"""
    aw = re.findall(r"[a-z']+", a.lower())
    bw = re.findall(r"[a-z']+", b.lower())
    matcher = difflib.SequenceMatcher(a=aw, b=bw, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        for x in aw[i1:i2]:
            for y in bw[j1:j2]:
                for pos in ("ADJ", "VERB", "ADV", "NOUN"):
                    lx = wn.morphy(x, _WN_POS[pos]) or x
                    ly = wn.morphy(y, _WN_POS[pos]) or y
                    if ly in antonyms(lx, pos):
                        return True
    return False


def quotas(rows: list[dict], label: int, total: int) -> dict[str, int]:
    """Proportional allocation over source datasets, at least 1 each."""
    counts = Counter(r["source"] for r in rows if r["label"] == label)
    alloc = {s: max(1, round(total * n / sum(counts.values()))) for s, n in counts.items()}
    # trim/pad the largest source so the quotas sum to exactly `total`
    biggest = max(alloc, key=alloc.get)
    alloc[biggest] += total - sum(alloc.values())
    return alloc


def assign_phases(rows: list[dict], rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[norm(row["a"])].append(row)
    keys = sorted(groups)
    rng.shuffle(keys)

    want = {
        "phase1": {(s, 1): n for s, n in quotas(rows, 1, PHASE1["negation"]).items()}
                  | {(s, 0): n for s, n in quotas(rows, 0, PHASE1["paraphrase"]).items()},
        "phase2": {(s, 1): n for s, n in quotas(rows, 1, PHASE2["negation"]).items()}
                  | {(s, 0): n for s, n in quotas(rows, 0, PHASE2["paraphrase"]).items()},
    }
    taken = {"phase1": [], "phase2": []}
    seen: dict[str, str] = {}  # normalised sentence -> phase that owns it

    for key in keys:
        for phase in ("phase1", "phase2"):
            need = want[phase]
            usable = [r for r in groups[key]
                      if need.get((r["source"], r["label"]), 0) > 0
                      and seen.get(norm(r["a"]), phase) == phase
                      and seen.get(norm(r["b"]), phase) == phase]
            if not usable:
                continue
            for row in usable:
                need[(row["source"], row["label"])] -= 1
                seen[norm(row["a"])] = phase
                seen[norm(row["b"])] = phase
                taken[phase].append(dict(row, phase=phase))
            break
    return taken["phase1"], taken["phase2"]


# ---------------------------------------------------------------------------
# Dissimilar positives: pairs that are neither negations nor paraphrases
# ---------------------------------------------------------------------------

def dissimilar_pairs(rng: np.random.Generator, blocked: set[str]) -> list[dict]:
    sts = [json.loads(line) for line in gzip.open(RAW / "stsb_test.jsonl.gz", "rt")]
    snli = pq.read_table(RAW / "snli_validation.parquet").to_pylist()

    def fresh(a: str, b: str) -> bool:
        return norm(a) not in blocked and norm(b) not in blocked and norm(a) != norm(b)

    out: list[dict] = []

    low = [r for r in sts if r["score"] < 2.0]
    rng.shuffle(low)
    for row in low:
        if len(out) >= DISSIMILAR["sts_low"]:
            break
        if fresh(row["sentence1"], row["sentence2"]):
            blocked |= {norm(row["sentence1"]), norm(row["sentence2"])}
            out.append({"a": row["sentence1"], "b": row["sentence2"], "label": 0,
                        "dataset": "stsb", "source": f"stsb:{row['dataset']}",
                        "subtype": "sts_low", "neg_type": None,
                        "note": f"sts score {row['score']}"})

    neutral = [r for r in snli if r["label"] == 1]
    rng.shuffle(neutral)
    n0 = len(out)
    for row in neutral:
        if len(out) - n0 >= DISSIMILAR["snli_neutral"]:
            break
        if fresh(row["premise"], row["hypothesis"]):
            blocked |= {norm(row["premise"]), norm(row["hypothesis"])}
            out.append({"a": row["premise"], "b": row["hypothesis"], "label": 0,
                        "dataset": "snli", "source": "snli:neutral",
                        "subtype": "snli_neutral", "neg_type": None})

    # random re-pairings: sentences from different pairs of different corpora
    pool_a = [r["sentence1"] for r in sts] + [r["premise"] for r in snli[:4000]]
    pool_b = [r["sentence2"] for r in sts] + [r["hypothesis"] for r in snli[:4000]]
    rng.shuffle(pool_a)
    rng.shuffle(pool_b)
    n0 = len(out)
    for a, b in zip(pool_a, pool_b):
        if len(out) - n0 >= DISSIMILAR["random_pair"]:
            break
        if fresh(a, b):
            blocked |= {norm(a), norm(b)}
            out.append({"a": a, "b": b, "label": 0, "dataset": "random",
                        "source": "random_repair", "subtype": "random_pair",
                        "neg_type": None})
    return [dict(r, phase="phase2") for r in out]


def xnot360_pairs() -> list[dict]:
    rows = pq.read_table(RAW / "xnot360.parquet").to_pylist()
    return [{"a": r["sentence1"], "b": r["sentence2"], "label": int(r["label"]),
             "dataset": "xnot360", "source": "xnot360",
             "subtype": "negation" if r["label"] else "not_negation",
             "neg_type": None, "phase": "external"} for r in rows]


def write(rows: list[dict], path: Path) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"{len(rows):>5}  {path.name}  "
          f"{dict(Counter((r['subtype'], r['label']) for r in rows))}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = cannot_rows()
    print(f"CANNOT: {len(rows)} pairs, sources "
          f"{dict(Counter(r['source'] for r in rows))}")
    phase1, phase2 = assign_phases(rows, rng)

    blocked = {norm(r[k]) for r in phase1 + phase2 for k in ("a", "b")}
    phase2 += dissimilar_pairs(rng, blocked)

    write(phase1, REPO / "datasets" / "splits" / "phase1.jsonl")
    write(phase2, REPO / "datasets" / "splits" / "phase2.jsonl")
    write(xnot360_pairs(), REPO / "datasets" / "splits" / "xnot360.jsonl")

    overlap = ({norm(r[k]) for r in phase1 for k in ("a", "b")}
               & {norm(r[k]) for r in phase2 for k in ("a", "b")})
    print(f"sentence overlap between phases: {len(overlap)}")
    print("phase1 by source:", dict(Counter((r["source"], r["label"]) for r in phase1)))
    print("phase1 neg_type:", dict(Counter(r["neg_type"] for r in phase1 if r["label"])))


if __name__ == "__main__":
    main()
