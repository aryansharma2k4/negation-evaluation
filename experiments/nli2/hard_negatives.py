"""1.2 Topic-matched hard negatives for the clean DEV set: same subject, different claim.

    STS-B   gold similarity 1.5-3.5 and word-set Jaccard overlap >= 0.5, minus any pair whose
            negation-cue counts differ (a cue-level negation must not be labelled 0); 250 sampled
    SNLI    neutral (gold: not a contradiction) with Jaccard overlap >= 0.5; 150 sampled

Only sentences not already in DEV, VALIDATION or TEST.
"""

from __future__ import annotations

import sys

from common import REPO

sys.path.insert(0, str(REPO / "experiments" / "negation_direction"))
from experiment import cue_count  # noqa: E402  (the lexicon cue counter)

import build_dev_clean as B  # noqa: E402

MIN_OVERLAP = 0.5
STS_RANGE = (1.5, 3.5)
TAKE = {"sts_hard": 250, "snli_hard": 150}


def hard_negatives(rng, blocked: set[str]) -> tuple[list[dict], dict]:
    sts, snli = B.load_sources()
    mid = [r for r in sts if STS_RANGE[0] <= r["score"] <= STS_RANGE[1]]
    sts_c = [r for r in mid if B.overlap(r["sentence1"], r["sentence2"]) >= MIN_OVERLAP]
    cue_diff = [r for r in sts_c if cue_count(r["sentence1"]) != cue_count(r["sentence2"])]
    sts_c = [r for r in sts_c if cue_count(r["sentence1"]) == cue_count(r["sentence2"])]
    neu = [r for r in snli if r["label"] == 1]
    snli_c = [r for r in neu if B.overlap(r["premise"], r["hypothesis"]) >= MIN_OVERLAP]

    out = []

    def take(cands, n, make):
        order = rng.permutation(len(cands))
        got = 0
        for i in order:
            if got >= n:
                break
            r = make(cands[i])
            if B.norm(r["a"]) in blocked or B.norm(r["b"]) in blocked or B.norm(r["a"]) == B.norm(r["b"]):
                continue
            blocked.update({B.norm(r["a"]), B.norm(r["b"])})
            out.append(r); got += 1
        return got

    got_sts = take(sts_c, TAKE["sts_hard"], lambda r: {
        "a": r["sentence1"], "b": r["sentence2"], "label": 0, "dataset": "stsb", "source": f"stsb:{r['dataset']}",
        "subtype": "hard_negative", "neg_type": None,
        "note": f"sts score {r['score']}; overlap {B.overlap(r['sentence1'], r['sentence2']):.2f}"})
    got_snli = take(snli_c, TAKE["snli_hard"], lambda r: {
        "a": r["premise"], "b": r["hypothesis"], "label": 0, "dataset": "snli", "source": "snli:neutral",
        "subtype": "hard_negative", "neg_type": None,
        "note": f"overlap {B.overlap(r['premise'], r['hypothesis']):.2f}"})
    info = {"rule": {"min_jaccard_overlap": MIN_OVERLAP, "sts_score_range": STS_RANGE, "take": TAKE},
            "sts_candidates_score_in_range": len(mid),
            "sts_candidates_after_overlap": len(sts_c) + len(cue_diff),
            "sts_dropped_cue_count_differs": len(cue_diff),
            "snli_neutral_total": len(neu), "snli_candidates_after_overlap": len(snli_c),
            "taken": {"sts_hard": got_sts, "snli_hard": got_snli},
            "examples": [{"a": r["a"], "b": r["b"], "note": r["note"]} for r in out[:3] + out[-3:]]}
    return out, info
