"""Shared plumbing for experiments/f1 (report_f1.md): the nli2 machinery, one place to log TEST
reads, and the evaluation every configuration in report_f1.md gets.

Everything is reused from experiments/nli2 (rows, features, GBM fitting, cascade rules); this file
only adds the f1 results directory and the ablation-row evaluation (precision / recall / F1 on top
of nli2's breakdowns).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
CACHE = HERE / "cache"
SEED = 0
sys.path.insert(0, str(REPO / "experiments" / "nli2"))

import common as N  # noqa: E402
from common import C  # noqa: E402

TEST_LOG = RESULTS / "test_reads.json"


def log_test_read(step: str, what: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    log = json.loads(TEST_LOG.read_text()) if TEST_LOG.exists() else []
    log.append({"step": step, "what": what, "time": time.strftime("%Y-%m-%d %H:%M")})
    TEST_LOG.write_text(json.dumps(log, indent=2))


def save(name: str, obj) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=2, default=float))


def load(name: str):
    return json.loads((RESULTS / name).read_text())


STOPPED = -1.0   # score of a pair decided "not a negation" before (or after) the combiner


def system_row(sets: dict, scores: dict, thr: float) -> dict:
    """Every ablation column for one system. ``scores`` maps set name -> score array (same order
    as ``sets``); a pair is called a negation iff score >= thr."""
    test, val, dev = sets["test"], sets["validation"], sets["dev_clean"]
    ev = N.evaluate(test, scores["test"], thr)
    ev["validation"] = N.evaluate(val, scores["validation"], thr, parts=("all",))["all"]
    ev["dev_clean"] = N.evaluate(dev, scores["dev_clean"], thr, parts=("all",))["all"]
    ev["k1_fpr"] = round(float((np.asarray(scores["k1"]) >= thr).mean()), 4)
    ev["k1_n"] = len(sets["k1"])
    sem = np.concatenate([np.asarray(scores["validation"])[[i for i, r in enumerate(val) if r["kind"] == "SemAntoNeg"]],
                          np.asarray(scores["test"])[[i for i, r in enumerate(test) if r["kind"] == "SemAntoNeg"]]])
    ev["semantoneg_phase2_fpr"] = round(float((sem >= thr).mean()), 4)
    return ev


def gain(sets, base: tuple, new: tuple) -> dict:
    """Accuracy gain of new over base on TEST parts, paired permutation p (uncorrected)."""
    return {p: N.compare(sets["test"], base, new, p) for p in ("all", "cannot_phase2", "xnot360")}


pc = lambda x: "—" if x is None else f"{100 * x:.1f}%"

ABLATION_HEADER = (
    "| configuration | phase-2 acc (95% CI) | all-TEST acc | AUC | precision | recall | F1 | FPR paraphrase | "
    "sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | VAL acc |\n"
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")


def ablation_line(label: str, ev: dict) -> str:
    t, f = ev, ev["fpr_by_kind"]
    p2 = t["cannot_phase2"]
    return (f"| {label} | {pc(p2['accuracy'])} ({pc(p2['ci95'][0])}–{pc(p2['ci95'][1])}) | {pc(t['all']['accuracy'])} | "
            f"{t['all']['auc']:.3f} | {pc(t['all']['precision'])} | {pc(t['all']['recall'])} | {pc(t['all']['f1'])} | "
            + " | ".join(pc(f[k]["fpr"]) for k in ("paraphrase", "sts_low", "snli_neutral", "random_pair",
                                                   "xnot360_not_negation"))
            + f" | {pc(t['recall_by_type']['antonym']['recall'])} | {pc(ev['k1_fpr'])} | {pc(ev['validation']['accuracy'])} |")
