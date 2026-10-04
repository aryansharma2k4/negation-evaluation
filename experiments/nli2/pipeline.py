"""The configurable cascade used from Phase 2 on, and the evaluation every config gets.

Stages, in order:
  4/5 conditional rule (3.2)  both sides conditional: antecedent-only flip -> stop (4, not a
                             negation); consequent flip or contraposition -> negation (5)
  0  antonym bypass (2.3)    antonym_aligned(A, B) -> skip gate and veto, go to NLI
  1  content gate            cos(A, B) < t1 (global, or by min(len) bucket, 2.4) -> not a negation
  2  cancellation veto       "b_only" (Report_NLI.md) or "symmetric" (2.1); optionally also the
                             not + antonym shape (2.2) -> not a negation
  3  NLI                     P(contradiction) >= stage-3 threshold (Youden's J on clean DEV, per config)

Slices reported for every config: TEST (all, CANNOT phase 2, xNot360), VALIDATION, clean DEV,
K1 (our 91 generator pairs: no longer in any fitting set, so a held-out slice, generator caveat),
SemAntoNeg pooled over phase 2 (n = 41).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

import conditionals as K
import rules as R
from common import C, compare, evaluate, load_clean, youden_threshold

BASE_T1 = 0.37   # step 1.4


@dataclass(frozen=True)
class Config:
    name: str
    veto: str | None = "b_only"           # None, "b_only", "symmetric"
    not_antonym: bool = False             # 2.2
    bypass: bool = False                  # 2.3
    t1: float = BASE_T1
    length_buckets: tuple = ()            # 2.4: ((max_min_len, t1), ...) ascending; last bucket open
    gate: bool = True
    conditionals: bool = False            # 3.2
    cond_both: str = "negation"           # 3.3 post-hoc: what "both halves flipped" maps to
    cond_cons: str = "negation"           # 3.3 post-hoc: what "consequent flipped" maps to


def n_words(tokens) -> int:
    return sum(t.pos != "PUNCT" for t in tokens)


def gate_threshold(cfg: Config, la: int, lb: int) -> float:
    if not cfg.length_buckets:
        return cfg.t1
    m = min(la, lb)
    for upper, t1 in cfg.length_buckets:
        if upper is None or m <= upper:
            return t1
    raise AssertionError("buckets must end with an open bucket")


def stages(rows, analyses, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """(score, stage): 0 bypassed to NLI, 1 gated, 2 vetoed, 3 NLI, 4 conditional: not a
    negation, 5 conditional: negation (score 1 + P(contradiction), above any threshold)."""
    parsed = R.parse([s for r in rows for s in (r["a"], r["b"])])
    conds = K.conditionals([s for r in rows for s in (r["a"], r["b"])]) if cfg.conditionals else {}
    score = np.empty(len(rows)); stage = np.empty(len(rows), int)
    for i, r in enumerate(rows):
        ta, tb = parsed[r["a"]], parsed[r["b"]]
        st = 3
        verdict = (K.decide(K.primary(conds[r["a"]]), K.primary(conds[r["b"]]), analyses[r["a"]], analyses[r["b"]])
                   if cfg.conditionals else None)
        verdict = {"contraposition": cfg.cond_both, "negation": cfg.cond_cons}.get(verdict, verdict)
        if verdict == "not_negation":
            st = 4
        elif verdict == "negation":
            st = 5
        elif cfg.bypass and R.antonym_aligned(ta, tb):
            st = 0
        elif cfg.gate and r["cos"] < gate_threshold(cfg, n_words(ta), n_words(tb)):
            st = 1
        elif cfg.veto:
            aa, ab = analyses[r["a"]], analyses[r["b"]]
            vetoed = C.cancellation_detected(aa, ab) or (cfg.veto == "symmetric" and C.cancellation_detected(ab, aa))
            if vetoed or (cfg.not_antonym and R.not_antonym_cancel(ta, tb)):
                st = 2
        stage[i] = st
        score[i] = r["nli_c"] if st in (0, 3) else 1.0 + r["nli_c"] if st == 5 else C.STOPPED
    return score, stage


def slices(splits, analyses_old_dev) -> dict[str, list[dict]]:
    """The fitting / evaluation slices. K1 rows come from Report_NLI.md's DEV (ours)."""
    k1 = [dict(r, nli_c=r["nli"]) for r in analyses_old_dev if r["kind"] == "K1"]
    phase2 = splits["validation"] + [r for r in splits["test"] if r["dataset"] == "cannot_phase2"]
    return {"dev_clean": splits["dev"], "validation": splits["validation"], "test": splits["test"],
            "k1": k1, "semantoneg_phase2": [r for r in phase2 if r["kind"] == "SemAntoNeg"]}


def load():
    splits, analyses = load_clean()
    old, _ = C.load()
    return splits, analyses, slices(splits, old["dev"])


def run(cfg: Config, splits, analyses, sl) -> dict:
    dev_score, _ = stages(sl["dev_clean"], analyses, cfg)
    thr = youden_threshold(sl["dev_clean"], dev_score)
    out = {"config": asdict(cfg), "stage3_threshold": float(thr), "slices": {}, "stage_counts": {}}
    scores = {}
    for name, rows in sl.items():
        s, st = stages(rows, analyses, cfg)
        scores[name] = s
        if name in ("dev_clean", "validation", "test"):
            out["slices"][name] = evaluate(rows, s, thr)
        else:   # all-negative slices: FPR only
            pred = s >= thr
            out["slices"][name] = {"n": len(rows), "fpr": round(float(pred.mean()), 4)}
        out["stage_counts"][name] = {str(k): int(v) for k, v in zip(*np.unique(st, return_counts=True))}
    out["_scores"] = scores
    return out


def gains(base: dict, new: dict, sl) -> dict:
    t = sl["test"]
    return {p: compare(t, (base["_scores"]["test"], base["stage3_threshold"]),
                       (new["_scores"]["test"], new["stage3_threshold"]), p) for p in ("all", "cannot_phase2")}


def strip(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


HEADER = ("| config | phase-2 acc (95% CI) | all-TEST acc (95% CI) | AUC | recall | FPR paraphrase | sts_low | "
          "snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | "
          "SemAntoNeg FPR (n=41) | VAL acc | gain vs baseline, phase 2 (p) |\n"
          "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")


def row(label: str, res: dict, gain: dict | None = None) -> str:
    t = res["slices"]["test"]
    f = t["fpr_by_kind"]
    pc = lambda x: f"{100 * x:.1f}%"
    g = "—" if gain is None else f"{100 * gain['cannot_phase2']['gain']:+.1f} pts (p = {gain['cannot_phase2']['p_permutation']})"
    return (f"| {label} | {pc(t['cannot_phase2']['accuracy'])} ({pc(t['cannot_phase2']['ci95'][0])}–"
            f"{pc(t['cannot_phase2']['ci95'][1])}) | {pc(t['all']['accuracy'])} ({pc(t['all']['ci95'][0])}–"
            f"{pc(t['all']['ci95'][1])}) | {t['all']['auc']:.3f} | {pc(t['all']['recall'])} | "
            + " | ".join(pc(f[k]["fpr"]) for k in ("paraphrase", "sts_low", "snli_neutral", "random_pair",
                                                   "xnot360_not_negation"))
            + f" | {pc(t['recall_by_type']['antonym']['recall'])} | {pc(res['slices']['k1']['fpr'])} | "
              f"{pc(res['slices']['semantoneg_phase2']['fpr'])} | {pc(res['slices']['validation']['all']['accuracy'])} | {g} |")
