"""Evaluation used for every configuration in report_f2.md, and the step-0.4 baseline row.

    ./venv/bin/python experiments/f2/eval2.py     -> results/0_4_baseline.json

A configuration is a dict of scores per set (dev_clean, validation, test, k1, val_ood) and a threshold;
a pair is a negation iff score >= threshold. Columns added to report_f1.md's:
  val_ood (select half) accuracy, plus its fit half and the whole;
  low-overlap negation recall  = recall on TEST negations with overlap < 0.6 (n = 129, human-written);
  low-overlap non-negation FPR = FPR on TEST non-negations with overlap < 0.4 (n = 396, human-written).
"""

from __future__ import annotations

import numpy as np

from core2 import C, N, STOPPED, baseline, log_test_read, pc, save, system_row

import val_ood as VO

SETS = ("dev_clean", "validation", "test", "k1", "val_ood")


def ood_rows():
    return VO.read()


def row(sets, scores, thr, ood) -> dict:
    ev = system_row(sets, scores, thr)
    b = baseline()
    ov = b["overlap"]["test"]
    y = C.labels(sets["test"])
    pred = np.asarray(scores["test"]) >= thr
    m_neg, m_non = (y == 1) & (ov < 0.6), (y == 0) & (ov < 0.4)
    ev["lowov_neg_recall"] = {"n": int(m_neg.sum()), "recall": round(float(pred[m_neg].mean()), 4)}
    ev["lowov_non_fpr"] = {"n": int(m_non.sum()), "fpr": round(float(pred[m_non].mean()), 4)}
    yo = C.labels(ood)
    po = np.asarray(scores["val_ood"]) >= thr
    for part in ("select", "fit", "all"):
        m = np.array([part == "all" or r["part"] == part for r in ood])
        ev[f"val_ood_{part}"] = {"n": int(m.sum()), "accuracy": round(float((po[m] == yo[m]).mean()), 4),
                                 "recall": round(float(po[m & (yo == 1)].mean()), 4) if (m & (yo == 1)).any() else None,
                                 "fpr": round(float(po[m & (yo == 0)].mean()), 4) if (m & (yo == 0)).any() else None}
    ev["val_ood_by_dataset"] = {}
    for ds in sorted({r["dataset"] for r in ood}):
        m = np.array([r["dataset"] == ds and r["part"] == "select" for r in ood])
        ev["val_ood_by_dataset"][ds] = {"n": int(m.sum()), "accuracy": round(float((po[m] == yo[m]).mean()), 4)}
    return ev


def adopt(base: dict, new: dict) -> bool:
    """The step-0.3 rule: VALIDATION and val_ood (select) both non-decreasing."""
    return (new["validation"]["accuracy"] >= base["validation"]["accuracy"]
            and new["val_ood_select"]["accuracy"] >= base["val_ood_select"]["accuracy"])


def gain_score(base: dict, new: dict) -> float:
    return (new["validation"]["accuracy"] - base["validation"]["accuracy"]) + \
           (new["val_ood_select"]["accuracy"] - base["val_ood_select"]["accuracy"])


HEADER = ("| configuration | phase-2 acc (95% CI) | all-TEST | xNot360 | precision | recall | F1 | FPR paraphrase | sts_low | "
          "snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91, machine-gen.) | "
          "low-ov neg recall (n=129) | low-ov non-neg FPR (n=396) | VAL | val_ood (select, n=383) |\n"
          "| --- | --- |" + " ---: |" * 16)


def line(label: str, ev: dict) -> str:
    f, p2 = ev["fpr_by_kind"], ev["cannot_phase2"]
    return (f"| {label} | {pc(p2['accuracy'])} ({pc(p2['ci95'][0])[:-1]}–{pc(p2['ci95'][1])}) | {pc(ev['all']['accuracy'])} | "
            f"{pc(ev['xnot360']['accuracy'])} | {pc(ev['all']['precision'])} | {pc(ev['all']['recall'])} | {pc(ev['all']['f1'])} | "
            + " | ".join(pc(f[k]["fpr"]) for k in ("paraphrase", "sts_low", "snli_neutral", "random_pair", "xnot360_not_negation"))
            + f" | {pc(ev['recall_by_type']['antonym']['recall'])} | {pc(ev['k1_fpr'])} | {pc(ev['lowov_neg_recall']['recall'])} | "
              f"{pc(ev['lowov_non_fpr']['fpr'])} | {pc(ev['validation']['accuracy'])} | {pc(ev['val_ood_select']['accuracy'])} |")


def baseline_scores():
    """Phase 0 baseline scores on every set, val_ood included."""
    import phase0 as P0
    b = baseline()
    ood = ood_rows()
    X, extra = VO.featurise(ood)
    s = b["fit"]["_model"].predict_proba(X)[:, 1]
    scores = dict(b["scores"])
    scores["val_ood"] = np.where(extra["override"], STOPPED, s)
    sets = dict(b["sets"])
    sets["val_ood"] = ood
    Xs = dict(b["X"])
    Xs["val_ood"] = X
    masks = dict(b["masks"])
    masks["val_ood"] = extra["override"]
    return sets, Xs, scores, masks, ood


def main() -> None:
    sets, X, scores, masks, ood = baseline_scores()
    b = baseline()
    log_test_read("0.4", "Phase 0 baseline on TEST, VALIDATION and val_ood")
    ev = row(sets, scores, b["thr"], ood)
    nli = {k: np.array([r["nli_c"] for r in sets[k]]) for k in SETS}
    ev_nli = row(sets, nli, b["nli_thr"], ood)
    save("0_4_baseline.json", {"phase0": ev, "nli_alone": ev_nli, "threshold": b["thr"], "nli_threshold": b["nli_thr"]})
    print(HEADER)
    print(line("Phase 0 baseline", ev))
    print(line("NLI alone (reference)", ev_nli))
    for k in ("val_ood_select", "val_ood_fit", "val_ood_all"):
        print(k, ev[k], ev_nli[k])


if __name__ == "__main__":
    main()
