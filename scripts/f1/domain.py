"""Phase 5.3 / 5.4: the adopted detector on the domain case study (OS / CN / DBMS definitions).

    ./venv/bin/python scripts/f1/domain.py      -> results/5_3_domain.json

Two subsets, never merged: machine-generated answers (datasets/generated/student_answers.jsonl) and the
placeholder "human" subset (datasets/eval/student_answers_human.jsonl, written by the model as carefully as
it could; to be replaced by people). Pair = (teacher definition, student answer); label from the
category. These numbers are kept out of every headline accuracy.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from core import REPO, save
from detector import Detector


def evaluate(items, det) -> dict:
    pred = det.predict([(i["teacher"], i["student"]) for i in items])
    y = np.array([i["label"] for i in items])
    p = np.array([o["negation"] for o in pred])
    stage = [o["stage"] for o in pred]

    def block(idx):
        idx = np.asarray(idx)
        yy, pp = y[idx], p[idx]
        tp, fp, fn = int((pp & (yy == 1)).sum()), int((pp & (yy == 0)).sum()), int((~pp & (yy == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        return {"n": len(idx), "accuracy": round(float((pp == yy).mean()), 4),
                "precision": None if prec is None else round(prec, 4), "recall": None if rec is None else round(rec, 4),
                "called_negation": round(float(pp.mean()), 4)}

    out = {"all": block(range(len(items))), "by_category": {}, "by_subject": {}, "by_category_subject": {},
           "stage_counts": dict(zip(*np.unique(stage, return_counts=True)))}
    groups = defaultdict(list)
    for k, it in enumerate(items):
        groups[("cat", it["category"])].append(k)
        groups[("subj", it["subject"])].append(k)
        groups[("both", f"{it['category']}|{it['subject']}")].append(k)
    for (kind, name), idx in sorted(groups.items()):
        out[{"cat": "by_category", "subj": "by_subject", "both": "by_category_subject"}[kind]][name] = block(idx)
    core5 = [k for k, it in enumerate(items) if not it.get("extra_category")]
    out["five_brief_categories"] = block(core5)
    out["errors"] = [{"category": it["category"], "subject": it["subject"], "teacher": it["teacher"], "student": it["student"],
                      "label": it["label"], "prob": round(o["prob"], 3), "stage": o["stage"]}
                     for it, o, yy, pp in zip(items, pred, y, p) if yy != pp]
    out["stage_counts"] = {str(k): int(v) for k, v in out["stage_counts"].items()}
    return out


def main() -> None:
    det = Detector.adopted()
    load = lambda path: [json.loads(l) for l in path.open() if l.strip()]
    res = {"machine": evaluate(load(REPO / "datasets" / "generated" / "student_answers.jsonl"), det),
           "human_placeholder": evaluate(load(REPO / "datasets" / "eval" / "student_answers_human.jsonl"), det)}
    m, h = res["machine"]["five_brief_categories"], res["human_placeholder"]["five_brief_categories"]
    res["gap_machine_minus_human_pts"] = round(100 * (m["accuracy"] - h["accuracy"]), 1)
    save("5_3_domain.json", res)
    for sub, r in res.items():
        if not isinstance(r, dict):
            print(sub, r); continue
        print("==", sub, r["all"], r["five_brief_categories"], r["stage_counts"])
        for k, v in r["by_category"].items():
            print("  ", k, v)
        for k, v in r["by_subject"].items():
            print("  ", k, v)


if __name__ == "__main__":
    main()
