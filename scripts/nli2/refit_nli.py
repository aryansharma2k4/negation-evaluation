"""1.3 Refit the NLI threshold on the clean DEV set; how much of the gain survives?

    ./venv/bin/python scripts/nli2/refit_nli.py

Old threshold: reports/06-composite-nli.md's 0.00125 (Youden's J on the generator-heavy DEV).
New threshold: Youden's J on datasets/splits/dev_clean.csv (no generator output).
The gate is held at reports/06-composite-nli.md's t1 = 0.36 here; step 1.4 refits it.
"""

from __future__ import annotations

import json

import numpy as np

from common import C, REPO, compare, evaluate, load_clean, log_test_read, save, youden_threshold

OLD = json.loads((REPO / "reports/results/composite/results_composite.json").read_text())["config"]


def nli_gate(rows, t1):
    return C.cascade(rows, None, "cos", t1)[0]


def main() -> None:
    splits, _ = load_clean()
    dev, val, test = splits["dev"], splits["validation"], splits["test"]
    old_thr, t1 = OLD["nli_threshold"], OLD["t1"]
    new_thr = youden_threshold(dev, [r["nli_c"] for r in dev])
    nli = lambda rows: np.array([r["nli_c"] for r in rows])
    configs = {"nli": nli, "nli_gate": lambda rows: nli_gate(rows, t1)}
    log_test_read("1.3", "NLI alone and NLI+gate(t1=0.36) at old and new NLI thresholds")
    res = {"old_threshold": old_thr, "new_threshold": new_thr, "t1_held": t1, "rows": {}, "gains": {}}
    for name, fn in configs.items():
        for tag, thr in (("old", old_thr), ("new", new_thr)):
            res["rows"][f"{name}|{tag}"] = {s: evaluate(rows, fn(rows), thr)
                                            for s, rows in (("dev_clean", dev), ("validation", val), ("test", test))}
    for tag, thr in (("old", old_thr), ("new", new_thr)):
        for part in ("all", "cannot_phase2", "xnot360"):
            res["gains"][f"gate_over_nli|{tag}|{part}"] = compare(test, (nli(test), thr), (nli_gate(test, t1), thr), part)
    for part in ("all", "cannot_phase2"):
        res["gains"][f"nli_new_over_old|{part}"] = compare(test, (nli(test), old_thr), (nli(test), new_thr), part)
    save("1_3_refit_nli.json", res)

    def row(name, tag):
        t = res["rows"][f"{name}|{tag}"]["test"]
        f = t["fpr_by_kind"]
        return (f"| {name} | {tag} ({res[tag + '_threshold']:.5f}) | {100*t['cannot_phase2']['accuracy']:.1f}% "
                f"({100*t['cannot_phase2']['ci95'][0]:.1f}–{100*t['cannot_phase2']['ci95'][1]:.1f}) | "
                f"{100*t['all']['accuracy']:.1f}% ({100*t['all']['ci95'][0]:.1f}–{100*t['all']['ci95'][1]:.1f}) | "
                f"{t['all']['auc']:.3f} | {100*t['all']['recall']:.1f}% | "
                + " | ".join(f"{100*f[k]['fpr']:.1f}%" for k in ("paraphrase", "SemAntoNeg", "sts_low", "snli_neutral",
                                                                 "random_pair", "xnot360_not_negation"))
                + f" | {100*t['recall_by_type']['antonym']['recall']:.1f}% |")
    print("| config | NLI threshold | phase-2 acc (95% CI) | all-TEST acc (95% CI) | AUC | recall | "
          "FPR paraphrase | SemAntoNeg (n=27) | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall |")
    print("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in configs:
        for tag in ("old", "new"):
            print(row(name, tag))
    print(json.dumps(res["gains"], indent=1))
    print("DEV clean / VAL acc:", {k: (v["dev_clean"]["all"]["accuracy"], v["validation"]["all"]["accuracy"])
                                   for k, v in res["rows"].items()})


if __name__ == "__main__":
    main()
