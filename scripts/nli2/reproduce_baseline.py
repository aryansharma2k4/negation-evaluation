"""0.2 Re-run the reports/06-composite-nli.md system exactly and check its headline numbers.

    ./venv/bin/python scripts/nli2/reproduce_baseline.py

Same rows, same DEV fits (run_composite.configure), same scorers. Does not
touch scripts/composite's results or test_scorings.json.
"""

from __future__ import annotations

import json

import numpy as np

from common import C, REPO, bootstrap_ci, log_test_read, metrics, save

import run_composite as RC  # noqa: E402

EXPECTED = {("cannot_phase2", "nli"): 0.788, ("cannot_phase2", "nli_gate"): 0.926,
            ("all", "nli"): 0.762, ("all", "nli_gate"): 0.872}


def main() -> None:
    splits, analyses = C.load()
    cfg = RC.configure(splits, analyses)
    old = json.loads((REPO / "reports/results/composite/results_composite.json").read_text())["config"]
    same_cfg = {k: (cfg[k], old[k]) for k in ("nli_threshold", "sim", "t1", "nli_gate_threshold", "full_threshold")}
    test = splits["test"]
    log_test_read("0.2", "reproduce Report_NLI.md rows nli, nli_gate, full on TEST")
    out = {"config_now_vs_report": same_cfg, "rows": {}}
    for name, fn in RC.scorers(cfg, analyses).items():
        score, thr, _ = fn(test)
        for part in ("all", "cannot_phase2", "xnot360"):
            idx = [i for i, r in enumerate(test) if part == "all" or r["dataset"] == part]
            y = C.labels([test[i] for i in idx])
            m = metrics(y, score[idx], thr)
            m["ci95"] = bootstrap_ci((score[idx] >= thr) == y)
            out["rows"][f"{name}|{part}"] = m
    out["checks"] = {f"{p}|{r}": {"expected": e, "got": out["rows"][f"{r}|{p}"]["accuracy"],
                                  "match": abs(out["rows"][f"{r}|{p}"]["accuracy"] - e) < 0.0006}
                     for (p, r), e in EXPECTED.items()}
    save("0_2_reproduce.json", out)
    print(json.dumps(out["checks"], indent=1))
    print(json.dumps(same_cfg))
    print("| config | part | n | accuracy (95% CI) | AUC |\n| --- | --- | ---: | --- | ---: |")
    for k, m in out["rows"].items():
        name, part = k.split("|")
        if name in ("nli", "nli_gate", "full"):
            print(f"| {name} | {part} | {m['n']} | {100*m['accuracy']:.1f}% ({100*m['ci95'][0]:.1f}–{100*m['ci95'][1]:.1f}) | {m['auc']:.3f} |")


if __name__ == "__main__":
    main()
