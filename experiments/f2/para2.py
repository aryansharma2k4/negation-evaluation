"""Step 1.4: report_f1.md's paragraph case study re-run with the adopted configuration.

    ./venv/bin/python experiments/f2/para2.py     -> results/1_4_paragraphs.json

Same matching as report_f1.md step 4.5 (optimal assignment, cosine cells, tau 0.661 fit on clean DEV).
The 28 items are machine-authored: these numbers are a case study of the sentence-matching pipeline, not
evidence for any detection claim (that rests on the human-written slices of step 0.1).
"""

from core2 import save

import paragraphs as PG  # noqa: E402  (f1)
from detector import Detector  # noqa: E402


def main() -> None:
    its = PG.items()
    tau = PG.fit_tau_cached("cos")
    out = {}
    for name, det in (("Phase 0 baseline (report_f1)", Detector(__import__("detector").Config())),
                      ("adopted (step 1.3)", Detector.adopted()), ("NLI alone", PG.NLIOnly())):
        res = PG.run(its, det, "cos", "optimal", tau, True)
        sc = PG.score(its, res)
        out[name] = {k: sc[k] for k in ("teacher_verdict_accuracy", "all_unit_accuracy", "contradiction", "confusion_teacher", "extra")}
        out[name]["stages"] = {}
        for r in res:
            for row in r["stages"]:
                for st in row:
                    out[name]["stages"][st] = out[name]["stages"].get(st, 0) + 1
        print(name, out[name]["teacher_verdict_accuracy"], out[name]["contradiction"], out[name]["confusion_teacher"])
    save("1_4_paragraphs.json", out)


if __name__ == "__main__":
    main()
