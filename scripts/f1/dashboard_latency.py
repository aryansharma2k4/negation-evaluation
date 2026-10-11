"""Phase 6.1: warm per-pair latency of what the dashboard's single-pair tab computes.

    ./venv/bin/python scripts/f1/dashboard_latency.py   -> results/6_1_latency.json

Pairs carry a random token, so none is in any cache. Warm-up: one pair through every panel (loads the NLI
model, spaCy, mpnet, the generator, the fitted combiner). Then 20 unseen
(teacher, student) pairs, one at a time, each through Detector.predict(explain=True) plus the
generated-negation panel and the affirmation panel, i.e. everything the tab shows. CPU only.
"""

from __future__ import annotations

import time

import numpy as np

from core import save
from detector import Detector, conditional, scopes
from negation.generators.n_affirmation import affirm
import variants as V

_RUN = __import__("secrets").token_hex(3)       # fresh every run, so no pair is in any cache
PAIRS = [(f"The scheduler {v} process {_RUN}{i}.", f"The scheduler does not {v2} process {_RUN}{i}.") for i, (v, v2) in
         enumerate([("runs", "run"), ("stops", "stop"), ("selects", "select"), ("blocks", "block"), ("wakes", "wake")] * 4)]


def panel(det, t, s) -> None:
    det.predict([(t, s)], explain=True)
    scopes(t); scopes(s); conditional(t); conditional(s)
    neg = V.negations([t])[t]
    V.embed([t, s] + [v for v, _ in neg])
    affirm(t)


def main() -> None:
    det = Detector.adopted()
    panel(det, f"Warm-up {_RUN} is not one.", f"Warm-up {_RUN} is two.")      # every panel, once
    times = []
    for t, s in PAIRS:
        t0 = time.perf_counter()
        panel(det, t, s)
        times.append(1000 * (time.perf_counter() - t0))
    res = {"pairs": len(times), "median_ms": round(float(np.median(times)), 1), "p90_ms": round(float(np.quantile(times, 0.9)), 1),
           "max_ms": round(float(max(times)), 1), "setup": "CPU (Ryzen 7 7730U), warm, one pair at a time, all panels of tab 1"}
    save("6_1_latency.json", res)
    print(res)


if __name__ == "__main__":
    main()
