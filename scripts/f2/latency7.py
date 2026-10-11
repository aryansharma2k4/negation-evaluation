"""Step 7: dashboard tab-1 latency with the adopted configuration and the severity panel.

    ./venv/bin/python scripts/f2/latency7.py    -> results/7_latency.json
Same method as reports/08-guardrail-ensemble.md step 6.1 (20 unseen pairs with a random token, one full warm-up pair, every panel).
"""
import time

import numpy as np

import core2  # noqa: F401
import dashboard_latency as L  # noqa: E402  (f1)
from detector import Detector  # noqa: E402
from severity import label_pairs


def main() -> None:
    det = Detector.adopted()
    t0, s0 = L.PAIRS[0][0].replace("runs", "idles"), L.PAIRS[0][1].replace("run", "idle")
    L.panel(det, t0, s0); label_pairs([(t0, s0)])
    times = []
    for t, s in L.PAIRS:
        a = time.perf_counter()
        L.panel(det, t, s)
        label_pairs([(t, s)])
        times.append(1000 * (time.perf_counter() - a))
    res = {"pairs": len(times), "median_ms": round(float(np.median(times)), 1), "p90_ms": round(float(np.quantile(times, 0.9)), 1),
           "max_ms": round(float(max(times)), 1), "config": det.cfg.__dict__ if hasattr(det.cfg, "__dict__") else str(det.cfg),
           "setup": "CPU (Ryzen 7 7730U), warm, one pair at a time, every tab-1 panel incl. severity"}
    core2.save("7_latency.json", res)
    print(res)


if __name__ == "__main__":
    main()
