"""Shared plumbing for experiments/f2 (report_f2.md). Reuses experiments/f1 and experiments/nli2.

TEST reads are logged to experiments/f2/results/test_reads.json; report_f2.md counts them cumulatively
with nli2 (14), f1 (8) and this round.
"""

from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
CACHE = HERE / "cache"
SEED = 0
sys.path.insert(0, str(REPO / "experiments" / "f1"))
sys.path.insert(0, str(HERE))

import core as F1  # noqa: E402  (f1 core: also puts nli2 on sys.path)
from core import C, N, STOPPED, system_row  # noqa: E402,F401

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


pc = lambda x: "—" if x is None else f"{100 * x:.1f}%"


@lru_cache(maxsize=1)
def baseline() -> dict:
    """report_f1.md's adopted configuration (Phase 0 GBM + cancellation override), with out-of-fold
    clean-DEV predictions from the identical fitting procedure."""
    import ensemble as E
    import phase0 as P0
    import phase4 as P4
    b = P0.baseline()
    g = E.fit_gbm(b["sets"], b["X"]["dev_clean"])
    assert abs(g["threshold"] - b["thr"]) < 1e-12, "fit_gbm must reproduce the Phase 0 model"
    oof = np.where(b["masks"]["dev_clean"], STOPPED, g["oof"])
    j = P4.FEATURES.index("token_overlap")
    return {**b, "oof_dev": oof, "overlap": {k: b["X"][k][:, j] for k in b["X"]},
            "nli_thr": N.youden_threshold(b["sets"]["dev_clean"], [r["nli_c"] for r in b["sets"]["dev_clean"]])}
