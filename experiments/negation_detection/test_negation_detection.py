"""Guardrails for the negation-detection experiment.

Everything downstream is worthless if a sentence leaks from phase 1 into phase 2
or if a base sentence spans two CV folds, so those are checked here, along with
the polarity rule that sorts generated variants into the NEG and SAME reference
sets.

    ./venv/bin/python -m pytest experiments/negation_detection -q
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from data import norm
from detect import HERE, group_folds, load, load_refsets


def test_no_sentence_leaks_between_phases():
    p1, p2 = load("phase1"), load("phase2")
    s1 = {norm(r[k]) for r in p1 for k in ("a", "b")}
    s2 = {norm(r[k]) for r in p2 for k in ("a", "b")}
    assert not (s1 & s2)


def test_phase1_and_phase2_have_no_duplicate_pairs():
    rows = load("phase1") + load("phase2")
    keys = {(norm(r["a"]), norm(r["b"])) for r in rows}
    assert len(keys) == len(rows)


def test_cv_folds_are_grouped_by_sentence_a():
    rows = load("phase1")
    folds = group_folds(rows, 5, np.random.default_rng(0))
    assert sum(len(f) for f in folds) == len(rows)
    seen: dict[str, int] = {}
    for k, fold in enumerate(folds):
        for i in fold:
            a = rows[int(i)]["a"]
            assert seen.setdefault(a, k) == k


def test_reference_sets_split_on_net_polarity():
    """SAME keeps A's own cue parity, NEG flips it; cancellations stay in SAME."""
    refsets = load_refsets()
    cancellations = 0
    for record in list(refsets.values())[:500]:
        parity = record["a_cue_parity"]
        for variant in record["same"]:
            assert variant["net_negation"] % 2 == parity
            cancellations += variant["family"] == "K1_cancellation"
        for variant in record["neg"]:
            assert variant["net_negation"] % 2 != parity
    assert cancellations, "expected K1 cancellations to land in the SAME sets"


def test_every_pair_has_a_reference_set():
    refsets = load_refsets()
    for name in ("phase1", "phase2", "xnot360"):
        for row in load(name):
            assert row["a"] in refsets


@pytest.mark.parametrize("cache", sorted((HERE / "cache").glob("*.pkl")))
def test_cached_embeddings_are_l2_normalised(cache: Path):
    blob = pickle.loads(cache.read_bytes())
    vectors = [v for v in blob.values() if isinstance(v, np.ndarray)]
    if not vectors:
        pytest.skip(f"{cache.name} holds no embeddings")
    sample = np.stack(vectors[:2000])
    assert np.allclose(np.linalg.norm(sample, axis=1), 1.0, atol=1e-4)
