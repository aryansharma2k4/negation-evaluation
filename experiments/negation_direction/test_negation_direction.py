"""Guardrails for the negation-direction experiment.

Two things must hold or every number downstream is wrong: the train/test split
must not leak a base sentence across sides, and the embeddings must be
L2-normalised (every cosine here is a plain dot product).

    ./venv/bin/python -m pytest experiments/negation_direction -q
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from experiment import GROUP_OF_FAMILY, HERE, load_pairs, split_bases


def test_split_has_no_base_sentence_leakage():
    pairs, _ = load_pairs()
    bases = sorted({p["base_sentence"] for p in pairs})
    train, test = split_bases(bases)
    assert not (train & test)
    assert train | test == set(bases)
    assert len(test) >= 1 and len(train) >= 1
    train_pairs = [p for p in pairs if p["base_sentence"] in train]
    test_pairs = [p for p in pairs if p["base_sentence"] in test]
    assert len(train_pairs) + len(test_pairs) == len(pairs)
    assert not ({p["base_sentence"] for p in train_pairs}
                & {p["base_sentence"] for p in test_pairs})


def test_split_is_deterministic():
    bases = sorted({p["base_sentence"] for p in load_pairs()[0]})
    assert split_bases(bases, 0) == split_bases(bases, 0)


def test_every_family_has_a_group():
    pairs, affirms = load_pairs()
    for pair in pairs + affirms:
        assert pair["group" if "group" in pair else "family"] is not None
        assert pair["family"] in GROUP_OF_FAMILY


@pytest.mark.parametrize("cache", sorted((HERE / "cache").glob("*.pkl")))
def test_cached_embeddings_are_l2_normalised(cache: Path):
    vectors = np.stack(list(pickle.loads(cache.read_bytes()).values()))
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)


def test_cache_covers_every_sentence_in_the_pairs():
    caches = sorted((HERE / "cache").glob("*.pkl"))
    if not caches:
        pytest.skip("no embedding cache yet; run experiment.py first")
    pairs, _ = load_pairs()
    wanted = {p["base_sentence"] for p in pairs} | {p["variant"] for p in pairs}
    for cache in caches:
        assert wanted <= set(pickle.loads(cache.read_bytes()))
