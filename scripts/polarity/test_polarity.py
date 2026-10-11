"""Split discipline, labels, and the scorer's contract.

    ./venv/bin/python -m pytest scripts/polarity -q
"""

import json
from pathlib import Path

import pytest

import data
import polarity as P

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def splits():
    return data.load_splits()[0]


# --- split discipline -------------------------------------------------------------

def test_no_sentence_crosses_splits(splits):
    names = list(splits)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            assert not data.sentences(splits[x]) & data.sentences(splits[y]), (x, y)


def test_splits_are_the_ones_the_brief_names(splits):
    assert {r["dataset"] for r in splits["dev"]} == {"ours", "cannot_phase1"}
    assert {r["dataset"] for r in splits["validation"]} == {"cannot_phase2"}
    assert {r["dataset"] for r in splits["test"]} == {"cannot_phase2", "xnot360"}
    phase2 = len(data.load_cannot("phase2"))
    val = len(splits["validation"])
    assert val + sum(r["dataset"] == "cannot_phase2" for r in splits["test"]) == phase2
    assert abs(val / phase2 - data.VAL_SHARE) < 0.02


def test_disjointness_check_catches_a_leak(splits):
    leaky = {"dev": splits["dev"], "validation": splits["validation"] + [splits["dev"][0]]}
    with pytest.raises(AssertionError):
        data.assert_disjoint(leaky)


def test_iterations_never_scored_the_test_set():
    path = HERE.parents[1] / "reports" / "results" / "polarity" / "results_polarity.json"
    if not path.exists():
        pytest.skip("no results yet")
    res = json.loads(path.read_text())
    for version in res.get("iterations", {}).values():
        for block in version["grid"].values():
            assert "test" not in block
    for block in res.get("baselines", {}).values():
        assert "test" not in block


# --- labels -----------------------------------------------------------------------

def test_k1_is_labelled_zero_and_negated_input_k1_is_excluded(splits):
    ours = [r for r in splits["dev"] if r["dataset"] == "ours"]
    assert all(r["label"] == 0 for r in ours if r["family"] == "K1_cancellation")
    assert all(r["label"] == 0 for r in ours if r["family"].startswith("ANCHOR"))
    excluded = data.load_ours()[1]["excluded_k1_negated_input"]
    assert excluded and not {tuple(e) for e in excluded} & {(r["a"], r["b"]) for r in ours}


def test_affirm_records_are_negations(splits):
    ours = [r for r in splits["dev"] if r["dataset"] == "ours"]
    affirm = [r for r in ours if r["operation"] == "affirm"]
    assert affirm and all(r["label"] == 1 for r in affirm)


# --- the scorer is a function, and cancellation lowers it --------------------------

@pytest.mark.parametrize("version", ["v2", "v5"])
def test_cancellation_lowers_the_score(version):
    assert P.polarity("The task is not impossible.", version) < P.polarity("The task is impossible.", version)
    assert P.polarity("The function never fails to sort.", version) < P.polarity("The function fails to sort.", version)


def test_scores_are_in_unit_interval_on_unseen_text():
    for s in ["Nobody ever said it would be easy, did they?", "Not everyone agrees.", "Wow!", ""]:
        for v in ("v1", "v2", "v3", "v4", "v5"):
            assert 0.0 <= P.polarity(s or ".", v) <= 1.0


def test_soft_xor_parity():
    assert P.soft_xor([1.0]) == 1.0
    assert P.soft_xor([1.0, 1.0]) == 0.0
    assert P.soft_xor([1.0, 1.0, 1.0]) == 1.0
