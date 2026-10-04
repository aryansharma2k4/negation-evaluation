"""Split discipline and the composite's two traps.

    ./venv/bin/python -m pytest experiments/composite -q
"""

import pytest

import composite as C


@pytest.fixture(scope="module")
def loaded():
    return C.load()


def test_no_sentence_crosses_splits(loaded):
    splits, _ = loaded
    names = list(splits)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            assert not C.pdata.sentences(splits[x]) & C.pdata.sentences(splits[y]), (x, y)


def test_every_row_has_the_cached_inputs(loaded):
    splits, analyses = loaded
    for rows in splits.values():
        for r in rows:
            assert {"cos", "c_resid", "nli", "kind"} <= r.keys()
            assert r["a"] in analyses and r["b"] in analyses


def test_kinds_the_report_breaks_down_on_exist(loaded):
    splits, _ = loaded
    assert sum(r["kind"] == "K1" for r in splits["dev"]) == 91
    assert {r["kind"] for r in splits["test"]} >= {"paraphrase", "sts_low", "snli_neutral",
                                                    "random_pair", "SemAntoNeg", "negation"}
    phase2 = splits["validation"] + [r for r in splits["test"] if r["dataset"] == "cannot_phase2"]
    assert sum(r["kind"] == "SemAntoNeg" for r in phase2) == 41
