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


# --- stage 1: content gate --------------------------------------------------------

def _row(split="dev", label=1, cos=0.5, **kw):
    return {"split": split, "label": label, "cos": cos, "c_resid": cos, "nli": 0.9, **kw}


def test_gate_stops_below_t1_and_passes_at_or_above():
    rows = [_row(cos=0.29), _row(cos=0.30), _row(cos=0.9)]
    assert C.gate_pass(rows, "cos", 0.30).tolist() == [False, True, True]


def test_gate_fit_keeps_dev_recall_floor(loaded):
    splits, _ = loaded
    y = C.labels(splits["dev"])
    for sim in C.SIMILARITIES:
        fit = C.fit_gate(splits["dev"], sim, nli_threshold=0.5)
        passed = C.gate_pass(splits["dev"], sim, fit["t1"])
        assert passed[y == 1].mean() >= C.MIN_GATE_RECALL
        assert fit["gate_recall"] == round(float(passed[y == 1].mean()), 4)


def test_no_threshold_is_fit_on_validation_or_test(loaded):
    splits, _ = loaded
    for name in ("validation", "test"):
        with pytest.raises(AssertionError, match="fit attempted"):
            C.fit_gate(splits[name], "cos", 0.5)
        with pytest.raises(AssertionError, match="fit attempted"):
            C.fit_gate(splits["dev"] + splits[name][:1], "cos", 0.5)


# --- stage 2: cancellation veto ---------------------------------------------------

def veto(a, b):
    return C.cancellation_detected(C.P._analysis(a), C.P._analysis(b))


def test_trap1_no_cue_lexical_negation_is_not_vetoed():
    # |delta_p| = 0 because neither side has a cue: a real negation NLI catches
    a = "The current constituency is thus smaller in area than the old one."
    b = "The current constituency is thus larger in area than the old one."
    assert C.P.v2(C.P._analysis(a)) == C.P.v2(C.P._analysis(b)) == 0.0
    assert not veto(a, b)


def test_trap1_holds_on_every_cue_free_pair_in_the_data(loaded):
    splits, analyses = loaded
    rows = [r for s in splits.values() for r in s if not analyses[r["b"]].cues]
    assert len(rows) > 1000
    assert not C.veto(rows, analyses).any()


def test_same_clause_cancellation_is_vetoed():
    assert veto("The function sorts the array.", "The function never fails to sort the array.")
    assert veto("The task is possible.", "The task is not impossible.")


def test_cancellation_that_flips_polarity_is_not_vetoed():
    # A already negated: "not impossible" does negate "impossible"
    assert not veto("The task is impossible.", "The task is not impossible.")


def test_cues_in_different_clauses_do_not_cancel():
    assert not veto("He left and she stayed.", "He did not leave and she never stayed.")


def test_veto_catches_most_k1(loaded):
    splits, analyses = loaded
    k1 = [r for r in splits["dev"] if r["kind"] == "K1"]
    assert C.veto(k1, analyses).mean() > 0.9
