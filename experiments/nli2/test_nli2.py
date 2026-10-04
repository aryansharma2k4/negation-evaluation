"""nli2 tests: conditional splitting (3.1), the four decisions (3.2), the Phase 2 rules,
and split discipline. Constructed sentences only: no TEST data is used to design a rule.

    ./venv/bin/python -m pytest experiments/nli2 -q
"""

import pytest

import conditionals as K
import rules as R
from common import C


def words(sentence, span):
    doc = K.parse_batch([sentence])[0]
    return " ".join(doc[i].text for i in span)


def cond(sentence):
    return K.primary(K.conditionals([sentence])[sentence])


def decision(a, b):
    return K.decide(cond(a), cond(b), C.P._analysis(a), C.P._analysis(b))


# --- 3.1 splitting --------------------------------------------------------------------

@pytest.mark.parametrize("sentence,marker,ante,cons", [
    ("If you sign the contract, you will receive benefits.", "if", "If you sign the contract", "you will receive benefits"),
    ("You will receive benefits if you sign.", "if", "if you sign", "You will receive benefits"),       # inverted
    ("Unless you sign, you will not receive benefits.", "unless", "Unless you sign", "you will not receive benefits"),
    ("When it rains, the street gets wet.", "when", "When it rains", "the street gets wet"),
    ("Provided that you sign, you will receive benefits.", "provided", "that you sign", None),
    ("Assuming you sign, you will receive benefits.", "assuming", "you sign", None),
])
def test_split_finds_marker_antecedent_and_consequent(sentence, marker, ante, cons):
    c = cond(sentence)
    assert c is not None and c.marker == marker
    assert words(sentence, c.ante_span) == ante
    if cons:
        assert words(sentence, c.cons_span) == cons


def test_coordinated_antecedent_keeps_both_conjuncts():
    s = "If you sign the contract and pay the fee, you will receive benefits."
    assert "pay the fee" in words(s, cond(s).ante_span)


def test_nested_conditional_is_flagged_and_primary_is_outermost():
    s = "If you sign, you will receive benefits if you pay."      # two antecedents, one consequent: siblings
    conds = K.conditionals([s])[s]
    assert len(conds) == 2 and not any(c.nested for c in conds)
    s2 = "If the court finds that you will pay if you sign, the case is closed."
    c2 = K.conditionals([s2])[s2]
    assert len(c2) == 2 and any(c.nested for c in c2)
    assert K.primary(c2).marker == "if" and words(s2, K.primary(c2).cons_span).startswith("the case")


def test_non_conditional_has_no_split():
    assert cond("The function sorts the array.") is None
    assert cond("He asked if it was raining.") is None          # "if" introducing a ccomp, not a condition


# --- 3.2 the four decisions -------------------------------------------------------------

def test_antecedent_only_flip_is_not_a_negation():
    assert decision("If you sign the contract, you will receive benefits.",
                    "If you do not sign the contract, you will receive benefits.") == "not_negation"


def test_consequent_flip_is_a_negation():
    assert decision("If you sign the contract, you will receive benefits.",
                    "If you sign the contract, you will not receive benefits.") == "negation"


def test_both_flipped_is_tagged_contraposition():
    assert decision("If you sign the contract, you will receive benefits.",
                    "If you do not sign the contract, you will not receive benefits.") == "contraposition"


def test_unless_is_if_not():
    # "unless you sign" == "if you do not sign": same antecedent polarity, nothing flipped
    assert decision("Unless you sign, you will receive benefits.",
                    "If you do not sign, you will receive benefits.") is None
    assert decision("Unless you sign, you will receive benefits.",
                    "If you sign, you will receive benefits.") == "not_negation"


def test_no_decision_when_one_side_is_not_conditional():
    assert decision("If you sign the contract, you will receive benefits.",
                    "You sign the contract and do not receive benefits.") is None


# --- Phase 2 rules ----------------------------------------------------------------------

def test_antonym_bypass_is_strict():
    p = R.parse(["It's natural.", "It's artificial.", "A man is playing guitar.", "Two cows eat grass.",
                 "The cat sat on the mat.", "The dog sat on the mat."])
    assert R.antonym_aligned(p["It's natural."], p["It's artificial."])
    assert not R.antonym_aligned(p["A man is playing guitar."], p["Two cows eat grass."])
    assert not R.antonym_aligned(p["The cat sat on the mat."], p["The dog sat on the mat."])


def test_not_antonym_cancellation_needs_a_cue_on_one_side_only():
    p = R.parse(["It's a good idea.", "It's not an evil idea.", "She's not out.", "She's not safe."])
    assert R.not_antonym_cancel(p["It's a good idea."], p["It's not an evil idea."])
    assert not R.not_antonym_cancel(p["She's not out."], p["She's not safe."])


# --- discipline -------------------------------------------------------------------------

def test_clean_dev_has_no_generator_output_and_no_heldout_sentence():
    from common import load_clean
    splits, _ = load_clean()
    assert {r["dataset"] for r in splits["dev"]} <= {"cannot_phase1", "snli", "stsb", "random"}
    for name in ("validation", "test"):
        assert not C.pdata.sentences(splits["dev"]) & C.pdata.sentences(splits[name]), name


def test_thresholds_cannot_be_fit_on_test():
    from common import load_clean, youden_threshold
    splits, _ = load_clean()
    with pytest.raises(AssertionError, match="fit attempted"):
        youden_threshold(splits["test"], [r["nli_c"] for r in splits["test"]])
