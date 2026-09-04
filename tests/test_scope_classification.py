"""K1 vs K2 is decided by scope containment, never by counting cues.

Every case here has exactly **two** negation cues.  A cue-counting classifier
would label them all identically; the correct answer alternates.
"""

from __future__ import annotations

import pytest

from negation.nlp_core import clause_heads, clause_scope, root_of
from negation.scope import classify_double, net_negation_for
from negation.schema import FAM_CANCELLATION, FAM_COMPOUND
from tests.conftest import variants


def test_clause_scope_stops_at_a_coordination_boundary(parse):
    """The conjunct verb is in the root's *subtree* but not in its *clause*."""
    doc = parse("The function sorts the array and returns the result.")
    root = root_of(doc)
    conj = next(t for t in doc if t.dep_ == "conj" and t.pos_ == "VERB")
    assert conj.i in {t.i for t in root.subtree}  # the naive test says "contained"
    assert conj.i not in clause_scope(root)  # the correct test says "separate"


def test_classify_contained_cue_as_cancellation(parse):
    doc = parse("The task is impossible.")
    root = root_of(doc)
    negative = next(t for t in doc if t.lower_ == "impossible")
    assert classify_double(root, negative) == FAM_CANCELLATION
    assert net_negation_for(FAM_CANCELLATION) == 0


def test_classify_separate_clause_cue_as_compound(parse):
    doc = parse("The function sorts the array and returns the result.")
    first, second = clause_heads(doc)
    assert classify_double(first, second) == FAM_COMPOUND
    assert net_negation_for(FAM_COMPOUND) == 2


@pytest.mark.parametrize("base,expected_family", [
    # Two cues, one clause -> they cancel.
    ("The task is impossible.", FAM_CANCELLATION),
    ("The result is important.", FAM_CANCELLATION),
    # Two cues, two clauses -> they do not.
    ("The function sorts the array and returns the result.", FAM_COMPOUND),
    ("She finished the report and he reviewed it.", FAM_COMPOUND),
])
def test_two_cue_records_are_classified_by_containment(
    parse, run_family, base, expected_family
):
    doc = parse(base)
    records = run_family(doc, expected_family)
    assert records, f"no {expected_family} record for {base!r}"
    assert all(len(r.cue_tokens) == 2 for r in records)
    assert all(r.family == expected_family for r in records)


def test_same_cue_count_opposite_net_negation(parse, run_family):
    """The pair the project exists to keep apart."""
    cancelling = run_family(parse("The result is important."), FAM_CANCELLATION)
    compounding = run_family(
        parse("The result is important and the method works."), FAM_COMPOUND
    )
    assert cancelling and compounding

    k1 = next(r for r in cancelling if r.subtype.startswith("not_prefix"))
    k2 = next(r for r in compounding if r.subtype.startswith("not+prefix"))

    assert len(k1.cue_tokens) == len(k2.cue_tokens) == 2  # identical cue counts
    assert k1.net_negation == 0
    assert k2.net_negation == 2
    assert k1.variant == "The result is not unimportant."
    assert k2.variant == "The result is unimportant and the method does not work."


def test_cancellation_from_an_implicative_trigger(parse, run_family):
    doc = parse("The function sorts the array.")
    got = variants(run_family(doc, FAM_CANCELLATION))
    assert "The function never fails to sort the array." in got


def test_ccomp_is_a_separate_clause(parse):
    doc = parse("He said that the model works.")
    matrix, embedded = clause_heads(doc)
    assert classify_double(matrix, embedded) == FAM_COMPOUND


def test_no_variant_stacks_three_negations(parse):
    from negation.driver import generate_all
    from negation.schema import MAX_DEPTH

    records = generate_all(
        [
            "The result is important and the method works.",
            "The function sorts the array.",
            "The task is impossible.",
        ]
    )
    assert records
    assert max(r.depth for r in records) <= MAX_DEPTH
    assert all(abs(r.net_negation) <= 2 for r in records)
