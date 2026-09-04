"""One test per generator family on a fixture sentence."""

from __future__ import annotations

from negation.schema import (
    FAM_AFFIXAL,
    FAM_ANTONYM,
    FAM_CANCELLATION,
    FAM_COMPOUND,
    FAM_CONTRASTIVE,
    FAM_HEDGED,
    FAM_IMPLICIT,
    FAM_INTENSIFIED,
    FAM_NEG_ADVERB,
    FAM_PREPOSITIONAL,
    FAM_QUANTIFIER,
    FAM_SCOPE_POSITION,
    FAM_SYNTACTIC,
)
from tests.conftest import variants


def test_a_syntactic_copula(parse, run_family):
    doc = parse("The task is simple.")
    assert "The task is not simple." in variants(run_family(doc, FAM_SYNTACTIC))


def test_a_syntactic_existing_aux(parse, run_family):
    doc = parse("She has finished the report.")
    assert "She has not finished the report." in variants(run_family(doc, FAM_SYNTACTIC))


def test_a_syntactic_can_is_lexicalised(parse, run_family):
    doc = parse("The model can handle noise.")
    got = variants(run_family(doc, FAM_SYNTACTIC))
    assert "The model cannot handle noise." in got
    assert "The model can not handle noise." not in got


def test_b_quantifier(parse, run_family):
    doc = parse("Some students solved the problem.")
    assert "No students solved the problem." in variants(run_family(doc, FAM_QUANTIFIER))


def test_b_quantifier_partitive_keeps_one_determiner(parse, run_family):
    doc = parse("All the students passed.")
    assert "None of the students passed." in variants(run_family(doc, FAM_QUANTIFIER))


def test_c_neg_adverb_has_both_subtypes(parse, run_family):
    doc = parse("The function sorts the array.")
    records = run_family(doc, FAM_NEG_ADVERB)
    assert "The function never sorts the array." in variants(records)
    assert "The function rarely sorts the array." in variants(records)
    partial = [r for r in records if r.subtype == "partial"]
    assert partial and all(r.intensity_hint == "partial" for r in partial)


def test_d_affixal(parse, run_family):
    doc = parse("The result is important.")
    records = run_family(doc, FAM_AFFIXAL)
    assert "The result is unimportant." in variants(records)
    assert all(r.subtype.startswith(("prefix_", "suffix_")) for r in records)


def test_d_affixal_rejects_unattested_derivations(parse, run_family):
    """``dis`` + ``array`` is a real word but not a negation of *array*."""
    doc = parse("The function sorts the array.")
    got = variants(run_family(doc, FAM_AFFIXAL))
    assert not any("disarray" in v or "disfunction" in v for v in got)


def test_e_antonym_matches_inflection(parse, run_family):
    doc = parse("The score increased.")
    assert "The score decreased." in variants(run_family(doc, FAM_ANTONYM))


def test_f_implicit(parse, run_family):
    doc = parse("The function sorts the array.")
    got = variants(run_family(doc, FAM_IMPLICIT))
    assert "The function fails to sort the array." in got
    assert "The function is unable to sort the array." in got


def test_f_implicit_lack(parse, run_family):
    doc = parse("The system has redundancy.")
    assert "The system lacks redundancy." in variants(run_family(doc, FAM_IMPLICIT))


def test_g_prepositional(parse, run_family):
    doc = parse("The parser runs with a cache.")
    got = variants(run_family(doc, FAM_PREPOSITIONAL))
    assert "The parser runs without a cache." in got
    assert "The parser runs in the absence of a cache." in got


def test_g_prepositional_possession(parse, run_family):
    doc = parse("The system has redundancy.")
    got = variants(run_family(doc, FAM_PREPOSITIONAL))
    assert "The system is devoid of redundancy." in got


def test_h_hedged(parse, run_family):
    doc = parse("The function sorts the array.")
    records = run_family(doc, FAM_HEDGED)
    got = variants(records)
    assert "The function probably does not sort the array." in got
    assert "The function might not sort the array." in got
    assert all(r.intensity_hint == "hedged" for r in records)
    assert all(r.depth == 2 for r in records)


def test_h_hedged_does_not_stack_a_modal_on_do_support(parse, run_family):
    doc = parse("The function sorts the array.")
    got = variants(run_family(doc, FAM_HEDGED))
    assert not any("does might" in v or "might does" in v for v in got)


def test_i_intensified(parse, run_family):
    doc = parse("The task is simple.")
    records = run_family(doc, FAM_INTENSIFIED)
    got = variants(records)
    assert "The task is definitely not simple." in got
    assert "The task is by no means simple." in got
    assert all(r.intensity_hint == "intensified" for r in records)


def test_i_post_aux_phrase_skips_do_support(parse, run_family):
    """*by no means* needs a real auxiliary; do-support clauses skip it."""
    doc = parse("The function sorts the array.")
    subtypes = {r.subtype for r in run_family(doc, FAM_INTENSIFIED)}
    assert "by_no_means" not in subtypes
    assert "not_at_all" not in subtypes


def test_j_contrastive(parse, run_family):
    doc = parse("The task is simple.")
    got = variants(run_family(doc, FAM_CONTRASTIVE))
    assert "The task is anything but simple." in got
    assert "The task is far from simple." in got


def test_j_the_opposite_of_is_nominal_only(parse, run_family):
    adj = parse("The task is simple.")
    noun = parse("The result is a solution.")
    assert "the_opposite_of" not in {r.subtype for r in run_family(adj, FAM_CONTRASTIVE)}
    assert "the_opposite_of" in {r.subtype for r in run_family(noun, FAM_CONTRASTIVE)}


def test_k1_cancellation(parse, run_family):
    doc = parse("The task is impossible.")
    records = run_family(doc, FAM_CANCELLATION)
    assert "The task is not impossible." in variants(records)
    assert all(r.net_negation == 0 for r in records)


def test_k2_compound(parse, run_family):
    doc = parse("The function sorts the array and returns the result.")
    records = run_family(doc, FAM_COMPOUND)
    assert (
        "The function does not sort the array and does not return the result."
        in variants(records)
    )
    assert all(r.net_negation == 2 for r in records)


def test_l_scope_position_emits_both_readings(parse, run_family):
    doc = parse("Everyone passed the exam.")
    records = run_family(doc, FAM_SCOPE_POSITION)
    by_subtype = {r.subtype: r for r in records}
    assert by_subtype["subject_scope"].variant == "Not everyone passed the exam."
    assert by_subtype["subject_scope"].scope_target == "subject"
    assert by_subtype["predicate_scope"].variant == "Everyone did not pass the exam."
    assert by_subtype["predicate_scope"].scope_target == "predicate"
