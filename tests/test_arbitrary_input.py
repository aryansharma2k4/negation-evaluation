"""The arbitrary-input layer: operations, sentence types, and the depth-1 rule.

Three things are pinned here.

*coverage*   one fixture per ``clause_type`` x ``polarity`` combination, so no
             input shape silently stops producing anything.
*invariants* the depth-1 rule, asserted over a whole generated corpus rather
             than on hand-picked cases.
*regression* stage 1's 448 variants, frozen against a copy of the base corpus
             that lives with the tests -- ``data/sample_sentences.txt`` is a
             scratch file users edit, and a regression baseline must not move
             when they do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from negation.base import registry
from negation.classify import (
    CLAUSE_DECLARATIVE,
    CLAUSE_EXCLAMATIVE,
    CLAUSE_IMPERATIVE,
    CLAUSE_INTERROGATIVE,
    POL_AFFIRMATIVE,
    POL_NEGATED,
    classify_input,
)
from negation.driver import generate_all, parse_bases
from negation.filters import check_operation_depth
from negation.nlp_core import parse_batch
from negation.schema import (
    DepthViolation,
    FAM_SYNTACTIC,
    MAX_OP_DEPTH,
    OP_AFFIRM,
    OP_NEGATE,
    OP_RESCOPE,
    OPERATIONS,
)

BASE_CORPUS = Path(__file__).parent / "data" / "base_sentences_v1.txt"

#: The stage-1 corpus, frozen.  Twenty sentences, 448 variants.
STAGE1_BASELINE = 448

#: One input per clause_type x polarity cell.
FIXTURES: dict[tuple[str, str], str] = {
    (CLAUSE_DECLARATIVE, POL_AFFIRMATIVE): "The function sorts the array.",
    (CLAUSE_DECLARATIVE, POL_NEGATED): "The function does not sort the array.",
    (CLAUSE_INTERROGATIVE, POL_AFFIRMATIVE): "Does it compile?",
    (CLAUSE_INTERROGATIVE, POL_NEGATED): "Does it not compile?",
    (CLAUSE_IMPERATIVE, POL_AFFIRMATIVE): "Sort the array.",
    (CLAUSE_IMPERATIVE, POL_NEGATED): "Do not sort the array.",
    (CLAUSE_EXCLAMATIVE, POL_AFFIRMATIVE): "What a mess this is!",
    (CLAUSE_EXCLAMATIVE, POL_NEGATED): "How unimportant this is!",
}


def base_sentences() -> list[str]:
    return [line.strip() for line in BASE_CORPUS.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def full_corpus() -> list:
    """Every record the whole registry produces for the frozen base corpus."""
    return generate_all(base_sentences())


@pytest.fixture(scope="module")
def fixture_corpus() -> list:
    """Every record produced for the clause_type x polarity fixtures."""
    return generate_all(list(FIXTURES.values()))


# -- clause_type x polarity coverage ---------------------------------------


@pytest.mark.parametrize("cell,sentence", sorted(FIXTURES.items()))
def test_fixture_classifies_as_its_cell(parse, cell, sentence):
    clause_type, polarity = cell
    profile = classify_input(parse(sentence))
    assert (profile.clause_type, profile.polarity) == (clause_type, polarity)


@pytest.mark.parametrize("cell,sentence", sorted(FIXTURES.items()))
def test_every_cell_produces_variants(cell, sentence):
    """No input shape may silently stop producing anything."""
    assert generate_all([sentence]), f"no variants for {cell}: {sentence!r}"


@pytest.mark.parametrize("cell,sentence", sorted(FIXTURES.items()))
def test_records_report_their_input_faithfully(cell, sentence):
    clause_type, polarity = cell
    for record in generate_all([sentence]):
        assert record.clause_type == clause_type
        assert record.input_polarity == polarity
        assert record.base_sentence == sentence


def test_negated_input_gets_the_operations_stage_one_could_not_offer():
    """A negated declarative affirms; an affirmative one has nothing to affirm."""
    negated = {r.operation for r in generate_all(["The function does not sort the array."])}
    affirmative = {r.operation for r in generate_all(["The function sorts the array."])}
    assert OP_AFFIRM in negated
    assert OP_AFFIRM not in affirmative


def test_all_three_operations_are_reachable():
    corpus = generate_all([
        "The function sorts the array.",
        "The function does not sort the array.",
        "Everyone did not pass the exam.",
    ])
    assert {r.operation for r in corpus} == OPERATIONS


# -- sentence-type handling -------------------------------------------------


def test_interrogative_preserves_inversion_and_the_question_mark():
    got = {r.variant for r in generate_all(["Does it compile?"])}
    assert "Does it not compile?" in got
    assert "Doesn't it compile?" in got
    assert all(v.endswith("?") for v in got)
    # The bug this replaced: the negator after the *fronted auxiliary*.
    assert not any(v.startswith("Does not") for v in got)


def test_imperative_uses_bare_do_support():
    got = {r.variant for r in generate_all(["Sort the array."])}
    assert "Do not sort the array." in got
    assert "Don't sort the array." in got


def test_passive_negation_attaches_to_the_passive_auxiliary():
    records = generate_all(["The array was sorted by the function."])
    got = {r.variant for r in records}
    assert "The array was not sorted by the function." in got
    assert all(r.voice == "passive" for r in records)


def test_existential_prefers_quantifier_negation():
    got = {r.variant for r in generate_all(["There is a solution."])}
    assert "There is no solution." in got


@pytest.mark.parametrize("sentence,expected", [
    ("There is a solution.", "There is no solution."),
    ("There are solutions.", "There are no solutions."),
    # *no* takes the NP's left edge, so it replaces a numeral rather than
    # stacking after it, and precedes an adjective rather than following it.
    ("There are three solutions.", "There are no solutions."),
    ("There are stale entries.", "There are no stale entries."),
    ("There is a simple solution.", "There is no simple solution."),
])
def test_existential_places_no_at_the_np_left_edge(sentence, expected):
    assert expected in {r.variant for r in generate_all([sentence])}


@pytest.mark.parametrize("sentence,expected", [
    ("This parser is faster than the old one.", "This parser is no faster than the old one."),
    # Periphrastic: *no* goes in front of the degree word, not the adjective.
    ("The parser is more robust than the old one.", "The parser is no more robust than the old one."),
    ("This is less robust than that.", "This is no less robust than that."),
])
def test_comparative_handles_periphrastic_degree_marking(sentence, expected):
    assert expected in {r.variant for r in generate_all([sentence])}


def test_neither_affirms_with_its_noun_pluralised():
    """*neither* takes a singular noun, *both* a plural one."""
    got = {r.variant for r in generate_all(["Neither test passed."])}
    assert "Both tests passed." in got
    assert "Both test passed." not in got


def test_modal_emits_both_readings_as_distinct_records():
    records = [r for r in generate_all(["The parser must handle noise."]) if r.family == "M_modal"]
    by_variant = {r.variant: r for r in records}
    assert "The parser must not handle noise." in by_variant       # prohibition
    assert "The parser need not handle noise." in by_variant       # no obligation
    subtypes = {r.subtype for r in records}
    assert len(subtypes) == len(records), "readings must not share a subtype"
    assert all(r.modality == "deontic" for r in records)


def test_comparative_emits_not_and_no_as_different_records():
    records = generate_all(["This parser is faster than the old one."])
    got = {r.variant: r for r in records}
    assert "This parser is not faster than the old one." in got
    assert "This parser is no faster than the old one." in got
    # Different meanings, so they must not collapse into one record.
    assert (
        got["This parser is not faster than the old one."].subtype
        != got["This parser is no faster than the old one."].subtype
    )


def test_conditional_negates_each_clause_in_a_separate_record():
    records = generate_all(["If the test passes, the build succeeds."])
    got = {r.variant for r in records}
    assert "If the test does not pass, the build succeeds." in got
    assert "If the test passes, the build does not succeed." in got


def test_no_depth_one_record_negates_both_clauses_of_a_conditional():
    """Antecedent and consequent together is two operations, so depth 1 forbids it.

    K2_compound does emit exactly that sentence, and is allowed to: it declares
    ``op_depth = 2`` precisely because it applies two operations. The rule being
    checked here is that nothing claiming depth 1 sneaks it in.
    """
    both = "If the test does not pass, the build does not succeed."
    records = generate_all(["If the test passes, the build succeeds."])
    culprits = [r for r in records if r.variant == both and r.op_depth == 1]
    assert not culprits
    # ...and the sentence is still produced, by the family that owns it.
    assert [r for r in records if r.variant == both and r.op_depth == 2]


def test_multi_clause_emits_one_record_per_clause_targeted():
    records = generate_all(["The function sorts the array and returns the result."])
    targeted = {
        r.target_clause_idx
        for r in records
        if r.family == FAM_SYNTACTIC and r.operation == OP_NEGATE
    }
    assert targeted == {0, 1}


def test_target_clause_idx_points_at_the_clause_actually_negated():
    records = generate_all(["If the test passes, the build succeeds."])
    by_variant = {r.variant: r for r in records if r.family == FAM_SYNTACTIC}
    assert by_variant["If the test does not pass, the build succeeds."].target_clause_idx == 0
    assert by_variant["If the test passes, the build does not succeed."].target_clause_idx == 1


def test_a_second_cue_on_a_negated_input_is_classified_k1_or_k2():
    """Adding a cue to an N-cue input is still one operation; only the label changes."""
    records = generate_all(["Unless the test fails, the build succeeds."])
    by_variant = {r.variant: r for r in records}
    cancelled = by_variant["Unless the test does not fail, the build succeeds."]
    compounded = by_variant["Unless the test fails, the build does not succeed."]
    assert (cancelled.family, cancelled.net_negation) == ("K1_cancellation", 0)
    assert (compounded.family, compounded.net_negation) == ("K2_compound", 2)
    assert cancelled.op_depth == compounded.op_depth == 1


# -- affirmation ------------------------------------------------------------


@pytest.mark.parametrize("negated,affirmed", [
    ("The function does not sort the array.", "The function sorts the array."),
    ("The function did not sort the array.", "The function sorted the array."),
    ("She has not finished the report.", "She has finished the report."),
    ("She cannot finish the report.", "She can finish the report."),
    ("The array was not sorted.", "The array was sorted."),
    ("The result is not important.", "The result is important."),
    ("No students passed the exam.", "Some students passed the exam."),
    ("He never arrives early.", "He arrives early."),
    ("He rarely arrives early.", "He often arrives early."),
    ("The result is unimportant.", "The result is important."),
    ("The function fails to sort the array.", "The function sorts the array."),
    ("The system lacks redundancy.", "The system has redundancy."),
])
def test_affirmation(negated, affirmed):
    got = {r.variant for r in generate_all([negated]) if r.operation == OP_AFFIRM}
    assert affirmed in got


def test_affirming_a_single_cue_sentence_gives_net_negation_zero():
    records = [
        r for r in generate_all(["The function does not sort the array."])
        if r.operation == OP_AFFIRM
    ]
    assert records
    assert all(r.net_negation == 0 for r in records)
    assert all(r.cue_count == 0 for r in records)


def test_affirming_a_two_cue_sentence_leaves_the_other_cue_standing():
    """Depth 1 affirming: one cue out, not both."""
    records = [
        r for r in generate_all(["The function does not fail to sort the array."])
        if r.operation == OP_AFFIRM
    ]
    got = {r.variant: r for r in records}
    assert "The function fails to sort the array." in got
    assert "The function does not sort the array." in got
    for record in records:
        assert record.input_cue_count == 2
        assert record.net_negation == 1
        assert record.cue_count == 1


def test_affirmation_declines_an_affirmative_input():
    assert not [
        r for r in generate_all(["The function sorts the array."])
        if r.operation == OP_AFFIRM
    ]


# -- round trip -------------------------------------------------------------


@pytest.mark.parametrize("original", [
    "The function does not sort the array.",
    "The function did not sort the array.",
    "She has not finished the report.",
    "The result is not important.",
    "The array was not sorted.",
])
def test_negate_of_affirm_recovers_the_original(original):
    """``negate(affirm(x))`` reproduces ``x`` for the syntactic family.

    The two directions are separate implementations -- do-support is built in
    ``frames`` and collapsed in ``affirm`` -- so this is a real check that the
    morphology survives the round trip rather than a tautology.
    """
    affirmed = {
        r.variant for r in generate_all([original])
        if r.operation == OP_AFFIRM and r.family == FAM_SYNTACTIC
    }
    assert affirmed, f"nothing affirmed {original!r}"

    recovered = set()
    for sentence in affirmed:
        recovered |= {
            r.variant for r in generate_all([sentence])
            if r.operation == OP_NEGATE and r.family == FAM_SYNTACTIC
        }
    assert original in recovered


def test_rescope_round_trips_between_the_two_scope_readings():
    subject_scope = "Not everyone passed the exam."
    predicate_scope = "Everyone did not pass the exam."

    forward = {r.variant for r in generate_all([subject_scope]) if r.operation == OP_RESCOPE}
    backward = {r.variant for r in generate_all([predicate_scope]) if r.operation == OP_RESCOPE}
    assert predicate_scope in forward
    assert subject_scope in backward


def test_rescope_keeps_the_cue_count_and_changes_the_scope_target():
    records = [r for r in generate_all(["Not everyone passed the exam."]) if r.operation == OP_RESCOPE]
    assert records
    for record in records:
        assert record.input_cue_count == record.cue_count == 1
        assert record.scope_target == "predicate"  # input was subject-scoped


# -- the depth-1 rule -------------------------------------------------------


def test_every_arbitrary_input_generator_declares_depth_one():
    for generator in registry():
        if generator.stage == 2:
            assert generator.op_depth == MAX_OP_DEPTH, generator.name


def test_stage_two_corpus_is_entirely_depth_one(full_corpus, fixture_corpus):
    for record in full_corpus + fixture_corpus:
        if record.generator in {g.name for g in registry() if g.stage == 2}:
            assert record.op_depth == 1, record


def test_no_record_exceeds_n_plus_one_cues(full_corpus, fixture_corpus):
    """The operative form of the depth-1 rule, over a whole corpus.

    Counted by the classifier on both sides, so a multiword cue counts once.
    """
    records = full_corpus + fixture_corpus
    docs = parse_batch([r.variant for r in records])
    for record, doc in zip(records, docs):
        observed = classify_input(doc).existing_count
        assert observed <= record.max_output_cues, (record.variant, observed)


def test_the_guardrail_raises_rather_than_dropping():
    """A miscounting generator must fail loudly."""
    record = generate_all(["The function sorts the array."])[0]
    object.__setattr__(record, "op_depth", 0)
    object.__setattr__(record, "input_cue_count", 0)
    doc = parse_batch(["The function does not sort the array."])[0]
    with pytest.raises(DepthViolation):
        check_operation_depth(record, doc)


def test_every_record_carries_the_full_input_description(full_corpus):
    for record in full_corpus:
        assert record.operation in OPERATIONS
        assert record.input_polarity in ("affirmative", "negated")
        assert record.input_cue_count >= 0
        assert record.op_depth >= 1
        assert record.target_clause_idx >= 0
        assert record.cue_count == len(record.cue_char_spans) == len(record.cue_tokens)
        for token, (start, end) in zip(record.cue_tokens, record.cue_char_spans):
            assert record.variant[start:end] == token


# -- regression -------------------------------------------------------------


def test_stage_one_still_produces_exactly_448_variants():
    """The frozen stage-1 battery, unchanged by everything above."""
    records = generate_all(base_sentences(), stages=(1,))
    assert len(records) == STAGE1_BASELINE


def test_stage_one_corpus_is_a_subset_of_the_full_corpus(full_corpus):
    """The new layer only ever adds; it never displaces a stage-1 record."""
    stage1 = generate_all(base_sentences(), stages=(1,))
    stage1_keys = {(r.base_id, r.variant, r.family, r.subtype) for r in stage1}
    full_keys = {(r.base_id, r.variant, r.family, r.subtype) for r in full_corpus}
    assert stage1_keys <= full_keys


def test_the_base_corpus_is_the_twenty_sentences_it_was():
    assert len(base_sentences()) == 20
