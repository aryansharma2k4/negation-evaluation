"""Input classification across clause types, voices, polarities and cue families."""

from __future__ import annotations

import pytest

from negation.classify import (
    CLAUSE_DECLARATIVE,
    CLAUSE_EXCLAMATIVE,
    CLAUSE_IMPERATIVE,
    CLAUSE_INTERROGATIVE,
    COPULA_BE,
    COPULA_NONE,
    COPULA_SEEM,
    POL_AFFIRMATIVE,
    POL_NEGATED,
    VOICE_ACTIVE,
    VOICE_PASSIVE,
    classify_input,
)
from negation.lexicons import MOD_ABILITY, MOD_DEONTIC, MOD_EPISTEMIC, NO_MODALITY
from negation.schema import (
    FAM_AFFIXAL,
    FAM_IMPLICIT,
    FAM_NEG_ADVERB,
    FAM_PREPOSITIONAL,
    FAM_QUANTIFIER,
    FAM_SYNTACTIC,
)


# -- clause type ------------------------------------------------------------


@pytest.mark.parametrize("sentence,expected", [
    ("The function sorts the array.", CLAUSE_DECLARATIVE),
    ("Does it compile?", CLAUSE_INTERROGATIVE),
    ("Is the task impossible?", CLAUSE_INTERROGATIVE),
    ("Has she finished the report?", CLAUSE_INTERROGATIVE),
    ("Sort the array.", CLAUSE_IMPERATIVE),
    ("What a mess this is!", CLAUSE_EXCLAMATIVE),
])
def test_clause_type(parse, sentence, expected):
    assert classify_input(parse(sentence)).clause_type == expected


def test_exclamation_mark_does_not_override_an_imperative():
    """*"Sort the array!"* is still a command, not an exclamative."""
    from negation.driver import parse_bases

    doc = parse_bases(["Sort the array!"])[0]
    assert classify_input(doc).clause_type == CLAUSE_IMPERATIVE


# -- voice ------------------------------------------------------------------


@pytest.mark.parametrize("sentence,expected", [
    ("The function sorted the array.", VOICE_ACTIVE),
    ("The array was sorted by the function.", VOICE_PASSIVE),
    ("The array has been sorted.", VOICE_PASSIVE),
])
def test_voice(parse, sentence, expected):
    assert classify_input(parse(sentence)).voice == expected


# -- polarity and cue families ---------------------------------------------


@pytest.mark.parametrize("sentence,family", [
    ("The function does not sort the array.", FAM_SYNTACTIC),
    ("She cannot finish the report.", FAM_SYNTACTIC),
    ("No students passed the exam.", FAM_QUANTIFIER),
    ("Nobody passed the exam.", FAM_QUANTIFIER),
    ("Neither test passed.", FAM_QUANTIFIER),
    ("He rarely arrives early.", FAM_NEG_ADVERB),
    ("He never arrives early.", FAM_NEG_ADVERB),
    ("The result is unimportant.", FAM_AFFIXAL),
    ("The function fails to sort the array.", FAM_IMPLICIT),
    ("The system lacks redundancy.", FAM_IMPLICIT),
    ("The parser runs without a cache.", FAM_PREPOSITIONAL),
    ("The argument is devoid of merit.", FAM_PREPOSITIONAL),
])
def test_cue_detection_spans_every_family(parse, sentence, family):
    profile = classify_input(parse(sentence))
    assert profile.polarity == POL_NEGATED
    assert profile.existing_count == 1
    assert profile.existing_cues[0].family_guess == family


def test_affirmative_input_has_no_cues(parse):
    profile = classify_input(parse("The function sorts the array."))
    assert profile.polarity == POL_AFFIRMATIVE
    assert profile.existing_cues == ()
    assert profile.existing_count == 0


def test_cue_carries_its_token_index_and_text(parse):
    doc = parse("The function does not sort the array.")
    cue = classify_input(doc).existing_cues[0]
    assert doc[cue.token_idx].text == cue.cue_text == "not"
    # Unpacks positionally as the (token_idx, cue_text, family_guess) triple.
    idx, text, family = cue
    assert (idx, text, family) == (cue.token_idx, cue.cue_text, cue.family_guess)


def test_multiword_quantifier_counts_as_one_cue(parse):
    profile = classify_input(parse("No one passed the exam."))
    assert profile.existing_count == 1
    assert profile.existing_cues[0].cue_text == "No one"


def test_two_cues_are_both_reported(parse):
    profile = classify_input(parse("The function does not fail to sort the array."))
    assert profile.existing_count == 2
    assert {c.family_guess for c in profile.existing_cues} == {FAM_SYNTACTIC, FAM_IMPLICIT}


def test_a_positive_word_with_a_negative_prefix_is_not_a_cue(parse):
    """*income* is not ``in`` + *come*; the WordNet antonymy gate rejects it."""
    profile = classify_input(parse("The income increased."))
    assert profile.polarity == POL_AFFIRMATIVE


def test_bare_free_is_not_privative(parse):
    """*free* only negates with its *of* complement."""
    assert classify_input(parse("The build is free.")).polarity == POL_AFFIRMATIVE
    assert classify_input(parse("The build is free of errors.")).polarity == POL_NEGATED


def test_cues_of_selects_by_family(parse):
    profile = classify_input(parse("The function does not fail to sort the array."))
    assert len(profile.cues_of(FAM_SYNTACTIC)) == 1
    assert profile.cues_of(FAM_QUANTIFIER) == ()


# -- modality ---------------------------------------------------------------


@pytest.mark.parametrize("sentence,expected", [
    ("The function sorts the array.", NO_MODALITY),
    ("The parser must handle noise.", MOD_DEONTIC),
    ("The parser should handle noise.", MOD_DEONTIC),
    ("The model can handle noise.", MOD_ABILITY),
    ("The model might handle noise.", MOD_EPISTEMIC),
])
def test_modality(parse, sentence, expected):
    assert classify_input(parse(sentence)).modality == expected


def test_modality_ignores_a_modal_in_a_subordinate_clause(parse):
    profile = classify_input(parse("The parser that can handle noise sorts the array."))
    assert profile.modality == NO_MODALITY


# -- structural flags -------------------------------------------------------


def test_existential(parse):
    assert classify_input(parse("There is a solution.")).is_existential
    assert not classify_input(parse("The solution is here.")).is_existential


def test_comparative(parse):
    assert classify_input(parse("This parser is faster than the old one.")).is_comparative
    assert not classify_input(parse("This parser is fast.")).is_comparative


def test_conditional(parse):
    assert classify_input(parse("If the test passes, the build succeeds.")).is_conditional
    assert classify_input(parse("Unless the test fails, the build succeeds.")).is_conditional
    assert not classify_input(parse("The test passes.")).is_conditional


def test_clause_count_and_subordination(parse):
    single = classify_input(parse("The function sorts the array."))
    assert single.clause_count == 1 and not single.has_subordinate

    coordinated = classify_input(parse("The function sorts the array and returns the result."))
    assert coordinated.clause_count == 2

    subordinated = classify_input(parse("Although the test passed, the build failed."))
    assert subordinated.has_subordinate


@pytest.mark.parametrize("sentence,expected", [
    ("The task is impossible.", COPULA_BE),
    ("The build seems stable.", COPULA_SEEM),
    ("The function sorts the array.", COPULA_NONE),
    ("The array was sorted.", COPULA_NONE),
])
def test_copula_type(parse, sentence, expected):
    assert classify_input(parse(sentence)).copula_type == expected


# -- caching ----------------------------------------------------------------


def test_profile_is_computed_once_per_doc(parse):
    doc = parse("The function sorts the array.")
    assert classify_input(doc) is classify_input(doc)


def test_profile_rejects_unknown_enum_values():
    from negation.classify import InputProfile

    with pytest.raises(ValueError):
        InputProfile(
            clause_type="bogus",
            voice=VOICE_ACTIVE,
            polarity=POL_AFFIRMATIVE,
            existing_cues=(),
            existing_count=0,
            modality=NO_MODALITY,
            clause_count=1,
            has_subordinate=False,
            is_existential=False,
            is_comparative=False,
            is_conditional=False,
            copula_type=COPULA_NONE,
        )
