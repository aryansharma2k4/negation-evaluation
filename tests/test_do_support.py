"""Do-support morphology: the auxiliary must inherit exactly what the verb loses.

These are the cases naive implementations get wrong -- emitting a fixed
``"does not"`` regardless of tense, or leaving the lexical verb inflected
(``*"does not sorts"``).
"""

from __future__ import annotations

import pytest

from negation.frames import DO_SUPPORT, detect_frame
from negation.nlp_core import root_of

DO_SUPPORT_CASES = [
    # (base sentence, expected negation)
    ("The function sorts the array.", "The function does not sort the array."),
    ("The function sorted the array.", "The function did not sort the array."),
    ("The functions sort the array.", "The functions do not sort the array."),
    ("I sort the array.", "I do not sort the array."),
    ("They solved the problem.", "They did not solve the problem."),
    ("She writes tests.", "She does not write tests."),
    ("The parser went wrong.", "The parser did not go wrong."),
    ("Sort the array.", "Do not sort the array."),
]


@pytest.mark.parametrize("base,expected", DO_SUPPORT_CASES)
def test_do_support_tense_and_agreement(parse, run_generator, base, expected):
    doc = parse(base)
    records = run_generator(doc, "syn_do_support_v1")
    assert [r.variant for r in records] == [expected]


@pytest.mark.parametrize("base,aux", [
    ("The function sorts the array.", "does"),
    ("The function sorted the array.", "did"),
    ("The functions sort the array.", "do"),
])
def test_do_form_is_read_from_morphology(parse, base, aux):
    frame = detect_frame(root_of(parse(base)))
    assert frame is not None and frame.subtype == DO_SUPPORT
    assert frame.do_form == aux


def test_lexical_verb_reverts_to_lemma(parse, run_generator):
    doc = parse("The function sorts the array.")
    variant = run_generator(doc, "syn_do_support_v1")[0].variant
    assert "sort " in variant and "sorts" not in variant


def test_do_support_not_used_when_an_auxiliary_exists(parse, run_generator):
    for base in ("She has finished.", "The model can handle noise.", "The task is simple."):
        assert run_generator(parse(base), "syn_do_support_v1") == []


def test_auxiliary_goes_before_a_preverbal_adverb(parse, run_generator):
    """*"he always arrives"* negates as *"he does not always arrive"*."""
    doc = parse("He always arrives early.")
    assert [r.variant for r in run_generator(doc, "syn_do_support_v1")] == [
        "He does not always arrive early."
    ]


def test_cue_span_points_at_the_negator(parse, run_generator):
    doc = parse("The function sorted the array.")
    record = run_generator(doc, "syn_do_support_v1")[0]
    assert record.cue_tokens == ["not"]
    start, end = record.cue_char_spans[0]
    assert record.variant[start:end] == "not"
    assert record.cue_count == 1
    assert record.net_negation == 1
