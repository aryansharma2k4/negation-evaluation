"""The derived lexicons stay in step with the tables they are derived from.

Several of the new lexicons exist because a word list already in the module has
to be readable in the other direction.  These tests are what keeps the two
directions from drifting when someone adds a word to only one of them.
"""

from __future__ import annotations

from negation.lexicons import (
    COPULAR_PRIVATIVES,
    IMPLICIT_TRIGGERS,
    IMPLICIT_TRIGGER_LEMMAS,
    IMPLICIT_TRIGGER_PREDICATES,
    LACK_TRIGGER,
    MODAL_NEGATION_READINGS,
    MODALITY_BY_MODAL,
    NEGATIVE_ADVERBS,
    NEGATIVE_AUX_BASES,
    NEGATIVE_QUANTIFIER_BIGRAMS,
    NEGATIVE_QUANTIFIER_FLIPS,
    NEGATIVE_QUANTIFIERS,
    NEG_ADVERB_AFFIRMATIONS,
    PRIVATIVE_HEADS,
    QUANTIFIER_SWAPS,
    WITH_REPLACEMENTS,
    CONTRACTED_NEGATIVE_AUX,
)


def test_every_negative_quantifier_can_be_flipped_back():
    """B can produce it, so the affirmation direction must be able to undo it."""
    for swap in QUANTIFIER_SWAPS.values():
        head = swap.replacement.split()[0]
        if head in NEGATIVE_ADVERBS:
            continue  # "always" -> "never" is family C's word, not a quantifier
        assert head in NEGATIVE_QUANTIFIER_FLIPS, head


def test_multiword_negative_quantifiers_have_a_bigram_entry():
    multiword = {
        tuple(s.replacement.split()[:2])
        for s in QUANTIFIER_SWAPS.values()
        if len(s.replacement.split()) > 1 and s.replacement.split()[0] != "none"
    }
    assert multiword <= set(NEGATIVE_QUANTIFIER_BIGRAMS)


def test_negative_quantifiers_exclude_negative_adverbs():
    """*never* is family C even though B reaches it from *always*."""
    assert not (NEGATIVE_QUANTIFIERS & set(NEGATIVE_ADVERBS))


def test_every_negative_adverb_has_an_affirmation():
    assert set(NEGATIVE_ADVERBS) == set(NEG_ADVERB_AFFIRMATIONS)


def test_deletion_is_offered_for_every_negative_adverb():
    assert all("" in forms for forms in NEG_ADVERB_AFFIRMATIONS.values())


def test_every_privative_phrase_contains_a_detectable_head():
    for phrase, _ in WITH_REPLACEMENTS + COPULAR_PRIVATIVES:
        assert set(phrase.split()) & PRIVATIVE_HEADS, phrase


def test_implicit_trigger_lemmas_cover_the_trigger_table():
    expected = {t.lemma for t in IMPLICIT_TRIGGERS if t.lemma != "be"}
    assert expected <= IMPLICIT_TRIGGER_LEMMAS
    assert LACK_TRIGGER.lemma in IMPLICIT_TRIGGER_LEMMAS
    # The copula is only a trigger together with its predicate, so it is
    # detected through IMPLICIT_TRIGGER_PREDICATES instead.
    assert "be" not in IMPLICIT_TRIGGER_LEMMAS
    assert "unable" in IMPLICIT_TRIGGER_PREDICATES


def test_negative_aux_bases_invert_the_contracted_table():
    for base, negated in CONTRACTED_NEGATIVE_AUX.items():
        assert NEGATIVE_AUX_BASES[negated] == base


def test_every_split_reading_modal_has_a_modality():
    assert set(MODAL_NEGATION_READINGS) <= set(MODALITY_BY_MODAL)


def test_split_readings_are_distinct_per_modal():
    for modal, readings in MODAL_NEGATION_READINGS.items():
        surfaces = [r.surface for r in readings]
        subtypes = [r.subtype for r in readings]
        assert len(set(surfaces)) == len(surfaces), modal
        assert len(set(subtypes)) == len(subtypes), modal
        assert len({r.reading for r in readings}) == len(readings), modal
