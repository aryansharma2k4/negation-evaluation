"""The corpus loaders and the family mapping onto our taxonomy.

The mapping tests run anywhere.  The loader tests need the corpora, which are
not vendored -- licences differ and two state none at all -- so they skip with a
message telling you how to fetch them rather than failing on a clean checkout.
"""

from __future__ import annotations

import pytest

from negation.data.family_map import cue_family, normalise_cue, scope_position_family
from negation.data.loaders import LOADERS, CorpusNotFetched
from negation.data.schema import (
    ANN_NEGATION,
    ANN_SPECULATION,
    PROV_PAIR_ONLY,
    UNMAPPED,
    AnnotatedSentence,
)
from negation.data.sources import CORPORA, STATUS_DEAD, STATUS_OK
from negation.schema import FAMILIES


# -- schema -----------------------------------------------------------------


def test_spans_outside_the_token_list_are_refused():
    with pytest.raises(ValueError):
        AnnotatedSentence("a b", ["a", "b"], [(0, 5)], [], "A_syntactic", "x", "train")


def test_unknown_annotation_type_is_refused():
    with pytest.raises(ValueError):
        AnnotatedSentence(
            "a", ["a"], [], [], "A_syntactic", "x", "train", annotation_type="bogus"
        )


def test_discontinuous_spans_expand_to_indices():
    record = AnnotatedSentence(
        "a b c d", ["a", "b", "c", "d"], [(0, 1), (3, 4)], [(1, 3)],
        "A_syntactic", "x", "train",
    )
    assert record.cue_token_indices == [0, 3]
    assert record.scope_token_indices == [1, 2]


# -- family mapping ---------------------------------------------------------


@pytest.mark.parametrize("cue,token,expected", [
    ("not", "not", "A_syntactic"),
    ("cannot", "cannot", "A_syntactic"),
    ("no", "no", "B_quantifier"),
    ("nobody", "nobody", "B_quantifier"),
    ("neither", "neither", "B_quantifier"),
    ("never", "never", "C_neg_adverb"),
    ("rarely", "rarely", "C_neg_adverb"),
    ("without", "without", "G_prepositional"),
    ("fails", "fails", "F_implicit"),
    ("lack", "lack", "F_implicit"),
])
def test_single_word_cues_map_to_their_family(cue, token, expected):
    assert cue_family(cue, token) == expected


@pytest.mark.parametrize("cue,token", [("in", "infrequent"), ("un", "unhappy"),
                                       ("less", "useless"), ("im", "impossible")])
def test_an_affix_cue_maps_to_affixal(cue, token):
    """CD-SCO annotates the affix alone, not the word carrying it."""
    assert cue_family(cue, token) == "D_affixal"


def test_a_whole_word_is_never_read_as_its_own_affix():
    """*in* as a preposition is a whole token, so it is not an affixal cue."""
    assert cue_family("in", "in") != "D_affixal"


@pytest.mark.parametrize("cue", ["do n't", "does n't", "ca n't", "can not"])
def test_split_contractions_still_map(cue):
    """SFU and BioScope tokenise contractions apart; the head word is not the cue."""
    assert cue_family(cue, cue) == "A_syntactic"


def test_curly_apostrophes_are_normalised():
    """The SFU XML carries Windows-1252 apostrophes that survive parsing."""
    assert normalise_cue("don’t") == "don't"
    assert cue_family("does n’t", "does n’t") == "A_syntactic"


@pytest.mark.parametrize("phrase,expected", [
    ("by no means", "I_intensified"),
    ("far from", "J_contrastive"),
    ("anything but", "J_contrastive"),
])
def test_phrases_beat_their_component_words(phrase, expected):
    """*by no means* is emphatic, not the quantifier *no*."""
    assert cue_family(phrase, phrase) == expected


def test_speculation_cues_route_to_the_hedged_family():
    assert cue_family("may", "may", annotation_type=ANN_SPECULATION) == "H_hedged"
    assert cue_family("suggest", "suggest", annotation_type=ANN_SPECULATION) == "H_hedged"


@pytest.mark.parametrize("cue", ["rather than", "except", "on the contrary"])
def test_constructions_outside_our_taxonomy_stay_unmapped(cue):
    """Exceptives have no family here; saying so beats forcing a wrong label."""
    assert cue_family(cue, cue) == UNMAPPED


def test_empty_cue_is_unmapped():
    assert cue_family("", None) == UNMAPPED


def test_every_mapped_family_is_one_of_ours():
    probes = ["not", "no", "never", "without", "fails", "by no means", "far from"]
    for cue in probes:
        family = cue_family(cue, cue)
        assert family in FAMILIES or family == UNMAPPED


def test_preposed_negator_on_a_quantifier_becomes_scope_position():
    assert scope_position_family(
        "A_syntactic", ["Not", "everyone", "passed"], [0]
    ) == "L_scope_position"


def test_scope_position_does_not_relabel_other_families():
    assert scope_position_family(
        "B_quantifier", ["No", "everyone", "x"], [0]
    ) == "B_quantifier"


# -- sources registry -------------------------------------------------------


def test_every_corpus_has_a_loader():
    assert set(CORPORA) == set(LOADERS)


def test_every_corpus_records_where_its_licence_came_from():
    for corpus in CORPORA.values():
        assert corpus.licence_source, corpus.key
        # A corpus with no licence must not be marked redistributable.
        if corpus.licence is None:
            assert not corpus.redistributable, corpus.key


def test_dead_sources_are_kept_and_marked():
    """The URLs that stopped working are recorded, not quietly dropped."""
    dead = [
        s for c in CORPORA.values() for s in c.sources if s.status == STATUS_DEAD
    ]
    assert dead, "the dead *SEM and BioScope URLs should still be documented"
    assert all(s.note for s in dead)


def test_every_corpus_has_at_least_one_working_source():
    for corpus in CORPORA.values():
        assert any(s.status == STATUS_OK for s in corpus.sources), corpus.key


# -- loaders (need the corpora) ---------------------------------------------


def _load(key):
    try:
        return LOADERS[key]()
    except CorpusNotFetched as exc:
        pytest.skip(str(exc).splitlines()[0])


@pytest.mark.parametrize("key", sorted(LOADERS))
def test_loader_produces_wellformed_records(key):
    records = _load(key)
    assert records
    for record in records[:500]:
        assert record.sentence and record.tokens
        assert record.source_dataset and record.split and record.sentence_id
        assert record.annotation_type in (ANN_NEGATION, ANN_SPECULATION)
        assert record.cue_family in FAMILIES or record.cue_family == UNMAPPED
        for start, end in record.cue_spans + record.scope_spans:
            assert 0 <= start < end <= len(record.tokens)


@pytest.mark.parametrize("key", ["cdsco", "sfu", "bioscope"])
def test_span_corpora_carry_gold_cues(key):
    records = _load(key)
    assert all(r.provenance == "gold" for r in records)
    assert all(r.cue_spans for r in records), "a gold record must locate its cue"


@pytest.mark.parametrize("key", ["cannot", "negnli", "nannli"])
def test_pair_corpora_declare_that_they_have_no_spans(key):
    """Pair-level corpora must not pretend to span annotation."""
    records = _load(key)
    assert all(r.provenance == PROV_PAIR_ONLY for r in records)
    assert all(not r.cue_spans and not r.scope_spans for r in records)
    assert all(r.extra.get("premise") is not None for r in records[:50])


@pytest.mark.parametrize("key,expected", [
    ("cdsco", 1421), ("sfu", 8891), ("negnli", 4500), ("nannli", 258),
])
def test_record_counts_match_what_was_verified(key, expected):
    """Pinned against the counts in docs/dataset_evaluation.md."""
    assert len(_load(key)) == expected


def test_cue_text_matches_the_tokens_it_points_at():
    """Except for affixal cues, where the corpus records only the affix."""
    for record in _load("sfu")[:300]:
        spanned = " ".join(record.tokens[i] for i in record.cue_token_indices)
        assert normalise_cue(record.cue_text) == normalise_cue(spanned)


@pytest.mark.parametrize("key", ["sfu", "bioscope"])
def test_the_hedging_corpora_separate_speculation_from_negation(key):
    """The whole reason these two are worth having."""
    types = {r.annotation_type for r in _load(key)}
    assert types == {ANN_NEGATION, ANN_SPECULATION}


def test_cdsco_has_no_speculation_layer():
    """Negation only -- it cannot train the intensity regressor."""
    assert {r.annotation_type for r in _load("cdsco")} == {ANN_NEGATION}


def test_bioscope_refuses_its_redacted_clinical_subcorpus():
    """The distributed clinical XML annotates text it does not contain."""
    from negation.data.loaders import RedactedSubcorpus, load_bioscope

    try:
        load_bioscope("abstracts")
    except CorpusNotFetched as exc:
        pytest.skip(str(exc).splitlines()[0])
    with pytest.raises(RedactedSubcorpus):
        load_bioscope("clinical")


def test_bioscope_default_excludes_the_redacted_subcorpus():
    records = _load("bioscope")
    assert {r.extra["subcorpus"] for r in records} == {"abstracts", "full_papers"}
