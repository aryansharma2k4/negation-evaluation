"""The antonym-vector module: data hygiene, method correctness, and the fallback.

The data and method tests run anywhere. Anything needing a 1 GB vector file
skips with a fetch hint.

The most important test here is :func:`test_methods_recover_a_synthetic_reflection`.
The real-data numbers in ``docs/antonym_vectors.md`` are poor, and a poor number
is only worth reporting if the implementation is known to be correct -- so the
learnable methods are checked against a task that genuinely *is* a linear
reflection, where they must score near-perfectly. If that test passes and the
real numbers stay low, the fault is the embedding space, not the gradient.
"""

from __future__ import annotations

import numpy as np
import pytest

from negation.antonym_vec.data import (
    MORPHOLOGICAL,
    POS_TAGS,
    SUPPLETIVE,
    Pair,
    build_dataset,
    extract_pairs,
    lemma_disjoint_split,
)
from negation.antonym_vec.embeddings import EmbeddingsNotFetched, load_static
from negation.antonym_vec.evaluate import (
    Result,
    gold_antonyms,
    slice_pairs,
)
from negation.antonym_vec.methods import (
    BaselineNegate,
    BaselineWordNet,
    Candidates,
    CounterFitting,
    LinearMap,
    MLPMap,
    Reflection,
    l2_normalise,
)


# -- data -------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset():
    return build_dataset(POS_TAGS, test_fraction=0.2, seed=0)


def test_pairs_are_extracted_in_both_directions():
    pairs = extract_pairs(("a",))
    index = {(p.word, p.antonym) for p in pairs}
    assert ("good", "bad") in index and ("bad", "good") in index


def test_no_multiword_lemmas():
    for pair in extract_pairs(POS_TAGS):
        assert "_" not in pair.word and "_" not in pair.antonym


def test_split_is_lemma_disjoint(dataset):
    """The property the whole evaluation rests on."""
    assert not (dataset.train_lemmas() & dataset.test_lemmas())


def test_no_test_lemma_appears_in_any_training_pair(dataset):
    """Stronger restatement: not merely disjoint sets, but no pairing either."""
    test_lemmas = dataset.test_lemmas()
    for pair in dataset.train:
        assert pair.word not in test_lemmas
        assert pair.antonym not in test_lemmas


def test_split_keeps_roughly_the_requested_fraction(dataset):
    total = len(dataset.train_lemmas()) + len(dataset.test_lemmas())
    assert 0.12 < len(dataset.test_lemmas()) / total < 0.30


def test_component_split_does_not_throw_pairs_away(dataset):
    """Assigning whole components means no pair straddles the boundary."""
    everything = extract_pairs(POS_TAGS)
    assert len(dataset.train) + len(dataset.test) == len(everything)


def test_a_star_of_antonyms_stays_on_one_side():
    """*good* is antonymous with several words; they must not be split."""
    pairs = extract_pairs(("a",))
    train, test = lemma_disjoint_split(pairs, 0.2, seed=0)
    partners = {p.antonym for p in pairs if p.word == "good"}
    train_lemmas = {w for p in train for w in (p.word, p.antonym)}
    test_lemmas = {w for p in test for w in (p.word, p.antonym)}
    for partner in partners:
        assert not (partner in train_lemmas and partner in test_lemmas)


def test_morphological_and_suppletive_are_both_present(dataset):
    relations = {p.relation for p in dataset.train}
    assert relations == {MORPHOLOGICAL, SUPPLETIVE}


@pytest.mark.parametrize("word,antonym", [
    ("happy", "unhappy"), ("possible", "impossible"), ("active", "inactive"),
])
def test_affixal_pairs_are_tagged_morphological(word, antonym):
    pairs = {(p.word, p.antonym): p for p in extract_pairs(("a",))}
    pair = pairs.get((word, antonym))
    if pair is None:
        pytest.skip(f"{word}/{antonym} not in this WordNet build")
    assert pair.relation == MORPHOLOGICAL


@pytest.mark.parametrize("word,antonym", [("hot", "cold"), ("good", "bad")])
def test_suppletive_pairs_are_tagged_suppletive(word, antonym):
    pairs = {(p.word, p.antonym): p for p in extract_pairs(("a",))}
    pair = pairs.get((word, antonym))
    if pair is None:
        pytest.skip(f"{word}/{antonym} not in this WordNet build")
    assert pair.relation == SUPPLETIVE


def test_controls_are_built_and_disjoint_from_gold(dataset):
    assert dataset.synonyms and dataset.random_pairs
    gold = {(p.word, p.antonym) for p in dataset.train + dataset.test}
    overlap = {(p.word, p.antonym) for p in dataset.synonyms} & gold
    # A handful of WordNet lemmas are listed as both; it must be a trickle.
    assert len(overlap) / len(dataset.synonyms) < 0.02


# -- methods ----------------------------------------------------------------


class _SyntheticSource:
    """A space in which antonymy really is a reflection across one hyperplane."""

    dim = 32

    def __init__(self, n: int = 400, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.normal = rng.normal(size=self.dim)
        self.normal /= np.linalg.norm(self.normal)
        self.base = l2_normalise(rng.normal(size=(n // 2, self.dim)))
        self.words = [f"w{i}" for i in range(n)]
        self.n = n

    def vector(self, word, pos="a", context=None):
        i = int(word[1:])
        if i < self.n // 2:
            return self.base[i]
        v = self.base[i - self.n // 2]
        return v - 2 * (v @ self.normal) * self.normal

    def vocabulary(self):
        return self.words


@pytest.fixture(scope="module")
def synthetic():
    source = _SyntheticSource()
    half = source.n // 2
    pairs = [Pair(f"w{i}", f"w{i+half}", "a", SUPPLETIVE) for i in range(half)]
    train, test = pairs[: int(half * 0.8)], pairs[int(half * 0.8) :]
    matrix = np.stack([source.vector(w) for w in source.words])
    candidates = Candidates(tuple(source.words), matrix)
    return source, train, test, candidates


@pytest.mark.parametrize("factory", [
    lambda: LinearMap(alpha=0.1),
    lambda: MLPMap(epochs=120, seed=0),
    lambda: Reflection(steps=600, seed=0),
])
def test_methods_recover_a_synthetic_reflection(synthetic, factory):
    """Correctness check for the learnable maps.

    If this fails the real-data numbers mean nothing; if it passes and the real
    numbers are still poor, the embedding space is the problem.
    """
    source, train, test, candidates = synthetic
    method = factory()
    method.fit(train, source)
    hits = 0
    for pair in test:
        ranked = method.suggest(pair.word, pair.pos, source, candidates, top=1)
        if ranked and ranked[0][0] == pair.antonym:
            hits += 1
    assert hits / len(test) > 0.9, f"{method.name} only got {hits}/{len(test)}"


def test_baseline_negate_returns_the_negated_vector(synthetic):
    source, _, _, _ = synthetic
    predicted = BaselineNegate().predict("w0", "a", source)
    assert np.allclose(predicted, -np.asarray(source.vector("w0"), dtype=np.float64))


def test_candidates_rank_excludes_the_query(synthetic):
    source, _, _, candidates = synthetic
    vector = np.asarray(source.vector("w0"), dtype=np.float64)
    ranked = candidates.rank(vector, exclude=("w0",), top=3)
    assert "w0" not in [w for w, _ in ranked]


def test_candidates_rank_of_is_consistent_with_rank(synthetic):
    source, _, _, candidates = synthetic
    vector = np.asarray(source.vector("w0"), dtype=np.float64)
    top = candidates.rank(vector, exclude=("w0",), top=1)[0][0]
    assert candidates.rank_of(vector, top, exclude=("w0",)) == 1


def test_wordnet_baseline_returns_real_antonyms():
    got = BaselineWordNet().suggest("good", "a", None, None, top=10)
    assert "bad" in [w for w, _ in got]


def test_wordnet_baseline_is_empty_for_an_uncovered_word():
    """The gap the vector fallback exists to fill."""
    assert BaselineWordNet().suggest("sonorous", "a", None, None, top=5) == []


def test_counterfitting_pushes_constrained_antonyms_apart(synthetic):
    source, train, _, candidates = synthetic
    cf = CounterFitting(steps=30, lr=0.2)
    ant = [(p.word, p.antonym) for p in train[:50]]
    fitted = cf.fit_space(candidates.words, candidates.matrix, ant, [])
    index = {w: i for i, w in enumerate(candidates.words)}
    before = [
        float(candidates.matrix[index[a]] @ candidates.matrix[index[b]]) for a, b in ant
    ]
    after = [float(fitted[index[a]] @ fitted[index[b]]) for a, b in ant]
    assert np.mean(after) < np.mean(before)


def test_counterfitting_leaves_unconstrained_words_where_they_were(synthetic):
    """The structural reason it cannot help unseen lemmas.

    Vectors move only under the constraints they appear in; a word in none moves
    only under the space-preservation term, which pulls it back to its origin.
    """
    source, train, test, candidates = synthetic
    cf = CounterFitting(steps=30, lr=0.2)
    ant = [(p.word, p.antonym) for p in train[:50]]
    fitted = cf.fit_space(candidates.words, candidates.matrix, ant, [])
    index = {w: i for i, w in enumerate(candidates.words)}
    constrained = {w for pair in ant for w in pair}
    untouched = [w for w in candidates.words if w not in constrained][:40]
    similarity = [
        float(fitted[index[w]] @ candidates.matrix[index[w]]) for w in untouched
    ]
    assert min(similarity) > 0.99


# -- evaluation -------------------------------------------------------------


def test_gold_antonyms_collects_every_partner():
    pairs = [Pair("good", "bad", "a", SUPPLETIVE), Pair("good", "evil", "a", SUPPLETIVE)]
    assert gold_antonyms(pairs)[("good", "a")] == {"bad", "evil"}


def test_slices_cover_pos_and_formation():
    pairs = [
        Pair("happy", "unhappy", "a", MORPHOLOGICAL),
        Pair("rise", "fall", "v", SUPPLETIVE),
    ]
    names = set(slice_pairs(pairs))
    assert "all" in names
    assert {"pos=adjective", "pos=verb"} <= names
    assert {f"form={MORPHOLOGICAL}", f"form={SUPPLETIVE}"} <= names


def test_result_rates_are_zero_safe():
    empty = Result(method="m", slice_name="s")
    assert empty.p_at_1 == 0.0 and empty.coverage == 0.0
    assert empty.mean_rank is None


def test_result_reports_synonym_contamination_beside_precision():
    """Precision alone cannot distinguish a near-miss from a wrong answer."""
    result = Result(method="m", slice_name="s", n=10, hits_at_1=2, synonym_hits_at_1=5)
    assert result.p_at_1 == 0.2
    assert result.synonym_at_1 == 0.5
    assert "synonym@1" in result.as_dict()


# -- integration ------------------------------------------------------------


def test_fallback_is_off_by_default(monkeypatch):
    """A clean checkout has no vectors; the pipeline must not care."""
    from negation.antonym_vec import api

    monkeypatch.delenv(api.ENABLE_ENV, raising=False)
    api.reset_cache()
    assert api.generate_antonym("sonorous", "a") is None
    assert not api.is_available()
    api.reset_cache()


def test_e_antonym_still_produces_wordnet_records():
    """The fallback must not disturb the path that already worked."""
    from negation.driver import generate_all

    variants = {r.variant for r in generate_all(["The score increased."])}
    assert "The score decreased." in variants


def test_wordnet_records_carry_no_confidence():
    """Confidence marks model output; rule output is either right or absent."""
    from negation.driver import generate_all

    for record in generate_all(["The score increased."]):
        if record.generator == "antonym_wordnet_v1":
            assert record.confidence is None


def test_schema_rejects_confidence_outside_the_unit_interval():
    from negation.schema import NegationVariant

    with pytest.raises(ValueError):
        NegationVariant(
            base_id="b", base_sentence="x", variant="y", family="E_antonym",
            subtype="s", cue_tokens=[], cue_char_spans=[], cue_count=0,
            net_negation=1, scope_target="clause", intensity_hint="neutral",
            generator="g", confidence=1.5,
        )


# -- anything needing the vector file ---------------------------------------


def _load_or_skip():
    try:
        return load_static(restrict_to=["hot", "cold", "good", "bad"], max_vocab=20_000)
    except EmbeddingsNotFetched as exc:
        pytest.skip(str(exc).splitlines()[0])


def test_static_vectors_load_and_are_finite():
    source = _load_or_skip()
    vector = source.vector("hot")
    assert vector is not None and np.isfinite(vector).all()
    assert source.dim > 0


def test_antonyms_are_close_together_in_the_raw_space():
    """The premise of the whole exercise, asserted rather than assumed.

    If this ever fails, the distributional problem has gone away and the module
    should be re-evaluated from scratch.
    """
    source = _load_or_skip()
    hot, cold = source.vector("hot"), source.vector("cold")
    cosine = float(hot @ cold) / float(np.linalg.norm(hot) * np.linalg.norm(cold))
    assert cosine > 0.3, "antonyms are no longer close; re-run the whole evaluation"
