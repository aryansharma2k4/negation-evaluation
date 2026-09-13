"""Antonym pairs from WordNet, split so that no test lemma is ever seen in training.

Three things here are easy to get subtly wrong, and each of them inflates the
headline number if you do.

**The split must be by lemma, not by pair.** Antonymy is symmetric and often
many-to-one: *good* is the antonym of *bad*, *evil* and *ill*. Splitting pairs at
random puts *good/bad* in training and *good/evil* in test, so a model that has
memorised where *good* points scores on test without generalising at all.
:func:`lemma_disjoint_split` partitions the antonymy *graph* into connected
components and assigns whole components, which guarantees the stated property --
no test lemma appears in training under any pairing -- and, because most
components are a single edge, costs almost no data.

**Morphological pairs are a different problem from suppletive ones.**
*happy/unhappy* is recoverable by string edit; *hot/cold* is not. Every pair is
tagged :data:`MORPHOLOGICAL` or :data:`SUPPLETIVE`, using the same WordNet-gated
affix machinery the generators use, and every metric is reported split by it.

The split was added on the expectation that morphological pairs would be the
easy ones. Measured, the opposite holds -- the best method scores 0.247 P@1 on
suppletive pairs against 0.157 on morphological ones. Word frequency dominates
formation type: *hot* and *cold* are common and have well-estimated vectors,
while *unabridged* and *nonadsorbent* are rare and do not. The reporting split
is still worth keeping, just not for the reason it was introduced.

**Controls matter more than usual here.** The characteristic failure of
distributional antonymy is returning a *synonym*, because synonyms and antonyms
share contexts. So a synonym set and a random-pair set are built alongside, not
as an afterthought: without them a nearest-neighbour hit rate is uninterpretable.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from nltk.corpus import wordnet as wn

from ..affixes import is_negative_derivative
from ..nlp_core import wn_pos as _wn_pos_of_spacy  # noqa: F401  (kept for symmetry)

#: WordNet POS codes we build pairs for.  Nouns are excluded: the brief asks for
#: adjective, verb and adverb antonymy, and nominal "antonyms" in WordNet are
#: mostly complementary pairs (*man/woman*) whose embeddings behave differently.
POS_ADJ, POS_VERB, POS_ADV = "a", "v", "r"
POS_TAGS: tuple[str, ...] = (POS_ADJ, POS_VERB, POS_ADV)

#: Human-readable POS names, for reports.
POS_NAMES = {POS_ADJ: "adjective", POS_VERB: "verb", POS_ADV: "adverb"}

#: How the antonym relates to its base form.
MORPHOLOGICAL = "morphological"
SUPPLETIVE = "suppletive"

#: Affix pairs that make an antonym recoverable by string edit alone.
_NEGATIVE_PREFIXES = ("un", "in", "im", "ir", "il", "non", "dis", "a", "anti", "de")
_SUFFIX_PAIR = ("ful", "less")


@dataclass(frozen=True)
class Pair:
    """One antonym pair, with the POS it holds under and how it is formed."""

    word: str
    antonym: str
    pos: str
    relation: str

    def key(self) -> tuple[str, str, str]:
        a, b = sorted((self.word, self.antonym))
        return (a, b, self.pos)


@dataclass
class Dataset:
    """Train/test antonym pairs plus the two control conditions."""

    train: list[Pair] = field(default_factory=list)
    test: list[Pair] = field(default_factory=list)
    #: Synonym pairs, for measuring contamination.  Never used for fitting.
    synonyms: list[Pair] = field(default_factory=list)
    #: Randomly matched pairs, the floor any method must clear.
    random_pairs: list[Pair] = field(default_factory=list)

    def train_lemmas(self) -> set[str]:
        return {w for p in self.train for w in (p.word, p.antonym)}

    def test_lemmas(self) -> set[str]:
        return {w for p in self.test for w in (p.word, p.antonym)}

    def synonyms_of(self, word: str, pos: str) -> set[str]:
        return {
            p.antonym
            for p in self.synonyms
            if p.word == word and p.pos == pos
        }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _clean(name: str) -> Optional[str]:
    """Reject multiword lemmas: they have no single embedding to look up."""
    lower = name.lower()
    if "_" in lower or "-" in lower or not lower.isalpha():
        return None
    return lower


def _morphology(word: str, antonym: str, pos: str) -> str:
    """``morphological`` when one form is a negative derivation of the other.

    Checked in both directions, and confirmed against WordNet's own antonymy
    link by :func:`negation.affixes.is_negative_derivative` rather than trusted
    on string shape alone -- otherwise *press/depress* and *part/apart* would be
    counted as morphological when the affix is not doing the negating.
    """
    wn_tag = {POS_ADJ: wn.ADJ, POS_VERB: wn.VERB, POS_ADV: wn.ADV}[pos]
    for stem, derived in ((word, antonym), (antonym, word)):
        if derived == stem:
            continue
        if derived.endswith(_SUFFIX_PAIR[1]) and stem.endswith(_SUFFIX_PAIR[0]):
            if derived[: -len(_SUFFIX_PAIR[1])] == stem[: -len(_SUFFIX_PAIR[0])]:
                return MORPHOLOGICAL
        for prefix in _NEGATIVE_PREFIXES:
            if derived == prefix + stem and is_negative_derivative(stem, derived, wn_tag):
                return MORPHOLOGICAL
    return SUPPLETIVE


def extract_pairs(pos_tags: Iterable[str] = POS_TAGS) -> list[Pair]:
    """Every single-word antonym pair WordNet records for ``pos_tags``.

    Emitted in both directions: the task is "given a word, produce its
    opposite", so *hot -> cold* and *cold -> hot* are two prediction problems
    even though WordNet stores one relation.  Adjective satellites (``s``) are
    folded into ``a``, which is how WordNet actually files gradable adjectives.
    """
    wanted = set(pos_tags)
    seen: set[tuple[str, str, str]] = set()
    out: list[Pair] = []

    for synset in wn.all_synsets():
        pos = POS_ADJ if synset.pos() in ("a", "s") else synset.pos()
        if pos not in wanted:
            continue
        for lemma in synset.lemmas():
            word = _clean(lemma.name())
            if word is None:
                continue
            for antonym_lemma in lemma.antonyms():
                antonym = _clean(antonym_lemma.name())
                if antonym is None or antonym == word:
                    continue
                relation = _morphology(word, antonym, pos)
                for a, b in ((word, antonym), (antonym, word)):
                    if (a, b, pos) in seen:
                        continue
                    seen.add((a, b, pos))
                    out.append(Pair(a, b, pos, relation))
    return out


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def _components(pairs: list[Pair]) -> list[set[str]]:
    """Connected components of the antonymy graph, per POS kept separate.

    Components, not individual lemmas, are the unit of assignment: *good* is
    antonymous with *bad*, *evil* and *ill*, and splitting that star would leak
    *good*'s direction from training into test.
    """
    adjacency: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        adjacency[pair.word].add(pair.antonym)
        adjacency[pair.antonym].add(pair.word)

    seen: set[str] = set()
    out: list[set[str]] = []
    for node in adjacency:
        if node in seen:
            continue
        stack, component = [node], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        seen |= component
        out.append(component)
    return out


def lemma_disjoint_split(
    pairs: list[Pair], test_fraction: float = 0.2, seed: int = 0
) -> tuple[list[Pair], list[Pair]]:
    """Split so no lemma occurs on both sides, under any pairing.

    Whole connected components are assigned, shuffled and accumulated until the
    test side holds ``test_fraction`` of the lemmas.  Because the assignment is
    by component, every pair lands wholly in one side and none is discarded --
    which a naive per-lemma random split could not manage: with most components
    being a single edge it would throw away roughly a third of the data to
    cross-boundary pairs and leave only ~4% as test.
    """
    components = _components(pairs)
    rng = random.Random(seed)
    rng.shuffle(components)

    total = sum(len(c) for c in components)
    target = int(round(total * test_fraction))
    test_lemmas: set[str] = set()
    for component in components:
        if len(test_lemmas) >= target:
            break
        test_lemmas |= component

    train = [p for p in pairs if p.word not in test_lemmas and p.antonym not in test_lemmas]
    test = [p for p in pairs if p.word in test_lemmas and p.antonym in test_lemmas]
    return train, test


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def extract_synonyms(
    lemmas: Iterable[str], pos_tags: Iterable[str] = POS_TAGS, max_per_word: int = 8
) -> list[Pair]:
    """Synonyms of ``lemmas`` from their own synsets.

    The contamination control.  A distributional model asked for the opposite of
    *hot* will often answer *warm*, because antonyms and synonyms occupy the same
    neighbourhood; without this list that failure is invisible, since *warm* is
    simply "not the gold answer" like any other miss.
    """
    wanted = set(pos_tags)
    out: list[Pair] = []
    seen: set[tuple[str, str, str]] = set()

    for word in lemmas:
        for synset in wn.synsets(word):
            pos = POS_ADJ if synset.pos() in ("a", "s") else synset.pos()
            if pos not in wanted:
                continue
            count = 0
            for lemma in synset.lemmas():
                candidate = _clean(lemma.name())
                if candidate is None or candidate == word:
                    continue
                key = (word, candidate, pos)
                if key in seen:
                    continue
                seen.add(key)
                out.append(Pair(word, candidate, pos, "synonym"))
                count += 1
                if count >= max_per_word:
                    break
    return out


def random_pairs(
    lemmas: Iterable[str], pos_tags: Iterable[str] = POS_TAGS, seed: int = 0
) -> list[Pair]:
    """Randomly matched pairs: the floor every method must clear.

    Any method scoring near this is doing nothing.  Sampled within POS so the
    control is not trivially easy to beat on part of speech alone.
    """
    rng = random.Random(seed)
    by_pos: dict[str, list[str]] = defaultdict(list)
    for word in lemmas:
        for synset in wn.synsets(word):
            pos = POS_ADJ if synset.pos() in ("a", "s") else synset.pos()
            if pos in set(pos_tags):
                by_pos[pos].append(word)
                break

    out: list[Pair] = []
    for pos, words in by_pos.items():
        if len(words) < 2:
            continue
        shuffled = words[:]
        rng.shuffle(shuffled)
        for word, other in zip(words, shuffled):
            if word != other:
                out.append(Pair(word, other, pos, "random"))
    return out


def build_dataset(
    pos_tags: Iterable[str] = POS_TAGS,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> Dataset:
    """The full experimental dataset: split antonyms plus both controls."""
    pairs = extract_pairs(pos_tags)
    train, test = lemma_disjoint_split(pairs, test_fraction, seed)
    lemmas = {w for p in pairs for w in (p.word, p.antonym)}
    return Dataset(
        train=train,
        test=test,
        synonyms=extract_synonyms(lemmas, pos_tags),
        random_pairs=random_pairs(lemmas, pos_tags, seed),
    )
