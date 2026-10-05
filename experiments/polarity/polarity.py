"""Sentence-level negation polarity: an arbitrary sentence -> a score in [0, 1].

    from polarity import polarity
    polarity("The task is not impossible.", "v2")   # ~0: the two cues cancel

Built from the project's own machinery, not a lookup:
  - cues:         negation.classify.existing_cues (every cue family the
                  generator knows, WordNet-gated affixes, multiword cues once),
                  plus the contrastive frames from negation.lexicons
  - clauses:      negation.nlp_core.clause_scope -- two cues cancel only if
                  they sit in the same clause scope
  - modifiers:    negation.lexicons.HEDGES / INTENSIFIERS / NEGATIVE_ADVERBS

A sentence is parsed once and reduced to a list of cues with their clause and
modifiers (``extract``); each scorer version is a pure function of that list
(``score``). ``polarity`` does both, so it works on any sentence.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from negation.classify import existing_cues  # noqa: E402
from negation.lexicons import (  # noqa: E402
    CONTRASTIVE_FRAMES, HEDGES, INTENSIFIERS, NEGATIVE_ADVERBS,
)
from negation.nlp_core import CLAUSE_BOUNDARY_DEPS, clause_scope, parse_batch  # noqa: E402
from negation.schema import (  # noqa: E402
    FAM_AFFIXAL, FAM_CONTRASTIVE, FAM_IMPLICIT, FAM_NEG_ADVERB, FAM_PREPOSITIONAL,
    FAM_QUANTIFIER, FAM_SYNTACTIC,
)

HEDGE_WORDS = frozenset(m.text for m in HEDGES)
INTENSIFIER_WORDS = frozenset(m.text for m in INTENSIFIERS if " " not in m.text)
PARTIAL_ADVERBS = frozenset(w for w, (sub, _) in NEGATIVE_ADVERBS.items() if sub == "partial")


@dataclass(frozen=True)
class Cue:
    text: str             # lower-cased cue text
    family: str           # which generator family would have produced it
    clause: int           # token index of the clause head whose scope holds it
    main_clause: bool     # that clause is the root clause
    hedged: bool          # a hedge (maybe, probably, might, ...) in the same clause
    intensified: bool     # an intensifier in the same clause, or by no means / not at all


@dataclass(frozen=True)
class Analysis:
    cues: tuple[Cue, ...] = field(default_factory=tuple)
    n_tokens: int = 0


def _clause_head(token):
    """Nearest ancestor (or self) that heads a clause: a boundary edge or the root."""
    # spaCy builds a new Token object per access, so compare indices, not identity
    while token.dep_ not in CLAUSE_BOUNDARY_DEPS and token.head.i != token.i:
        token = token.head
    return token


def extract(doc) -> Analysis:
    lower = [t.lower_ for t in doc]
    raw = [(c.token_idx, c.cue_text.lower(), c.family_guess) for c in existing_cues(doc)]
    text = doc.text.lower()
    for phrase, _ in CONTRASTIVE_FRAMES:
        start = text.find(phrase)
        if start >= 0:
            span = doc.char_span(start, start + len(phrase), alignment_mode="expand")
            if span is not None:
                raw.append((span.start, phrase, FAM_CONTRASTIVE))

    cues = []
    for i, cue_text, family in raw:
        head = _clause_head(doc[i])
        scope = clause_scope(head)
        assert i in scope
        words = {lower[j] for j in scope}
        by_no_means = cue_text == "no" and i > 0 and lower[i - 1] == "by" \
            and i + 1 < len(doc) and lower[i + 1] == "means"
        not_at_all = cue_text in ("not", "n't") and lower[i + 1:i + 3] == ["at", "all"]
        cues.append(Cue(
            text=cue_text, family=family, clause=head.i, main_clause=head.dep_ == "ROOT",
            hedged=bool(words & HEDGE_WORDS),
            intensified=bool(words & INTENSIFIER_WORDS) or by_no_means or not_at_all,
        ))
    return Analysis(tuple(cues), len(doc))


def analyse_many(sentences: list[str]) -> dict[str, Analysis]:
    unique = list(dict.fromkeys(sentences))
    return {s: extract(doc) for s, doc in zip(unique, parse_batch(unique))}


# ---------------------------------------------------------------------------
# Scorer versions. Each: Analysis -> [0, 1].
# ---------------------------------------------------------------------------

def soft_xor(weights: list[float]) -> float:
    """Parity with graded cues: one cue of weight w -> w; two full cues -> 0.

    (1 - prod(1 - 2w)) / 2. Two cues of weight 1 in one scope cancel exactly;
    *not* (0.9) over *impossible* (0.8) leaves 0.26; three cues flip back.
    """
    prod = 1.0
    for w in weights:
        prod *= 1 - 2 * w
    return (1 - prod) / 2


def by_clause(cues) -> dict[int, list[Cue]]:
    out: dict[int, list[Cue]] = {}
    for c in cues:
        out.setdefault(c.clause, []).append(c)
    return out


def v1(a: Analysis, _w=None) -> float:
    """Any cue present -> 1."""
    return 1.0 if a.cues else 0.0


def v2(a: Analysis, _w=None) -> float:
    """Cues in one clause scope combine by parity; the sentence takes its most
    negated clause."""
    return max((soft_xor([1.0] * len(cs)) for cs in by_clause(a.cues).values()), default=0.0)


INTENSITY = {"hedged": 0.6, "neutral": 0.9, "intensified": 1.0, "partial": 0.4}


def intensity_of(c: Cue) -> str:
    if c.text in PARTIAL_ADVERBS:
        return "partial"
    if c.intensified:
        return "intensified"
    if c.hedged:
        return "hedged"
    return "neutral"


def v3(a: Analysis, w=None) -> float:
    """v2, each cue weighted by its intensity."""
    w = w or INTENSITY
    return max((soft_xor([w[intensity_of(c)] for c in cs]) for cs in by_clause(a.cues).values()),
               default=0.0)


#: v4: explicit cues keep the intensity weight; lexical ones get their own
FAMILY_WEIGHT = {FAM_SYNTACTIC: 1.0, FAM_NEG_ADVERB: 1.0, FAM_QUANTIFIER: 1.0,
                 FAM_IMPLICIT: 0.8, FAM_AFFIXAL: 0.7, FAM_PREPOSITIONAL: 0.8, FAM_CONTRASTIVE: 0.8}


def cue_weight_v4(c: Cue, fam=None, inten=None) -> float:
    fam, inten = fam or FAMILY_WEIGHT, inten or INTENSITY
    return fam[c.family] * inten[intensity_of(c)]


def v4(a: Analysis, w=None) -> float:
    w = w or {}
    return max((soft_xor([cue_weight_v4(c, w.get("family"), w.get("intensity")) for c in cs])
                for cs in by_clause(a.cues).values()), default=0.0)


def v5(a: Analysis, w=None) -> float:
    """Parity first, weight second. An even number of cues in a clause cancels
    exactly; an odd number leaves the weight of the strongest cue (v4 weights).

    v4 weighted each cue *inside* the parity, so two cues of weight < 1 no
    longer cancel: *not impossible* kept 0.26-0.4 and the K1 false-positive
    rate rose from 0.05 to 0.24.
    """
    w = w or {}
    best = 0.0
    for cs in by_clause(a.cues).values():
        if len(cs) % 2:
            best = max(best, max(cue_weight_v4(c, w.get("family"), w.get("intensity")) for c in cs))
    return best


VERSIONS = {"v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5}


def score(a: Analysis, version: str, weights=None) -> float:
    return float(VERSIONS[version](a, weights))


@lru_cache(maxsize=4096)
def _analysis(sentence: str) -> Analysis:
    return extract(parse_batch([sentence])[0])


def polarity(sentence: str, version: str = "v4", weights=None) -> float:
    """How negated ``sentence`` is, 0.0 (affirmative) to 1.0 (negated)."""
    return score(_analysis(sentence), version, weights)


#: versions whose weights are fit on DEV: name -> the version they parameterise
#: and the hand-set weights the search starts from
FITTED: dict[str, dict] = {
    # v5's family and intensity weights, fit on DEV instead of hand-set
    "v6": {"base": "v5", "start": {"family": dict(FAMILY_WEIGHT), "intensity": dict(INTENSITY)}},
}

#: the worked table in the report
WORKED_EXAMPLES = [
    "The function sorts the array.",
    "The function does not sort the array.",
    "The function doesn't sort the array.",
    "The function probably does not sort the array.",
    "The function definitely does not sort the array.",
    "The function rarely sorts the array.",
    "The function never sorts the array.",
    "The function fails to sort the array.",
    "The function never fails to sort the array.",
    "The task is impossible.",
    "The task is not impossible.",
    "The result is unimportant.",
    "The result is anything but simple.",
    "It runs without a cache.",
    "Nobody passed the exam.",
    "The function sorts the array and does not return the result.",
    "The function does not sort the array and does not return the result.",
]
