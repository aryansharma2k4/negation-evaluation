"""Pair-level rules for Phase 2 (and the conditional logic of Phase 3 builds on the same parse).

Every sentence is parsed once (negation.nlp_core.parse_batch) and reduced to light token
records, cached in .cache/nli2/parse.pkl:

    Tok(lower, lemma, pos, dep, head, i)

Alignment is difflib's SequenceMatcher over lemma sequences, the same tool
negation_detection used to recover CANNOT's negation types.
"""

from __future__ import annotations

import pickle
from difflib import SequenceMatcher
from functools import lru_cache
from typing import NamedTuple

from common import CACHE, REPO  # noqa: F401  (REPO on sys.path for negation.*)

from negation.nlp_core import antonyms, parse_batch  # noqa: E402

NEG_WORDS = frozenset({"not", "n't", "never", "no", "cannot", "nobody", "nothing", "none", "neither", "nor"})
DETERMINERS = frozenset({"a", "an", "the"})

#: Antonym pairs WordNet misses, found as failing aligned antonym negations on the clean
#: DEV set only (never VALIDATION or TEST), step 2.3. The brief's four (alive/dead,
#: present/absent, include/exclude, same/different) are already in WordNet; listed anyway.
CURATED_ANTONYMS: frozenset[frozenset[str]] = frozenset(map(frozenset, [
    ("alive", "dead"), ("present", "absent"), ("include", "exclude"), ("same", "different"),
    ("right", "incorrect"), ("wrong", "correct"), ("hard", "easy"), ("important", "insignificant"),
    ("important", "noncrucial"), ("serious", "playful"), ("large", "tiny"), ("increase", "diminish"),
    ("increase", "decline"), ("continue", "stop"), ("cause", "prevent"), ("broad", "narrow"),
    ("sure", "uncertain"), ("true", "untruthful"), ("predecessor", "successor"),
    ("stupid", "intelligent"), ("open", "covert"), ("rarely", "commonly"), ("seldom", "commonly"),
    ("manageable", "difficult"), ("establish", "discontinue"), ("significant", "slight"),
]))


class Tok(NamedTuple):
    lower: str
    lemma: str
    pos: str
    dep: str
    head: int
    i: int


_PARSE_PATH = CACHE / "parse.pkl"
_parsed: dict[str, tuple[Tok, ...]] | None = None


def parse(sentences) -> dict[str, tuple[Tok, ...]]:
    global _parsed
    if _parsed is None:
        _parsed = pickle.loads(_PARSE_PATH.read_bytes()) if _PARSE_PATH.exists() else {}
    todo = [s for s in dict.fromkeys(sentences) if s not in _parsed]
    if todo:
        for s, doc in zip(todo, parse_batch(todo)):
            _parsed[s] = tuple(Tok(t.lower_, t.lemma_.lower(), t.pos_, t.dep_, t.head.i, t.i) for t in doc)
        CACHE.mkdir(exist_ok=True)
        _PARSE_PATH.write_bytes(pickle.dumps(_parsed))
    return {s: _parsed[s] for s in sentences}


_WN = {"ADJ": ("a", "s"), "VERB": ("v",), "NOUN": ("n",), "ADV": ("r",), "AUX": ("v",)}


@lru_cache(maxsize=None)
def _ants(word: str) -> frozenset[str]:
    out = set()
    for pos in ("a", "s", "v", "n", "r"):
        out |= {a.lower() for a in antonyms(word, pos)}
    return frozenset(out)


def are_antonyms(x: Tok, y: Tok) -> bool:
    """WordNet antonyms (any POS, surface or lemma, either direction) or a curated pair."""
    xs, ys = {x.lower, x.lemma}, {y.lower, y.lemma}
    if xs & ys:
        return False
    if any(frozenset({a, b}) in CURATED_ANTONYMS for a in xs for b in ys):
        return True
    return any(b in _ants(a) for a in xs for b in ys) or any(a in _ants(b) for a in xs for b in ys)


def diff(ta: tuple[Tok, ...], tb: tuple[Tok, ...]) -> list[tuple[str, list[Tok], list[Tok]]]:
    """Non-equal alignment blocks: (tag, tokens from A, tokens from B), punctuation ignored."""
    a = [t for t in ta if t.pos != "PUNCT"]
    b = [t for t in tb if t.pos != "PUNCT"]
    sm = SequenceMatcher(a=[t.lemma for t in a], b=[t.lemma for t in b], autojunk=False)
    return [(tag, a[i1:i2], b[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


def antonym_aligned(ta, tb) -> bool:
    """2.3: exactly one differing aligned position, one token each side, same dependency
    slot, and the two are antonyms. Nothing else may differ (a/an excepted), so random
    pairs cannot trigger it."""
    blocks = [blk for blk in diff(ta, tb) if not _only_determiners(blk)]
    if len(blocks) != 1:
        return False
    tag, xa, xb = blocks[0]
    return tag == "replace" and len(xa) == len(xb) == 1 and xa[0].dep == xb[0].dep and are_antonyms(xa[0], xb[0])


def _only_determiners(block) -> bool:
    _, xa, xb = block
    return all(t.lower in DETERMINERS for t in xa + xb)


def not_antonym_cancel(ta, tb) -> bool:
    """2.2: one sentence adds a negation cue the other lacks, and the only content change is
    a single aligned antonym swap: *a good idea* / *not an evil idea*. not(antonym(x)) ~ x,
    so the meaning is kept: a cancellation, not a negation. Either direction."""
    neg_a = sum(t.lower in NEG_WORDS for t in ta)
    neg_b = sum(t.lower in NEG_WORDS for t in tb)
    if neg_a == neg_b:
        return False
    swaps = 0
    for tag, xa, xb in diff(ta, tb):
        content_a = [t for t in xa if t.lower not in NEG_WORDS | DETERMINERS and t.pos not in ("AUX", "PART")]
        content_b = [t for t in xb if t.lower not in NEG_WORDS | DETERMINERS and t.pos not in ("AUX", "PART")]
        if not content_a and not content_b:
            continue                              # the inserted cue (plus do-support, a/an)
        if len(content_a) == len(content_b) == 1 and are_antonyms(content_a[0], content_b[0]):
            swaps += 1
            continue
        return False
    return swaps == 1
