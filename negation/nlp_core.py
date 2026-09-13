"""Shared NLP resources: one spaCy pipeline, cached WordNet, morphology helpers.

Loading rules for this project:

* the spaCy model is loaded exactly once, at import time, with the NER
  component disabled (we only ever need tagger/parser/lemmatizer);
* every parse goes through :func:`parse_batch`, which wraps ``nlp.pipe`` -- no
  generator or filter is allowed to call ``nlp()`` inside a loop;
* WordNet membership and antonym lookups are memoised, because the affixal and
  antonym generators hammer them with the same handful of lemmas.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional

import spacy
from spacy.tokens import Doc, Token

_MODEL = "en_core_web_sm"

#: Module-level singleton.  Never reloaded, never reconfigured.
nlp = spacy.load(_MODEL, disable=["ner"])

BATCH_SIZE = 256


def parse_batch(texts: Iterable[str], *, batch_size: int = BATCH_SIZE) -> list[Doc]:
    """Parse many strings in one pipe pass.  The only entry point to the model."""
    return list(nlp.pipe(list(texts), batch_size=batch_size))


# ---------------------------------------------------------------------------
# WordNet
# ---------------------------------------------------------------------------

from nltk.corpus import wordnet as wn  # noqa: E402  (import after spaCy on purpose)

_WN_POS = {"ADJ": wn.ADJ, "NOUN": wn.NOUN, "VERB": wn.VERB, "ADV": wn.ADV}


def wn_pos(spacy_pos: str) -> Optional[str]:
    """Map a spaCy coarse POS tag onto a WordNet POS constant."""
    return _WN_POS.get(spacy_pos)


@lru_cache(maxsize=50000)
def in_wordnet(word: str, pos: Optional[str] = None) -> bool:
    """True if ``word`` is a real English word WordNet knows about.

    Used to gate affixal derivation: ``un`` + ``important`` is admissible only
    because ``unimportant`` is a WordNet lemma, while ``un`` + ``possible`` is
    rejected before it can be emitted.
    """
    try:
        return bool(wn.synsets(word.lower().replace(" ", "_"), pos=pos))
    except Exception:  # pragma: no cover - corpus lookup edge cases
        return False


@lru_cache(maxsize=50000)
def antonyms(lemma: str, pos: str) -> tuple[str, ...]:
    """Direct WordNet antonyms of ``lemma`` under ``pos``, most frequent first.

    ``lemma.antonyms()`` is defined on lemmas, not synsets, so we walk every
    synset for the word and collect the antonyms attached to the lemma that
    actually spells ``lemma``.  Multiword antonyms are dropped: splicing an
    underscore-joined WordNet lemma into a sentence produces junk.
    """
    key = lemma.lower()
    found: list[str] = []
    seen: set[str] = set()
    try:
        synsets = wn.synsets(key, pos=pos)
    except Exception:  # pragma: no cover
        return ()
    for syn in synsets:
        for lem in syn.lemmas():
            if lem.name().lower() != key:
                continue
            for ant in lem.antonyms():
                name = ant.name()
                if "_" in name or name.lower() in seen:
                    continue
                seen.add(name.lower())
                found.append(name)
    return tuple(found)


# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------

#: Irregular past / past-participle forms we may need when re-inflecting a
#: WordNet antonym lemma to match the original token.  Regular ``-ed``/``-s``
#: rules cover the rest.
IRREGULAR_VERBS: dict[str, tuple[str, str]] = {
    # lemma: (past, past participle)
    "be": ("was", "been"),
    "become": ("became", "become"),
    "begin": ("began", "begun"),
    "break": ("broke", "broken"),
    "bring": ("brought", "brought"),
    "build": ("built", "built"),
    "buy": ("bought", "bought"),
    "catch": ("caught", "caught"),
    "choose": ("chose", "chosen"),
    "come": ("came", "come"),
    "cost": ("cost", "cost"),
    "cut": ("cut", "cut"),
    "do": ("did", "done"),
    "draw": ("drew", "drawn"),
    "drive": ("drove", "driven"),
    "eat": ("ate", "eaten"),
    "fall": ("fell", "fallen"),
    "feed": ("fed", "fed"),
    "feel": ("felt", "felt"),
    "fight": ("fought", "fought"),
    "find": ("found", "found"),
    "forget": ("forgot", "forgotten"),
    "get": ("got", "gotten"),
    "give": ("gave", "given"),
    "go": ("went", "gone"),
    "grow": ("grew", "grown"),
    "have": ("had", "had"),
    "hear": ("heard", "heard"),
    "hide": ("hid", "hidden"),
    "hit": ("hit", "hit"),
    "hold": ("held", "held"),
    "keep": ("kept", "kept"),
    "know": ("knew", "known"),
    "lead": ("led", "led"),
    "leave": ("left", "left"),
    "lend": ("lent", "lent"),
    "let": ("let", "let"),
    "lose": ("lost", "lost"),
    "make": ("made", "made"),
    "mean": ("meant", "meant"),
    "meet": ("met", "met"),
    "pay": ("paid", "paid"),
    "put": ("put", "put"),
    "read": ("read", "read"),
    "rise": ("rose", "risen"),
    "run": ("ran", "run"),
    "say": ("said", "said"),
    "see": ("saw", "seen"),
    "sell": ("sold", "sold"),
    "send": ("sent", "sent"),
    "set": ("set", "set"),
    "show": ("showed", "shown"),
    "shut": ("shut", "shut"),
    "sing": ("sang", "sung"),
    "sit": ("sat", "sat"),
    "sleep": ("slept", "slept"),
    "speak": ("spoke", "spoken"),
    "spend": ("spent", "spent"),
    "stand": ("stood", "stood"),
    "take": ("took", "taken"),
    "teach": ("taught", "taught"),
    "tell": ("told", "told"),
    "think": ("thought", "thought"),
    "throw": ("threw", "thrown"),
    "understand": ("understood", "understood"),
    "wear": ("wore", "worn"),
    "win": ("won", "won"),
    "write": ("wrote", "written"),
}

_VOWELS = frozenset("aeiou")
_SIBILANT_ENDINGS = ("s", "sh", "ch", "x", "z", "o")


def third_person_singular(lemma: str) -> str:
    """``sort`` -> ``sorts``, ``pass`` -> ``passes``, ``carry`` -> ``carries``."""
    if lemma == "be":
        return "is"
    if lemma == "have":
        return "has"
    if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in _VOWELS:
        return lemma[:-1] + "ies"
    if lemma.endswith(_SIBILANT_ENDINGS):
        return lemma + "es"
    return lemma + "s"


def _double_final_consonant(lemma: str) -> bool:
    return (
        len(lemma) >= 3
        and lemma[-1] not in _VOWELS
        and lemma[-1] not in "wxy"
        and lemma[-2] in _VOWELS
        and lemma[-3] not in _VOWELS
    )


def past_tense(lemma: str) -> str:
    """``sort`` -> ``sorted``, ``stop`` -> ``stopped``, ``go`` -> ``went``."""
    if lemma in IRREGULAR_VERBS:
        return IRREGULAR_VERBS[lemma][0]
    if lemma.endswith("e"):
        return lemma + "d"
    if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in _VOWELS:
        return lemma[:-1] + "ied"
    if _double_final_consonant(lemma):
        return lemma + lemma[-1] + "ed"
    return lemma + "ed"


def past_participle(lemma: str) -> str:
    if lemma in IRREGULAR_VERBS:
        return IRREGULAR_VERBS[lemma][1]
    return past_tense(lemma)


def present_participle(lemma: str) -> str:
    """``sort`` -> ``sorting``, ``make`` -> ``making``, ``stop`` -> ``stopping``."""
    if lemma.endswith("ie"):
        return lemma[:-2] + "ying"
    if lemma.endswith("e") and not lemma.endswith(("ee", "oe", "ye")):
        return lemma[:-1] + "ing"
    if _double_final_consonant(lemma):
        return lemma + lemma[-1] + "ing"
    return lemma + "ing"


def inflect_verb(lemma: str, tag: str) -> str:
    """Inflect ``lemma`` to the Penn tag ``tag`` (VB/VBP/VBZ/VBD/VBN/VBG)."""
    if tag == "VBZ":
        return third_person_singular(lemma)
    if tag == "VBD":
        return past_tense(lemma)
    if tag == "VBN":
        return past_participle(lemma)
    if tag == "VBG":
        return present_participle(lemma)
    return lemma  # VB, VBP


def match_inflection(new_lemma: str, model: Token) -> str:
    """Re-inflect ``new_lemma`` so it fills ``model``'s morphological slot.

    Verbs go through the Penn tag; plural nouns get an ``-s``; adjectives and
    adverbs are used bare (WordNet antonyms are already citation forms and the
    comparative/superlative cases are handled by the caller refusing them).
    """
    if model.pos_ in ("VERB", "AUX"):
        return inflect_verb(new_lemma, model.tag_)
    if model.pos_ == "NOUN" and model.tag_ in ("NNS", "NNPS"):
        return third_person_singular(new_lemma)
    return new_lemma


def matches_case(new_word: str, model: Token) -> str:
    """Copy ``model``'s capitalisation pattern onto ``new_word``."""
    if model.text[:1].isupper():
        return new_word[:1].upper() + new_word[1:]
    return new_word


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

#: Dependency labels that open a *new* clause under their head.  Crossing one of
#: these edges leaves the current negation's scope -- the distinction K1/K2
#: turns on.
CLAUSE_BOUNDARY_DEPS: frozenset[str] = frozenset(
    {"conj", "advcl", "ccomp", "csubj", "csubjpass", "parataxis", "relcl", "acl"}
)

#: Auxiliary dependency labels.
AUX_DEPS: frozenset[str] = frozenset({"aux", "auxpass"})


def root_of(doc: Doc) -> Optional[Token]:
    """The single ROOT token, or ``None`` if the parse has none."""
    for token in doc:
        if token.dep_ == "ROOT":
            return token
    return None


def roots(doc: Doc) -> list[Token]:
    return [t for t in doc if t.dep_ == "ROOT"]


def finite_aux(verb: Token) -> Optional[Token]:
    """The leftmost finite auxiliary governed by ``verb``, if any."""
    auxes = [c for c in verb.children if c.dep_ in AUX_DEPS]
    return min(auxes, key=lambda t: t.i) if auxes else None


def subject_of(verb: Token) -> Optional[Token]:
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass", "expl"):
            return child
    return None


def clause_scope(head: Token) -> frozenset[int]:
    """Token indices in ``head``'s clause, stopping at clause boundaries.

    This is the operative notion of *scope* for K1 vs K2.  ``head.subtree``
    would wrongly swallow a coordinated second clause -- in *"the function sorts
    the array and prints the result"* the ``conj`` verb *prints* is inside
    ``sorts.subtree``.  Descending only through non-boundary edges gives the
    clause the negator actually commands.
    """
    scope: set[int] = set()
    stack = [head]
    while stack:
        token = stack.pop()
        if token.i in scope:
            continue
        scope.add(token.i)
        for child in token.children:
            if child.dep_ in CLAUSE_BOUNDARY_DEPS:
                continue
            stack.append(child)
    return frozenset(scope)


def clause_heads(doc: Doc) -> list[Token]:
    """Every finite verbal clause head in ``doc``, document order.

    A clause head is the ROOT plus any token reached across a clause-boundary
    edge that itself heads a verbal predicate.
    """
    heads: list[Token] = []
    for token in doc:
        if token.dep_ == "ROOT":
            heads.append(token)
        elif token.dep_ in CLAUSE_BOUNDARY_DEPS and token.pos_ in ("VERB", "AUX"):
            heads.append(token)
    return sorted(heads, key=lambda t: t.i)


def clause_index_at(doc: Doc, char_offset: int) -> int:
    """Index into :func:`clause_heads` of the clause covering ``char_offset``.

    Used to fill a record's ``target_clause_idx`` from the edit it spliced, so
    every generator reports which clause it touched without having to be told.
    Walks up from the token at the offset to the first token that heads a
    clause; falls back to the ROOT's clause when the chain reaches nothing else.
    """
    heads = clause_heads(doc)
    if not heads:
        return 0
    positions = {head.i: index for index, head in enumerate(heads)}

    token = None
    for candidate in doc:
        if candidate.idx <= char_offset < candidate.idx + len(candidate.text):
            token = candidate
            break
        if candidate.idx >= char_offset:
            token = candidate
            break
    if token is None:
        token = doc[-1]

    seen = {token.i}
    cur = token
    while True:
        if cur.i in positions:
            return positions[cur.i]
        if cur.head.i == cur.i:
            break
        cur = cur.head
        if cur.i in seen:  # pragma: no cover - malformed parse guard
            break
        seen.add(cur.i)
    root = root_of(doc)
    return positions.get(root.i, 0) if root is not None else 0


def tree_depth(doc: Doc) -> int:
    """Maximum head-chain length in the parse, or ``-1`` on a malformed tree."""
    best = 0
    for token in doc:
        depth = 0
        cur = token
        seen = {cur.i}
        while cur.head.i != cur.i:
            cur = cur.head
            if cur.i in seen:
                return -1  # cycle
            seen.add(cur.i)
            depth += 1
            if depth > len(doc):
                return -1
        best = max(best, depth)
    return best


def subtree_char_span(token: Token) -> tuple[int, int]:
    """``(start, end)`` character offsets covering ``token``'s whole subtree."""
    toks = list(token.subtree)
    return toks[0].idx, toks[-1].idx + len(toks[-1].text)


def span_without_punct(tokens: list[Token]) -> tuple[int, int]:
    """Char span of ``tokens`` with trailing sentence punctuation trimmed."""
    while tokens and tokens[-1].is_punct:
        tokens = tokens[:-1]
    if not tokens:
        return (0, 0)
    return tokens[0].idx, tokens[-1].idx + len(tokens[-1].text)
