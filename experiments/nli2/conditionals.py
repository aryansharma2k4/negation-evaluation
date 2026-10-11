"""Phase 3: split conditionals into antecedent and consequent, and decide on the halves.

Spans come from negation.nlp_core.clause_scope (the walk the rest of the codebase uses for
K1 vs K2); nothing here defines scope again. A conditional is an adverbial clause
attached to its consequent's head, marked by one of MARKERS. spaCy parses the markers three ways:

    if / unless / whenever   mark child of the advcl head         "If you sign, ..."
      (not after *as*; *even if* / *only if* are kept as distinct connectives, f1 step 0.2)
    when / whenever          advmod child of the advcl head       "When it rains, ..."
    provided / assuming      the marker heads the clause; the     "Provided that you sign, ..."
                             antecedent is its ccomp

The antecedent is clause_scope of the antecedent head plus clause_scope of its conj heads
(clause_scope stops at conj, which would drop "and pay" from "If you sign and pay").

Polarity of a half = parity of the v2 cues (experiments/polarity) whose clause is one of
that half's heads. "unless" counts as "if not": the antecedent starts negated.

Pair decision (only when A and B are both conditionals):
    consequent flipped, antecedent not   -> negation
    antecedent flipped only              -> NOT a negation
    both flipped                         -> negation, tagged contraposition
    neither                              -> no decision (cascade decides)
"""

from __future__ import annotations

import pickle
from typing import NamedTuple

from common import C, CACHE, REPO  # noqa: F401  (REPO on sys.path for negation.*)

from negation.nlp_core import clause_scope, parse_batch  # noqa: E402

MARKERS = frozenset({"if", "unless", "when", "whenever", "provided", "assuming"})
DISTINCT_CONNECTIVES = frozenset({"even if", "only if"})
_HEADED = frozenset({"provide", "provided", "assume", "assuming"})


class Cond(NamedTuple):
    marker: str
    ante_heads: tuple[int, ...]
    ante_span: tuple[int, ...]
    cons_heads: tuple[int, ...]
    cons_span: tuple[int, ...]
    nested: bool                 # another conditional sits inside this one's antecedent or consequent


def _with_conj(head) -> list:
    """head plus its conj heads, recursively (coordinated clauses share the role)."""
    out, stack = [], [head]
    while stack:
        t = stack.pop()
        out.append(t)
        stack.extend(c for c in t.children if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX", "ADJ", "NOUN"))
    return out


def _span(heads) -> tuple[int, ...]:
    idx = set()
    for h in heads:
        idx |= clause_scope(h)
    doc = heads[0].doc
    return tuple(sorted(i for i in idx if not doc[i].is_punct))


def _if_connective(if_tok) -> str | None:
    """f1 step 0.2: *as if* is a comparison, not a condition (None); *even if* and *only if*
    are their own connectives, so a pair that swaps one for the other is not read as a
    polarity flip of plain *if*."""
    prev = if_tok.doc[if_tok.i - 1].lower_ if if_tok.i > 0 else ""
    if prev == "as":
        return None
    return f"{prev} if" if prev in ("even", "only") else "if"


def find(doc) -> list[Cond]:
    found = []
    for tok in doc:
        marker, ante = None, None
        if tok.dep_ == "advcl":
            kids = {c.lower_: c for c in tok.children if c.dep_ in ("mark", "advmod")}
            hit = next((m for m in ("if", "unless", "whenever", "when") if m in kids), None)
            if hit == "if":
                hit = _if_connective(kids["if"])
            if hit:
                marker, ante = hit, tok
        if marker is None and tok.lower_ in ("provided", "assuming") and tok.dep_ in ("advcl", "prep"):
            comp = next((c for c in tok.children if c.dep_ == "ccomp"), None)
            if comp is not None:
                marker, ante = tok.lower_, comp
        if marker is None:
            continue
        cons = tok.head
        ante_heads, cons_heads = _with_conj(ante), _with_conj(cons)
        found.append([marker, tuple(t.i for t in ante_heads), _span(ante_heads),
                      tuple(t.i for t in cons_heads), _span(cons_heads),
                      frozenset(t.i for t in ante.subtree)])
    out = []
    for m, ah, asp, ch, csp, sub in found:
        # nested: this conditional's consequent sits inside another's antecedent (containment is
        # a subtree question; clause_scope deliberately stops at the embedded clause)
        nested = any(ch[0] in f[5] for f in found if f[1] != ah)
        out.append(Cond(m, ah, asp, ch, csp, nested))
    return out


_PATH = CACHE / "conditionals_v2.pkl"     # v2: after the f1 step 0.2 fix (as if / even if / only if)
_cache: dict | None = None


def conditionals(sentences) -> dict[str, list[Cond]]:
    global _cache
    if _cache is None:
        _cache = pickle.loads(_PATH.read_bytes()) if _PATH.exists() else {}
    todo = [s for s in dict.fromkeys(sentences) if s not in _cache]
    if todo:
        for s, doc in zip(todo, parse_batch(todo)):
            _cache[s] = find(doc)
        CACHE.mkdir(exist_ok=True)
        _PATH.write_bytes(pickle.dumps(_cache))
    return {s: _cache[s] for s in sentences}


def primary(conds: list[Cond]) -> Cond | None:
    """The outermost conditional (not nested in another's antecedent), leftmost first."""
    if not conds:
        return None
    outer = [c for c in conds if not c.nested]
    return min(outer or conds, key=lambda c: c.ante_heads[0])


def half_polarity(analysis, heads: tuple[int, ...], negated_start: bool = False) -> int:
    """Parity of v2 cues in the clauses headed by ``heads`` (1 = negated)."""
    n = sum(c.clause in heads for c in analysis.cues)
    return (n + negated_start) % 2


def polarities(cond: Cond, analysis) -> tuple[int, int]:
    return (half_polarity(analysis, cond.ante_heads, cond.marker == "unless"),
            half_polarity(analysis, cond.cons_heads))


def decide(ca: Cond | None, cb: Cond | None, aa, ab) -> str | None:
    """'negation', 'contraposition', 'not_negation' or None (no decision)."""
    if ca is None or cb is None:
        return None
    if ca.marker != cb.marker and {ca.marker, cb.marker} & DISTINCT_CONNECTIVES:
        return None              # the connective itself changed (even if -> only if): no polarity rule applies
    pa, pb = polarities(ca, aa), polarities(cb, ab)
    ante_flip, cons_flip = pa[0] != pb[0], pa[1] != pb[1]
    if ante_flip and cons_flip:
        return "contraposition"
    if cons_flip:
        return "negation"
    if ante_flip:
        return "not_negation"
    return None
