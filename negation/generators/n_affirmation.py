"""N_affirmation -- the positive form of a negative sentence, for the teacher / student use case.

*"Delhi is not the capital of India."* -> *"Delhi is the capital of India."*

The one-cue removals already exist as stage-2 generators in :mod:`.affirm` (do-support collapse,
aux / copula negator, quantifier flip, negative adverb, affixal strip, implicit trigger).  This
module does not re-implement them.  It runs them, keeps only what removes a cue **in the main
clause's scope**, and adds what they lack:

* the main-clause restriction: *"A process that is not running is blocked"* is not affirmed by
  editing the relative clause.  Scope is :func:`negation.nlp_core.clause_scope` of the ROOT, the
  same walk K1 / K2 and the conditional splitter use;
* the step-0.1 eligibility rules for affixal stripping (predicative adjective or whitelisted noun,
  never inside a compound or hyphenated term) and the :data:`~negation.affixes.AFFIXAL_BLOCKLIST`;
* implicit negatives by lexicon (*refuse* -> *agree*, *deny* -> *admit*, ...), with the
  complement-promotion of :class:`.affirm.ImplicitStrip` kept as a secondary reading, plus *avoid*,
  which the cue lexicon does not list;
* litotes (*not uncommon* -> *common*), tagged ``approximate`` because the hedge is lost, and
  tagged ``polarity_flip=False`` because it is a rewording of the input, not its negation.

Readings that are logically weaker than a flip are tagged too: *never* -> *sometimes* is the
secondary reading of *never* (never *always*; see ``tests/test_affirmation.py``), and stripping
*hardly* / *barely* / *scarcely* is ``approximate`` (*barely passed* already entails *passed*).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from spacy.tokens import Doc, Token

from ..affixes import AFFIXAL_BLOCKLIST, NOUN_STEM_WHITELIST, _in_hyphenated_word, _predicative, negative_base_forms
from ..base import registry
from ..classify import classify_input
from ..filters import run_filters
from ..nlp_core import clause_scope, inflect_lexical, match_inflection, matches_case, parse_batch, root_of, tree_depth
from ..schema import FAM_AFFIXAL, FAM_IMPLICIT, FAM_NEG_ADVERB, FAM_QUANTIFIER, FAM_SYNTACTIC, OP_AFFIRM
from ..splice import Edit, apply_edits, normalize_spacing
from .affirm import _delete_negator, _infinitival_complement, _gerund_complement, clausal_negator_removal

FAMILY = "N_affirmation"

#: Implicit negatives and their positive counterparts.  Curated, not guessed: each pair keeps the
#: complement's syntax (*refuses to X* / *agrees to X*; *denies X-ing* / *admits X-ing*).  *fail*
#: maps to *manage*, not the brief's *succeed*, because *succeeds to X* is ungrammatical and
#: *succeeds in X-ing* would need the complement rebuilt.
IMPLICIT_SWAPS: dict[str, str] = {
    "refuse": "agree", "deny": "admit", "fail": "manage", "neglect": "remember", "cease": "continue",
    "forget": "remember",
}
#: Triggers that only count when they embed a clause (*fails to sort*, not *the disk fails*).
_NEEDS_COMPLEMENT = frozenset({"refuse", "fail", "neglect", "cease", "forget"})
#: Implicit negatives the cue lexicon does not list; affirmed by promoting the gerund.
_PROMOTE_ONLY = frozenset({"avoid"})
_APPROXIMATE_ADVERBS = frozenset({"hardly", "barely", "scarcely"})
_LITOTES_NEGATORS = frozenset({"not", "n't", "never"})


@dataclass(frozen=True)
class Affirmation:
    text: str
    kind: str              # cue type removed (see :func:`cue_type`)
    cue: str               # the cue's surface text
    primary: bool          # the main reading for this cue; False for secondary readings
    approximate: bool      # meaning only approximately kept (litotes, hardly / barely)
    polarity_flip: bool    # True: a negation of the input. False: a rewording (litotes)
    rule: str              # generator / subtype that produced it
    family: str = FAMILY
    clause: str = "main"   # report_f2 4.1: main | coordinated | advcl | ccomp (the removed cue's clause)


def main_scope(doc: Doc) -> frozenset[int]:
    root = root_of(doc)
    return clause_scope(root) if root is not None else frozenset()


_VERBAL = frozenset({"VERB", "AUX", "ADJ", "NOUN"})


def clause_kinds(doc: Doc) -> dict[int, str]:
    """token index -> the affirmable clause it belongs to (report_f2 step 4.1).

    main         clause_scope(ROOT)
    coordinated  clause_scope of a verbal conj of ROOT (recursively): *X sorts the list and does not print it*
    advcl/ccomp  clause_scope of an adverbial or complement clause of a main or coordinated head

    Relative clauses and other noun modifiers (relcl, acl) are never included: affirming *A process that is
    not running is blocked* would change the subject, not the claim (the step-1.1 restriction).
    Every scope comes from negation.nlp_core.clause_scope; nothing here defines scope again.
    """
    root = root_of(doc)
    if root is None:
        return {}
    kinds: dict[int, str] = {i: "main" for i in clause_scope(root)}
    heads, stack = [root], [root]
    while stack:
        h = stack.pop()
        for c in h.children:
            if c.dep_ == "conj" and c.pos_ in _VERBAL and c.pos_ != "NOUN" or (c.dep_ == "conj" and h.i == root.i and c.pos_ == "AUX"):
                for i in clause_scope(c):
                    kinds.setdefault(i, "coordinated")
                heads.append(c)
                stack.append(c)
    for h in heads:
        for c in h.children:
            if c.dep_ in ("advcl", "ccomp") and c.pos_ in ("VERB", "AUX", "ADJ"):
                for i in clause_scope(c):
                    kinds.setdefault(i, c.dep_)
    return kinds


def cue_type(doc: Doc, idx: int) -> str:
    """A readable cue type for the inventory (report_f1.md step 1.3)."""
    t = doc[idx]
    low = t.lower_
    if low in ("not", "n't", "cannot"):
        head = t.head
        auxes = [c for c in head.children if c.dep_ in ("aux", "auxpass") and c.i < t.i]
        aux = auxes[-1] if auxes else (head if head.lemma_ == "be" and head.i < t.i else None)
        if low == "cannot" or (aux is not None and aux.lower_ in ("ca", "can")):
            return "can + not / can't / cannot"
        if aux is None:
            return "not, other position (e.g. not + NP / adverb)"
        lem = aux.lemma_.lower()
        if lem == "do":
            return {"VBZ": "does not V", "VBD": "did not V"}.get(aux.tag_, "do not V")
        if lem in ("will", "wo"):
            return "will not / won't"
        if lem in ("would", "should", "could", "must", "may", "might", "shall", "need"):
            return "other modal + not"
        if lem == "have":
            return "has / have / had not (perfect)"
        if lem == "be":
            return "be + not (copula)" if aux is head else "be + not (aux / passive / progressive)"
        return "not, other position (e.g. not + NP / adverb)"
    if low == "never":
        return "never"
    if low in ("rarely", "seldom", "hardly", "barely", "scarcely"):
        return "rarely / seldom / hardly / barely"
    if low in ("no", "none", "nobody", "nothing", "neither", "nor", "nowhere", "no-one", "noone"):
        return "negative quantifier (no / none / nobody / nothing / neither / nor)"
    if t.lemma_.lower() in IMPLICIT_SWAPS or t.lemma_.lower() in ("lack", "unable") or low == "unable":
        return "implicit (refuse / deny / fail / lack / unable ...)"
    if low.endswith("less") and t.pos_ == "ADJ":
        return "-less suffix"
    if negative_base_forms(t):
        return "affixal prefix (un- / in- / non- / dis- ...)"
    if low in ("without",):
        return "without"
    return f"other ({low})"


def _strip_eligible(t: Token) -> bool:
    """The step-0.1 rule, applied to the token an affixal strip would rewrite."""
    if t.dep_ == "compound" or _in_hyphenated_word(t):
        return False
    if t.pos_ == "NOUN":
        return any(stem in NOUN_STEM_WHITELIST for stem, _ in negative_base_forms(t))
    return _predicative(t)


def _splice(doc: Doc, edits: list[Edit]) -> str:
    text, cues = apply_edits(doc.text, edits)
    return normalize_spacing(text, cues)[0]


def _from_generators(doc: Doc, scope) -> list[Affirmation]:
    profile = classify_input(doc)
    candidates = []
    for g in registry():
        if g.operation == OP_AFFIRM and g.licensed_for(doc) and g.applies(doc):
            candidates.extend(g.generate(doc))
    kept, _ = run_filters(candidates, {doc._.base_id: tree_depth(doc)}, per_family_dedup=False) if candidates else ([], None)
    out = []
    for v in kept:
        cues = profile.cues_of(v.family)
        if not cues:
            continue
        idx = cues[0].token_idx                      # every affirm generator removes its family's first cue
        if idx not in scope:
            continue
        t = doc[idx]
        if v.family == FAM_AFFIXAL and not _strip_eligible(t):
            continue
        secondary = (v.family == FAM_NEG_ADVERB and not v.subtype.startswith("strip_")) or \
                    (v.family == FAM_IMPLICIT and t.lemma_.lower() in IMPLICIT_SWAPS)
        clause = scope.get(idx, "main") if isinstance(scope, dict) else "main"
        out.append(Affirmation(
            text=v.variant, kind=cue_type(doc, idx), cue=t.text, primary=not secondary,
            approximate=v.family == FAM_NEG_ADVERB and t.lower_ in _APPROXIMATE_ADVERBS,
            polarity_flip=clause in ("main", "coordinated"), rule=f"{v.generator}/{v.subtype}", clause=clause))
    return out


def _implicit(doc: Doc, scope) -> list[Affirmation]:
    out = []
    for t in doc:
        lem = t.lemma_.lower()
        if t.i not in scope or t.pos_ != "VERB":
            continue
        if lem in IMPLICIT_SWAPS:
            if lem in _NEEDS_COMPLEMENT and _infinitival_complement(t) is None:
                continue
            new = matches_case(inflect_lexical(IMPLICIT_SWAPS[lem], t.tag_), t)
            out.append(Affirmation(_splice(doc, [Edit.replace(t.idx, t.idx + len(t.text), new)]),
                                   kind=cue_type(doc, t.i), cue=t.text, primary=True, approximate=False,
                                   polarity_flip=_flip(scope, t.i), rule=f"implicit_swap/{lem}_to_{IMPLICIT_SWAPS[lem]}",
                                   clause=_clause(scope, t.i)))
        elif lem in _PROMOTE_ONLY:
            comp = _gerund_complement(t)
            if comp is None:
                continue
            restored = inflect_lexical(comp.lemma_, t.tag_)
            out.append(Affirmation(_splice(doc, [Edit.replace(t.idx, comp.idx + len(comp.text), restored)]),
                                   kind="implicit (avoid)", cue=t.text, primary=True, approximate=False,
                                   polarity_flip=_flip(scope, t.i), rule=f"implicit_promote/{lem}", clause=_clause(scope, t.i)))
    return out


def _clause(scope, i: int) -> str:
    return scope.get(i, "main") if isinstance(scope, dict) else "main"


def _flip(scope, i: int) -> bool:
    """Affirming a main or coordinated clause flips the sentence's claim; affirming an adverbial or
    complement clause flips a subordinate claim (for *if not P* that is an antecedent flip), which is
    not a negation of the sentence."""
    return _clause(scope, i) in ("main", "coordinated")


def litotes(doc: Doc, scope: Optional[frozenset[int]] = None) -> Optional[Affirmation]:
    """*not un-X* -> *X*: one negator and one affixal negative in the main clause."""
    scope = main_scope(doc) if scope is None else scope
    negs = [t for t in doc if t.i in scope and t.lower_ in _LITOTES_NEGATORS]
    if len(negs) != 1:
        return None
    neg = negs[0]
    # the affixal word must be what the negator negates (its head, or a predicate of that head):
    # *Unfortunately it will not recharge* has both cues but is not a litotes
    affs = [t for t in doc if t.i in scope and t.pos_ in ("ADJ", "NOUN") and negative_base_forms(t)
            and (t.i == neg.head.i or t.head.i == neg.head.i) and _strip_eligible(t)]
    if not affs:
        return None
    x = min(affs, key=lambda a: abs(a.i - neg.i))
    edits = clausal_negator_removal(neg) or [_delete_negator(neg)]
    stem = negative_base_forms(x)[0][0]
    edits = [e for e in edits if not (e.start <= x.idx < e.end)]
    edits.append(Edit.replace(x.idx, x.idx + len(x.text), matches_case(match_inflection(stem, x), x)))
    return Affirmation(_splice(doc, edits), kind="litotes (not un-X)", cue=f"{neg.text} {x.text}", primary=True,
                       approximate=True, polarity_flip=False, rule="litotes")


def affirm_doc(doc: Doc, extended: bool = True) -> list[Affirmation]:
    """``extended=False`` reproduces report_f1.md's main-clause-only behaviour (for the old-vs-new table)."""
    scope = main_scope(doc)
    kinds = clause_kinds(doc) if extended else {i: "main" for i in scope}
    out = []
    lit = litotes(doc, scope)
    if lit is not None:
        out.append(lit)
    out.extend(_implicit(doc, kinds))
    out.extend(_from_generators(doc, kinds))
    if lit is not None:   # the one-cue removals of a litotes are negations of it, not its positive form
        out = [out[0]] + [a if a is lit else Affirmation(**{**a.__dict__, "primary": False}) for a in out[1:]]
    seen, uniq = set(), []
    for a in out:
        if a.text != doc.text and a.text not in seen:
            seen.add(a.text)
            uniq.append(a)
    return uniq


def affirm(sentence: str) -> list[Affirmation]:
    from ..driver import parse_bases
    return affirm_doc(parse_bases([sentence])[0])


def failure_reason(doc: Doc, extended: bool = True) -> Optional[str]:
    """Why a sentence with a cue got no affirmation (named bucket), or None if it got one."""
    if affirm_doc(doc, extended):
        return None
    profile = classify_input(doc)
    scope = set(clause_kinds(doc)) if extended else main_scope(doc)
    if not profile.existing_cues:
        return "no cue the generator's classifier recognises"
    inside = [c for c in profile.existing_cues if c.token_idx in scope]
    if not inside:
        if extended:
            return "every cue is in a relative clause or attached to a noun phrase (not an affirmable clause)"
        return "every cue is outside the main clause (relative / subordinate / coordinated clause)"
    t = doc[inside[0].token_idx]
    fam = inside[0].family_guess
    if fam == FAM_AFFIXAL:
        if t.lower_ in AFFIXAL_BLOCKLIST:
            return "affixal: on the exception blocklist"
        return "affixal: attributive / term position (step-0.1 rule) or no WordNet-attested stem"
    if fam == FAM_SYNTACTIC:
        if t.lower_ in ("not", "n't") and not any(c.dep_ in ("aux", "auxpass") for c in t.head.children) \
                and t.head.lemma_ != "be":
            return "not without an auxiliary to hang on (not + NP / adverb / infinitive)"
        return "syntactic negator: parse shape the removal rules do not cover"
    if fam == FAM_QUANTIFIER:
        return "negative quantifier: no flip for this form or position"
    if fam == FAM_IMPLICIT:
        return "implicit trigger: no complement to promote / not in the swap lexicon"
    if fam == FAM_NEG_ADVERB:
        return "negative adverb: removal rejected by the filters"
    return f"cue family {fam}: no affirmation rule"


def demo() -> None:   # python -m negation.generators.n_affirmation
    for s in ["Delhi is not the capital of India.", "He never calls.", "It is not uncommon for a process to block."]:
        for a in affirm(s):
            print(s, "->", a)


if __name__ == "__main__":
    demo()
