"""Phase 6: a severity label for every pair called a negation, derived mechanically from the cue shape.

    ./venv/bin/python scripts/f2/severity.py     -> results/6_severity.json

**This does not change any verdict or any accuracy number**: it describes a negation already detected.

The negating cue = the cues (polarity analysis, scripts/polarity) present in one sentence and not the
other, compared by (text, family) multiset; B's extra cues when B has them, else the cues A has and B lacks
(an affirmation). Each cue carries its generator family and whether its clause holds a hedge or an
intensifier (clause_scope, as everywhere). Precedence when several apply, most specific first:

    scope        a differing cue of family B_quantifier or L_scope_position,    not all segments are delivered
                 unless it is part of an intensifier idiom (*by no means*)
    intensified  a differing cue that is intensified (I_intensified, 'certainly not', 'not at all')
    hedged       a differing cue that is hedged (H_hedged, M_modal: 'probably not', 'may not', 'I don't think')
    flat         a plain differing cue (A_syntactic, C_neg_adverb; G_prepositional 'without', J_contrastive 'far from')
    lexical      no plain cue: D_affixal, E_antonym, F_implicit, or no cue differs at all (antonym / rewording)

Scope outranks strength because it changes the truth conditions; strength (intensified, hedged) outranks
'flat' because it is the information a teacher would otherwise lose.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from core2 import C, REPO, log_test_read, save

SEVERITIES = ("scope", "intensified", "hedged", "flat", "lexical")
SCOPE_FAM = {"B_quantifier", "L_scope_position"}
LEXICAL_FAM = {"D_affixal", "E_antonym", "F_implicit"}
HEDGE_WORDS = {"might", "may", "maybe", "perhaps", "probably", "possibly", "think", "believe", "could", "likely", "seem", "seems"}


def differing_cues(a, b) -> list:
    key = lambda c: (c.text, c.family)
    ca, cb = Counter(map(key, a.cues)), Counter(map(key, b.cues))
    extra_b = [c for c in b.cues if (cb - ca)[key(c)] > 0]
    if extra_b:
        return extra_b
    return [c for c in a.cues if (ca - cb)[key(c)] > 0]


def severity(a, b, b_text: str = "") -> tuple[str, list[str]]:
    cues = differing_cues(a, b)
    fams = {c.family for c in cues}
    shown = [f"{c.text} ({c.family}{', hedged' if c.hedged else ''}{', intensified' if c.intensified else ''})" for c in cues]
    # a quantifier cue inside an intensifier idiom (*by no means*) is intensified, not a scope change (v2 fix,
    # found on the machine-generated domain answers; v1 numbers kept in results/6_severity_v1.json)
    if any(c.family in SCOPE_FAM and not c.intensified for c in cues):
        return "scope", shown
    if any(c.intensified for c in cues) or "I_intensified" in fams:
        return "intensified", shown
    words = set(b_text.lower().replace("n't", " not").split())
    if any(c.hedged for c in cues) or fams & {"H_hedged", "M_modal"} or (cues and words & HEDGE_WORDS and not words.isdisjoint({"not", "no", "never"})):
        return "hedged", shown
    if cues and not fams <= LEXICAL_FAM:
        return "flat", shown
    return "lexical", shown


def label_pairs(pairs) -> list[tuple[str, list[str]]]:
    an = C.P.analyse_many([s for p in pairs for s in p])
    return [severity(an[a], an[b], b) for a, b in pairs]


GOLD = {"A_syntactic": "flat", "C_neg_adverb": "flat", "G_prepositional": "flat", "J_contrastive": "flat",
        "H_hedged": "hedged", "M_modal": "hedged", "I_intensified": "intensified", "B_quantifier": "scope",
        "L_scope_position": "scope", "D_affixal": "lexical", "E_antonym": "lexical", "F_implicit": "lexical"}


def main() -> None:
    from detector import Detector
    det = Detector.adopted()
    ans = [json.loads(l) for l in (REPO / "datasets" / "generated" / "student_answers.jsonl").open()]
    negs = [a for a in ans if a["label"] == 1]
    gold = [GOLD[a["how"].split()[1].split("/")[0]] for a in negs]
    pred = label_pairs([(a["teacher"], a["student"]) for a in negs])
    out = {"note": "severity is descriptive: it changes no verdict and no accuracy"}
    det_out = det.predict([(a["teacher"], a["student"]) for a in negs])
    called = np.array([o["negation"] for o in det_out])
    conf = Counter((g, p[0]) for g, p in zip(gold, pred))
    out["domain_machine"] = {"n": len(negs), "accuracy_all": round(float(np.mean([g == p[0] for g, p in zip(gold, pred)])), 4),
                             "n_called_negation": int(called.sum()),
                             "accuracy_on_called": round(float(np.mean([g == p[0] for g, p, c in zip(gold, pred, called) if c])), 4),
                             "confusion": {f"{g}->{p}": v for (g, p), v in sorted(conf.items())},
                             "per_gold": {s: {"n": sum(g == s for g in gold), "correct": sum(g == s and p[0] == s for g, p in zip(gold, pred))} for s in SEVERITIES},
                             "errors": [{"gold": g, "pred": p[0], "cues": p[1], "teacher": a["teacher"][:120], "student": a["student"][:160]}
                                        for g, p, a in zip(gold, pred, negs) if g != p[0]][:25]}
    probs = np.array([o["prob"] for o in det_out])
    out["domain_machine"]["combiner_prob_by_predicted_severity"] = {
        s: {"n": int(sum(p[0] == s for p in pred)), "mean_prob": round(float(np.mean([pr for pr, p in zip(probs, pred) if p[0] == s])), 4) if any(p[0] == s for p in pred) else None,
            "recall": round(float(np.mean([c for c, p in zip(called, pred) if p[0] == s])), 4) if any(p[0] == s for p in pred) else None} for s in SEVERITIES}
    # TEST negations (human-written): severity of each, and the adopted system's output by severity
    from core2 import baseline
    t = [r for r in baseline()["sets"]["test"] if r["label"] == 1]
    log_test_read("6.3", "adopted system's output on TEST negations, grouped by severity")
    tp = label_pairs([(r["a"], r["b"]) for r in t])
    to = det.predict([(r["a"], r["b"]) for r in t])
    out["test_negations_human"] = {s: {"n": sum(p[0] == s for p in tp),
                                       "mean_combiner_output": round(float(np.mean([o["prob"] for o, p in zip(to, tp) if p[0] == s])), 4) if any(p[0] == s for p in tp) else None,
                                       "recall": round(float(np.mean([o["negation"] for o, p in zip(to, tp) if p[0] == s])), 4) if any(p[0] == s for p in tp) else None,
                                       "share_cannot": round(float(np.mean([r["dataset"] == "cannot_phase2" for r, p in zip(t, tp) if p[0] == s])), 3) if any(p[0] == s for p in tp) else None}
                                   for s in SEVERITIES}
    out["test_examples"] = {s: [{"a": r["a"], "b": r["b"], "cues": p[1]} for r, p in zip(t, tp) if p[0] == s][:2] for s in SEVERITIES}
    # paragraph set: distribution only (no generating family exists for these items; machine-authored)
    its = [json.loads(l) for l in (REPO / "datasets" / "eval" / "paragraph_eval.jsonl").open()]
    cp = [(it["teacher"][i], it["student"][g["student"]]) for it in its for i, g in enumerate(it["gold_teacher"]) if g["verdict"] == "contradicted"]
    out["paragraph_contradictions_machine"] = {"n": len(cp), "distribution": dict(Counter(p[0] for p in label_pairs(cp))),
                                               "items": [{"t": a, "s": b, "severity": p[0], "cues": p[1]} for (a, b), p in zip(cp, label_pairs(cp))]}
    save("6_severity.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("test_examples",)}, indent=1)[:7000])


if __name__ == "__main__":
    main()
