"""Phase 5.2: machine-generated student answers for the 500 sourced definitions.

    ./venv/bin/python scripts/f1/make_student_answers.py   -> datasets/generated/student_answers.jsonl, results/5_2_answers.json

One answer per (definition, category); labels follow from the category. **Every answer here is
machine-generated** (generator or template), so a score on this set measures the detector against
the generator's distribution, not against real student writing.

| category | label | how it is made |
| --- | --- | --- |
| paraphrase | 0 | template rewrite of the definition (one of four templates, chosen by a seeded draw) |
| negation | 1 | the generator: one stage-1 variant whose negation parity differs from the definition's, cue families only (A, B, C, D, F, H, I, M) |
| partial | 0 | the definition cut back to its first predicate (relative / participial / prepositional tail dropped) |
| wrong_related | 0 | another definition from the same subject, with this term swapped in as its subject |
| cancellation | 0 | the generator's K1_cancellation variant; a template ("It is not untrue that ...") only where K1 produced none, flagged |
| antonym_negation (extra) | 1 | the generator's E_antonym variant (a content word swapped for its antonym) |

Definitions for which a category cannot be built (no variant, no cut point) get no item in that
category; counts are reported. Seed 0.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter

from core import REPO, SEED, save

import common as N  # noqa: F401  (nli2: REPO on sys.path)

from negation.classify import classify_input  # noqa: E402
from negation.driver import generate_all, parse_bases  # noqa: E402

CUE_FAMILIES = ("A_syntactic", "B_quantifier", "C_neg_adverb", "D_affixal", "F_implicit", "H_hedged", "I_intensified", "M_modal")
COPULA = re.compile(r"^(?P<lead>(?:In [^,]{1,60}, )?)(?P<subj>.+?) (?P<cop>is|are|refers to|describes|denotes) (?P<rest>.+)$")
TAIL = re.compile(r"(,| that | which | who | where | in which | used | using | for | by | to | with | when | whose )")


def split(definition: str):
    m = COPULA.match(definition)
    return (m.group("lead"), m.group("subj"), m.group("cop"), m.group("rest").rstrip(".")) if m else None


def lower_first(text: str) -> str:
    """Lower-case a sentence-initial word for use mid-sentence, unless it looks like a name or an
    acronym (second letter upper-case, or not an ordinary WordNet word)."""
    from nltk.corpus import wordnet as wn
    w = text.split()[0]
    if w in ("A", "An", "The") or (len(w) > 1 and w[1:2].islower() and wn.synsets(w.lower())
                                   and w.lower() not in ("unix", "linux", "windows")):
        return text[0].lower() + text[1:]
    return text


def cap(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def paraphrase(d: dict, rng: random.Random) -> str | None:
    p = split(d["definition"])
    if not p:
        return None
    lead, subj, cop, rest = p
    subj_l = lower_first(subj)
    templates = [f"{subj} can be described as {rest}.",
                 f"The term {subj_l} refers to {rest}.",
                 f"Put simply, {subj_l} {cop} {rest}.",
                 f"{rest[0].upper() + rest[1:]}: that is what {subj_l} {'means' if cop != 'are' else 'mean'}."]
    if cop != "is":
        templates = templates[2:3]
    return rng.choice(templates)


def partial(d: dict) -> str | None:
    p = split(d["definition"])
    if not p:
        return None
    lead, subj, cop, rest = p
    m = TAIL.search(rest)
    if not m or m.start() < 3:
        return None
    short = rest[:m.start()].strip()
    if len(short.split()) < 2 or short == rest:
        return None
    return f"{subj} {cop} {short}."


def wrong_related(d: dict, pool: list[dict], rng: random.Random) -> str | None:
    p = split(d["definition"])
    if not p:
        return None
    others = [o for o in pool if o["subject"] == d["subject"] and o["id"] != d["id"] and split(o["definition"])]
    o = rng.choice(others)
    _, _, ocop, orest = split(o["definition"])
    cop = "are" if p[2] == "are" else "is"
    return f"{p[1]} {cop} {orest}.", o["id"]


def main() -> None:
    defs = [json.loads(l) for l in (REPO / "datasets" / "raw" / "definitions.jsonl").open()]
    rng = random.Random(SEED)
    items, missing = [], Counter()
    for d in defs:
        doc = parse_bases([d["definition"]])[0]
        parity = classify_input(doc).existing_count % 2
        try:
            variants = generate_all([d["definition"]], stages=(1,))
        except AssertionError:
            variants = []
            missing["generator_error"] += 1
        flips = [v for v in variants if v.net_negation % 2 != parity]
        neg = [v for v in flips if v.family in CUE_FAMILIES]
        ant = [v for v in flips if v.family == "E_antonym"]
        k1 = [v for v in variants if v.family == "K1_cancellation" and v.net_negation % 2 == parity]
        add = lambda cat, label, text, how, **kw: items.append(
            {"definition_id": d["id"], "subject": d["subject"], "term": d["term"], "teacher": d["definition"],
             "category": cat, "label": label, "student": cap(text), "how": how, "authorship": "machine", **kw})
        t = paraphrase(d, rng)
        add("paraphrase", 0, t, "template") if t else missing.update(["paraphrase"])
        if neg:
            fams = sorted({v.family for v in neg})
            fam = rng.choice(fams)
            v = rng.choice([v for v in neg if v.family == fam])
            add("negation", 1, v.variant, f"generator {v.family}/{v.subtype}")
        else:
            missing["negation"] += 1
        t = partial(d)
        add("partial", 0, t, "template: tail cut") if t else missing.update(["partial"])
        t = wrong_related(d, defs, rng)
        add("wrong_related", 0, t[0], f"template: predicate of {t[1]}") if t else missing.update(["wrong_related"])
        if k1:
            v = rng.choice(k1)
            add("cancellation", 0, v.variant, f"generator {v.family}/{v.subtype}")
        else:
            p = split(d["definition"])
            body = lower_first(d["definition"])
            add("cancellation", 0, f"It is not untrue that {body}", "template (no K1 variant)", template_fallback=True)
        if ant:
            v = rng.choice(ant)
            add("antonym_negation", 1, v.variant, f"generator {v.family}/{v.subtype}", extra_category=True)
        else:
            missing["antonym_negation"] += 1
    with (REPO / "datasets" / "generated" / "student_answers.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    summary = {"definitions": len(defs), "items": len(items), "by_category": dict(Counter(i["category"] for i in items)),
               "by_category_subject": dict(Counter(f"{i['category']}|{i['subject']}" for i in items)),
               "not_buildable": dict(missing), "cancellation_template_fallbacks": sum(bool(i.get("template_fallback")) for i in items),
               "negation_families": dict(Counter(i["how"].split()[1].split("/")[0] for i in items if i["category"] == "negation")),
               "examples": {c: [i for i in items if i["category"] == c][:4] for c in sorted({i["category"] for i in items})}}
    save("5_2_answers.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=1))
    for c, ex in summary["examples"].items():
        for e in ex[:3]:
            print(c, "|", e["teacher"][:100], "=>", e["student"][:140])


if __name__ == "__main__":
    main()
