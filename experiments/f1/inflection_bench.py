"""Phase 1.1 choice: which inflector restores a verb's morphology most accurately?

The do-support collapse (*does not run* -> *runs*) rebuilds a surface verb from its lemma plus the
Tense / Person / Number the auxiliary carried. Every inflected verb already in clean DEV is a gold
example of that operation: give each inflector (lemma, Penn tag) and check it reproduces the word
that was written. Candidates: the hand-rolled negation.nlp_core.inflect_verb (rules + irregular
table), lemminflect.getInflection, pyinflect (spaCy extension). Tags: VBZ, VBD, VBN, VBG (VB/VBP are
the lemma for every candidate). Sentences: every A and B of clean DEV, de-duplicated.

    ./venv/bin/python experiments/f1/inflection_bench.py   -> results/1_1_inflection.json
"""

from __future__ import annotations

from collections import Counter

from core import REPO, save

import common as N  # noqa: F401  (nli2; puts REPO on sys.path)

from negation.nlp_core import inflect_verb, parse_batch  # noqa: E402

TAGS = ("VBZ", "VBD", "VBN", "VBG")


def main() -> None:
    import lemminflect  # before pyinflect: both register Token._.inflect, pyinflect with force
    import pyinflect
    dev = N.read_dev_clean()
    sents = list(dict.fromkeys(s for r in dev for s in (r["a"], r["b"])))
    cases = []
    for doc in parse_batch(sents):
        for t in doc:
            if t.pos_ == "VERB" and t.tag_ in TAGS and t.is_alpha and t.lemma_.isalpha():
                cases.append((t.lemma_.lower(), t.tag_, t.lower_))
    inflectors = {
        "hand-rolled nlp_core.inflect_verb": lambda lem, tag: inflect_verb(lem, tag),
        "lemminflect.getInflection": lambda lem, tag: (lemminflect.getInflection(lem, tag=tag) or [lem])[0],
        "pyinflect.getInflection": lambda lem, tag: (pyinflect.getInflection(lem, tag=tag) or [lem])[0],
    }
    res = {"sentences": len(sents), "verb_tokens": len(cases), "by_tag": dict(Counter(c[1] for c in cases)),
           "inflectors": {}}
    for name, f in inflectors.items():
        wrong = [(lem, tag, gold, f(lem, tag).lower()) for lem, tag, gold in cases if f(lem, tag).lower() != gold]
        per_tag = {tag: round(1 - sum(w[1] == tag for w in wrong) / max(1, sum(c[1] == tag for c in cases)), 4)
                   for tag in TAGS}
        types = set(cases)
        wrong_types = {w[:3] for w in wrong}
        res["inflectors"][name] = {"accuracy": round(1 - len(wrong) / len(cases), 4), "errors": len(wrong),
                                   "type_accuracy": round(1 - len(wrong_types) / len(types), 4),
                                   "types": len(types), "type_errors": len(wrong_types),
                                   "accuracy_by_tag": per_tag,
                                   "error_examples": [list(w) for w in Counter(wrong).most_common(15)]}
        print(name, res["inflectors"][name]["accuracy"], res["inflectors"][name]["type_accuracy"], per_tag, Counter(wrong).most_common(8))
    res["rule_as_first_stated"] = "highest token accuracy over all clean-DEV inflected verbs"
    res["chosen_by_first_rule"] = max(res["inflectors"], key=lambda n: res["inflectors"][n]["accuracy"])
    # Revised on DEV after the first run (reported as such): token accuracy is dominated by one SNLI
    # word (spaCy lemmatises *riding* as *rid*; 53 tokens), which pyinflect maps back by luck, and
    # pyinflect's lexicon has work -> wrought. Distinct (lemma, tag, form) types weigh each verb once.
    res["rule"] = "highest type accuracy (distinct lemma, tag, written form); token accuracy breaks a tie"
    best = max(res["inflectors"], key=lambda n: (res["inflectors"][n]["type_accuracy"], res["inflectors"][n]["accuracy"]))
    res["chosen"] = best
    save("1_1_inflection.json", res)
    print("chosen:", best)


if __name__ == "__main__":
    main()
