"""Phase 1: measure the N_affirmation generator on clean DEV.

    ./venv/bin/python experiments/f1/phase1.py --step 1.2   # coverage, failure buckets, spot-check sample, round trip
    ./venv/bin/python experiments/f1/phase1.py --step 1.3   # inventory table (needs the spot-check verdicts)

Negative sentence = odd number of cues found by the generator's own classifier
(negation.classify.classify_input), over every distinct A and B sentence of clean DEV.
Spot-check: 30 primary affirmations sampled at random (seed 0); verdicts are in
experiments/f1/data/affirmation_spotcheck.json, written by hand against the criteria below.
No TEST data is read in Phase 1.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict

from core import HERE, SEED, load, save

import common as N  # noqa: F401  (nli2: REPO on sys.path)

from negation.affixes import AFFIXAL_BLOCKLIST, negative_base_forms  # noqa: E402
from negation.classify import classify_input  # noqa: E402
from negation.driver import generate_all, parse_bases  # noqa: E402
from negation.generators import n_affirmation as NA  # noqa: E402
from negation.schema import FAM_AFFIXAL, FAM_IMPLICIT, FAM_NEG_ADVERB, FAM_QUANTIFIER, FAM_SYNTACTIC  # noqa: E402

SPOTCHECK = HERE / "data" / "affirmation_spotcheck.json"
CRITERIA = [
    "1. grammatical English (agreement, tense, articles) with no leftover fragment of the cue",
    "2. the removed cue is the main clause's negation, and nothing else in the sentence changed",
    "3. the output is the logical negation of the input (for litotes: the input's meaning without the hedge)",
    "4. secondary / approximate tags are on the readings that need them",
]
UNDOABLE = (FAM_SYNTACTIC, FAM_QUANTIFIER, FAM_NEG_ADVERB, FAM_AFFIXAL, FAM_IMPLICIT)


def dev_sentences() -> list[str]:
    return list(dict.fromkeys(s for r in N.read_dev_clean() for s in (r["a"], r["b"])))


def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", s.lower()))


def word_edit_distance(a: str, b: str) -> int:
    x, y = norm(a).split(), norm(b).split()
    d = list(range(len(y) + 1))
    for i in range(1, len(x) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(y) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x[i - 1] != y[j - 1]))
    return d[-1]


def step_1_2() -> None:
    sents = dev_sentences()
    docs = parse_bases(sents)
    negative, litotes_docs = [], []
    for d in docs:
        p = classify_input(d)
        if p.existing_count % 2 == 1:
            negative.append(d)
        elif NA.litotes(d) is not None:
            litotes_docs.append(d)
    produced, failures, records = 0, Counter(), []
    affixal_pairs = Counter()
    for d in negative:
        outs = NA.affirm_doc(d)
        scope = NA.main_scope(d)
        cues = classify_input(d).existing_cues
        main = [c for c in cues if c.token_idx in scope]
        ctype = NA.cue_type(d, (main or cues)[0].token_idx)
        rec = {"sentence": d.text, "cue_type": ctype, "outputs": [a.__dict__ for a in outs]}
        if outs:
            produced += 1
        else:
            rec["failure"] = NA.failure_reason(d)
            failures[rec["failure"]] += 1
        for c in cues:
            t = d[c.token_idx]
            for stem, _ in negative_base_forms(t):
                affixal_pairs[(t.lower_, stem)] += 1
        records.append(rec)
    lit_records = [{"sentence": d.text, "outputs": [a.__dict__ for a in NA.affirm_doc(d)]} for d in litotes_docs]

    # spot-check sample: 30 primary affirmations, one per sentence, random (seed 0)
    rng = random.Random(SEED)
    pool = [(r["sentence"], next(o for o in r["outputs"] if o["primary"]), r["cue_type"])
            for r in records if any(o["primary"] for o in r["outputs"])]
    sample = rng.sample(pool, 30)

    # round trip: affirmative clean-DEV sentences, negated by a family the affirmer can undo
    affirmative = [d.text for d in docs if classify_input(d).existing_count == 0]
    bases = random.Random(SEED).sample(affirmative, 600)
    variants, gen_errors = [], 0
    for b in bases:                    # per base: a generator's depth assertion aborts the whole batch
        try:
            variants += [v for v in generate_all([b], stages=(1,)) if v.family in UNDOABLE and v.net_negation == 1]
        except AssertionError:
            gen_errors += 1
    by_base = defaultdict(list)
    for v in variants:
        by_base[v.base_sentence].append(v)
    rt = []
    for base, vs in by_base.items():
        for v in vs:
            outs = [a.text for a in NA.affirm(v.variant) if a.polarity_flip]
            if not outs:
                rt.append({"family": v.family, "base": base, "negated": v.variant, "result": "no affirmation"})
                continue
            best = min(outs, key=lambda o: word_edit_distance(o, base))
            dist = word_edit_distance(best, base)
            rt.append({"family": v.family, "base": base, "negated": v.variant, "affirmed": best, "word_edits": dist,
                       "result": "exact" if dist == 0 else "one word off" if dist == 1 else "differs"})
    rt_summary = {}
    for fam in sorted({r["family"] for r in rt}):
        rows = [r for r in rt if r["family"] == fam]
        rt_summary[fam] = {"n": len(rows), **{k: sum(r["result"] == k for r in rows)
                                              for k in ("exact", "one word off", "differs", "no affirmation")}}
    both = [r for r in rt if r["result"] != "no affirmation"]
    from negation.filters import blacklist_hit
    res = {
        "dev_sentences": len(sents), "negative_by_cue_parity": len(negative),
        "dev_sentences_matching_a_blacklist_pattern_in_the_base": sum(blacklist_hit(s) is not None for s in sents),
        "with_at_least_one_affirmation": produced, "with_none": len(negative) - produced,
        "failure_buckets": dict(failures.most_common()),
        "cue_types": dict(Counter(r["cue_type"] for r in records).most_common()),
        "produced_by_cue_type": dict(Counter(r["cue_type"] for r in records if r["outputs"]).most_common()),
        "litotes_sentences_even_parity": len(lit_records), "litotes": lit_records[:40],
        "affixal_pairs_seen": [[w, s, n] for (w, s), n in affixal_pairs.most_common()],
        "blocklist": sorted(AFFIXAL_BLOCKLIST),
        "spotcheck_criteria": CRITERIA,
        "spotcheck_sample": [{"input": s, "output": o["text"], "cue_type": c, "rule": o["rule"],
                              "approximate": o["approximate"]} for s, o, c in sample],
        "round_trip": {"bases_sampled": len(bases), "bases_with_generator_error": gen_errors,
                       "bases_with_undoable_negation": len(by_base),
                       "pairs": len(rt), "pairs_affirmed": len(both),
                       "exact_rate_of_affirmed": round(sum(r["result"] == "exact" for r in both) / max(1, len(both)), 4),
                       "exact_or_one_word_rate_of_affirmed": round(sum(r["result"] in ("exact", "one word off") for r in both) / max(1, len(both)), 4),
                       "by_family": rt_summary,
                       "non_exact_examples": [r for r in rt if r["result"] in ("one word off", "differs")][:40],
                       "no_affirmation_examples": [r for r in rt if r["result"] == "no affirmation"][:20]},
        "examples_failures": {b: [r["sentence"] for r in records if r.get("failure") == b][:5] for b in failures},
    }
    save("1_2_affirmation.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("litotes", "affixal_pairs_seen", "spotcheck_sample",
                                                                 "examples_failures")}, indent=1)[:6000])
    print("affixal pairs:", res["affixal_pairs_seen"][:80])


def step_1_3() -> None:
    res = load("1_2_affirmation.json")
    verdicts = json.loads(SPOTCHECK.read_text())["items"]
    assert [v["input"] for v in verdicts] == [s["input"] for s in res["spotcheck_sample"]], "spot-check file out of date"
    by_type = defaultdict(lambda: [0, 0])
    for v in verdicts:
        by_type[v["cue_type"]][0] += v["correct"]
        by_type[v["cue_type"]][1] += 1
    rows = []
    for ct, n in res["cue_types"].items():
        made = res["produced_by_cue_type"].get(ct, 0)
        ok, seen = by_type.get(ct, (0, 0))
        rows.append({"cue_type": ct, "count": n, "affirmation_produced": made,
                     "produced_rate": round(made / n, 4), "spotcheck_correct": ok, "spotcheck_n": seen})
    out = {"rows": rows, "spotcheck_correct": sum(v["correct"] for v in verdicts), "spotcheck_n": len(verdicts)}
    save("1_3_inventory.json", out)
    print("| cue type | count in clean DEV | affirmation produced | spot-check correct |\n| --- | ---: | ---: | ---: |")
    for r in rows:
        sc = f"{r['spotcheck_correct']} / {r['spotcheck_n']}" if r["spotcheck_n"] else "— (none sampled)"
        print(f"| {r['cue_type']} | {r['count']} | {r['affirmation_produced']} ({100 * r['produced_rate']:.0f}%) | {sc} |")
    print(out["spotcheck_correct"], "/", out["spotcheck_n"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["1.2", "1.3"])
    {"1.2": step_1_2, "1.3": step_1_3}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
