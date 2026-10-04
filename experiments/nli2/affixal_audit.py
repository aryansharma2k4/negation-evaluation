"""0.1 How many existing affixal variants did the old eligibility rule corrupt?

    ./venv/bin/python experiments/nli2/affixal_audit.py

An existing variant made by an affixal generator (affix_derive_v1 for
D_affixal, k1_affixal_cancel_v1 for K1) counts as corrupted when the fixed
generator, rerun on the same base sentence, no longer produces it: the prefix
sat on an attributive adjective inside a term, a non-whitelisted noun, a
compound or a hyphenated word. Strip-direction variants (affirm_*) are a
different mechanism and are not counted.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from negation.base import registry  # noqa: E402
from negation.driver import parse_bases  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results" / "0_1_affixal_audit.json"
GENERATORS = {"affix_derive_v1": "D_affixal", "k1_affixal_cancel_v1": "K1_affixal"}


def regenerate(bases: list[str]) -> dict[str, set[str]]:
    gens = [g for g in registry() if g.name in GENERATORS]
    out = {}
    for doc in parse_bases(bases):
        got = set()
        for g in gens:
            if g.licensed_for(doc) and g.applies(doc):
                got |= {r.variant for r in g.generate(doc)}
        out[doc.text] = got
    return out


def audit(existing: list[tuple[str, str, str]]) -> dict:
    """existing: (base, variant, kind) with kind in GENERATORS.values()."""
    fresh = regenerate(sorted({b for b, _, _ in existing}))
    total, bad, examples = Counter(), Counter(), []
    for base, variant, kind in existing:
        total[kind] += 1
        if variant not in fresh.get(base, set()):
            bad[kind] += 1
            if len(examples) < 10:
                examples.append({"kind": kind, "base": base, "variant": variant})
    return {k: {"n": total[k], "corrupted": bad[k], "pct": round(100 * bad[k] / total[k], 1)}
            for k in sorted(total)} | {"examples": examples}


def main() -> None:
    pattern = [json.loads(l) for l in (REPO / "experiments/pattern/data/variants.jsonl").read_text().splitlines()]
    ours = [(r["base_sentence"], r["variant"], GENERATORS[r["generator"]])
            for r in pattern if r["generator"] in GENERATORS]

    refs = []
    for line in (REPO / "experiments/negation_detection/data/refsets.jsonl").read_text().splitlines():
        r = json.loads(line)
        for v in r["neg"] + r["same"]:
            sub = v["subtype"]
            if v["family"] == "D_affixal" and sub.startswith(("prefix_", "suffix_")):
                refs.append((r["a"], v["text"], "D_affixal"))
            elif v["family"] == "K1_cancellation" and sub.startswith(("not_prefix_", "not_suffix_")):
                refs.append((r["a"], v["text"], "K1_affixal"))

    res = {"pattern_os_cn": audit(ours), "refsets_cannot_xnot": audit(refs)}
    OUT.write_text(json.dumps(res, indent=2))
    print("| corpus | variant kind | existing | corrupted | % |\n| --- | --- | ---: | ---: | ---: |")
    for corpus, block in res.items():
        for k, v in block.items():
            if k != "examples":
                print(f"| {corpus} | {k} | {v['n']} | {v['corrupted']} | {v['pct']}% |")
    for corpus, block in res.items():
        print(f"\n{corpus} examples:")
        for e in block["examples"]:
            print(f"| {e['kind']} | {e['base']} | {e['variant']} |")


if __name__ == "__main__":
    main()
