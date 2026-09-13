#!/usr/bin/env python
"""Per-corpus statistics under *our* taxonomy.

Three questions, for every corpus that loads:

**What families does it contain?**  Cue counts mapped onto our 13 families via
:mod:`negation.data.family_map`, so an external corpus's coverage is directly
comparable with what our generators produce -- and so the families it cannot
reach are visible.

**Does it annotate hedging separately?**  The negation / speculation split, per
corpus.  A corpus with no speculation column cannot train the intensity
regressor, whatever else it is good for.

**How does it treat multiply-negated sentences?**  Sentences carrying two or
more negation cues are re-parsed and each cue pair is put through
:func:`negation.scope.classify_double` -- the same containment test the pipeline
uses -- to see how many are cancellations (K1) and how many are compound (K2).

That last number is the point.  Every corpus here annotates each cue
independently, with no relation between them, so a sentence such as CD-SCO's
*"those not infrequent occasions"* is stored as two unrelated negations even
though one scopes over the other. The K1 count is the number of sentences where
that distinction is present in the text and absent from the annotation.

    ./venv/bin/python scripts/dataset_stats.py
    ./venv/bin/python scripts/dataset_stats.py --corpora cdsco sfu --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from negation.data.loaders import LOADERS, CorpusNotFetched  # noqa: E402
from negation.data.schema import (  # noqa: E402
    ANN_NEGATION,
    ANN_SPECULATION,
    PROV_PAIR_ONLY,
    AnnotatedSentence,
)
from negation.data.sources import CORPORA  # noqa: E402
from negation.nlp_core import clause_heads, parse_batch  # noqa: E402
from negation.schema import FAM_CANCELLATION, FAM_COMPOUND, FAMILIES  # noqa: E402
from negation.scope import classify_double  # noqa: E402


def _char_offsets(tokens: list[str]) -> list[int]:
    """Start offset of each token in ``" ".join(tokens)``."""
    offsets, at = [], 0
    for token in tokens:
        offsets.append(at)
        at += len(token) + 1
    return offsets


def _align(tokens: list[str], doc) -> dict[int, int]:
    """Corpus token index -> spaCy token index, by character offset.

    The corpora tokenise differently from spaCy (and from each other), so the
    only reliable bridge is the character position in the re-joined sentence.
    Corpus tokens that spaCy split are mapped to the first piece.
    """
    offsets = _char_offsets(tokens)
    mapping: dict[int, int] = {}
    for corpus_i, start in enumerate(offsets):
        for tok in doc:
            if tok.idx <= start < tok.idx + len(tok.text) or tok.idx == start:
                mapping[corpus_i] = tok.i
                break
    return mapping


def _clause_head_of(token):
    """The clause head governing ``token`` -- the anchor a negator attaches to."""
    heads = {t.i for t in clause_heads(token.doc)}
    cur, seen = token, {token.i}
    while True:
        if cur.i in heads:
            return cur
        if cur.head.i == cur.i:
            return cur
        cur = cur.head
        if cur.i in seen:
            return cur
        seen.add(cur.i)


def k1_k2_breakdown(groups: list[list[AnnotatedSentence]]) -> dict:
    """Classify each multi-cue sentence's cue pair as K1 or K2.

    One verdict per sentence, taken from its first two negation cues in text
    order: the question is whether the corpus's independent annotations stand in
    a containment relation, and the first pair answers it.
    """
    result = {
        "classified": 0,
        FAM_CANCELLATION: 0,
        FAM_COMPOUND: 0,
        "unalignable": 0,
        "k1_examples": [],
        "k2_examples": [],
    }
    if not groups:
        return result

    docs = parse_batch(g[0].sentence for g in groups)
    for group, doc in zip(groups, docs):
        records = sorted(group, key=lambda r: min(r.cue_token_indices))
        first, second = records[0], records[1]
        mapping = _align(first.tokens, doc)
        i = mapping.get(min(first.cue_token_indices))
        j = mapping.get(min(second.cue_token_indices))
        if i is None or j is None or i >= len(doc) or j >= len(doc):
            result["unalignable"] += 1
            continue
        family = classify_double(_clause_head_of(doc[i]), doc[j])
        result[family] += 1
        result["classified"] += 1
        bucket = "k1_examples" if family == FAM_CANCELLATION else "k2_examples"
        if len(result[bucket]) < 3:
            result[bucket].append(
                {
                    "sentence": first.sentence[:150],
                    "cues": [first.cue_text, second.cue_text],
                    "families": [first.cue_family, second.cue_family],
                }
            )
    return result


def analyse(key: str) -> dict:
    records = LOADERS[key]()
    by_sentence: dict[str, list[AnnotatedSentence]] = defaultdict(list)
    for record in records:
        by_sentence[record.sentence_id].append(record)

    families: Counter[str] = Counter()
    types: Counter[str] = Counter()
    for record in records:
        types[record.annotation_type] += 1
        if record.provenance != PROV_PAIR_ONLY:
            families[record.cue_family] += 1

    negation_groups = [
        [r for r in group if r.annotation_type == ANN_NEGATION]
        for group in by_sentence.values()
    ]
    multi = [g for g in negation_groups if len(g) >= 2 and g[0].provenance != PROV_PAIR_ONLY]
    mixed = sum(
        1
        for group in by_sentence.values()
        if any(r.annotation_type == ANN_NEGATION for r in group)
        and any(r.annotation_type == ANN_SPECULATION for r in group)
    )

    return {
        "corpus": CORPORA[key].name,
        "licence": CORPORA[key].licence,
        "records": len(records),
        "sentences": len(by_sentence),
        "gold_spans": sum(1 for r in records if r.provenance != PROV_PAIR_ONLY),
        "annotation_types": dict(types),
        "has_speculation_layer": types.get(ANN_SPECULATION, 0) > 0,
        "families": dict(families),
        "multi_cue_sentences": len(multi),
        "neg_and_spec_same_sentence": mixed,
        "k1_k2": k1_k2_breakdown(multi),
    }


def _bar(n: int, total: int, width: int = 24) -> str:
    filled = 0 if not total else round(width * n / total)
    return "#" * filled + "." * (width - filled)


def report(key: str, stats: dict, *, examples: bool) -> None:
    print(f"\n{'=' * 78}\n{stats['corpus']}\n{'=' * 78}")
    print(f"  licence              : {stats['licence'] or 'NONE STATED'}")
    print(f"  annotation records   : {stats['records']:,}")
    print(f"  distinct sentences   : {stats['sentences']:,}")
    print(f"  records w/ gold spans: {stats['gold_spans']:,}")

    spec = stats["annotation_types"].get(ANN_SPECULATION, 0)
    neg = stats["annotation_types"].get(ANN_NEGATION, 0)
    flag = "YES" if stats["has_speculation_layer"] else "NO"
    print(f"\n  hedge/speculation layer separate from negation: {flag}")
    print(f"    negation cues    : {neg:,}")
    print(f"    speculation cues : {spec:,}")
    if stats["neg_and_spec_same_sentence"]:
        print(f"    sentences carrying both: {stats['neg_and_spec_same_sentence']:,}")

    if stats["gold_spans"]:
        print("\n  cue-family distribution under our 13-family taxonomy:")
        total = sum(stats["families"].values())
        for family, n in sorted(stats["families"].items(), key=lambda kv: -kv[1]):
            known = family in FAMILIES
            mark = " " if known else "*"
            print(f"    {mark}{family:<18} {n:>6}  {100*n/total:5.1f}%  {_bar(n,total)}")
        if any(f not in FAMILIES for f in stats["families"]):
            print("    (* not one of our families)")
    else:
        print("\n  cue-family distribution: N/A - no gold spans (pair-level corpus)")

    kk = stats["k1_k2"]
    print(f"\n  sentences with 2+ negation cues: {stats['multi_cue_sentences']:,}")
    if stats["multi_cue_sentences"]:
        print(f"    classified by our clause_scope logic: {kk['classified']:,}")
        print(f"      K1_cancellation (contained, net 0): {kk[FAM_CANCELLATION]:,}")
        print(f"      K2_compound     (separate, net 2) : {kk[FAM_COMPOUND]:,}")
        if kk["unalignable"]:
            print(f"      unalignable to a parse            : {kk['unalignable']:,}")
        print("    NOTE: the corpus itself makes no such distinction; every cue")
        print("          above is annotated independently of the others.")
        if examples:
            for label, bucket in (("K1", "k1_examples"), ("K2", "k2_examples")):
                for ex in kk[bucket]:
                    print(f"      [{label}] {ex['cues']} {ex['families']}")
                    print(f"           {ex['sentence']}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpora", nargs="*", default=list(LOADERS),
                        choices=list(LOADERS), help="which corpora to analyse")
    parser.add_argument("--json", type=Path, help="also write the stats as JSON")
    parser.add_argument("--examples", action="store_true",
                        help="print sample K1/K2 sentences")
    args = parser.parse_args(argv)

    out: dict[str, dict] = {}
    for key in args.corpora:
        try:
            stats = analyse(key)
        except CorpusNotFetched as exc:
            print(f"\n{'=' * 78}\n{CORPORA[key].name}\n{'=' * 78}")
            print(f"  NOT FETCHED: {exc}")
            continue
        out[key] = stats
        report(key, stats, examples=args.examples)

    if out:
        print(f"\n{'=' * 78}\nTOTALS\n{'=' * 78}")
        gold = sum(s["gold_spans"] for s in out.values())
        multi = sum(s["multi_cue_sentences"] for s in out.values())
        k1 = sum(s["k1_k2"][FAM_CANCELLATION] for s in out.values())
        k2 = sum(s["k1_k2"][FAM_COMPOUND] for s in out.values())
        spec = sum(s["annotation_types"].get(ANN_SPECULATION, 0) for s in out.values())
        print(f"  gold-span annotations across corpora : {gold:,}")
        print(f"  speculation/hedge cues available     : {spec:,}")
        print(f"  sentences with 2+ negation cues      : {multi:,}")
        print(f"    our logic calls K1 (cancellation)  : {k1:,}")
        print(f"    our logic calls K2 (compound)      : {k2:,}")
        print("  none of these corpora label that distinction themselves.")

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
