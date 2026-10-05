"""Steps 1-3: base sentences -> variants + anchors -> blind labelling sample.

    ./venv/bin/python experiments/pattern/pattern.py

Reads   data/base_sentences.txt        (subtopic<TAB>sentence, hand-written)
        data/paraphrase_candidates.tsv (hand-written, hand-reviewed, keep=0|1)
Writes  data/variants.jsonl            every record the generator produced
        data/to_label.jsonl            id, a, b -- and nothing else
        data/metadata.jsonl            everything else, keyed by id
        data/build_report.json         the counts steps 1-3 ask to report

The generator in ``negation/`` is called, never edited. ``to_label.jsonl`` is
the only file ``label.py`` reads, so the labeller cannot see a family name.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"
sys.path.insert(0, str(REPO))

SEED = 0
PER_FAMILY = 35
N_UNRELATED = 80
TOTAL_CAP = 600
LABEL_FIELDS = ("id", "a", "b")


def read_bases() -> list[dict]:
    rows = []
    for n, line in enumerate((DATA / "base_sentences.txt").read_text().splitlines(), 1):
        topic, sentence = line.split("\t")
        rows.append({"line": n, "subtopic": topic, "sentence": sentence.strip()})
    return rows


def read_paraphrases() -> tuple[list[dict], list[dict]]:
    kept, dropped = [], []
    for line in (DATA / "paraphrase_candidates.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        base_line, method, keep, text, note = (line.split("\t") + [""])[:5]
        row = {"base_line": int(base_line), "method": method, "text": text, "note": note}
        (kept if keep == "1" else dropped).append(row)
    return kept, dropped


def item_id(a: str, b: str) -> str:
    return "p" + hashlib.sha1(f"{a}\x00{b}".encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Step 1: describe the base sentences with the generator's own analysers
# ---------------------------------------------------------------------------

def profile_bases(bases: list[dict]) -> dict:
    from negation.classify import classify_input
    from negation.frames import detect_frame
    from negation.nlp_core import clause_heads, parse_batch

    docs = parse_batch([b["sentence"] for b in bases])
    root_frame, any_frame = Counter(), Counter()
    for base, doc in zip(bases, docs):
        profile = classify_input(doc)
        root = next(t for t in doc if t.dep_ == "ROOT")
        frame = detect_frame(root)
        base["root_frame"] = frame.subtype if frame else "none (parser found no verbal root)"
        base["negated"] = profile.is_negated
        base["clause_count"] = profile.clause_count
        base["words"] = len(base["sentence"].split())
        root_frame[base["root_frame"]] += 1
        for head in clause_heads(doc):
            f = detect_frame(head)
            if f:
                any_frame[f.subtype] += 1
    return {
        "n": len(bases),
        "subtopic": dict(Counter(b["subtopic"] for b in bases)),
        "root_clause_frame": dict(root_frame),
        "all_clause_heads_frame": dict(any_frame),
        "already_negated": sum(b["negated"] for b in bases),
        "multi_clause": sum(b["clause_count"] > 1 for b in bases),
        "words_min": min(b["words"] for b in bases),
        "words_max": max(b["words"] for b in bases),
    }


# ---------------------------------------------------------------------------
# Step 2: variants (unmodified generator) and features the analysis needs
# ---------------------------------------------------------------------------

def scope_span_ratios(records: list) -> list[float]:
    """Tokens in the first cue's clause scope / tokens in the variant.

    Scope is ``nlp_core.clause_scope`` of the cue token's head -- the same
    containment notion the generator uses for K1 vs K2. 0.0 when the variant
    carries no cue (an affirmation that removed the last one).
    """
    from negation.nlp_core import clause_scope, parse_batch

    docs = parse_batch([r.variant for r in records])
    out = []
    for record, doc in zip(records, docs):
        if not record.cue_char_spans:
            out.append(0.0)
            continue
        start, end = record.cue_char_spans[0]
        span = doc.char_span(start, end, alignment_mode="expand")
        if span is None:
            out.append(0.0)
            continue
        head = span.root.head if span.root.dep_ in ("neg", "advmod", "det") else span.root
        out.append(len(clause_scope(head)) / len(doc))
    return out


def generate_variants(bases: list[dict]) -> list[dict]:
    from negation import generate_all

    records = generate_all([b["sentence"] for b in bases])
    by_sentence = {b["sentence"]: b for b in bases}
    ratios = scope_span_ratios(records)
    rows = []
    for record, ratio in zip(records, ratios):
        row = record.to_json_dict()
        base = by_sentence[row["base_sentence"]]
        row.update(base_line=base["line"], subtopic=base["subtopic"],
                   scope_span_ratio=round(ratio, 4))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Step 3: stratified sample, anchors, blind split
# ---------------------------------------------------------------------------

def stratified(rows: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Up to ``k`` rows, round-robin across subtypes so small subtypes survive.

    Plain random sampling would give C_neg_adverb ~6 ``absolute`` items out of
    35 (380 partial vs 76 absolute in the pool), too few for the partial-vs-
    absolute test the brief asks for.
    """
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["subtype"]].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    order = sorted(buckets)
    picked = []
    while len(picked) < k and any(buckets.values()):
        for subtype in order:
            if buckets[subtype] and len(picked) < k:
                picked.append(buckets[subtype].pop())
    return picked


def assign_ambiguous(variants: list[dict]) -> tuple[list[dict], dict]:
    """One family per (base, variant) string, before sampling.

    Per-family dedup lets two families emit the same string (*inefficient* is
    both D_affixal and an E_antonym; *cannot* is both A_syntactic and M_modal).
    A string is rated once, so it must belong to one family. The more specific
    analysis wins, which on this data is always the family with the smaller
    pool (D over E, M over A/H, K2 and L over A); ties break alphabetically.
    """
    pool = Counter(v["family"] for v in variants)
    groups = defaultdict(list)
    for v in variants:
        groups[(v["base_sentence"], v["variant"])].append(v)
    kept, reassigned = [], Counter()
    for rows in groups.values():
        winner = min(rows, key=lambda r: (pool[r["family"]], r["family"]))
        kept.append(winner)
        for r in rows:
            if r is not winner:
                reassigned[f'{r["family"]} -> {winner["family"]}'] += 1
    return kept, dict(reassigned)


def negation_items(variants: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    by_family = defaultdict(list)
    for v in variants:
        by_family[v["family"]].append(v)
    items, short = [], {}
    for family in sorted(by_family):
        pool = by_family[family]
        if len(pool) < PER_FAMILY:
            short[family] = len(pool)
        for v in stratified(pool, PER_FAMILY, rng):
            items.append({
                "a": v["base_sentence"], "b": v["variant"],
                "meta": {
                    "kind": "negation", "family": family, "subtype": v["subtype"],
                    "base_line": v["base_line"], "subtopic": v["subtopic"],
                    "cue_count": v["cue_count"], "net_negation": v["net_negation"],
                    "intensity_hint": v["intensity_hint"], "operation": v["operation"],
                    "input_polarity": v["input_polarity"],
                    "input_cue_count": v["input_cue_count"],
                    "scope_target": v["scope_target"],
                    "scope_span_ratio": v["scope_span_ratio"],
                    "generator": v["generator"],
                },
            })
    return items, short


def paraphrase_items(bases: list[dict], kept: list[dict]) -> list[dict]:
    by_line = {b["line"]: b for b in bases}
    return [{
        "a": by_line[p["base_line"]]["sentence"], "b": p["text"],
        "meta": {"kind": "paraphrase", "family": "ANCHOR_paraphrase", "subtype": p["method"],
                 "base_line": p["base_line"],
                 "subtopic": by_line[p["base_line"]]["subtopic"]},
    } for p in kept]


def unrelated_items(bases: list[dict], variants: list[dict], rng: random.Random) -> list[dict]:
    """Base sentence i joined to a variant of base j, j from another subtopic."""
    by_line = defaultdict(list)
    for v in variants:
        by_line[v["base_line"]].append(v)
    items, seen = [], set()
    while len(items) < N_UNRELATED:
        a, donor = rng.sample(bases, 2)
        if a["subtopic"] == donor["subtopic"] or not by_line[donor["line"]]:
            continue
        v = rng.choice(by_line[donor["line"]])
        if (a["line"], v["variant"]) in seen:
            continue
        seen.add((a["line"], v["variant"]))
        items.append({
            "a": a["sentence"], "b": v["variant"],
            "meta": {"kind": "unrelated", "family": "ANCHOR_unrelated", "subtype": "cross_subtopic",
                     "base_line": a["line"], "subtopic": a["subtopic"],
                     "donor_line": donor["line"], "donor_subtopic": donor["subtopic"],
                     "donor_family": v["family"]},
        })
    return items


def write_split(items: list[dict]) -> None:
    """``to_label.jsonl`` gets id/a/b only; everything else goes to metadata."""
    with (DATA / "to_label.jsonl").open("w") as lab, (DATA / "metadata.jsonl").open("w") as meta:
        for item in items:
            lab.write(json.dumps({k: item[k] for k in LABEL_FIELDS}, ensure_ascii=False) + "\n")
            meta.write(json.dumps({"id": item["id"], **item["meta"]}, ensure_ascii=False) + "\n")


def main() -> None:
    rng = random.Random(SEED)
    bases = read_bases()
    step1 = profile_bases(bases)

    variants = generate_variants(bases)
    with (DATA / "variants.jsonl").open("w") as fh:
        for v in variants:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")
    kept, dropped = read_paraphrases()

    unique_variants, reassigned = assign_ambiguous(variants)
    neg, short = negation_items(unique_variants, rng)
    para = paraphrase_items(bases, kept)
    unrel = unrelated_items(bases, variants, rng)
    items = neg + para + unrel
    for item in items:
        item["id"] = item_id(item["a"], item["b"])
    unique = list({item["id"]: item for item in items}.values())
    assert len(unique) == len(items), "an anchor collides with a negation pair"
    rng.shuffle(unique)
    assert len(unique) <= TOTAL_CAP + 20, len(unique)
    write_split(unique)

    report = {
        "seed": SEED,
        "step1_base_sentences": step1,
        "step2_variants_per_family": dict(sorted(Counter(v["family"] for v in variants).items())),
        "step2_variants_total": len(variants),
        "step2_paraphrase_candidates": len(kept) + len(dropped),
        "step2_paraphrase_kept": len(kept),
        "step2_paraphrase_discarded": dropped,
        "step2_paraphrase_methods": dict(Counter(p["method"] for p in kept)),
        "step2_unrelated": len(unrel),
        "step3_per_family_cap": PER_FAMILY,
        "step3_families_below_cap": short,
        "step3_sampled_per_family": dict(sorted(Counter(i["meta"]["family"] for i in unique).items())),
        "step3_sampled_subtypes": dict(sorted(Counter(
            f'{i["meta"]["family"]}/{i["meta"]["subtype"]}' for i in unique
            if i["meta"]["kind"] == "negation").items())),
        "step3_ambiguous_strings_reassigned": reassigned,
        "step3_total_items": len(unique),
    }
    (DATA / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "step3_sampled_subtypes"}, indent=2))


if __name__ == "__main__":
    main()
