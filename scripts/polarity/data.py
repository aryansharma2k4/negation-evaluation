"""Rows and splits. Labels come from the generator or the released datasets only.

DEV        our dataset (scripts/pattern) + CANNOT phase 1
VALIDATION 30% of CANNOT phase 2, grouped so no sentence is in two splits
TEST       the other 70% of phase 2 + all of xNot360
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PATTERN = REPO / "scripts" / "pattern"
PATTERN_DATA = REPO / "datasets" / "generated" / "pattern"
SPLITS = REPO / "datasets" / "splits"
SEED = 0
VAL_SHARE = 0.30

POSITIVE_FAMILIES = {"A_syntactic", "B_quantifier", "C_neg_adverb", "D_affixal", "E_antonym",
                     "F_implicit", "G_prepositional", "H_hedged", "I_intensified", "J_contrastive",
                     "K2_compound", "M_modal",
                     # not in the brief's lists: every L record here negates an
                     # affirmative A ("Every ..." -> "Not every ..."), so it is a negation
                     "L_scope_position"}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_ours() -> tuple[list[dict], dict]:
    """Every generated variant (one family per string) plus both anchor classes."""
    sys.path.insert(0, str(PATTERN))
    from pattern import assign_ambiguous

    variants, _ = assign_ambiguous(jsonl(PATTERN_DATA / "variants.jsonl"))
    rows, excluded = [], []
    for v in variants:
        row = {"a": v["base_sentence"], "b": v["variant"], "dataset": "ours", "family": v["family"],
               "subtype": v["subtype"], "operation": v["operation"], "base_line": v["base_line"]}
        if v["family"] == "K1_cancellation":
            if v["input_polarity"] == "negated":
                # "is impossible" -> "is not impossible": B negates A, so the
                # brief's "cues cancel, meaning preserved" does not hold. Left
                # out rather than labelled wrongly.
                excluded.append(row)
                continue
            row["label"] = 0
        elif v["operation"] == "affirm" or v["family"] in POSITIVE_FAMILIES:
            row["label"] = 1
        else:
            raise ValueError(f"unlabelled family {v['family']}")
        rows.append(row)

    text = {r["id"]: r for r in jsonl(PATTERN_DATA / "to_label.jsonl")}
    for m in jsonl(PATTERN_DATA / "metadata.jsonl"):
        if m["kind"] in ("paraphrase", "unrelated"):
            rows.append({"a": text[m["id"]]["a"], "b": text[m["id"]]["b"], "dataset": "ours",
                         "family": m["family"], "subtype": m["subtype"], "operation": None,
                         "base_line": m["base_line"], "label": 0})
    return rows, {"excluded_k1_negated_input": [(r["a"], r["b"]) for r in excluded]}


def load_cannot(name: str) -> list[dict]:
    rows = jsonl(SPLITS / f"{name}.jsonl")
    for r in rows:
        r["dataset"] = "xnot360" if name == "xnot360" else f"cannot_{name}"
    return rows


def components(rows: list[dict]) -> list[list[int]]:
    """Row indices grouped so rows sharing any sentence (as A or B) stay together."""
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        parent[find(r["a"])] = find(r["b"])
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(find(r["a"]), []).append(i)
    return list(groups.values())


def split_phase2(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = components(rows)
    random.Random(SEED).shuffle(groups)
    val, test, target = [], [], VAL_SHARE * len(rows)
    for g in groups:
        (val if len(val) < target else test).extend(rows[i] for i in g)
    return val, test


def sentences(rows: list[dict]) -> set[str]:
    return {s for r in rows for s in (r["a"], r["b"])}


def assert_disjoint(splits: dict[str, list[dict]]) -> None:
    names = list(splits)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            shared = sentences(splits[x]) & sentences(splits[y])
            assert not shared, f"{len(shared)} sentences in both {x} and {y}, e.g. {next(iter(shared))!r}"


def load_splits() -> tuple[dict[str, list[dict]], dict]:
    ours, notes = load_ours()
    phase1, phase2, xnot = load_cannot("phase1"), load_cannot("phase2"), load_cannot("xnot360")
    val, test2 = split_phase2(phase2)
    splits = {"dev": ours + phase1, "validation": val, "test": test2 + xnot}
    assert_disjoint(splits)
    return splits, notes
