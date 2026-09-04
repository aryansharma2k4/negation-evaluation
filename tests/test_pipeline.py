"""Filters, schema invariants, driver and CLI behaviour."""

from __future__ import annotations

import json

import pytest

from negation.base import registry
from negation.driver import generate_all, make_base_id, parse_bases, summarize
from negation.filters import (
    FilterReport,
    blacklist_hit,
    dedup,
    normalize,
    run_filters,
)
from negation.nlp_core import parse_batch, tree_depth
from negation.schema import FAMILIES, MAX_DEPTH, NegationVariant
from negation.splice import Edit, OverlappingEditError, apply_edits


def _record(variant: str, family: str = "A_syntactic", base: str = "The task is simple.") -> NegationVariant:
    return NegationVariant(
        base_id="b0",
        base_sentence=base,
        variant=variant,
        family=family,
        subtype="copula",
        cue_tokens=["not"],
        cue_char_spans=[(12, 15)],
        cue_count=1,
        net_negation=1,
        scope_target="clause",
        intensity_hint="neutral",
        generator="test",
    )


# -- schema -----------------------------------------------------------------


def test_cue_count_is_derived_not_trusted():
    """A hand-supplied ``cue_count`` is overwritten from the spans."""
    record = NegationVariant(
        base_id="b0",
        base_sentence="The task is simple.",
        variant="The task is not simple.",
        family="A_syntactic",
        subtype="copula",
        cue_tokens=["not"],
        cue_char_spans=[(12, 15)],
        cue_count=99,
        net_negation=1,
        scope_target="clause",
        intensity_hint="neutral",
        generator="test",
    )
    assert record.cue_count == 1


def test_schema_rejects_unknown_enum_values():
    with pytest.raises(ValueError):
        _record("x", family="Z_bogus")


def test_verified_hook_defaults_to_none():
    assert _record("The task is not simple.").verified is None
    assert "verified" in _record("The task is not simple.").to_json_dict()


def test_json_dict_is_serialisable():
    payload = json.dumps(_record("The task is not simple.").to_json_dict())
    assert json.loads(payload)["cue_char_spans"] == [[12, 15]]


# -- splicing ---------------------------------------------------------------


def test_apply_edits_reports_cue_offsets_in_the_output():
    text, cues = apply_edits("The task is simple.", [Edit.insert(11, " not", cue=True)])
    assert text == "The task is not simple."
    word, start, end = cues[0]
    assert word == "not" and text[start:end] == "not"


def test_overlapping_edits_are_refused():
    with pytest.raises(OverlappingEditError):
        apply_edits("abcdef", [Edit.replace(0, 3, "x"), Edit.replace(2, 5, "y")])


def test_multiword_cue_yields_one_span_per_token():
    text, cues = apply_edits(
        "The task is simple.", [Edit.insert(11, " by no means", cue=True)]
    )
    assert [c[0] for c in cues] == ["by", "no", "means"]
    assert all(text[s:e] == w for w, s, e in cues)


# -- filters ----------------------------------------------------------------


def test_normalize_is_case_punctuation_and_space_insensitive():
    assert normalize("The  Task is NOT simple!") == normalize("the task is not simple")


@pytest.mark.parametrize("bad", [
    "The task is not not simple.",
    "The function does not has redundancy.",
    "The model might could work.",
    "The the task is not simple.",
])
def test_blacklist_catches_known_bad_surfaces(bad):
    assert blacklist_hit(bad) is not None


@pytest.mark.parametrize("good", [
    "The system does not have redundancy.",
    "The function did not do that.",
    "The task is not simple.",
])
def test_blacklist_leaves_well_formed_negations_alone(good):
    assert blacklist_hit(good) is None


def test_dedup_keeps_distinct_family_analyses_by_default():
    report = FilterReport()
    kept = dedup(
        [_record("The task is not simple."), _record("The task is not simple.", family="C_neg_adverb")],
        report,
    )
    assert len(kept) == 2


def test_dedup_global_mode_collapses_them():
    report = FilterReport()
    kept = dedup(
        [_record("The task is not simple."), _record("The task is not simple.", family="C_neg_adverb")],
        report,
        per_family=False,
    )
    assert len(kept) == 1 and report.dropped["duplicate"] == 1


def test_dedup_drops_a_variant_identical_to_its_base():
    report = FilterReport()
    assert dedup([_record("The task is simple.")], report) == []
    assert report.dropped["identical_to_base"] == 1


def test_parse_check_rejects_a_two_root_variant():
    base_doc = parse_batch(["The task is simple."])[0]
    report = FilterReport()
    bad = _record("The task is simple. The task is not simple.")
    kept, report = run_filters([bad], {"b0": tree_depth(base_doc)}, report=report)
    assert kept == []
    assert any(reason.startswith("parse:") for reason in report.dropped)


# -- driver / registry ------------------------------------------------------


def test_registry_names_and_families_are_valid_and_unique():
    names = [g.name for g in registry()]
    assert len(names) == len(set(names))
    assert all(g.family in FAMILIES for g in registry())
    assert all(g.depth <= MAX_DEPTH for g in registry())


def test_base_sentence_is_parsed_exactly_once(monkeypatch):
    import negation.driver as driver

    calls: list[int] = []
    original = driver.parse_batch

    def counting(texts, **kwargs):
        materialised = list(texts)
        calls.append(len(materialised))
        return original(materialised, **kwargs)

    monkeypatch.setattr(driver, "parse_batch", counting)
    driver.parse_bases(["The function sorts the array.", "The task is simple."])
    assert calls == [2]  # one batched pipe call for both sentences


def test_base_ids_are_stable_and_content_derived():
    assert make_base_id("The task is simple.", 0) == make_base_id("The task is simple.", 0)
    assert make_base_id("A.", 0) != make_base_id("B.", 0)


def test_generate_all_end_to_end():
    sentences = [
        "The function sorts the array.",
        "The task is impossible.",
        "Everyone passed the exam.",
        "The function sorts the array and returns the result.",
    ]
    records = generate_all(sentences)
    assert records
    assert {r.base_sentence for r in records} <= set(sentences)
    for record in records:
        assert record.family in FAMILIES
        assert record.cue_count == len(record.cue_char_spans) == len(record.cue_tokens)
        for token, (start, end) in zip(record.cue_tokens, record.cue_char_spans):
            assert record.variant[start:end] == token
        assert record.verified is None
        assert record.depth <= MAX_DEPTH


def test_summary_counts_add_up():
    records = generate_all(["The function sorts the array."])
    counts = summarize(records)
    assert sum(counts["family"].values()) == len(records)
    assert sum(counts["net_negation"].values()) == len(records)


def test_cli_writes_jsonl_and_summary(tmp_path, capsys):
    from negation.cli import main

    src = tmp_path / "in.txt"
    src.write_text("The function sorts the array.\nThe task is impossible.\n")
    out = tmp_path / "out.jsonl"

    assert main([str(src), "-o", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows and all("net_negation" in row for row in rows)

    printed = capsys.readouterr().out
    assert "by family:" in printed and "by net_negation:" in printed


def test_cli_reports_missing_input(tmp_path, capsys):
    from negation.cli import main

    assert main([str(tmp_path / "nope.txt")]) == 2


def test_no_generator_builds_strings_in_applies():
    """``applies`` must stay cheap: it may not allocate variant records."""
    docs = parse_bases(["The function sorts the array and returns the result."])
    for generator in registry():
        before = generator.applies(docs[0])
        assert isinstance(before, bool)
