"""Stage-2 verification: parsing, caching, tiering, and the failure paths.

Almost everything here runs without a model. That is deliberate: the parts most
likely to break at 3am are the ones that handle a model behaving badly, and
those have to be testable without one. A fake backend stands in for the LLM so
malformed output, truncation and an unreachable server are all exercised
directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from negation.verification.cache import VerdictCache
from negation.verification.cola import format_sweep, threshold_sweep
from negation.verification.driver import (
    already_verified,
    read_records,
    run_tier1,
    run_tier2,
    summarise,
    write_records,
)
from negation.verification.llm import (
    LLMJudge,
    LlamaCppBackend,
    _is_local,
    extract_verdicts,
    verdict_from_row,
)
from negation.verification.prompts import build_prompt, item_for
from negation.verification.report import cluster, collect_disagreements, write_disagreement_report
from negation.verification.schema import (
    REASON_COLA_REJECT,
    REASON_PARSE_FAILURE,
    RunStats,
    Verdict,
    record_key,
    rejected_by_cola,
)
from negation.verification.validate import (
    KAPPA_GATE,
    build_constructed_set,
    cohens_kappa,
    compare,
    corrupt_grammar,
    index_by_key,
)


def make_record(**overrides) -> dict:
    record = {
        "base_id": "b0",
        "base_sentence": "The function sorts the array.",
        "variant": "The function does not sort the array.",
        "family": "A_syntactic",
        "subtype": "do_support",
        "operation": "negate",
        "net_negation": 1,
        "generator": "syn_do_support_v1",
        "verified": None,
    }
    record.update(overrides)
    return record


# -- schema ------------------------------------------------------------------


def test_verified_requires_all_three_checks():
    assert Verdict(True, True, True).verified is True
    assert Verdict(True, True, False).verified is False
    assert Verdict(False, True, True).verified is False


def test_unjudged_stays_none_not_false():
    """A missing answer is not a rejection; conflating them poisons the signal."""
    assert Verdict(True, None, True).verified is None
    assert Verdict().verified is None


def test_reason_names_which_checks_failed():
    fields = Verdict(True, False, False, model="m").as_fields()
    assert fields["verify_reason"] == "llm_reject:semantic+category"
    assert fields["verified"] is False


def test_passing_verdict_reads_as_pass():
    assert Verdict(True, True, True, model="m").as_fields()["verify_reason"] == "pass"


def test_record_key_ignores_fields_the_prompt_never_sees():
    """Two records asking the LLM the same question share a cache entry."""
    a = make_record(generator="x", cue_char_spans=[[1, 2]])
    b = make_record(generator="y", cue_char_spans=[[9, 9]])
    assert record_key(a) == record_key(b)


def test_record_key_separates_different_questions():
    assert record_key(make_record()) != record_key(make_record(variant="Other."))
    assert record_key(make_record()) != record_key(make_record(family="B_quantifier"))


def test_cola_rejection_does_not_invent_llm_answers():
    """The LLM never saw it, so its checks stay unknown rather than false."""
    fields = rejected_by_cola(0.02, "cola")
    assert fields["verified"] is False
    assert fields["verify_grammatical"] is False
    assert fields["verify_semantic"] is None
    assert fields["verify_category"] is None
    assert fields["verify_reason"] == REASON_COLA_REJECT


# -- response parsing --------------------------------------------------------


@pytest.mark.parametrize("text", [
    '{"verdicts":[{"id":0,"grammatical":true,"semantically_valid":true,"category_correct":true}]}',
    '[{"id":0,"grammatical":true,"semantically_valid":true,"category_correct":true}]',
    '{"results":[{"id":0,"grammatical":true,"semantically_valid":true,"category_correct":true}]}',
    '```json\n{"verdicts":[{"id":0,"grammatical":true,"semantically_valid":true,"category_correct":true}]}\n```',
    'Sure! {"verdicts":[{"id":0,"grammatical":true,"semantically_valid":true,"category_correct":true}]}',
])
def test_parser_tolerates_the_shapes_models_actually_return(text):
    parsed = extract_verdicts(text)
    assert 0 in parsed
    assert verdict_from_row(parsed[0], "m").verified is True


@pytest.mark.parametrize("text", ["", "   ", "I cannot help with that.", "{", "null"])
def test_parser_reports_failure_rather_than_guessing(text):
    assert extract_verdicts(text) == {}


@pytest.mark.parametrize("value,expected", [
    (True, True), ("true", True), ("yes", True), (1, True),
    (False, False), ("false", False), ("no", False), (0, False),
    ("maybe", None), (None, None),
])
def test_boolean_coercion(value, expected):
    row = {"grammatical": value, "semantically_valid": True, "category_correct": True}
    assert verdict_from_row(row, "m").grammatical is expected


def test_a_family_outside_the_taxonomy_is_dropped_not_stored():
    """An unactionable suggestion must not look like a label."""
    row = {"grammatical": True, "semantically_valid": True, "category_correct": False,
           "suggested_family": "Z_invented"}
    verdict = verdict_from_row(row, "m")
    assert verdict.category is False
    assert verdict.suggested_family is None


def test_a_valid_family_suggestion_survives():
    row = {"grammatical": True, "semantically_valid": True, "category_correct": False,
           "suggested_family": "D_affixal"}
    assert verdict_from_row(row, "m").suggested_family == "D_affixal"


def test_confidence_is_clamped():
    row = {"grammatical": True, "semantically_valid": True, "category_correct": True,
           "confidence": 4.2}
    assert verdict_from_row(row, "m").confidence == 1.0


# -- the judge's failure paths ----------------------------------------------


class FakeBackend:
    """Scripted responses, so model misbehaviour is reproducible."""

    def __init__(self, responses, name="fake/model"):
        self.responses = list(responses)
        self.name = name
        self.calls = 0

    def complete(self, prompt, max_tokens):
        self.calls += 1
        if not self.responses:
            return ""
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _ok(*ids):
    rows = [
        {"id": i, "grammatical": True, "semantically_valid": True,
         "category_correct": True, "suggested_family": None}
        for i in ids
    ]
    return json.dumps({"verdicts": rows})


def test_a_clean_batch_needs_one_call():
    judge = LLMJudge(FakeBackend([_ok(0, 1, 2)]))
    verdicts = judge.judge_batch([make_record() for _ in range(3)])
    assert [v.verified for v in verdicts] == [True, True, True]
    assert judge.backend.calls == 1


def test_malformed_output_is_retried_then_falls_back_per_record():
    """Two retries on the batch, then one call per record."""
    backend = FakeBackend(["garbage", "still garbage", "nope"] + [_ok(0)] * 2)
    retries, fallbacks = [], []
    judge = LLMJudge(backend, backoff=0.0)
    judge.on_retry = lambda: retries.append(1)
    judge.on_fallback = lambda: fallbacks.append(1)

    verdicts = judge.judge_batch([make_record(), make_record(variant="B.")])
    assert [v.verified for v in verdicts] == [True, True]
    assert len(retries) == 2
    assert len(fallbacks) == 2


def test_after_every_route_fails_the_record_is_null_not_false():
    """The distinction the whole schema turns on."""
    judge = LLMJudge(FakeBackend(["x"] * 10), backoff=0.0)
    verdicts = judge.judge_batch([make_record()])
    assert verdicts[0].verified is None
    assert verdicts[0].reason == REASON_PARSE_FAILURE


def test_a_truncated_batch_recovers_only_the_missing_records():
    """Three asked, two returned: only the third is re-asked alone."""
    backend = FakeBackend([_ok(0, 1), _ok(0, 1), _ok(0, 1), _ok(0)])
    judge = LLMJudge(backend, backoff=0.0)
    verdicts = judge.judge_batch([make_record(variant=f"V{i}.") for i in range(3)])
    assert [v.verified for v in verdicts] == [True, True, True]


def test_an_unreachable_server_does_not_crash_the_run():
    judge = LLMJudge(FakeBackend([OSError("connection refused")]), backoff=0.0)
    verdicts = judge.judge_batch([make_record()])
    assert verdicts[0].verified is None
    assert verdicts[0].reason == "llm_unavailable"


def test_judge_never_raises_on_arbitrary_exceptions():
    judge = LLMJudge(FakeBackend([ValueError("boom")] * 10), backoff=0.0)
    assert judge.judge_batch([make_record()])[0].verified is None


# -- no hosted API -----------------------------------------------------------


@pytest.mark.parametrize("url,ok", [
    ("http://localhost:8080", True),
    ("http://127.0.0.1:11434", True),
    ("https://api.openai.com", False),
    ("http://example.com:8080", False),
])
def test_only_local_endpoints_are_accepted(url, ok):
    assert _is_local(url) is ok


def test_llamacpp_refuses_a_remote_endpoint():
    """A non-local URL is a configuration error, not a feature."""
    with pytest.raises(ValueError):
        LlamaCppBackend(url="https://api.example.com")


# -- cache -------------------------------------------------------------------


def test_cache_round_trips_a_verdict(tmp_path):
    cache = VerdictCache(tmp_path / "c.sqlite")
    record = make_record()
    assert cache.get(record, "m") is None
    cache.put(record, Verdict(True, True, True, model="m"))
    assert cache.get(record, "m").verified is True
    cache.close()


def test_cache_is_keyed_by_model(tmp_path):
    """Verdicts are not interchangeable across models."""
    cache = VerdictCache(tmp_path / "c.sqlite")
    cache.put(make_record(), Verdict(True, True, True, model="model-a"))
    assert cache.get(make_record(), "model-b") is None
    cache.close()


def test_unjudged_verdicts_are_never_cached(tmp_path):
    """Caching a parse failure would make it permanent."""
    cache = VerdictCache(tmp_path / "c.sqlite")
    cache.put(make_record(), Verdict(None, None, None, model="m"))
    assert cache.count() == 0
    cache.close()


def test_cache_survives_reopening(tmp_path):
    path = tmp_path / "c.sqlite"
    with VerdictCache(path) as cache:
        cache.put(make_record(), Verdict(True, True, True, model="m"))
    with VerdictCache(path) as reopened:
        assert reopened.get(make_record(), "m").verified is True


def test_disabled_cache_is_inert(tmp_path):
    cache = VerdictCache(tmp_path / "c.sqlite", enabled=False)
    cache.put(make_record(), Verdict(True, True, True, model="m"))
    assert cache.get(make_record(), "m") is None


# -- tiering -----------------------------------------------------------------


class FakeScorer:
    name = "fake-cola"
    device = "cpu"

    def __init__(self, scores):
        self.scores = scores

    def score(self, sentences):
        return self.scores[: len(sentences)]


def test_tier1_rejects_below_threshold_and_passes_the_rest():
    records = [make_record(variant=f"V{i}.") for i in range(4)]
    stats = RunStats(total=4)
    survivors, scores = run_tier1(
        records, FakeScorer([0.9, 0.1, 0.8, 0.05]), 0.5, stats, log=lambda *_: None
    )
    assert len(survivors) == 2
    assert stats.cola_rejected == 2
    assert all(r["verify_reason"] == REASON_COLA_REJECT
               for r in records if r.get("verified") is False)


def test_tier1_absent_sends_everything_to_the_llm():
    records = [make_record() for _ in range(3)]
    stats = RunStats(total=3)
    survivors, scores = run_tier1(records, None, 0.5, stats, log=lambda *_: None)
    assert len(survivors) == 3 and scores == []


def test_threshold_sweep_reports_what_each_cut_would_cost():
    sweep = threshold_sweep([0.05, 0.2, 0.45, 0.8, 0.95], points=(0.1, 0.5, 0.9))
    by_point = {row["threshold"]: row for row in sweep}
    assert by_point[0.1]["rejected"] == 1
    assert by_point[0.5]["rejected"] == 3
    assert by_point[0.9]["rejected"] == 4
    assert "threshold" in format_sweep(sweep)


def test_tier2_serves_cache_hits_without_calling_the_model(tmp_path):
    cache = VerdictCache(tmp_path / "c.sqlite")
    record = make_record()
    cache.put(record, Verdict(True, True, True, model="fake/model"))

    backend = FakeBackend([])
    judge = LLMJudge(backend)
    stats = RunStats(total=1)
    run_tier2([record], judge, cache, 8, stats, log=lambda *_: None)

    assert stats.cache_hits == 1
    assert backend.calls == 0
    assert record["verified"] is True
    assert "cached" in record["verify_reason"]
    cache.close()


def test_resume_leaves_already_verified_records_alone():
    assert already_verified(make_record(verified=True))
    assert already_verified(make_record(verify_reason="pass"))
    assert not already_verified(make_record())


def test_summarise_counts_three_outcomes():
    records = [
        make_record(verified=True, family="A_syntactic"),
        make_record(verified=False, family="A_syntactic"),
        make_record(verified=None, verify_reason="llm_parse_failure"),
    ]
    stats = summarise(records, RunStats(total=3))
    assert (stats.passed, stats.failed, stats.unjudged) == (1, 1, 1)
    assert stats.parse_failures == 1
    assert stats.per_family["A_syntactic"] == [2, 1]


# -- round trip --------------------------------------------------------------


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "v.jsonl"
    write_records([make_record(), make_record(variant="Other.")], path)
    assert len(read_records(path)) == 2


def test_malformed_input_lines_are_skipped_not_fatal(tmp_path):
    path = tmp_path / "v.jsonl"
    path.write_text('{"a":1}\nnot json\n{"b":2}\n', encoding="utf-8")
    assert len(read_records(path)) == 2


# -- prompts -----------------------------------------------------------------


def test_prompt_shows_only_judgement_relevant_fields():
    """Provenance is withheld: a model shown a generator name grades the generator."""
    shown = item_for(make_record(), 0)
    assert set(shown) == {"id", "base_sentence", "variant", "family", "subtype",
                          "operation", "net_negation"}
    assert "generator" not in shown


def test_prompt_asks_three_separate_questions():
    prompt = build_prompt([make_record()])
    for key in ("grammatical", "semantically_valid", "category_correct"):
        assert key in prompt


def test_prompt_forbids_rewriting_and_protects_the_base():
    prompt = build_prompt([make_record()])
    assert "Do NOT rewrite" in prompt
    assert "BASE SENTENCE is NOT under judgement" in prompt


def test_prompt_carries_all_four_few_shot_outcomes():
    """Including grammatical-but-implausible, which qwen2.5:7b needs shown."""
    prompt = build_prompt([make_record()])
    assert prompt.count("VERDICT:") == 4


def test_few_shot_covers_grammatical_but_semantically_invalid():
    """The case whose absence made the model collapse the three questions."""
    from negation.verification.prompts import _FEW_SHOT

    combinations = {
        (s["verdict"]["grammatical"], s["verdict"]["semantically_valid"],
         s["verdict"]["category_correct"])
        for s in _FEW_SHOT
    }
    assert (True, False, True) in combinations
    assert (True, True, False) in combinations
    assert (False, False, False) in combinations
    assert (True, True, True) in combinations


def test_prompt_tells_the_model_the_answers_are_independent():
    assert "INDEPENDENT" in build_prompt([make_record()])


def test_prompt_version_is_part_of_the_cache_identity():
    """A prompt change must not silently reuse verdicts from the old prompt."""
    from negation.verification.llm import OllamaBackend
    from negation.verification.prompts import PROMPT_VERSION

    assert PROMPT_VERSION in OllamaBackend(model="m").name


def test_prompt_lists_the_closed_family_set():
    prompt = build_prompt([make_record()])
    assert "K1_cancellation" in prompt and "M_modal" in prompt


# -- disagreement report -----------------------------------------------------


def test_clusters_group_by_labelled_and_suggested_family():
    records = [
        make_record(family="D_affixal", verify_category=False, suggested_family="E_antonym"),
        make_record(family="D_affixal", verify_category=False, suggested_family="E_antonym",
                    variant="x"),
        make_record(family="C_neg_adverb", verify_category=False, suggested_family="B_quantifier"),
        make_record(family="A_syntactic", verify_category=True),
    ]
    found = collect_disagreements(records)
    assert len(found) == 3
    clusters = cluster(found)
    assert clusters[0]["count"] == 2
    assert clusters[0]["labelled_family"] == "D_affixal"
    assert clusters[0]["suggested_family"] == "E_antonym"


def test_disagreement_report_writes_even_with_nothing_to_report(tmp_path):
    path = tmp_path / "d.md"
    summary = write_disagreement_report([make_record(verify_category=True)], path)
    assert path.exists() and summary["total"] == 0


def test_disagreement_report_lists_clusters(tmp_path):
    path = tmp_path / "d.md"
    records = [
        make_record(family="D_affixal", verify_category=False,
                    suggested_family="E_antonym", variant=f"V{i}.")
        for i in range(3)
    ]
    write_disagreement_report(records, path, model="m", corpus_size=3)
    text = path.read_text()
    assert "D_affixal" in text and "E_antonym" in text


# -- validation --------------------------------------------------------------


def test_kappa_is_one_for_perfect_agreement():
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0


def test_kappa_is_zero_when_a_rater_never_varies():
    """The failure kappa exists to catch: a verifier that always says yes."""
    gold = [True] * 9 + [False]
    assert cohens_kappa(gold, [True] * 10) == 0.0


def test_kappa_punishes_agreement_that_accuracy_rewards():
    gold = [True] * 9 + [False]
    predicted = [True] * 10
    accuracy = sum(1 for g, p in zip(gold, predicted) if g == p) / len(gold)
    assert accuracy == 0.9
    assert cohens_kappa(gold, predicted) < KAPPA_GATE


def test_kappa_is_negative_for_systematic_disagreement():
    assert cohens_kappa([True, False, True, False], [False, True, False, True]) < 0


def test_compare_matches_reference_to_verified_by_key():
    verified = [make_record(verify_grammatical=True, verify_semantic=True,
                            verify_category=True)]
    gold = [{"key": record_key(verified[0]), "grammatical": True,
             "semantic": True, "category": True}]
    agreements, matched = compare(gold, index_by_key(verified))
    assert matched == 1
    assert all(a.accuracy == 1.0 for a in agreements if a.n)


def test_compare_ignores_unlabelled_questions():
    """A reference row may label only some questions."""
    verified = [make_record(verify_grammatical=True, verify_semantic=False,
                            verify_category=True)]
    gold = [{"key": record_key(verified[0]), "grammatical": True,
             "semantic": None, "category": None}]
    agreements, _ = compare(gold, index_by_key(verified))
    by_question = {a.question: a for a in agreements}
    assert by_question["grammatical"].n == 1
    assert by_question["semantic"].n == 0


# -- constructed reference set ----------------------------------------------


def test_corrupting_grammar_changes_the_sentence():
    import random

    broken = corrupt_grammar("The function does not sort the array today.", random.Random(0))
    assert broken is not None and broken != "The function does not sort the array today."


def test_short_sentences_are_not_corrupted():
    import random

    assert corrupt_grammar("It works.", random.Random(0)) is None


def test_constructed_set_has_all_three_groups():
    records = [make_record(variant=f"The function does not sort the array number {i}.")
               for i in range(30)]
    built = build_constructed_set(records, n=30, seed=0)
    kinds = {row["kind"] for row in built}
    assert kinds == {"clean", "grammar_corrupted", "family_swapped"}


def test_constructed_labels_follow_from_construction():
    records = [make_record(variant=f"The function does not sort the array number {i}.")
               for i in range(30)]
    for row in build_constructed_set(records, n=30, seed=0):
        if row["kind"] == "grammar_corrupted":
            assert row["grammatical"] is False
        if row["kind"] == "family_swapped":
            assert row["category"] is False
            assert row["family"] != row["original_family"]
        if row["kind"] == "clean":
            assert row["grammatical"] is True


def test_constructed_set_includes_clean_records():
    """Without them a verifier that rejects everything would score perfectly."""
    records = [make_record(variant=f"The function does not sort the array number {i}.")
               for i in range(30)]
    built = build_constructed_set(records, n=30, seed=0)
    assert sum(1 for r in built if r["kind"] == "clean") > 0
