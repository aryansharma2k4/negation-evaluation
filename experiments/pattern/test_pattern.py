"""The three properties the brief asks to be enforced, plus the agreement metric.

    ./venv/bin/python -m pytest experiments/pattern -q
"""

import json
from pathlib import Path

import numpy as np
import pytest

import analyse
import label

DATA = Path(__file__).resolve().parent / "data"
META_ONLY = {"family", "subtype", "cue_count", "net_negation", "intensity_hint", "kind",
             "operation", "input_polarity", "scope_target", "base_line", "subtopic"}


def jsonl(name):
    return [json.loads(l) for l in (DATA / name).read_text().splitlines() if l.strip()]


# --- metadata is not readable from the labelling input -------------------------------

def test_labelling_input_holds_only_id_and_sentences():
    rows = jsonl("to_label.jsonl")
    assert rows and all(set(r) == {"id", "a", "b"} for r in rows)


def test_label_script_never_opens_metadata():
    source = Path(label.__file__).read_text()
    assert "metadata" not in source.replace("Never metadata.jsonl", "")


def test_prompt_carries_no_family_or_negation_vocabulary():
    items = label.load_items()
    families = {m["family"] for m in jsonl("metadata.jsonl")}
    prompt = label.build_prompt([(i["a"], i["b"]) for i in items[:5]])
    assert "negat" not in label.PROMPT_HEAD.lower()
    assert "polarity" not in label.PROMPT_HEAD.lower()
    assert not any(f in prompt for f in families)
    assert not any(i["id"] in prompt for i in items[:5])


def test_metadata_is_keyed_by_the_same_ids():
    assert {r["id"] for r in jsonl("to_label.jsonl")} == {r["id"] for r in jsonl("metadata.jsonl")}


# --- no base sentence spans CV folds --------------------------------------------------

def test_no_base_sentence_spans_cv_folds():
    rows = [m for m in jsonl("metadata.jsonl") if m["kind"] == "negation"]
    groups = np.array([r["base_line"] for r in rows])
    splits = analyse.folds(rows)
    assert len(splits) == analyse.FOLDS
    for train, test in splits:
        assert not set(groups[train]) & set(groups[test])
    assert sorted(np.concatenate([t for _, t in splits])) == list(range(len(rows)))


# --- the three passes are independent -------------------------------------------------

class Recorder:
    def __init__(self, score):
        self.prompts, self.score = [], score

    def __call__(self, prompt, seed):
        self.prompts.append((prompt, seed))
        n = prompt.count("\nSentence 1: ") - 3  # minus the three worked examples
        return json.dumps({"scores": [{"pair": i, "score": self.score} for i in range(1, n + 1)]})


@pytest.fixture(scope="module")
def items():
    return label.load_items()[:23]


def test_a_pass_depends_only_on_the_items(items):
    # pass 3 asked after pass 1 scored everything 0 == pass 3 asked after pass 1 scored 5
    for pass_no in (1, 2, 3):
        first, second = Recorder(0), Recorder(5)
        label.run_pass(items, 1, ask=first, log=None)
        a = Recorder(3); label.run_pass(items, pass_no, ask=a, log=None)
        label.run_pass(items, 1, ask=second, log=None)
        b = Recorder(3); label.run_pass(items, pass_no, ask=b, log=None)
        assert a.prompts == b.prompts


def test_passes_differ_in_the_intended_way_only(items):
    rec = {k: Recorder(3) for k in (1, 2, 3)}
    out = {k: label.run_pass(items, k, ask=rec[k], log=None) for k in (1, 2, 3)}
    seeds = {k: {s for _, s in rec[k].prompts} for k in rec}
    assert len(set.union(*seeds.values())) == 3                      # a seed per pass
    ids = {k: [r["id"] for r in out[k]] for k in out}
    assert ids[1] == ids[2]                                           # 2: same order ...
    assert rec[2].prompts[0][0].count(items[0]["b"] + "\nSentence 2: " + items[0]["a"]) == 1  # ... swapped
    assert ids[3] != ids[1] and sorted(ids[3]) == sorted(ids[1])      # 3: reshuffled


def test_written_passes_cover_every_item_once():
    ids = {r["id"] for r in jsonl("to_label.jsonl")}
    for k in (1, 2, 3):
        path = DATA / f"labels_pass{k}.jsonl"
        if not path.exists():
            pytest.skip("labelling has not been run")
        rows = jsonl(path.name)
        assert [r["pass"] for r in rows] == [k] * len(rows)
        assert sorted(r["id"] for r in rows) == sorted(ids)


# --- the agreement metric ---------------------------------------------------------------

def test_krippendorff_ordinal_matches_reference_package():
    # values checked against `krippendorff` 0.8 (alpha(..., "ordinal")) on the same data
    rng = np.random.default_rng(1)
    expected = [1.0, 0.77956, 0.474135, 0.304389]
    for trial, want in enumerate(expected):
        truth = rng.integers(0, 6, 200)
        data = np.clip(truth[None, :] + rng.integers(-1, 2, (3, 200)) * trial, 0, 5).astype(float)
        data[rng.random(data.shape) < 0.05] = np.nan
        units = [[None if np.isnan(v) else v for v in col] for col in data.T]
        assert analyse.krippendorff_ordinal(units) == pytest.approx(want, abs=1e-6)


def test_parse_scores_never_guesses():
    assert label.parse_scores("not json", 2) == [None, None]
    assert label.parse_scores('{"scores": [{"pair": 2, "score": 9}]}', 2) == [None, None]
    assert label.parse_scores('{"scores": [{"pair": 1, "score": 3.6}]}', 2) == [4, None]
