"""Checks for report_f1.md's new logic: paragraph assignment / verdicts, the guardrail's
downgrade-only rule, and the cancellation override. Constructed inputs only.

    ./venv/bin/python -m pytest experiments/f1 -q
"""

import numpy as np

import guardrail as G
import paragraphs as PG


def test_optimal_beats_greedy_on_a_crossed_matrix():
    S = np.array([[0.9, 0.8], [0.85, 0.1]])       # greedy takes (0,0) then is stuck with (1,1)=0.1
    assert PG.assign(S, "greedy", None) == {0: 0, 1: 1}
    assert PG.assign(S, "optimal", None) == {0: 1, 1: 0}


def test_threshold_before_assignment_leaves_weak_cells_unmatched():
    S = np.array([[0.9, 0.2], [0.3, 0.4]])
    assert PG.assign(S, "optimal", 0.5) == {0: 0}


def test_verdicts_cover_all_four_outcomes():
    S = np.array([[0.9, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.2]])
    NEG = np.array([[False] * 3, [False, True, False], [False] * 3])
    tv, extra = PG.verdicts(S, NEG, {0: 0, 1: 1}, tau=0.5)
    assert tv == ["covered", "contradicted", "missing"] and extra == [2]


def test_guardrail_only_downgrades_and_abstains_without_variants():
    scores = {k: np.array([0.9, 0.9, 0.1, 0.9]) for k in G.SETS}
    m = {k: np.array([0.2, -0.1, -0.5, np.nan]) for k in G.SETS}
    out = G.apply(scores, 0.5, m, 0.0)["test"]
    assert out[0] == 0.9                 # margin above delta: kept
    assert out[1] < 0.5                  # margin below delta: downgraded
    assert out[2] == 0.1                 # never upgraded
    assert out[3] == 0.9                 # no variants: abstain


def test_cancellation_override_fires_both_directions():
    import phase0 as P0
    from common import C
    rows = [{"a": "The process is common.", "b": "The process is not uncommon."},
            {"a": "The process is not uncommon.", "b": "The process is common."},
            {"a": "The process is common.", "b": "The process is not common."}]
    analyses = C.P.analyse_many([s for r in rows for s in (r["a"], r["b"])])
    assert P0.override_mask(rows, analyses).tolist() == [True, True, False]
