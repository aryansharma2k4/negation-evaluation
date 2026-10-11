"""Iteration log prose. Words only: every number in the report comes from
results_polarity.json. Written after looking at each version's dev errors
(run.py --show-errors), before running the next version."""

ITERATIONS = {
    "v1": {
        "changed": "Any cue that `existing_cues` or a contrastive frame finds -> 1.0, else 0.0.",
        "hypothesis": "The crude baseline: presence of a cue anywhere.",
        "errors": "Every K1 pair is a false positive (no cancellation). False negatives: lexical "
                  "substitutions no cue detector sees (*minimizes/maximizes*, *appeared/disappeared*), "
                  "and pairs where A already carries a cue, so both sides score 1 (*unhappy* vs "
                  "*not unhappy*; a cue in A's subordinate clause).",
    },
    "v2": {
        "changed": "Cues inside one `clause_scope` combine by parity (two cancel, three flip back); "
                   "the sentence takes its most negated clause.",
        "hypothesis": "The core claim: cancellation-aware polarity separates K1 from real negations.",
        "errors": "K1 false positives nearly vanish. Remaining false positives are CANNOT paraphrases "
                  "that swap a negation for an antonym (*not alive* / *dead*, *not a good idea* / "
                  "*a bad idea*): a real polarity flip with the meaning kept, which a per-sentence "
                  "score cannot see. False negatives: negative words outside the lexicons (*missing*, "
                  "*absent*, *banned from*, *except*), antonym pairs, and A carrying a cue in another "
                  "clause.",
    },
    "v3": {
        "changed": "Each cue weighted by intensity (hedged, neutral, intensified, partial) inside the parity.",
        "hypothesis": "Graded intensity makes the polarity difference more informative.",
        "errors": "Dev AUC falls. The task is binary: *probably not* is labelled a negation like any "
                  "other, so shrinking its polarity only costs recall. Nothing in the errors asks for "
                  "intensity grading.",
    },
    "v4": {
        "changed": "Implicit, affixal, prepositional and contrastive cues get their own family weight "
                   "on top of the intensity weight.",
        "hypothesis": "Lexical cues are weaker evidence of negation than explicit ones.",
        "errors": "Dev AUC falls again and the K1 false-positive rate rises sharply. The soft parity "
                  "no longer cancels when both cues weigh less than 1: *not impossible* keeps a "
                  "residual score. The weighting fights the cancellation.",
    },
    "v5": {
        "changed": "Deviation from the suggested ladder (per-subtype weights). Parity is decided first, "
                   "crisply; the weight of the strongest cue is applied only to a clause left negated.",
        "hypothesis": "v4's K1 regression is the weights leaking through the parity, not the weights "
                      "themselves. Per-subtype weights were skipped: v3 and v4 already show that "
                      "hand-set grading lowers dev AUC, so more hand-set grades would test the same "
                      "direction with more parameters. v6 tests grading directly by fitting it.",
        "errors": "K1 is fixed again. Dev AUC sits between v2 and v4: with the leak removed, the "
                  "remaining cost is the grading itself.",
    },
    "v6": {
        "changed": "v5's family and intensity weights fit on DEV by coordinate ascent on DEV AUC "
                   "(grid 0.1-1.0, two sweeps).",
        "hypothesis": "If grading helps binary detection, fitted weights beat both hand-set and flat ones.",
        "errors": "Dev improves, validation barely moves, and the dev-validation gap widens: the stop "
                  "condition. The fitted weights rank hedged and intensified cues above plain ones, "
                  "which matches DEV's composition (hundreds of easy H and I positives in our dataset) "
                  "rather than anything about negation. Stopped adding complexity here; v7 and v8 "
                  "were not run.",
    },
}

NOT_RUN = ("Not run, because the v6 gap widened. The errors pointed to three things, all "
           "coverage rather than grading: (1) negative words outside the lexicons (*missing*, "
           "*absent*, *banned*, *except*); (2) the max-over-clauses rule saturating when A already "
           "carries a cue in another clause; (3) antonym substitution, which no per-sentence score "
           "can represent because the polarity lives in the *pair*.")
