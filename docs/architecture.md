# Architecture: the current pipeline

What the adopted system (report 09, `reports/09-overlap-shortcut.md`) does when it is asked whether a
student sentence **B** negates a teacher sentence **A**. The code is `scripts/f1/detector.py`
(`Detector.adopted()`); the configuration is `reports/results/f2/final_config.json`. Mermaid diagrams of the
stage-1 generator are in `pipeline_diagrams.md`; the project's history is in `reports/README.md`.

## 1. Signals computed for every pair

- **Polarity analysis** of each sentence (`scripts/polarity/polarity.py`): the spaCy parse is reduced to a
  list of negation cues, each with its family, the clause it sits in (`negation.nlp_core.clause_scope`)
  and whether that clause is hedged or intensified. From it come the cue counts, the net negation
  (polarity B − polarity A), cancellation (two cues cancelling in one clause, *not impossible*) and
  antonym alignment.
- **Conditionals** (`scripts/nli2/conditionals.py`): whether each side is a conditional, its antecedent
  and consequent, and whether only the antecedent's polarity flips.
- **Embedding similarity**: mpnet (`all-mpnet-base-v2`) cosine of A and B.
- **Surface features**: lengths, length difference, token overlap.
- **NLI**: P(contradiction / entailment / neutral) from four cross-encoders, premise A, hypothesis B
  (`deberta_base`, `roberta_base`, `ml_deberta_base_anli`, `bart_large`; pinned revisions in
  `scripts/f1/nli_models.py`). This is most of the cost (~275 ms per pair on CPU).

## 2. Decision, in order

1. **Cancellation override.** If the polarity analysis finds a cancelling cue pair, the answer is *not a
   negation* and nothing later can change it.
2. **Combiner.** A gradient-boosted tree model over 26 features (the 17 base features of
   `scripts/nli2/phase4.py:FEATURES` plus the three extra models' NLI probabilities) gives a probability;
   it is a negation if that is at least the threshold fit by Youden's J on out-of-fold clean-DEV
   predictions. Trained on `datasets/splits/dev_clean.csv` only.
3. **Low-overlap NLI override.** If the combiner said *no*, but token overlap < 0.5, the reference NLI
   model's P(contradiction) > 0.9 and the cosine ≥ 0.6 (same topic), the answer becomes *negation*. This
   exists because the combiner learned a word-overlap shortcut and misses reworded negations
   (report 09 §2); the topic floor keeps unrelated sentences from being flagged.

The output records which stage decided (`stage`), the combiner probability and threshold, and on
request the features, per-feature contributions and every NLI model's probabilities.

## 3. Around the detector

- **Severity** (`scripts/f2/severity.py`): for a pair already called a negation, a descriptive label
  (scope, intensified, hedged, flat, lexical) from the differing cues. It changes no verdict.
- **Paragraphs** (`scripts/f1/paragraphs.py`): teacher and student sentences are matched one-to-one by
  optimal assignment on cosine similarity (threshold fit on clean DEV), then each matched pair goes
  through the detector to give *covered*, *contradicted* or *missing* per teacher sentence.
- **Dashboard** (`dashboard/app.py`): single-pair, paragraph and results views over the same detector.
- **Stage-1 generator** (`negation/`, `scripts/run.sh`): produces negation variants of a sentence in 14
  families plus affirmation, with cue, scope and depth metadata. The detector does not call it at
  prediction time; it supplied the reference sets, the polarity lexicons and the evaluation families.

## 4. Data discipline

Thresholds and models are fit on clean DEV; selection uses VALIDATION and the `val_ood` select half; TEST
(CANNOT phase-2 test + xNot360) is only read to report, and every read is logged in the round's
`test_reads.json`. Machine-generated evaluation data is labelled wherever its numbers appear.
