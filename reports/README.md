# Reports

The project's audit record, in the order the work was done. Each report is unchanged from the commit
that wrote it except for one italic line under its title and updated filenames where it cites another
report; its reproduction commands use the layout of their own commits (`docs/PATHS.md` translates them).
"Date" is the date the report was first committed. 03–05 were committed together on 2026-10-05, after 06
and 07; they are numbered before 06 because 06 is built on 05's polarity pipeline and 03's dataset.

| # | report | old filename | date | what that round found |
| --- | --- | --- | --- | --- |
| 01 | [`01-negation-direction.md`](01-negation-direction.md) | `experiments/negation_direction/REPORT.md` | 2026-09-22 | A negation direction exists in sentence-embedding space, but it is mostly a cue detector: weak on implicit negation and antonyms, wrong on cancellation on two of three models, and carries almost no intensity information. |
| 02 | [`02-negation-detection.md`](02-negation-detection.md) | `EXPERIMENT_2_REPORT.md` | 2026-09-22 | Is B the negation of A? An off-the-shelf NLI cross-encoder beats every embedding- or generator-based scorer (CV ROC-AUC 0.993 vs 0.744; paraphrase FPR 3.0% vs 36–93%). Full write-up. |
| 02a | [`02a-negation-detection-architectures.md`](02a-negation-detection-architectures.md) | `experiments/negation_detection/REPORT.md` | 2026-09-22 | The shorter in-tree report of the same experiment; also documents the data sources and their licences. |
| 03 | [`03-pattern-dataset.md`](03-pattern-dataset.md) | `PATTERN.md` | 2026-10-05 | Is a graded negation penalty justified? Not determined: the LLM labels failed their validation gate (Krippendorff's alpha 0.586 < 0.6), so no formula was fitted. |
| 04 | [`04-pattern-exploration.md`](04-pattern-exploration.md) | `experiments/pattern/REPORT2.md` | 2026-10-05 | Exploration of the gate-failed labels; hypotheses only (alpha on negation pairs alone is 0.365). |
| 05 | [`05-polarity.md`](05-polarity.md) | `REPORT_POLARITY.md` | 2026-10-05 | A sentence-level polarity function beats cue counting (test AUC 0.638 vs 0.493) but loses to the reference-set architecture (0.743) and the NLI cross-encoder (0.840). |
| 06 | [`06-composite-nli.md`](06-composite-nli.md) | `Report_NLI.md` | 2026-10-04 | Content gate → cancellation veto → NLI: CANNOT phase-2 accuracy 78.8% → 92.5%, and all of the gain is the content gate. |
| 07 | [`07-honest-baseline.md`](07-honest-baseline.md) | `Report_NLI2.md` | 2026-10-04 | Clean DEV (no generator output), honest NLI threshold, rule fixes, conditionals; a gradient-boosted combiner over the pipeline's signals reaches 97.9% on phase 2 and 94.0% on all of TEST. |
| 08 | [`08-guardrail-ensemble.md`](08-guardrail-ensemble.md) | `report_f1.md` | 2026-10-11 | Cancellation override, affirmation generator, NLI ensemble, guardrail (rejected), paragraph matching, domain case study and dashboard; the Phase 0 baseline (GBM + cancellation override) is the configuration the evidence supports. |
| 09 | [`09-overlap-shortcut.md`](09-overlap-shortcut.md) | `report_f2.md` | 2026-10-11 | The combiner learned a word-overlap shortcut and misses reworded negations; a topic-gated low-overlap NLI override plus a 4-model ensemble lifts low-overlap recall 75.2% → 90.7% (n = 129, human-written) at 5.1 pts of low-overlap specificity. |

Other files here:

- `results/<round>/`: every committed results JSON, filenames unchanged; each report's `results/…` paths
  resolve to its own round's folder (`f1` = 08, `f2` = 09, `nli2` = 07, `composite` = 06, `polarity` = 05,
  `pattern` = 03–04, `negation_detection` = 02/02a, `negation_direction` = 01).
- `figures/<round>/`: the plots and figures the reports cite.
- `verification_disagreements.md`, `verifier_validation.md`, `verifier_*.jsonl`,
  `verifier_validation_p1_baseline.txt`: stage-2 LLM verifier artefacts, written by `negation/verification`
  and `scripts/verify_validate.py` (unchanged location).
