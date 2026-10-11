# Path map: old layout → current layout

The repository was reorganised in one commit, **`chore: reorganise repository layout`**, the first commit on
`chore/repo-layout` after `b352e7c` ("layout" in the tables below; a commit cannot contain its own hash, so
`git log --oneline -1 -- docs/PATHS.md` prints it).
Every move was a `git mv`, so `git log --follow <new path>` traces each file through its full history.

**The reports' reproduction commands are deliberately not rewritten.** They were correct at the commit hashes
they list. To run one today, translate its paths with the prefix rules below; the full table follows them.

## Prefix rules (cover every row of the tables)

| old prefix | new prefix |
| --- | --- |
| `experiments/<round>/*.py` (code, tests, conftest) | `scripts/<round>/*.py` |
| `experiments/{f1,f2,nli2}/results/` | `reports/results/{f1,f2,nli2}/` (filenames unchanged) |
| `experiments/<round>/results*.json`, `experiments/composite/test_scorings.json` | `reports/results/<round>/` (filenames unchanged) |
| `results/<file>.json` written inside a round's report | `reports/results/<that round>/<file>.json` |
| `experiments/<round>/plots/` | `reports/figures/<round>/` |
| `experiments/<round>/cache/` (gitignored) | `.cache/<round>/` (gitignored) |
| `experiments/negation_detection/data/{phase1,phase2,xnot360}.jsonl` | `datasets/splits/` |
| `experiments/negation_detection/data/raw/` (gitignored) | `datasets/raw/negation_detection/` (gitignored) |
| `experiments/negation_detection/data/refsets.jsonl` (gitignored) | `datasets/generated/negation_detection/refsets.jsonl` (gitignored) |
| `experiments/negation_direction/data/` | `datasets/generated/negation_direction/` |
| `experiments/pattern/data/human_seed.csv` | `datasets/eval/human_seed.csv` |
| `experiments/pattern/data/` | `datasets/generated/pattern/` |
| `experiments/f1/data/affirmation_spotcheck.json` | `datasets/eval/affirmation_spotcheck.json` |
| `experiments/f2/raw/` (gitignored) | `datasets/raw/nli/` (gitignored) |
| `data/dev_clean.csv` (gitignored), `data/val_ood.csv` | `datasets/splits/` |
| `data/paragraph_eval*.jsonl`, `data/student_answers_human.jsonl` | `datasets/eval/` |
| `data/student_answers.jsonl` | `datasets/generated/` |
| `data/definitions.jsonl` | `datasets/raw/` (committed; CC BY-SA 4.0) |
| `data/sample_sentences.txt` | `datasets/` |
| `data/README.md` | `datasets/eval/README.md` (a new `datasets/README.md` indexes every file) |
| `data/external/`, `data/embeddings/` (gitignored) | `datasets/raw/external/`, `datasets/raw/embeddings/` (gitignored) |
| `docs/figures/overlap_mechanism.png` | `reports/figures/f2/overlap_mechanism.png` |
| `docs/diagrams.md` | `docs/pipeline_diagrams.md` |
| `run.sh`, `run_test.sh`, `neg.py` | `scripts/run.sh`, `scripts/run_test.sh`, `scripts/neg.py` |
| report files | `reports/NN-name.md`, see the reports table |

## Reports

| old | new | moved in |
| --- | --- | --- |
| `experiments/negation_direction/REPORT.md` | `reports/01-negation-direction.md` | layout |
| `EXPERIMENT_2_REPORT.md` | `reports/02-negation-detection.md` | layout |
| `experiments/negation_detection/REPORT.md` | `reports/02a-negation-detection-architectures.md` | layout |
| `PATTERN.md` | `reports/03-pattern-dataset.md` | layout |
| `experiments/pattern/REPORT2.md` | `reports/04-pattern-exploration.md` | layout |
| `REPORT_POLARITY.md` | `reports/05-polarity.md` | layout |
| `Report_NLI.md` | `reports/06-composite-nli.md` | layout |
| `Report_NLI2.md` | `reports/07-honest-baseline.md` | layout |
| `report_f1.md` | `reports/08-guardrail-ensemble.md` | layout |
| `report_f2.md` | `reports/09-overlap-shortcut.md` | layout |

## Every tracked file, by old directory

### `(repository root)` (3)

| old | new | moved in |
| --- | --- | --- |
| `neg.py` | `scripts/neg.py` | layout |
| `run.sh` | `scripts/run.sh` | layout |
| `run_test.sh` | `scripts/run_test.sh` | layout |

### `data/` (8)

| old | new | moved in |
| --- | --- | --- |
| `data/README.md` | `datasets/eval/README.md` | layout |
| `data/definitions.jsonl` | `datasets/raw/definitions.jsonl` | layout |
| `data/paragraph_eval.jsonl` | `datasets/eval/paragraph_eval.jsonl` | layout |
| `data/paragraph_eval_human.jsonl` | `datasets/eval/paragraph_eval_human.jsonl` | layout |
| `data/sample_sentences.txt` | `datasets/sample_sentences.txt` | layout |
| `data/student_answers.jsonl` | `datasets/generated/student_answers.jsonl` | layout |
| `data/student_answers_human.jsonl` | `datasets/eval/student_answers_human.jsonl` | layout |
| `data/val_ood.csv` | `datasets/splits/val_ood.csv` | layout |

### `docs/` (1)

| old | new | moved in |
| --- | --- | --- |
| `docs/diagrams.md` | `docs/pipeline_diagrams.md` | layout |

### `docs/figures/` (1)

| old | new | moved in |
| --- | --- | --- |
| `docs/figures/overlap_mechanism.png` | `reports/figures/f2/overlap_mechanism.png` | layout |

### `experiments/composite/` (7)

| old | new | moved in |
| --- | --- | --- |
| `experiments/composite/composite.py` | `scripts/composite/composite.py` | layout |
| `experiments/composite/conftest.py` | `scripts/composite/conftest.py` | layout |
| `experiments/composite/report_composite.py` | `scripts/composite/report_composite.py` | layout |
| `experiments/composite/results_composite.json` | `reports/results/composite/results_composite.json` | layout |
| `experiments/composite/run_composite.py` | `scripts/composite/run_composite.py` | layout |
| `experiments/composite/test_composite.py` | `scripts/composite/test_composite.py` | layout |
| `experiments/composite/test_scorings.json` | `reports/results/composite/test_scorings.json` | layout |

### `experiments/f1/` (44)

| old | new | moved in |
| --- | --- | --- |
| `experiments/f1/.gitignore` | `scripts/f1/.gitignore` | layout |
| `experiments/f1/conftest.py` | `scripts/f1/conftest.py` | layout |
| `experiments/f1/core.py` | `scripts/f1/core.py` | layout |
| `experiments/f1/dashboard_latency.py` | `scripts/f1/dashboard_latency.py` | layout |
| `experiments/f1/data/affirmation_spotcheck.json` | `datasets/eval/affirmation_spotcheck.json` | layout |
| `experiments/f1/detector.py` | `scripts/f1/detector.py` | layout |
| `experiments/f1/domain.py` | `scripts/f1/domain.py` | layout |
| `experiments/f1/ensemble.py` | `scripts/f1/ensemble.py` | layout |
| `experiments/f1/fetch_definitions.py` | `scripts/f1/fetch_definitions.py` | layout |
| `experiments/f1/fetch_models.py` | `scripts/f1/fetch_models.py` | layout |
| `experiments/f1/final_f1.py` | `scripts/f1/final_f1.py` | layout |
| `experiments/f1/guardrail.py` | `scripts/f1/guardrail.py` | layout |
| `experiments/f1/inflection_bench.py` | `scripts/f1/inflection_bench.py` | layout |
| `experiments/f1/make_paragraphs.py` | `scripts/f1/make_paragraphs.py` | layout |
| `experiments/f1/make_student_answers.py` | `scripts/f1/make_student_answers.py` | layout |
| `experiments/f1/nli_models.py` | `scripts/f1/nli_models.py` | layout |
| `experiments/f1/paragraphs.py` | `scripts/f1/paragraphs.py` | layout |
| `experiments/f1/phase0.py` | `scripts/f1/phase0.py` | layout |
| `experiments/f1/phase1.py` | `scripts/f1/phase1.py` | layout |
| `experiments/f1/results/0_1_override.json` | `reports/results/f1/0_1_override.json` | layout |
| `experiments/f1/results/0_2_conditional_fix.json` | `reports/results/f1/0_2_conditional_fix.json` | layout |
| `experiments/f1/results/0_3_baseline.json` | `reports/results/f1/0_3_baseline.json` | layout |
| `experiments/f1/results/1_1_inflection.json` | `reports/results/f1/1_1_inflection.json` | layout |
| `experiments/f1/results/1_2_affirmation.json` | `reports/results/f1/1_2_affirmation.json` | layout |
| `experiments/f1/results/1_3_inventory.json` | `reports/results/f1/1_3_inventory.json` | layout |
| `experiments/f1/results/2_1_timing.json` | `reports/results/f1/2_1_timing.json` | layout |
| `experiments/f1/results/2_2_standalone.json` | `reports/results/f1/2_2_standalone.json` | layout |
| `experiments/f1/results/2_3_disagreement.json` | `reports/results/f1/2_3_disagreement.json` | layout |
| `experiments/f1/results/2_4_ensembles.json` | `reports/results/f1/2_4_ensembles.json` | layout |
| `experiments/f1/results/2_5_overfit.json` | `reports/results/f1/2_5_overfit.json` | layout |
| `experiments/f1/results/3_3_coverage.json` | `reports/results/f1/3_3_coverage.json` | layout |
| `experiments/f1/results/3_4_guardrail.json` | `reports/results/f1/3_4_guardrail.json` | layout |
| `experiments/f1/results/4_2_matching.json` | `reports/results/f1/4_2_matching.json` | layout |
| `experiments/f1/results/4_2_tau.json` | `reports/results/f1/4_2_tau.json` | layout |
| `experiments/f1/results/4_5_paragraphs.json` | `reports/results/f1/4_5_paragraphs.json` | layout |
| `experiments/f1/results/5_1_definitions.json` | `reports/results/f1/5_1_definitions.json` | layout |
| `experiments/f1/results/5_2_answers.json` | `reports/results/f1/5_2_answers.json` | layout |
| `experiments/f1/results/5_3_domain.json` | `reports/results/f1/5_3_domain.json` | layout |
| `experiments/f1/results/6_1_latency.json` | `reports/results/f1/6_1_latency.json` | layout |
| `experiments/f1/results/final_config.json` | `reports/results/f1/final_config.json` | layout |
| `experiments/f1/results/final_f1.json` | `reports/results/f1/final_f1.json` | layout |
| `experiments/f1/results/test_reads.json` | `reports/results/f1/test_reads.json` | layout |
| `experiments/f1/test_f1.py` | `scripts/f1/test_f1.py` | layout |
| `experiments/f1/variants.py` | `scripts/f1/variants.py` | layout |

### `experiments/f2/` (37)

| old | new | moved in |
| --- | --- | --- |
| `experiments/f2/.gitignore` | `scripts/f2/.gitignore` | layout |
| `experiments/f2/conditional_val.py` | `scripts/f2/conditional_val.py` | layout |
| `experiments/f2/core2.py` | `scripts/f2/core2.py` | layout |
| `experiments/f2/deberta.py` | `scripts/f2/deberta.py` | layout |
| `experiments/f2/eval2.py` | `scripts/f2/eval2.py` | layout |
| `experiments/f2/fetch_data.py` | `scripts/f2/fetch_data.py` | layout |
| `experiments/f2/final_f2.py` | `scripts/f2/final_f2.py` | layout |
| `experiments/f2/latency7.py` | `scripts/f2/latency7.py` | layout |
| `experiments/f2/overlap.py` | `scripts/f2/overlap.py` | layout |
| `experiments/f2/override.py` | `scripts/f2/override.py` | layout |
| `experiments/f2/para2.py` | `scripts/f2/para2.py` | layout |
| `experiments/f2/rebalance.py` | `scripts/f2/rebalance.py` | layout |
| `experiments/f2/results/0_1_overlap.json` | `reports/results/f2/0_1_overlap.json` | layout |
| `experiments/f2/results/0_2_mechanism.json` | `reports/results/f2/0_2_mechanism.json` | layout |
| `experiments/f2/results/0_3_val_ood.json` | `reports/results/f2/0_3_val_ood.json` | layout |
| `experiments/f2/results/0_4_baseline.json` | `reports/results/f2/0_4_baseline.json` | layout |
| `experiments/f2/results/1_1_rule_grid.json` | `reports/results/f2/1_1_rule_grid.json` | layout |
| `experiments/f2/results/1_3_override.json` | `reports/results/f2/1_3_override.json` | layout |
| `experiments/f2/results/1_4_paragraphs.json` | `reports/results/f2/1_4_paragraphs.json` | layout |
| `experiments/f2/results/2_1_swap_diagnosis.json` | `reports/results/f2/2_1_swap_diagnosis.json` | layout |
| `experiments/f2/results/2_2_deberta.json` | `reports/results/f2/2_2_deberta.json` | layout |
| `experiments/f2/results/2_4_latency.json` | `reports/results/f2/2_4_latency.json` | layout |
| `experiments/f2/results/3_1_family_overlap.json` | `reports/results/f2/3_1_family_overlap.json` | layout |
| `experiments/f2/results/3_2_augmentation.json` | `reports/results/f2/3_2_augmentation.json` | layout |
| `experiments/f2/results/3_3_rebalance.json` | `reports/results/f2/3_3_rebalance.json` | layout |
| `experiments/f2/results/4_2_affirmation_coverage.json` | `reports/results/f2/4_2_affirmation_coverage.json` | layout |
| `experiments/f2/results/5_1_conditionals.json` | `reports/results/f2/5_1_conditionals.json` | layout |
| `experiments/f2/results/5_2_variant_b.json` | `reports/results/f2/5_2_variant_b.json` | layout |
| `experiments/f2/results/6_severity.json` | `reports/results/f2/6_severity.json` | layout |
| `experiments/f2/results/6_severity_v1.json` | `reports/results/f2/6_severity_v1.json` | layout |
| `experiments/f2/results/7_latency.json` | `reports/results/f2/7_latency.json` | layout |
| `experiments/f2/results/final_config.json` | `reports/results/f2/final_config.json` | layout |
| `experiments/f2/results/final_f2.json` | `reports/results/f2/final_f2.json` | layout |
| `experiments/f2/results/raw_manifest.json` | `reports/results/f2/raw_manifest.json` | layout |
| `experiments/f2/results/test_reads.json` | `reports/results/f2/test_reads.json` | layout |
| `experiments/f2/severity.py` | `scripts/f2/severity.py` | layout |
| `experiments/f2/val_ood.py` | `scripts/f2/val_ood.py` | layout |

### `experiments/negation_detection/` (20)

| old | new | moved in |
| --- | --- | --- |
| `experiments/negation_detection/.gitignore` | `scripts/negation_detection/.gitignore` | layout |
| `experiments/negation_detection/conftest.py` | `scripts/negation_detection/conftest.py` | layout |
| `experiments/negation_detection/data.py` | `scripts/negation_detection/data.py` | layout |
| `experiments/negation_detection/data/phase1.jsonl` | `datasets/splits/phase1.jsonl` | layout |
| `experiments/negation_detection/data/phase2.jsonl` | `datasets/splits/phase2.jsonl` | layout |
| `experiments/negation_detection/data/xnot360.jsonl` | `datasets/splits/xnot360.jsonl` | layout |
| `experiments/negation_detection/detect.py` | `scripts/negation_detection/detect.py` | layout |
| `experiments/negation_detection/plots/margins_minilm.png` | `reports/figures/negation_detection/margins_minilm.png` | layout |
| `experiments/negation_detection/plots/margins_mpnet.png` | `reports/figures/negation_detection/margins_mpnet.png` | layout |
| `experiments/negation_detection/plots/resid_vs_deltap_minilm.png` | `reports/figures/negation_detection/resid_vs_deltap_minilm.png` | layout |
| `experiments/negation_detection/plots/resid_vs_deltap_mpnet.png` | `reports/figures/negation_detection/resid_vs_deltap_mpnet.png` | layout |
| `experiments/negation_detection/plots/scores_minilm.png` | `reports/figures/negation_detection/scores_minilm.png` | layout |
| `experiments/negation_detection/plots/scores_mpnet.png` | `reports/figures/negation_detection/scores_mpnet.png` | layout |
| `experiments/negation_detection/refsets.py` | `scripts/negation_detection/refsets.py` | layout |
| `experiments/negation_detection/report_experiment2.py` | `scripts/negation_detection/report_experiment2.py` | layout |
| `experiments/negation_detection/report_tables.py` | `scripts/negation_detection/report_tables.py` | layout |
| `experiments/negation_detection/results.json` | `reports/results/negation_detection/results.json` | layout |
| `experiments/negation_detection/results_mpnet.json` | `reports/results/negation_detection/results_mpnet.json` | layout |
| `experiments/negation_detection/test_negation_detection.py` | `scripts/negation_detection/test_negation_detection.py` | layout |
| `experiments/negation_detection/verify_report.py` | `scripts/negation_detection/verify_report.py` | layout |

### `experiments/negation_direction/` (20)

| old | new | moved in |
| --- | --- | --- |
| `experiments/negation_direction/.gitignore` | `scripts/negation_direction/.gitignore` | layout |
| `experiments/negation_direction/conftest.py` | `scripts/negation_direction/conftest.py` | layout |
| `experiments/negation_direction/data/control_candidates.jsonl` | `datasets/generated/negation_direction/control_candidates.jsonl` | layout |
| `experiments/negation_direction/data/control_keep.txt` | `datasets/generated/negation_direction/control_keep.txt` | layout |
| `experiments/negation_direction/data/control_pairs.jsonl` | `datasets/generated/negation_direction/control_pairs.jsonl` | layout |
| `experiments/negation_direction/data/variants.jsonl` | `datasets/generated/negation_direction/variants.jsonl` | layout |
| `experiments/negation_direction/experiment.py` | `scripts/negation_direction/experiment.py` | layout |
| `experiments/negation_direction/make_controls.py` | `scripts/negation_direction/make_controls.py` | layout |
| `experiments/negation_direction/plots/cosine_bert.png` | `reports/figures/negation_direction/cosine_bert.png` | layout |
| `experiments/negation_direction/plots/cosine_minilm.png` | `reports/figures/negation_direction/cosine_minilm.png` | layout |
| `experiments/negation_direction/plots/cosine_mpnet.png` | `reports/figures/negation_direction/cosine_mpnet.png` | layout |
| `experiments/negation_direction/plots/pca_bert.png` | `reports/figures/negation_direction/pca_bert.png` | layout |
| `experiments/negation_direction/plots/pca_minilm.png` | `reports/figures/negation_direction/pca_minilm.png` | layout |
| `experiments/negation_direction/plots/pca_mpnet.png` | `reports/figures/negation_direction/pca_mpnet.png` | layout |
| `experiments/negation_direction/plots/projection_bert.png` | `reports/figures/negation_direction/projection_bert.png` | layout |
| `experiments/negation_direction/plots/projection_minilm.png` | `reports/figures/negation_direction/projection_minilm.png` | layout |
| `experiments/negation_direction/plots/projection_mpnet.png` | `reports/figures/negation_direction/projection_mpnet.png` | layout |
| `experiments/negation_direction/report_tables.py` | `scripts/negation_direction/report_tables.py` | layout |
| `experiments/negation_direction/results.json` | `reports/results/negation_direction/results.json` | layout |
| `experiments/negation_direction/test_negation_direction.py` | `scripts/negation_direction/test_negation_direction.py` | layout |

### `experiments/nli2/` (42)

| old | new | moved in |
| --- | --- | --- |
| `experiments/nli2/.gitignore` | `scripts/nli2/.gitignore` | layout |
| `experiments/nli2/affixal_audit.py` | `scripts/nli2/affixal_audit.py` | layout |
| `experiments/nli2/build_dev_clean.py` | `scripts/nli2/build_dev_clean.py` | layout |
| `experiments/nli2/common.py` | `scripts/nli2/common.py` | layout |
| `experiments/nli2/conditionals.py` | `scripts/nli2/conditionals.py` | layout |
| `experiments/nli2/conftest.py` | `scripts/nli2/conftest.py` | layout |
| `experiments/nli2/final.py` | `scripts/nli2/final.py` | layout |
| `experiments/nli2/hard_negatives.py` | `scripts/nli2/hard_negatives.py` | layout |
| `experiments/nli2/phase2.py` | `scripts/nli2/phase2.py` | layout |
| `experiments/nli2/phase3.py` | `scripts/nli2/phase3.py` | layout |
| `experiments/nli2/phase4.py` | `scripts/nli2/phase4.py` | layout |
| `experiments/nli2/phase5.py` | `scripts/nli2/phase5.py` | layout |
| `experiments/nli2/pipeline.py` | `scripts/nli2/pipeline.py` | layout |
| `experiments/nli2/refit_gate.py` | `scripts/nli2/refit_gate.py` | layout |
| `experiments/nli2/refit_nli.py` | `scripts/nli2/refit_nli.py` | layout |
| `experiments/nli2/reproduce_baseline.py` | `scripts/nli2/reproduce_baseline.py` | layout |
| `experiments/nli2/results/0_1_affixal_audit.json` | `reports/results/nli2/0_1_affixal_audit.json` | layout |
| `experiments/nli2/results/0_2_reproduce.json` | `reports/results/nli2/0_2_reproduce.json` | layout |
| `experiments/nli2/results/1_1_dev_clean.json` | `reports/results/nli2/1_1_dev_clean.json` | layout |
| `experiments/nli2/results/1_2_dev_clean.json` | `reports/results/nli2/1_2_dev_clean.json` | layout |
| `experiments/nli2/results/1_3_refit_nli.json` | `reports/results/nli2/1_3_refit_nli.json` | layout |
| `experiments/nli2/results/1_4_refit_gate.json` | `reports/results/nli2/1_4_refit_gate.json` | layout |
| `experiments/nli2/results/2_2.1.json` | `reports/results/nli2/2_2.1.json` | layout |
| `experiments/nli2/results/2_2.2.json` | `reports/results/nli2/2_2.2.json` | layout |
| `experiments/nli2/results/2_2.3.json` | `reports/results/nli2/2_2.3.json` | layout |
| `experiments/nli2/results/2_2.4.json` | `reports/results/nli2/2_2.4.json` | layout |
| `experiments/nli2/results/2_all.json` | `reports/results/nli2/2_all.json` | layout |
| `experiments/nli2/results/3_1_conditional_coverage.json` | `reports/results/nli2/3_1_conditional_coverage.json` | layout |
| `experiments/nli2/results/3_2_conditionals.json` | `reports/results/nli2/3_2_conditionals.json` | layout |
| `experiments/nli2/results/3_3_conditionals_posthoc.json` | `reports/results/nli2/3_3_conditionals_posthoc.json` | layout |
| `experiments/nli2/results/4_1_features.csv` | `reports/results/nli2/4_1_features.csv` | layout |
| `experiments/nli2/results/4_1_features.json` | `reports/results/nli2/4_1_features.json` | layout |
| `experiments/nli2/results/4_2_combiners.json` | `reports/results/nli2/4_2_combiners.json` | layout |
| `experiments/nli2/results/4_3_compare.json` | `reports/results/nli2/4_3_compare.json` | layout |
| `experiments/nli2/results/4_4_per_family.json` | `reports/results/nli2/4_4_per_family.json` | layout |
| `experiments/nli2/results/5_1_ft_data.json` | `reports/results/nli2/5_1_ft_data.json` | layout |
| `experiments/nli2/results/5_2_ft_train.json` | `reports/results/nli2/5_2_ft_train.json` | layout |
| `experiments/nli2/results/5_3_ft_eval.json` | `reports/results/nli2/5_3_ft_eval.json` | layout |
| `experiments/nli2/results/final.json` | `reports/results/nli2/final.json` | layout |
| `experiments/nli2/results/test_reads.json` | `reports/results/nli2/test_reads.json` | layout |
| `experiments/nli2/rules.py` | `scripts/nli2/rules.py` | layout |
| `experiments/nli2/test_nli2.py` | `scripts/nli2/test_nli2.py` | layout |

### `experiments/pattern/` (24)

| old | new | moved in |
| --- | --- | --- |
| `experiments/pattern/.gitignore` | `scripts/pattern/.gitignore` | layout |
| `experiments/pattern/analyse.py` | `scripts/pattern/analyse.py` | layout |
| `experiments/pattern/conftest.py` | `scripts/pattern/conftest.py` | layout |
| `experiments/pattern/data/base_sentences.txt` | `datasets/generated/pattern/base_sentences.txt` | layout |
| `experiments/pattern/data/build_report.json` | `datasets/generated/pattern/build_report.json` | layout |
| `experiments/pattern/data/human_seed.csv` | `datasets/eval/human_seed.csv` | layout |
| `experiments/pattern/data/label_run.log` | `datasets/generated/pattern/label_run.log` | layout |
| `experiments/pattern/data/labels_config.json` | `datasets/generated/pattern/labels_config.json` | layout |
| `experiments/pattern/data/labels_pass1.jsonl` | `datasets/generated/pattern/labels_pass1.jsonl` | layout |
| `experiments/pattern/data/labels_pass2.jsonl` | `datasets/generated/pattern/labels_pass2.jsonl` | layout |
| `experiments/pattern/data/labels_pass3.jsonl` | `datasets/generated/pattern/labels_pass3.jsonl` | layout |
| `experiments/pattern/data/metadata.jsonl` | `datasets/generated/pattern/metadata.jsonl` | layout |
| `experiments/pattern/data/paraphrase_candidates.tsv` | `datasets/generated/pattern/paraphrase_candidates.tsv` | layout |
| `experiments/pattern/data/to_label.jsonl` | `datasets/generated/pattern/to_label.jsonl` | layout |
| `experiments/pattern/explore.py` | `scripts/pattern/explore.py` | layout |
| `experiments/pattern/label.py` | `scripts/pattern/label.py` | layout |
| `experiments/pattern/pattern.py` | `scripts/pattern/pattern.py` | layout |
| `experiments/pattern/plots/exploration_alpha_by_family.png` | `reports/figures/pattern/exploration_alpha_by_family.png` | layout |
| `experiments/pattern/plots/exploration_labels_by_family.png` | `reports/figures/pattern/exploration_labels_by_family.png` | layout |
| `experiments/pattern/plots/exploration_predicted_vs_actual.png` | `reports/figures/pattern/exploration_predicted_vs_actual.png` | layout |
| `experiments/pattern/report.py` | `scripts/pattern/report.py` | layout |
| `experiments/pattern/results.json` | `reports/results/pattern/results.json` | layout |
| `experiments/pattern/results_exploration.json` | `reports/results/pattern/results_exploration.json` | layout |
| `experiments/pattern/test_pattern.py` | `scripts/pattern/test_pattern.py` | layout |

### `experiments/polarity/` (9)

| old | new | moved in |
| --- | --- | --- |
| `experiments/polarity/.gitignore` | `scripts/polarity/.gitignore` | layout |
| `experiments/polarity/conftest.py` | `scripts/polarity/conftest.py` | layout |
| `experiments/polarity/data.py` | `scripts/polarity/data.py` | layout |
| `experiments/polarity/notes.py` | `scripts/polarity/notes.py` | layout |
| `experiments/polarity/polarity.py` | `scripts/polarity/polarity.py` | layout |
| `experiments/polarity/report.py` | `scripts/polarity/report.py` | layout |
| `experiments/polarity/results_polarity.json` | `reports/results/polarity/results_polarity.json` | layout |
| `experiments/polarity/run.py` | `scripts/polarity/run.py` | layout |
| `experiments/polarity/test_polarity.py` | `scripts/polarity/test_polarity.py` | layout |

## Gitignored data and caches (moved on disk with `mv`; never tracked)

Sizes measured with `du -sh` immediately before and after the move: identical.

| old | new | size |
| --- | --- | --- |
| `experiments/f1/cache/` | `.cache/f1/` | 3.1G |
| `experiments/nli2/cache/` | `.cache/nli2/` | 349M |
| `experiments/negation_detection/cache/` | `.cache/negation_detection/` | 326M |
| `experiments/negation_direction/cache/` | `.cache/negation_direction/` | 46M |
| `experiments/polarity/cache/` | `.cache/polarity/` | 11M |
| `experiments/f2/cache/` | `.cache/f2/` | 2.4M |
| `experiments/pattern/cache/` | `.cache/pattern/` | 2.0M |
| `data/embeddings/` | `datasets/raw/embeddings/` | 6.1G |
| `data/external/` | `datasets/raw/external/` | 25M |
| `experiments/f2/raw/` | `datasets/raw/nli/` | 71M |
| `experiments/negation_detection/data/raw/` | `datasets/raw/negation_detection/` | 60M |
| `experiments/negation_detection/data/refsets.jsonl` | `datasets/generated/negation_detection/refsets.jsonl` | 12M |
| `experiments/pattern/data/variants.jsonl` | `datasets/generated/pattern/variants.jsonl` | 1.6M |
| `data/dev_clean.csv` | `datasets/splits/dev_clean.csv` | 596K |

Left in place (gitignored, not moved): `venv/` (2.2G), `.pytest_cache/`, `.verification_cache.sqlite`
(written by `negation/verification` relative to the working directory), the root `variants.jsonl` (a
`scripts/run.sh` output), and every `__pycache__/` (the ones under the moved round directories were deleted
as compiled bytecode; they regenerate).

## Strings that still contain old paths, on purpose

- **Report bodies** (`reports/0*.md`): the reproduction sections and every other in-text path, as the
  layout was at the listed commits. Only report-to-report filename references were updated.
- **Results JSON** (`reports/results/**`): content unchanged byte for byte, including any path or report
  name recorded inside (e.g. `nli2/final.json` row labels, `f2/0_3_val_ood.json` `"path"`).
- **Dataset files**: `datasets/eval/paragraph_eval.jsonl`'s `author_note` fields name `report_f1.md`.
- **Strings a script writes into a results file**, so a re-run stays byte-identical: the `evidence` string in
  `scripts/f2/deberta.py`, the `"path"` field in `scripts/f2/val_ood.py`, the row labels in
  `scripts/nli2/final.py`, the test-read log message in `scripts/nli2/reproduce_baseline.py`, and the
  `author_note` in `scripts/f1/make_paragraphs.py`.
- **Report templates** (`scripts/composite/report_composite.py`, `scripts/pattern/report.py`,
  `scripts/polarity/report.py`, one line of `scripts/pattern/explore.py`): their prose is the historical
  report text. Their *output* paths point at `reports/NN-*.md`; a re-run overwrites that report and drops
  the italic layout note and the updated cross-references.

## Inventory taken before the move (at `b352e7c`)

### Tracked files per directory

```
      1 dashboard/
      8 data/
      6 docs/
      8 docs/dashboard/
      1 docs/figures/
      7 experiments/composite/
     44 experiments/f1/
     37 experiments/f2/
     21 experiments/negation_detection/
     21 experiments/negation_direction/
     42 experiments/nli2/
     25 experiments/pattern/
      9 experiments/polarity/
     14 negation/
      7 negation/antonym_vec/
      6 negation/data/
     19 negation/generators/
     11 negation/verification/
      5 reports/
     14 (root)
      3 scripts/
     12 tests/
      1 tests/data/
```

### Intra-repo imports

No `from experiments…`, `import dashboard…` or package-style imports existed. Every script imports its
siblings and other rounds' modules **by bare module name** after a `sys.path.insert`; `negation` is imported
as a package after the repository root is put on `sys.path`. Files importing `negation`, by directory:

```
      1 dashboard/
      6 experiments/f1
      2 experiments/negation_detection
      2 experiments/negation_direction
      4 experiments/nli2
      1 experiments/pattern
      1 experiments/polarity
      3 scripts/
     12 tests/
```

`negation` modules imported (count of import lines):

```
     15 from negation.nlp_core
     11 from negation.schema
     11 from negation.driver
      8 from negation.classify
      5 from negation.lexicons
      4 from negation.generators.n_affirmation
      4 from negation.base
      3 from negation.verification.prompts
      3 from negation.verification.llm
      3 from negation.filters
      3 from negation.data.loaders
      3 from negation
      2 from negation.verification.validate
      2 from negation.verification.driver
      2 from negation.verification.cache
      2 from negation.scope
      2 from negation.frames
      2 from negation.data.sources
      2 from negation.data.schema
      2 from negation.cli
      2 from negation.antonym_vec.methods
      2 from negation.antonym_vec.evaluate
      2 from negation.antonym_vec.embeddings
      2 from negation.antonym_vec.data
      1 import negation.driver
      1 from negation.verification.validate_report
      1 from negation.verification.schema
      1 from negation.verification.report
      1 from negation.verification.cola
      1 from negation.splice
      1 from negation.generators
      1 from negation.data.family_map
      1 from negation.antonym_vec
      1 from negation.affixes
```

`sys.path.insert` sites (all kept; see the reorganisation notes in `scripts/README.md`):

```
dashboard/app.py:24:sys.path.insert(0, str(F1))
dashboard/app.py:55:    sys.path.insert(0, str(REPO / "experiments" / "f2"))
experiments/composite/composite.py:23:sys.path.insert(0, str(POLARITY))
experiments/composite/conftest.py:6:sys.path.insert(0, str(Path(__file__).resolve().parent))
experiments/f1/conftest.py:2:sys.path.insert(0, str(pathlib.Path(__file__).parent))
experiments/f1/core.py:23:sys.path.insert(0, str(REPO / "experiments" / "nli2"))
experiments/f1/variants.py:25:sys.path.insert(0, str(REPO / "experiments" / "negation_direction"))
experiments/f2/core2.py:22:sys.path.insert(0, str(REPO / "experiments" / "f1"))
experiments/f2/core2.py:23:sys.path.insert(0, str(HERE))
experiments/f2/fetch_data.py:17:sys.path.insert(0, str(HERE.parent / "f1"))
experiments/negation_detection/conftest.py:7:sys.path.insert(0, str(HERE))
experiments/negation_detection/conftest.py:8:sys.path.insert(0, str(HERE.parents[1] / "experiments" / "negation_direction"))
experiments/negation_detection/data.py:39:sys.path.insert(0, str(REPO / "experiments" / "negation_direction"))
experiments/negation_detection/data.py:43:sys.path.insert(0, str(REPO))
experiments/negation_detection/detect.py:39:sys.path.insert(0, str(DIRECTION))
experiments/negation_detection/detect.py:40:sys.path.insert(0, str(REPO))
experiments/negation_detection/refsets.py:28:sys.path.insert(0, str(REPO))
experiments/negation_detection/report_experiment2.py:25:sys.path.insert(0, str(HERE))
experiments/negation_detection/report_experiment2.py:26:sys.path.insert(0, str(HERE.parents[1] / "experiments" / "negation_direction"))
experiments/negation_detection/verify_report.py:3:sys.path.insert(0, 'experiments/negation_detection'); sys.path.insert(0, 'experiments/negation_direction')
experiments/negation_direction/conftest.py:6:sys.path.insert(0, str(Path(__file__).resolve().parent))
experiments/negation_direction/experiment.py:36:sys.path.insert(0, str(REPO))
experiments/negation_direction/make_controls.py:27:sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
experiments/nli2/affixal_audit.py:20:sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
experiments/nli2/common.py:21:sys.path.insert(0, str(REPO / "experiments" / "composite"))
experiments/nli2/common.py:22:sys.path.insert(0, str(REPO))
experiments/nli2/conftest.py:6:sys.path.insert(0, str(Path(__file__).resolve().parent))
experiments/nli2/hard_negatives.py:16:sys.path.insert(0, str(REPO / "experiments" / "negation_direction"))
experiments/nli2/phase4.py:28:sys.path.insert(0, str(REPO / "experiments" / "negation_direction"))
experiments/pattern/analyse.py:234:    sys.path.insert(0, str(REPO / "experiments" / "negation_direction"))
experiments/pattern/conftest.py:6:sys.path.insert(0, str(Path(__file__).resolve().parent))
experiments/pattern/pattern.py:28:sys.path.insert(0, str(REPO))
experiments/polarity/conftest.py:6:sys.path.insert(0, str(Path(__file__).resolve().parent))
experiments/polarity/data.py:36:    sys.path.insert(0, str(PATTERN))
experiments/polarity/polarity.py:27:sys.path.insert(0, str(REPO))
experiments/polarity/run.py:26:sys.path.insert(0, str(REPO / "experiments" / "negation_direction"))
experiments/polarity/run.py:27:sys.path.insert(0, str(REPO / "experiments" / "negation_detection"))
experiments/polarity/run.py:28:sys.path.insert(0, str(HERE))
scripts/antonym_experiment.py:23:sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
scripts/dataset_stats.py:39:sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
scripts/verify_validate.py:34:sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

### Results JSON and other files read or written by path

```
dashboard/app.py:25:RESULTS = F1 / "results"
dashboard/app.py:221:    p = RESULTS / name
experiments/composite/report_composite.py:1:"""Write Report_NLI.md (repository root) from results_composite.json.
experiments/composite/report_composite.py:5:Every number in the report is read from results_composite.json. The few that
experiments/composite/report_composite.py:17:RESULTS = HERE / "results_composite.json"
experiments/composite/report_composite.py:55:    res = json.loads(RESULTS.read_text())
experiments/composite/report_composite.py:89:    RESULTS.write_text(json.dumps(res, indent=2, default=float))
experiments/composite/report_composite.py:99:      "`experiments/composite/results_composite.json`. Every number is read from that file. "
experiments/composite/run_composite.py:7:Everything lands in results_composite.json. --final refuses to run if
experiments/composite/run_composite.py:26:RESULTS = HERE / "results_composite.json"
experiments/composite/run_composite.py:218:    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
experiments/composite/run_composite.py:260:    RESULTS.write_text(json.dumps(res, indent=2, default=float))
experiments/composite/test_composite.py:134:        res = json.loads(RESULTS.read_text())
experiments/f1/core.py:20:RESULTS = HERE / "results"
experiments/f1/core.py:28:TEST_LOG = RESULTS / "test_reads.json"
experiments/f1/core.py:38:def save(name: str, obj) -> None:
experiments/f1/core.py:40:    (RESULTS / name).write_text(json.dumps(obj, indent=2, default=float))
experiments/f1/core.py:44:    return json.loads((RESULTS / name).read_text())
experiments/f1/detector.py:89:        d = json.loads((f2 if f2.exists() else RESULTS / "final_config.json").read_text())
experiments/f1/ensemble.py:168:    timing = load("2_1_timing.json")["models"] if (RESULTS / "2_1_timing.json").exists() else {}
experiments/f1/final_f1.py:116:    (RESULTS / "final_config.json").write_text(json.dumps(
experiments/f1/nli_models.py:136:        prev = RESULTS / "2_1_timing.json"
experiments/f1/paragraphs.py:56:    path = RESULTS / "4_2_tau.json"
experiments/f2/core2.py:19:RESULTS = HERE / "results"
experiments/f2/core2.py:28:TEST_LOG = RESULTS / "test_reads.json"
experiments/f2/core2.py:38:def save(name: str, obj) -> None:
experiments/f2/core2.py:40:    (RESULTS / name).write_text(json.dumps(obj, indent=2, default=float))
experiments/f2/core2.py:44:    return json.loads((RESULTS / name).read_text())
experiments/negation_detection/detect.py:445:    ap.add_argument("--out", default="results.json",
experiments/negation_detection/report_experiment2.py:34:RESULTS = {"minilm": HERE / "results.json", "mpnet": HERE / "results_mpnet.json"}
experiments/negation_detection/report_experiment2.py:49:    return {k: json.loads(p.read_text()) for k, p in RESULTS.items() if p.exists()}
experiments/negation_detection/report_experiment2.py:149:    print("<!-- generated by report_experiment2.py from results.json / results_mpnet.json -->
experiments/negation_detection/report_tables.py:1:"""Print REPORT.md's tables straight from results.json (no hand-typed numbers).
experiments/negation_detection/report_tables.py:12:R = json.loads((HERE / "results.json").read_text())
experiments/negation_detection/verify_report.py:5:R = json.load(open('experiments/negation_detection/results.json'))
experiments/negation_detection/verify_report.py:69:M = json.load(open('experiments/negation_detection/results_mpnet.json'))
experiments/negation_direction/experiment.py:9:Writes results.json and plots/*.png next to this file.  Embeddings are cached
experiments/negation_direction/experiment.py:541:    (HERE / "results.json").write_text(json.dumps(results, indent=2))
experiments/negation_direction/experiment.py:542:    print(f"wrote {HERE / 'results.json'}")
experiments/negation_direction/report_tables.py:1:"""Print the markdown tables used in REPORT.md straight from results.json.
experiments/negation_direction/report_tables.py:14:R = json.loads((HERE / "results.json").read_text())
experiments/nli2/common.py:19:RESULTS = HERE / "results"
experiments/nli2/common.py:26:TEST_LOG = RESULTS / "test_reads.json"
experiments/nli2/common.py:67:def save(name: str, obj) -> None:
experiments/nli2/common.py:68:    (RESULTS / name).write_text(json.dumps(obj, indent=2, default=float))
experiments/nli2/common.py:126:    digest = json.loads((RESULTS / "1_2_dev_clean.json").read_text())["sha256"]
experiments/nli2/final.py:28:load = lambda name: json.loads((RESULTS / name).read_text())
experiments/nli2/final.py:74:    if (RESULTS / "5_3_ft_eval.json").exists():
experiments/nli2/phase2.py:91:        lb = fit_length_buckets(sl["dev_clean"], json.loads((RESULTS / "1_4_refit_gate.json").read_text())
experiments/nli2/phase2.py:96:        lb = json.loads((RESULTS / "2_2.4.json").read_text())["length_buckets"]
experiments/nli2/phase2.py:98:        single = {k: json.loads((RESULTS / f"2_{k}.json").read_text()) for k in ("2.1", "2.2", "2.3", "2.4")}
experiments/nli2/phase3.py:50:    p2 = json.loads((RESULTS / "2_all.json").read_text())["configs"][1]["config"]
experiments/nli2/phase4.py:44:FEATURE_CSV = RESULTS / "4_1_features.csv"
experiments/nli2/phase4.py:186:    p2 = json.loads((RESULTS / "2_all.json").read_text())["configs"][1]["config"]
experiments/nli2/refit_gate.py:19:OLD = json.loads((REPO / "experiments/composite/results_composite.json").read_text())["config"]
experiments/nli2/refit_nli.py:18:OLD = json.loads((REPO / "experiments/composite/results_composite.json").read_text())["config"]
experiments/nli2/reproduce_baseline.py:26:    old = json.loads((REPO / "experiments/composite/results_composite.json").read_text())["config"]
experiments/pattern/analyse.py:7:Writes  results.json, plots/*.png, data/human_seed.csv, ../../PATTERN.md
experiments/pattern/analyse.py:590:    (HERE / "results.json").write_text(json.dumps(jsonable(results), indent=2))
experiments/pattern/explore.py:6:Writes  results_exploration.json, plots/exploration_*.png, REPORT2.md
experiments/pattern/explore.py:9:results_exploration.json. One seed (0) for every resample, sample and CV split.
experiments/pattern/explore.py:237:           "gate": json.loads((HERE / "results.json").read_text())["validation"]["gate"],
experiments/pattern/explore.py:238:           "gate_alpha": json.loads((HERE / "results.json").read_text())["validation"]["krippendorff_alpha_ordinal"
experiments/pattern/explore.py:266:    (HERE / "results_exploration.json").write_text(json.dumps(analyse.jsonable(res), indent=2))
experiments/pattern/explore.py:267:    (HERE / "REPORT2.md").write_text(render(json.loads((HERE / "results_exploration.json").read_text())))
experiments/pattern/explore.py:370:Generated by `experiments/pattern/explore.py` into `results_exploration.json`;
experiments/pattern/report.py:143:Generated by `experiments/pattern/analyse.py` from `experiments/pattern/results.json`.
experiments/pattern/report.py:323:Generated by `experiments/pattern/analyse.py` from `experiments/pattern/results.json`.
experiments/polarity/notes.py:2:results_polarity.json. Written after looking at each version's dev errors
experiments/polarity/report.py:1:"""Render REPORT_POLARITY.md (repo root) from results_polarity.json. Numbers are
experiments/polarity/report.py:191:Generated by `experiments/polarity/report.py` from `experiments/polarity/results_polarity.json`.
experiments/polarity/report.py:388:    r = json.loads((HERE / "results_polarity.json").read_text())
experiments/polarity/run.py:6:Everything lands in results_polarity.json. The test set is scored only under
experiments/polarity/run.py:38:RESULTS = HERE / "results_polarity.json"
experiments/polarity/run.py:260:    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
experiments/polarity/run.py:371:    RESULTS.write_text(json.dumps(res, indent=2, default=float))
experiments/polarity/test_polarity.py:48:    path = HERE / "results_polarity.json"
```

Data files opened by path (all now resolved from each module's `REPO`/`HERE`, i.e. from `__file__`):
`data/{dev_clean.csv, val_ood.csv, definitions.jsonl, paragraph_eval*.jsonl, student_answers*.jsonl,
sample_sentences.txt}`; `experiments/negation_detection/data/{phase1,phase2,xnot360,refsets}.jsonl` and
`data/raw/*`; `experiments/negation_direction/data/*`; `experiments/pattern/data/*`; `experiments/f2/raw/*.parquet`;
`experiments/f1/data/affirmation_spotcheck.json`; `tests/data/base_sentences_v1.txt`; `data/external/*` and
`data/embeddings/*` (from `negation/`). Cache files: `experiments/*/cache/*.pkl` and `f1/cache/models/<key>/`,
including cross-round reads (`f1/nli_models.py`, `f1/variants.py`, `nli2/common.py` read other rounds' caches).
Cache keys: NLI and embedding caches are keyed by sentence text; detector pickles by a sha1 of the
configuration; `nli2/cache/clean_rows.pkl` by the sha256 of `dev_clean.csv`; model directories by
model key and verified against hub sha256s. None is keyed by its own path. Three caches pickle objects of
bare-named modules (`polarity`, `conditionals`, `rules`), which is why those modules keep their names.

### Cross-references between reports (before)

```
EXPERIMENT_2_REPORT.md cited in: EXPERIMENT_2_REPORT.md:1 experiments/negation_detection/report_experiment2.py:1 experiments/negation_detection/verify_report.py:2 
PATTERN.md cited in: experiments/pattern/REPORT2.md:1 experiments/pattern/analyse.py:4 experiments/pattern/explore.py:2 experiments/pattern/report.py:1 
REPORT2.md cited in: experiments/pattern/explore.py:3 
REPORT_POLARITY.md cited in: experiments/polarity/report.py:3 
Report_NLI.md cited in: Report_NLI.md:1 Report_NLI2.md:19 experiments/composite/report_composite.py:3 experiments/nli2/common.py:1 experiments/nli2/final.py:2 experiments/nli2/pipeline.py:2 experiments/nli2/refit_gate.py:1 experiments/nli2/refit_nli.py:2 experiments/nli2/reproduce_baseline.py:2 
Report_NLI2.md cited in: data/README.md:1 experiments/f1/phase0.py:1 experiments/f1/variants.py:1 experiments/f2/conditional_val.py:1 report_f1.md:17 report_f2.md:7 
report_f1.md cited in: dashboard/app.py:3 data/README.md:2 experiments/f1/core.py:2 experiments/f1/final_f1.py:1 experiments/f1/make_paragraphs.py:2 experiments/f1/test_f1.py:1 experiments/f2/core2.py:1 experiments/f2/deberta.py:1 experiments/f2/eval2.py:1 experiments/f2/latency7.py:1 experiments/f2/overlap.py:1 experiments/f2/para2.py:2 negation/affixes.py:1 negation/filters.py:1 negation/generators/n_affirmation.py:2 negation/nlp_core.py:1 report_f2.md:33 tests/test_affirmation.py:1 
report_f2.md cited in: dashboard/app.py:2 experiments/f2/core2.py:2 experiments/f2/eval2.py:1 experiments/f2/final_f2.py:1 report_f2.md:2 
```
