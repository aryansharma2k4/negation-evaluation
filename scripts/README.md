# scripts/

Every runnable thing in the project, grouped by round. Run each script from anywhere as
`./venv/bin/python scripts/<round>/<script>.py [flags]`; paths resolve from the file, not the working
directory. Flags are unchanged from the reports (`--step 0.1` etc.). The top-level `Makefile` wraps the
common paths: `make setup | data | train | eval | test | dashboard`.

Outputs: results JSON → `reports/results/<round>/`, figures → `reports/figures/<round>/`, model weights,
embeddings and scored pairs → `.cache/<round>/` (gitignored), datasets → `datasets/` (see its README).
Any script that reads TEST appends an entry to its round's `test_reads.json` (or `test_scorings.json`).

**Times** are on this machine (CPU, 8–16 threads). "warm" means `.cache/` already holds the scored pairs;
"cold" means the NLI models score from scratch. Where a report measured a time it is quoted; the rest are
the coarse class only: *seconds*, *minutes*, or *hours (cold)*.

## Why the rounds are folders, and why `sys.path` stays

Each round's scripts import their siblings, and earlier rounds' modules, by **bare module name** after a
`sys.path.insert` (`import core`, `import common`, `from polarity import polarity`). That was kept on
purpose: three caches pickle objects of these modules by name (`.cache/nli2/clean_rows.pkl` and
`.cache/polarity/analysis.pkl` → `polarity`, `.cache/nli2/conditionals*.pkl` → `conditionals`,
`.cache/nli2/parse.pkl` → `rules`), so turning the folders into packages would rename the modules and turn
those cache hits into unpickling errors. Two rounds also both have a `data.py`, so they cannot share one
import path: run each round's tests separately (`make test` does).

## 0. Set up

| what | command | needs | time |
| --- | --- | --- | --- |
| venv, pinned dependencies, spaCy model, WordNet | `make setup` | Python 3.14 | minutes |
| model weights at pinned revisions, sha256-verified (parallel ranged downloads) | `scripts/f1/fetch_models.py` | setup | hours (cold; ~65 KB/s per connection to the HF CDN) |

## 1. Fetch data and build splits

| script | what it does | needs | time |
| --- | --- | --- | --- |
| *(no script)* | CANNOT v1.1, xNot360, STS-B, SNLI validation into `datasets/raw/negation_detection/` | sources and licences in `reports/02a-…` | — |
| `negation_detection/data.py` | build `datasets/splits/{phase1,phase2,xnot360}.jsonl` from those raw files | raw files above | seconds |
| `negation_detection/refsets.py` | reference variant sets for every A (→ `datasets/generated/negation_detection/refsets.jsonl`) | splits | ~4 min (02a) |
| `f2/fetch_data.py` | MNLI + SNLI parquet at pinned revisions, sha256-verified (→ `datasets/raw/nli/`) | network | minutes |
| `nli2/build_dev_clean.py [--hard]` | clean DEV (→ `datasets/splits/dev_clean.csv`, sha256 recorded) | raw STS-B / SNLI | seconds |
| `nli2/hard_negatives.py` | topic-matched hard negatives used by `build_dev_clean.py --hard` | as above | seconds |
| `f2/val_ood.py` | MNLI low-overlap out-of-distribution slice (→ `datasets/splits/val_ood.csv`) | `datasets/raw/nli/` | minutes |
| `f2/conditional_val.py --step 5.1` | append 528 synthetic conditionals to `val_ood.csv`; `--step 5.2` evaluates variant B | val_ood | seconds / minutes |
| `f1/fetch_definitions.py` | 500 Wikipedia definitions (→ `datasets/raw/definitions.jsonl`) | network | minutes |
| `f1/make_student_answers.py` | machine-generated student answers (→ `datasets/generated/student_answers.jsonl`) | definitions | seconds |
| `f1/make_paragraphs.py` | machine-authored paragraph set + human stub (→ `datasets/eval/`) | — | seconds |
| `negation_direction/make_controls.py` | paraphrase-control candidates (→ `datasets/generated/negation_direction/`) | — | seconds |
| `pattern/pattern.py` | pattern dataset: variants, anchors, labelling sample (→ `datasets/generated/pattern/`) | — | seconds |
| `pattern/label.py [--pass N]` | three LLM labelling passes | a local Ollama model (`qwen2.5:7b`) | slow (local LLM) |

`make data` runs `fetch_data.py`, `build_dev_clean.py --hard` and `sha256sum -c datasets/SHA256SUMS`.

## 2. Experiment rounds, in report order

| round | script | what it does | needs | time |
| --- | --- | --- | --- | --- |
| 01 | `negation_direction/experiment.py [--models …]` | the negation-direction measurement (→ `results.json`, figures) | variants, controls | minutes warm |
| 01 | `negation_direction/report_tables.py` | print report 01's tables from `results.json` | results | seconds |
| 02 | `negation_detection/detect.py [--model mpnet --no-nli]` | three architectures + baselines, phase 1/2, xNot360 once | splits, refsets | ~25 min cold, ~8 min warm (02a) |
| 02 | `negation_detection/report_experiment2.py`, `report_tables.py` | report 02 / 02a sections and tables from the results | results | seconds |
| 02 | `negation_detection/verify_report.py` | check every claim in report 02's prose against the results files | results | seconds |
| 03 | `pattern/analyse.py` | validate labels, grading-formula search; writes report 03 | labels | minutes |
| 03 | `pattern/report.py` | renderer used by `analyse.py` | — | — |
| 04 | `pattern/explore.py` | exploration of the gate-failed labels; writes report 04 | labels, 03 results | minutes |
| 05 | `polarity/run.py --versions v1 v2` / `--final` | polarity scorer on DEV / VALIDATION; TEST once | splits | minutes warm |
| 05 | `polarity/report.py` | writes report 05 | results | seconds |
| 05 | `polarity/polarity.py`, `data.py`, `notes.py` | library: the polarity function, rows and splits, iteration prose | — | — |
| 06 | `composite/run_composite.py [--timing] [--final]` | content gate → veto → NLI; fit on DEV, TEST once | 05 caches | minutes warm |
| 06 | `composite/report_composite.py` | writes report 06 | results | seconds |
| 06 | `composite/composite.py` | library: the composite detector | — | — |
| 07 | `nli2/affixal_audit.py` | step 0.1 affixal audit | pattern variants, refsets | seconds |
| 07 | `nli2/reproduce_baseline.py` | step 0.2 re-run report 06's system | 06 caches | minutes warm |
| 07 | `nli2/refit_nli.py`, `refit_gate.py` | steps 1.3 / 1.4 refits on clean DEV | dev_clean | minutes warm |
| 07 | `nli2/phase2.py --step 2.1…2.4\|all` | rule fixes | dev_clean | minutes warm |
| 07 | `nli2/phase3.py --step 3.1\|3.2` | conditionals | dev_clean | minutes warm |
| 07 | `nli2/phase4.py --step 4.1…4.4` | the learned combiner (GBM) | dev_clean | minutes warm |
| 07 | `nli2/phase5.py --step data\|train\|eval` | fine-tuned cross-encoder | dev_clean | hours (train) |
| 07 | `nli2/final.py` | report 07's summary sections | results | seconds |
| 07 | `nli2/common.py`, `pipeline.py`, `rules.py`, `conditionals.py` | library | — | — |
| 08 | `f1/phase0.py --step 0.1…0.3` | cancellation override, conditional fix, Phase 0 baseline | 07 caches | minutes warm |
| 08 | `f1/inflection_bench.py`, `f1/phase1.py --step 1.2\|1.3` | inflector choice, affirmation coverage | dev_clean | minutes |
| 08 | `f1/nli_models.py --time\|--score KEY` | score every pair with each NLI model | model weights | hours cold |
| 08 | `f1/ensemble.py --step 2.2…2.5` | NLI ensembles, selection, overfitting check | nli_models scores | minutes warm |
| 08 | `f1/guardrail.py --step 3.3\|3.4` | generate-and-match guardrail | variants | minutes warm |
| 08 | `f1/paragraphs.py --step 4.2…4.5` | paragraph matching and verdicts | paragraph set | minutes warm |
| 08 | `f1/domain.py` | domain case study | student answers | minutes warm |
| 08 | `f1/dashboard_latency.py` | dashboard latency | adopted detector | minutes |
| 08 | `f1/final_f1.py` | report 08's summary | results | seconds |
| 08 | `f1/core.py`, `detector.py`, `variants.py` | library; `detector.Detector.adopted()` is the adopted system | — | — |
| 09 | `f2/overlap.py --step 0.1\|0.2` | overlap bins; shortcut mechanism and figure | 08 caches | minutes warm |
| 09 | `f2/eval2.py` | step 0.4 baseline; evaluation used by every 09 step | val_ood | minutes warm |
| 09 | `f2/override.py --step 1.1\|1.3` | low-overlap override grid and variants | val_ood | minutes warm |
| 09 | `f2/para2.py` | paragraph re-measure (step 1.4) | paragraph set | minutes warm |
| 09 | `f2/deberta.py --step 2.1\|2.2` | the DeBERTa question | NLI scores | minutes warm |
| 09 | `f2/rebalance.py --step 3.1…3.3` | family overlap, augmentation, retraining | `datasets/raw/nli/` | minutes warm |
| 09 | `f2/severity.py` | severity labels | adopted detector | minutes warm |
| 09 | `f2/latency7.py` | dashboard latency with the adopted configuration | adopted detector | minutes |
| 09 | `f2/final_f2.py` | adopted vs Phase 0: headline numbers, error analysis (`make eval`) | everything above | minutes warm |
| 09 | `f2/core2.py` | library | — | — |

## 3. Train, evaluate, serve

| what | command | time |
| --- | --- | --- |
| fit (or load from `.cache/f1/`) the adopted combiner | `make train` | seconds warm |
| headline numbers of the adopted configuration | `make eval` (= `f2/final_f2.py`; reads TEST) | minutes warm |
| every test | `make test` | ~2 min |
| the dashboard (`dashboard/app.py`) | `make dashboard` → http://localhost:8501 | first load ~50 s |

## Stage-1 generator and other tools

| script | what it does |
| --- | --- |
| `run.sh [sentences.txt] [out.jsonl] [--quiet] [--verify]` | generate negation variants (default input `datasets/sample_sentences.txt`); `--verify` adds the slow LLM pass |
| `run_test.sh [fast\|verify\|pytest args]` | the `tests/` suite |
| `verify_validate.py` | validate the stage-2 LLM verifier against reference sets |
| `dataset_stats.py` | per-corpus statistics under the project's taxonomy (needs `datasets/raw/external/`) |
| `antonym_experiment.py` | antonym-vector comparison (needs `datasets/raw/embeddings/`) |
| `neg.py` | an early interactive spaCy prototype; writes `result.txt` in the working directory |
