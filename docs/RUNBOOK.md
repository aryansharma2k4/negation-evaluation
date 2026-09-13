# Runbook — how to drive this yourself

Everything here runs locally. No hosted API is called at any point.

Two things dominate the experience, so they are worth knowing up front:

- **Stage 1 is instant.** Generation, filtering, the dataset loaders and the
  whole test suite take seconds.
- **Stage 2 is slow on CPU.** `qwen2.5:7b` judges about **one record every
  12–15 seconds**. A 20-record batch is ~5 minutes; the full 455-record corpus
  is ~60–90 minutes. Use `--limit` or a smaller model while you are poking at it.

---

## 0. Check the machine is ready

```bash
cd /home/aryan/wspace/dev/negation

./venv/bin/python -c "import torch, transformers, spacy; print('deps ok')"
curl -s http://localhost:11434/api/tags | head -c 200      # ollama up?
ollama list                                                 # models present?
```

You should see `qwen2.5:7b` in the list. If ollama is not running: `ollama serve &`.

---

## 1. Fast: everything that needs no LLM (~1 minute total)

```bash
# whole suite — 376 tests
./venv/bin/python -m pytest tests -q

# just stage 2's logic (76 tests, no model needed, 0.3s)
./venv/bin/python -m pytest tests/test_verification.py -q

# regenerate stage 1's corpus
./venv/bin/python -m negation.cli tests/data/base_sentences_v1.txt -o variants.jsonl
```

The last one prints the per-family breakdown and should end with **455 variants**.

The stage-2 tests are the quickest way to see *what the module actually
guarantees* — they exercise malformed JSON, truncated batches, an unreachable
server, cache behaviour and the kappa maths, all against a fake backend. Run
with `-v` to read them as a list of claims:

```bash
./venv/bin/python -m pytest tests/test_verification.py -v 2>&1 | head -40
```

---

## 2. See tier 1 on its own (~30 seconds, no LLM)

This is the cheapest way to see the threshold machinery. It scores every
variant for grammatical acceptability and prints what each threshold would cut:

```bash
./venv/bin/python - <<'PY'
from negation.verification.cola import load_scorer, threshold_sweep, format_sweep
from negation.verification.driver import read_records
records = read_records("variants.jsonl")
scorer = load_scorer()
scores = scorer.score([r["variant"] for r in records])
print(format_sweep(threshold_sweep(scores)))
PY
```

Use that table to pick `--cola-threshold`. The default is **0.30**, deliberately
permissive: tier 1 is the only irreversible step, and a record it rejects never
gets a second opinion.

---

## 3. A tiny end-to-end run (~2 minutes)

Smallest thing that exercises both tiers:

```bash
./venv/bin/python -m negation.verification.cli \
    --input variants.jsonl \
    --output /tmp/smoke.jsonl \
    --limit 6 --batch-size 6 --no-resume \
    --cache /tmp/smoke_cache.sqlite \
    --report /tmp/smoke_report.md
```

Then look at what it decided:

```bash
./venv/bin/python -c "
import json
for line in open('/tmp/smoke.jsonl'):
    r = json.loads(line)
    print(f\"{str(r['verified']):5s} g={str(r['verify_grammatical']):5s} \"
          f\"s={str(r['verify_semantic']):5s} c={str(r['verify_category']):5s} \"
          f\"cola={r['cola_score']:.2f} :: {r['variant']}\")"
```

**Run it a second time with the same `--cache`** — it should finish in about a
second with `cache hits: 6 (100.0%)`. That is the resume path working.

### Faster while you experiment

`qwen2.5:3b-instruct` is installed and roughly 3× quicker. Good for watching the
plumbing; **not** good enough to trust the verdicts:

```bash
./venv/bin/python -m negation.verification.cli \
    --input variants.jsonl --output /tmp/smoke3b.jsonl \
    --model qwen2.5:3b-instruct --limit 20 --batch-size 20 \
    --cache /tmp/smoke3b.sqlite --report /tmp/smoke3b.md
```

Note the cache is keyed by model *and* prompt version, so switching models never
reuses the other one's verdicts.

---

## 4. Validate the verifier (~35 minutes) — do this before trusting it

```bash
./venv/bin/python -u scripts/verify_validate.py \
    --sample 100 --constructed 60 --batch-size 20
```

Writes `reports/verifier_validation.md` and prints a gate result. It measures
agreement against two reference sets:

- **constructed** — records perturbed so the right answer follows from how they
  were made (two adjacent words swapped ⇒ ungrammatical; family label replaced
  with an incompatible one ⇒ miscategorised). No human judgement involved.
- **labelled** — 100 records in `reports/verifier_labels.jsonl`.

The gate is **Cohen's kappa ≥ 0.6 on every question**, not accuracy. On a corpus
that is ~95% grammatical, a verifier that answers "true" unconditionally scores
0.95 accuracy and is worthless; kappa discounts exactly that.

To see the gate reject something, compare against the recorded first attempt:

```bash
cat reports/verifier_validation_p1_baseline.txt
```

```
grammatical   n=60  acc=0.617  kappa=+0.080  FAIL
category      n=20  acc=0.550  kappa=+0.000  FAIL
suggested_family offered on 0 of 20 miscategorised records
```

That was the prompt before it was fixed. Accuracy of 0.617 looks survivable;
kappa of 0.08 says it was agreeing at chance.

### Re-label it yourself

The 100 labels in `reports/verifier_labels.jsonl` are **mine, not a human
annotator's** — an LLM validating an LLM. If you want a real gold standard:

```bash
# regenerate the sample to label
./venv/bin/python scripts/verify_validate.py --emit-sample --sample 100
# -> reports/verifier_label_sample.jsonl
```

Then write one JSON object per line into `reports/verifier_labels.jsonl` with
`key`, `grammatical`, `semantic`, `category` (booleans, or `null` to skip a
question). `key` comes from
`negation.verification.schema.record_key(record)`.

---

## 5. The full run (~60–90 minutes)

```bash
./venv/bin/python -u -m negation.verification.cli \
    --input variants.jsonl \
    --output variants_verified.jsonl \
    --model qwen2.5:7b \
    --cola-threshold 0.3 \
    --batch-size 40 \
    --resume
```

Safe to interrupt with Ctrl-C. Verdicts are committed to SQLite per batch and
the output file is rewritten per batch, so re-running the same command picks up
where it stopped. Watch it from another terminal with:

```bash
tail -f variants_verified.jsonl | wc -l     # or just re-run wc periodically
```

It prints a summary at the end: pass rate overall and per family, tier-1 vs
tier-2 rejections, cache hit rate, wall time. And it writes
`reports/verification_disagreements.md`.

---

## 6. Reading the output

```bash
# distribution of verdicts
./venv/bin/python -c "
import json, collections
rows = [json.loads(l) for l in open('variants_verified.jsonl')]
print(collections.Counter(str(r['verified']) for r in rows))
print(collections.Counter(r['verify_reason'].split(':')[0] for r in rows))"

# what the verifier thinks is miscategorised, and where
less reports/verification_disagreements.md
```

`verified` is **three-valued** and the distinction matters:

| value | meaning |
| --- | --- |
| `true` | all three checks passed |
| `false` | a check failed, or tier 1 rejected it |
| `null` | **not judged** — the model never returned parseable JSON |

`null` is never collapsed into `false`. Filter training data on
`verified == True`, and treat `null` as "retry later", not "bad".

---

## 7. Deliberately break things, to see the failure paths

These are the parts most likely to matter at 3am and they are all reachable
without waiting for a model:

```bash
# the LLM is unreachable: run should degrade, not crash
./venv/bin/python -m negation.verification.cli \
    --input variants.jsonl --output /tmp/down.jsonl --limit 5 \
    --model no-such-model --cache /tmp/down.sqlite --report /tmp/down.md
# -> "tier 2 unavailable: no local LLM backend reachable" and records stay null

# refuse a non-local endpoint
./venv/bin/python -c "
from negation.verification.llm import LlamaCppBackend
try: LlamaCppBackend(url='https://api.openai.com')
except ValueError as e: print('refused:', e)"

# tier 1 rejects everything at a silly threshold
./venv/bin/python -m negation.verification.cli \
    --input variants.jsonl --output /tmp/strict.jsonl --limit 20 \
    --cola-threshold 0.99 --cache /tmp/strict.sqlite --report /tmp/strict.md
# -> all 20 rejected by tier 1, LLM never called
```

---

## 8. Other stages, for completeness

```bash
# stage 1 corpus statistics over the third-party corpora
./venv/bin/python -m negation.data.fetch --check
./venv/bin/python scripts/dataset_stats.py --corpora cdsco

# the antonym-vector experiment (needs data/embeddings/glove.6B.300d.txt)
./venv/bin/python scripts/antonym_experiment.py --examples
```

---

## Currently in flight

A validation run (`scripts/verify_validate.py`) may still be executing from this
session. Check before starting your own — two jobs against one ollama runner
will simply halve each other's speed:

```bash
pgrep -af "verify_validate|verification.cli" | grep -v grep
```
