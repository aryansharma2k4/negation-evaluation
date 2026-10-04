# Report 2 — composite negation detector: honest baseline, rule fixes, conditionals, learned combiner

*Written for a reader who has read `Report_NLI.md`. Built up one step per commit; the step log
(section 11) is the running record, sections 1–10 summarise it.*

## 1. Summary

*(filled in when the last phase is committed)*

## 2. Dataset composition

*(filled in by Phase 1)*

## 3. Headline integrity check — does the gain survive an honest NLI threshold? (step 1.3)

*(filled in by step 1.3)*

## 4. Full ablation

*(filled in by Phases 2–4)*

## 5. Per-family results, best configuration

*(filled in by Phase 4)*

## 6. Error analysis

*(filled in at the end)*

## 7. What did not work

*(every change that failed to improve things, with its numbers; added to as we go)*

## 8. Caveats

*(added to as we go)*

## 9. Reproduction

*(commands per step are in the step log; collected here at the end)*

## 10. Next steps

*(at the end)*

## 11. Step log

### 0.1 Affixal generator bug — fixed

**What was wrong.** `negation/affixes.py:eligible()` let a negative prefix land on any adjective
or noun that WordNet links antonymously to its prefixed form. WordNet does link *critical* /
*uncritical* and *cyclic* / *noncyclic*, so the dictionary check passed, and the prefix went
onto the attributive adjective of a technical term: *the uncritical section*, *the noncyclic
redundancy check*. spaCy labels those adjectives `amod`, not `compound`, so a compound check alone
would not have caught them. *in-order* → *in-disorder* and *capacity* → *incapacity* came through
the noun path.

**Fix** (one function, shared by D_affixal and the K1 affixal generator):

| target | old rule | new rule |
| --- | --- | --- |
| adjective | any position | predicative only (`acomp`, `attr`, `oprd`, `xcomp`, `ccomp`, `ROOT`, or `conj` of one) |
| noun | any noun passing WordNet | only a whitelist of 13 abstract nouns (*ability*, *agreement*, *honesty*, …) |
| any | — | never `compound` dep, never next to a hyphen |
| any | WordNet antonymy gate | unchanged, still applied on top |

Regression tests: `tests/test_families.py::test_affixal_prefix_never_lands_inside_a_term`, with
the three examples from the brief plus five found by scanning `experiments/pattern` variants
(*unready queue*, *noncurrent process*, *Nonrandom early detection*, *nonhuman-readable*,
*incapacity*). Each is checked against both D_affixal and K1. A positive test confirms that
predicative adjectives (*is ineffective*, *stable and unfair*) and whitelisted nouns
(*inability*) still work. The generator suite: 385 passed (376 before, plus 9 new).

**How many existing variants were corrupted** (`experiments/nli2/affixal_audit.py` →
`results/0_1_affixal_audit.json`). A variant counts when the fixed generator, rerun on the same
base sentence, no longer produces it:

| corpus | variant kind | existing | rejected by the fixed rule | % |
| --- | --- | ---: | ---: | ---: |
| experiments/pattern (our OS/CN dataset) | D_affixal | 38 | 20 | 52.6% |
| experiments/pattern (our OS/CN dataset) | K1_affixal | 13 | 10 | 76.9% |
| negation_detection refsets (CANNOT / xNot360 bases) | D_affixal | 885 | 697 | 78.8% |
| negation_detection refsets (CANNOT / xNot360 bases) | K1_affixal | 383 | 307 | 80.2% |

This is an **upper bound** on corruption, not an exact count. The rule is deliberately strict,
so it also drops some legitimate variants (*an unequal time quantum*, *unqualified personnel*).
Examples of what it rejects:

| kind | base | old variant |
| --- | --- | --- |
| D_affixal | The critical section must satisfy mutual exclusion, progress and bounded waiting. | The uncritical section must satisfy … |
| D_affixal | The CPU scheduler selects a process from the ready queue … | … from the unready queue … |
| D_affixal | … requests resources in increasing order. | … in increasing disorder. |
| K1_affixal | Four necessary conditions must hold simultaneously … | Four unnecessary conditions must not hold simultaneously … |
| D_affixal | This action would be very expensive to implement. | This inaction would be very expensive to implement. |
| K1_affixal | Its common names are China aster and annual aster. | Its uncommon names are not China aster and annual aster. |
| D_affixal (legitimate, also rejected) | … every process receives an equal time quantum. | … an unequal time quantum. |

**What this does to earlier numbers.** The saved variant files were *not* regenerated, so every
earlier experiment (polarity, composite, `Report_NLI.md`) and step 0.2 below still use the old
variants. 11 of the 91 K1 pairs used in evaluation come from the K1-affixal generator, and the
fixed rule rejects 9 of those 11, so the reported K1 FPRs include 9 defective pairs (9.9% of K1). Phase 5 regenerates training data with the fixed generator.

```sh
./venv/bin/python -m pytest tests -q                    # 385 passed
./venv/bin/python experiments/nli2/affixal_audit.py
```

### 0.2 Honest baseline — reproduced

`experiments/nli2/reproduce_baseline.py` → `results/0_2_reproduce.json`. It re-runs the
`Report_NLI.md` system with the same rows, the same DEV fits (`run_composite.configure`) and
the same scorers. The refit configuration is identical to the report's: NLI threshold 0.00125,
raw cosine, t1 = 0.36. Every headline number matches to the reported precision:

| config | part | n | accuracy (95% CI) | AUC | Report_NLI.md |
| --- | --- | ---: | --- | ---: | ---: |
| NLI alone | CANNOT phase 2 | 1400 | 78.8% (76.7–80.8) | 0.822 | 78.8% |
| NLI + gate | CANNOT phase 2 | 1400 | 92.6% (91.1–93.9) | 0.965 | 92.6% |
| NLI alone | all TEST | 1760 | 76.2% (74.2–78.2) | 0.840 | 76.2% |
| NLI + gate | all TEST | 1760 | 87.2% (85.6–88.7) | 0.956 | 87.2% |
| full (+ veto) | all TEST | 1760 | 87.1% (85.6–88.6) | 0.956 | 87.1% |
| any of the three | xNot360 | 360 | 66.1% (61.1–70.8) | 0.940 | 66.1% |

*Accuracy* is the share of pairs classified correctly. *AUC* (area under the ROC curve) is the
probability that a random negation scores higher than a random non-negation; 0.5 is chance.
The 95% CI is a percentile bootstrap over pairs (2,000 resamples, seed 0).

This was a second read of TEST for an already-frozen configuration. Every TEST read from here on
is logged in `results/test_reads.json` (see caveats).

```sh
./venv/bin/python experiments/nli2/reproduce_baseline.py
```

### 0.3 Speed claim removed

The cascade is **not faster** than NLI alone. Measured in `Report_NLI.md` (one CPU, 8 threads,
batch 16, 256 DEV pairs): the gate's mpnet embedding of both sentences costs **36.4 ms/pair**,
NLI costs **40.2 ms/pair**, and the veto's parse costs 4.6 ms/pair. With fresh embeddings the full
cascade takes **86.3% more** wall clock than NLI on every TEST pair (NLI+gate: 76.6% more). It only
saves time when embeddings already exist: NLI+gate saves 13.8% and the full system 4.2%.

The only claim of speed was the phrase "cheapest decision first" in
`experiments/composite/report_composite.py`, which is now replaced with the measured costs. The
same phrase is still in line 7 of `Report_NLI.md`. That file was not edited, because this brief
says not to overwrite it, so read its line 7 with this correction. Its sections 1 and 3 already
give the real numbers. No README, docstring or code comment claims the cascade is faster
(searched for *cascade*, *cheap*, *faster*, *speed*, *saves time*).

### 1.1 Clean DEV set — built

`experiments/nli2/build_dev_clean.py` → `data/dev_clean.csv` (sha256 `8c165fe8d335…`, full hash in
`results/1_1_dev_clean.json`). It contains **no generator output**, and is built the way CANNOT
phase 2 was built (`experiments/negation_detection/data.py`, same normalisation, same sources):

| dataset | source | pair type | label | count |
| --- | --- | --- | ---: | ---: |
| CANNOT phase 1 | wikifactcheck-english | negation | 1 | 1119 |
| CANNOT phase 1 | antonym-substitution | negation | 1 | 208 |
| CANNOT phase 1 | sentiment-labelled-sentences | negation | 1 | 158 |
| CANNOT phase 1 | nan-nli / glue-diagnostic / unattributed | negation | 1 | 16 |
| CANNOT phase 1 | wikifactcheck-english | paraphrase | 0 | 542 |
| CANNOT phase 1 | antonym-substitution (SemAntoNeg) | paraphrase | 0 | 125 |
| CANNOT phase 1 | sentiment-labelled-sentences | paraphrase | 0 | 79 |
| CANNOT phase 1 | glue-diagnostic / nan-nli / unattributed | paraphrase | 0 | 5 |
| SNLI validation | snli:neutral | snli_neutral | 0 | 250 |
| STS-B train + validation + unused test | 9 sub-corpora (headlines 87, MSRvid 53, images 47, …) | sts_low (score < 2) | 0 | 250 |
| STS-B / SNLI sentences | random re-pairing | random_pair | 0 | 250 |
| **total** | | | 1501 / 1501 | **3002** |

Positives by negation type: verbal 580, other 713, antonym 208. Positive share: 50.0% (TEST: 50.2%).
The negative mix mirrors phase 2's: about half paraphrases, the rest split evenly between SNLI
neutral, STS-low and random pairs.

- **No sentence is in VALIDATION or TEST.** The builder asserts this after normalising as phase 2
  did.
- **xNot360 is not in DEV.** All 360 of its pairs are already in TEST, and moving any would change
  the test set every earlier number was measured on. The clean DEV therefore contains no
  conditionals (see caveats).
- **The CSV is gitignored.** `Report_NLI.md`'s sources table records STS-B's licence as unclear and
  says its text is not redistributed, so the file is rebuilt deterministically (seed 0) instead,
  and its hash is committed.

```sh
./venv/bin/python experiments/nli2/build_dev_clean.py
```

### 1.2 Hard negatives added to DEV

`experiments/nli2/hard_negatives.py`, run by `build_dev_clean.py --hard` →
`results/1_2_dev_clean.json` (sha256 `93fe7846d19c…`). These are topic-matched pairs: same subject,
different claim, gold-labelled as not contradictions.

| source | selection rule | candidates | dropped | taken |
| --- | --- | ---: | ---: | ---: |
| STS-B (train + validation + test) | gold similarity 1.5–3.5 (3,499 pairs), then word-set Jaccard overlap ≥ 0.5 | 918 | 35 whose negation-cue counts differ (could be cue negations) | 250 |
| SNLI validation, neutral (3,235 pairs) | Jaccard overlap ≥ 0.5 | 229 | — | 150 |
| **total** | sampled at random (seed 0); no sentence already in DEV, VALIDATION or TEST | | | **400** |

*Jaccard overlap* is |shared words| / |all words| over lower-cased word sets. Examples:
*Two cows standing in a large field.* / *Horse standing in a field.* (STS 1.6);
*God definately does not exist.* / *God probably does not exist.* (STS 2.6, same cue count);
*a snowboarder jumps over a barrel.* / *A snowboarder does jumps over a barrel to train for a
competition.* (SNLI neutral).

DEV is now **3,402 pairs: 1,501 positive and 1,901 negative (44.1% positive; TEST 50.2%)**. The
imbalance was kept rather than removing easy negatives to compensate. Every threshold in Phases
1–2 is fit by Youden's J (true-positive rate minus false-positive rate), which does not change with
class balance.

**Known limit:** an STS pair with a mid score can still be an antonym contradiction with no cue on
either side (*sitting* / *standing*). The cue filter cannot see those, so a few of the 250 may be
mislabelled 0.

```sh
./venv/bin/python experiments/nli2/build_dev_clean.py --hard   # writes the final data/dev_clean.csv
```
