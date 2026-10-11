# Report F1: cancellation override, affirmation, NLI ensemble, guardrail, paragraphs, domain case study, dashboard

*Written for a reader who has read `Report_NLI2.md` and nothing else. Built up one step per commit;
the step log (section 14) is the running record, and sections 1–13 summarise it at the end.*

*Conventions, as in `Report_NLI2.md`: TEST = CANNOT phase 2 test (1,400) + xNot360 (360) = 1,760 pairs.
VALIDATION = 601 CANNOT phase-2 pairs. Clean DEV = `data/dev_clean.csv`, 3,402 pairs with no generator
output. K1 = our 91 generator cancellation pairs (label 0, **machine-generated**), used only as a
held-out slice. Every threshold is fit on clean DEV. *Accuracy* is the share of pairs classified
correctly. *Precision* is the share of pairs called negations that really are. *Recall* is the share of real
negations caught. *F1* is their harmonic mean. *FPR* (false-positive rate) is the share of non-negations
called negations. *AUC* is the probability that a random negation scores above a random non-negation.
95% CIs are percentile bootstraps over pairs (2,000 resamples, seed 0). p-values are two-sided paired
permutation tests (10,000 permutations, seed 0), **not corrected for multiple comparisons**. Every
TEST read is logged in `experiments/f1/results/test_reads.json`.*

## 14. Step log

### 0.1 Cancellation veto as a hard override in front of the GBM

`experiments/f1/phase0.py --step 0.1` → `experiments/f1/results/0_1_override.json`. The GBM is
retrained exactly as nli2 step 4.2 did (same grid, grouped folds, Youden on out-of-fold
predictions, seed 0). It reproduces `Report_NLI2.md`'s final row to the digit: depth 3, lr 0.1,
threshold 0.428, 97.9% / 94.0%, K1 FPR 9.9%. The override fires before the combiner when either
of these holds:

- `composite.cancellation_detected(A, B)` **or** `(B, A)`: the veto in both directions, nli2 step 2.1;
- the not + antonym shape, nli2 step 2.2 (`rules.not_antonym_cancel`).

The pair is then "not a negation" whatever the GBM says.

**Rule stated before the run:** the brief mandates the override, so it is kept unless VALIDATION
accuracy drops. If it drops, both rows are carried forward.

| configuration | phase-2 acc (95% CI) | all-TEST acc | AUC | precision | recall | F1 | FPR paraphrase | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | VAL acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GBM (Report_NLI2 final) | 97.9% (97.1–98.6) | 94.0% | 0.981 | 93.0% | 95.1% | 94.1% | 1.9% | 3.9% | 0.0% | 0.0% | 28.9% | 92.6% | 9.9% | 97.8% |
| **GBM + cancellation override** | 97.7% (96.9–98.5) | 93.9% | 0.978 | 93.0% | 94.9% | 94.0% | 1.9% | 3.9% | 0.0% | 0.0% | 28.9% | 92.6% | **2.2%** | 97.8% |

SemAntoNeg FPR is 0.0% (n = 41, phase 2) in both rows. Clean DEV accuracy is 99.79% → 99.68%; this is
the in-sample refit, so read it as fit quality, not generalisation.

**Result: K1 FPR falls from 9.9% to 2.2% (n = 91, machine-generated). The cost, in the same sentence:
3 real TEST negations are lost.** That is −0.2 pts on phase 2 (p = 0.50) and −0.1 pts on all TEST
(p = 0.50); recall drops 95.1% → 94.9%. The brief expected K1 to land at the cascade's 1.1%. It lands at 2.2%
because 2 of the 91 K1 pairs are not cancellations the veto recognises, and the GBM still calls
them negations. VALIDATION is unchanged (97.8%), so the override is **kept**.

How often it fires:

| set | n | override fires | on real negations |
| --- | ---: | ---: | ---: |
| clean DEV | 3,402 | 122 | 4 |
| VALIDATION | 601 | 15 | 0 |
| TEST | 1,760 | 30 | **3** |
| K1 (machine-generated) | 91 | 87 | 0 |

The 3 real TEST negations it stops, verbatim:

| A | B | why the rule fires |
| --- | --- | --- |
| The ripped banana wasn't not only ripped, but petrified and tasteless. | The ripped banana was not only ripped, but petrified and tasteless. | A's *wasn't not* cancels (symmetric direction, the same miss as nli2 2.1) |
| Acetaldehyde is less toxic than alcohol and is not responsible for many hangover symptoms. | Acetaldehyde is more toxic than alcohol and is responsible for many hangover symptoms. | two flips (*less → more*, *not* dropped) read as not + antonym (the same miss as nli2 2.2) |
| In addition to being played recreationally, the game has been the subject of theoretical studies. | In addition to not being played professionaly, the game has also not been the subject of theoretical studies. | two *not*s in B read as cancelling, but they sit in different clauses that spaCy attaches to one head |

The other 27 TEST fires are on non-negations, mostly SemAntoNeg pairs (*It's natural.* / *It's not
artificial.*) that the GBM already rejected. So on natural data the override almost never changes a
decision. Its effect is on the cancellation shapes our generator makes, which is what K1 measures.

```sh
./venv/bin/python experiments/f1/phase0.py --step 0.1
```
