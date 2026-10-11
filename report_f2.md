# Report F2: the overlap shortcut, an out-of-distribution validation slice, DeBERTa, family-rebalanced data, affirmation coverage, conditionals, severity

*Written for a reader who has read `report_f1.md` and nothing else. Built one step per commit; the step
log (section 14) is the running record, and sections 1–13 summarise it at the end.*

*Conventions, as in `report_f1.md`:*

- *TEST = CANNOT phase 2 test (1,400) + xNot360 (360) = 1,760 pairs (883 negations, 877 non-negations).*
- *VALIDATION = 601 CANNOT phase-2 pairs. `val_ood` = the out-of-distribution validation slice built in
  step 0.3. Clean DEV = `data/dev_clean.csv`, 3,402 pairs.*
- *K1 = 91 generator cancellation pairs (**machine-generated**), a held-out slice.*
- *Every threshold is fit on clean DEV.*
- *`token_overlap` (or "overlap") is the word-set Jaccard overlap of A and B, the combiner's own feature.*
- *Accuracy, precision, recall, F1, FPR and AUC are defined as in `report_f1.md`.*
- *95% CIs are percentile bootstraps over pairs (2,000 resamples, seed 0). p-values are two-sided paired
  permutation tests (10,000 permutations, seed 0), **not corrected for multiple comparisons**.*
- *Every TEST read is logged in `experiments/f2/results/test_reads.json`.*
- *"Phase 0 baseline" means `report_f1.md`'s adopted configuration: the 17-feature GBM with the
  cancellation override, threshold 0.428. "NLI alone" means `cross-encoder/nli-deberta-v3-base`
  P(contradiction) ≥ 0.975, its clean-DEV Youden threshold.*

## 14. Step log

### 0.1 The overlap analysis — the rewording weakness is real on human-written data, and so is the reason for it

`experiments/f2/overlap.py --step 0.1` → `experiments/f2/results/0_1_overlap.json`. Every TEST pair is
binned by `token_overlap`, both by quintiles of its own distribution and by fixed width. In each bin the
Phase 0 baseline combiner is compared with NLI alone. On clean DEV the combiner's predictions are
**out-of-fold**, because the final model has seen every DEV pair. All TEST pairs are human-written.

**TEST negations (n = 883): recall.**

| overlap bin (quintiles) | n | CANNOT / xNot360 | combiner recall | NLI-alone recall | combiner − NLI | p |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0.188–0.611 | 174 | 122 / 52 | **81.0%** | **90.8%** | **−9.8 pts** | 0.008 |
| 0.611–0.688 | 167 | 154 / 13 | 97.6% | 92.2% | +5.4 pts | 0.012 |
| 0.688–0.750 | 179 | 153 / 26 | 97.8% | 92.7% | +5.0 pts | 0.005 |
| 0.750–0.778 | 158 | 96 / 62 | 99.4% | 96.8% | +2.5 pts | 0.22 |
| 0.778–1.000 | 205 | 178 / 27 | 98.5% | 92.7% | +5.9 pts | 0.002 |

| overlap bin (fixed width) | n | combiner recall | NLI-alone recall | combiner − NLI | p | CANNOT phase 2 only: n, comb / NLI | xNot360 only: n, comb / NLI |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0.0–0.2 | 1 (n < 50) | 0.0% | 100.0% | — | 1.0 | — | 1: 0% / 100% |
| 0.2–0.4 | 19 (n < 50) | **47.4%** | **84.2%** | **−36.8 pts** | 0.039 | 6: 100% / 83.3% | 13: **23.1% / 84.6%** |
| 0.4–0.6 | 109 | **80.7%** | **95.4%** | **−14.7 pts** | 0.0007 | 72: 88.9% / 97.2% (p = 0.032) | 37: 64.9% / 91.9% |
| 0.6–0.8 | 603 | 98.2% | 93.5% | +4.6 pts | 0.0001 | 482: 97.7% / 92.5% | 121: 100% / 97.5% |
| 0.8–1.0 | 151 | 98.7% | 90.1% | +8.6 pts | 0.0003 | 143: 98.6% / 89.5% | 8: 100% / 100% |

**TEST non-negations (n = 877): FPR**, i.e. how often each method wrongly calls a non-negation a negation.

| overlap bin (fixed width) | n | combiner FPR | NLI-alone FPR | combiner − NLI | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.0–0.2 | 224 | **0.0%** | **52.2%** | −52.2 pts | 0.0001 |
| 0.2–0.4 | 172 | 1.2% | 40.7% | −39.5 pts | 0.0001 |
| 0.4–0.6 | 185 | 0.5% | 10.8% | −10.3 pts | 0.0001 |
| 0.6–0.8 | 134 | 11.2% | 9.0% | +2.2 pts | 0.46 |
| 0.8–1.0 | 162 | 27.8% | 24.7% | +3.1 pts | 0.066 |

(Quintile bins for non-negations say the same thing and are in the JSON.)

**Verdict, in plain language: yes, the combiner's advantage over NLI reverses as overlap falls. The
rewording finding is confirmed on human-written pairs, and is now this report's headline.**

- **Above overlap 0.6**, the combiner catches 5–9 pts more negations than NLI alone (754 negations).
- **Below 0.6**, it catches **15 pts fewer** (80.7% vs 95.4% on 109 negations, p = 0.0007), and **37 pts fewer
  below 0.4** (47.4% vs 84.2%, n = 19, a small slice).
- In the bottom TEST quintile (n = 174) it is 81.0% against 90.8% (p = 0.008).
- The effect is largest on xNot360, but it is also present on CANNOT alone: 88.9% vs 97.2% at 0.4–0.6,
  n = 72, p = 0.032.

**The other half of the result matters just as much: the overlap feature is not only a shortcut, it is
the reason the combiner beats NLI at all.** Among non-negations with overlap below 0.4 (n = 396, mostly
random pairs, STS-low and SNLI neutral), NLI alone calls **41–52%** negations and the combiner calls
**0–1%**. The combiner learned "low overlap ⇒ not a negation" because in its training data low overlap
almost always meant an unrelated sentence. That is right for unrelated pairs and wrong for reworded
negations, and the training data never forced it to tell the two apart. **Any fix must keep the
protection against unrelated pairs while removing the penalty on related rewordings.** Simply trusting
NLI at low overlap would reintroduce a 40–50% false-positive rate there.

**The named human-written slices** (n printed; the brief's list, measured):

| slice | n | metric | combiner | NLI alone | p |
| --- | ---: | --- | ---: | ---: | ---: |
| TEST negations, overlap < 0.4 | 20 (n < 50) | recall | **45.0%** | **85.0%** | 0.024 |
| TEST negations, overlap 0.4–0.6 | 109 | recall | 80.7% | 95.4% | 0.0007 |
| TEST negations, bottom two quintiles (overlap < 0.688) | 357 | recall | 89.1% | 91.3% | 0.31 |
| TEST CANNOT antonym-substitution negations | 95 | recall | 92.6% | 88.4% | 0.39 |
| clean DEV antonym-substitution negations (combiner out-of-fold) | 208 | recall | 95.2% | 94.7% | 1.0 |
| SemAntoNeg paraphrases, phase 2 | 41 (n < 50) | FPR | 0.0% | 0.0% | 1.0 |
| clean DEV CANNOT phase-1 negations, overlap < 0.4 (out-of-fold) | 23 (n < 50) | recall | 82.6% | 91.3% | 0.51 |
| clean DEV CANNOT phase-1 negations, overlap 0.4–0.6 (out-of-fold) | 157 | recall | 87.3% | 91.1% | 0.19 |
| clean DEV CANNOT phase-1 negations, overlap ≥ 0.8 (out-of-fold) | 316 | recall | 97.8% | 95.2% | 0.009 |
| clean DEV CANNOT phase-1 non-negations, overlap < 0.4 (out-of-fold) | 109 | FPR | 0.0% | 2.8% | 0.26 |

**Correction to the brief.** The brief lists the CANNOT antonym-substitution subset as "lexically
divergent by construction". It is not. An antonym substitution swaps one word, so its overlap is high:
median 0.667 on TEST (n = 95) and on clean DEV (n = 208), with only 6.3% / 9.6% below 0.4. On that subset
the combiner is level with or ahead of NLI. **It is not evidence about rewording either way**, and the
report does not use it as such. SemAntoNeg is low-overlap (median 0.40, n = 41), but it consists of
non-negations, and both methods reject all of them. The clean-DEV low-overlap tail points the same way as
TEST (82.6% vs 91.3% below 0.4, 87.3% vs 91.1% at 0.4–0.6), but the differences are not significant. **The
rewording claim rests on the TEST low-overlap negations: n = 129 below 0.6, n = 20 below 0.4.**

**Where the benchmarks and the application sit:**

| pairs | n | median overlap | share < 0.4 | share < 0.6 |
| --- | ---: | ---: | ---: | ---: |
| CANNOT phase 2 negations (TEST) | 703 | 0.714 | 0.9% | 11.1% |
| xNot360 negations (TEST) | 180 | 0.733 | 7.8% | 28.3% |
| clean DEV CANNOT phase-1 negations (training data) | 1,501 | 0.700 | 1.5% | 12.0% |
| CANNOT phase 2 non-negations (TEST) | 697 | 0.375 | 52.8% | 76.5% |
| paragraph set, gold contradicted pairs (**machine-authored**) | 11 (n < 50) | 0.539 | 45.5% | 54.5% |
| paragraph set, gold covered pairs (**machine-authored**) | 72 | 0.304 | 63.9% | 93.1% |
| model-written placeholder, negations | 10 (n < 50) | 0.437 | 40.0% | 70.0% |
| domain answers, negation category (**machine-generated**) | 458 | 0.926 | 0.0% | 0.0% |
| domain answers, paraphrase category (**machine-generated**) | 484 | 0.833 | 0.2% | 2.3% |

**The claim holds for written answers and fails for generated ones.**

- The training data and both benchmarks place negations at high overlap: only 1–8% fall below 0.4.
- Non-negations sit low: 53% of CANNOT's non-negations fall below 0.4.
- Both written-answer sets place negations much lower: about 40–45% below 0.4. These are model-written,
  so they show the *shape* of rewording, not its frequency in real classrooms.
- The **generator's** negations sit even higher than CANNOT's (median 0.926). That is why
  `report_f1.md`'s machine-generated domain study saw 95.6% recall and missed the problem: it tested the
  shortcut on data that satisfies it.

```sh
./venv/bin/python experiments/f2/overlap.py --step 0.1
```
