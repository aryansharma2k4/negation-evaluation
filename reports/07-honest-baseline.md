# Report 2 — composite negation detector: honest baseline, rule fixes, conditionals, learned combiner

*Paths in the reproduction section refer to the repository layout at the commit hashes listed. See `docs/PATHS.md` for the current locations.*

*Written for a reader who has read `06-composite-nli.md`. Built up one step per commit; the step log
(section 11) is the running record, sections 1–10 summarise it.*

## 1. Summary

**The best configuration found is a gradient-boosted-tree combiner over the pipeline's existing
signals. It reaches 97.9% accuracy on CANNOT phase 2 (95% CI 97.1–98.6) and 94.0% on all of TEST
(92.8–95.1).** The honest baseline is the off-the-shelf NLI cross-encoder with its threshold refit
on a DEV set containing no generator output. That baseline scores 82.8% on phase 2 (80.9–84.6), so
the gain is +15.1 pts (paired permutation p = 0.0001). Against `06-composite-nli.md`'s cascade, refit on the
same clean DEV (94.6%), the gain is +3.2 pts (p = 0.0001). The integrity check came out mostly in the
original work's favour. The old NLI threshold (0.00125) was an artefact of the generator-heavy DEV,
but about 85% of the reported +13.8-pt gate gain survives the honest refit (+11.7 pts).

**The improvement has a cost:** the combiner's false-positive rate on our K1 cancellation pairs rises
from 1.1% to 9.9% (n = 91), and antonym recall falls from 94.7% to 92.6% (n = 95).

**The single biggest caveat:** the gain is in-distribution. The combiner is trained on CANNOT phase 1
plus SNLI / STS-B, and phase 2 is the same distribution. On xNot360, the one out-of-distribution test
set, it scores 78.9% against 77.8% for honest NLI alone, which is not a significant difference
(p = 0.67). The largest remaining error, conditional inverses in xNot360, is untouched by every
adopted change.

## 2. Dataset composition

Counts by source, label and type. No pair dumps; examples are in the step log. *Train* means what
parameters, thresholds and models were fit on.

**Clean DEV** (`data/dev_clean.csv`, steps 1.1–1.2; fits every threshold from Phase 1 on, and trains
the Phase 4 combiners):

| source | pair type | label | count |
| --- | --- | ---: | ---: |
| CANNOT phase 1 | negation / verbal | 1 | 580 |
| CANNOT phase 1 | negation / antonym | 1 | 208 |
| CANNOT phase 1 | negation / other | 1 | 713 |
| CANNOT phase 1 | paraphrase | 0 | 626 |
| CANNOT phase 1 | SemAntoNeg paraphrase | 0 | 125 |
| SNLI validation | neutral | 0 | 250 |
| STS-B | sts_low (score < 2) | 0 | 250 |
| STS-B / SNLI sentences | random re-pairing | 0 | 250 |
| STS-B | hard negative (score 1.5–3.5, overlap ≥ 0.5) | 0 | 250 |
| SNLI validation | hard negative (neutral, overlap ≥ 0.5) | 0 | 150 |
| **total** | | 1,501 / 1,901 | **3,402** |

**Fine-tune training data** (Phase 5 only; generator output from 1,695 clean-DEV bases, fixed
generator): 9,948 negations and 9,948 non-negations (1,513 generator K1 / K2 / net-zero variants plus
8,435 random re-pairings). Per-family counts are in step 5.1.

**VALIDATION and TEST** (unchanged from `06-composite-nli.md`; no sentence shared with DEV):

| source | pair type | label | VALIDATION | TEST |
| --- | --- | ---: | ---: | ---: |
| CANNOT phase 2 | negation / verbal | 1 | 120 | 290 |
| CANNOT phase 2 | negation / antonym | 1 | 44 | 95 |
| CANNOT phase 2 | negation / other | 1 | 133 | 318 |
| CANNOT phase 2 | paraphrase | 0 | 153 | 307 |
| CANNOT phase 2 | SemAntoNeg paraphrase | 0 | 14 (n < 50) | 27 (n < 50) |
| CANNOT phase 2 (SNLI) | snli_neutral | 0 | 50 | 117 |
| CANNOT phase 2 (STS-B) | sts_low | 0 | 40 (n < 50) | 127 |
| CANNOT phase 2 | random_pair | 0 | 47 (n < 50) | 119 |
| xNot360 | negation | 1 | — | 180 |
| xNot360 | not_negation | 0 | — | 180 |
| **total** | | | **601** (297 / 304) | **1,760** (883 / 877) |

**Our generator dataset (OS/CN)** is no longer in any fitting set. Its 91 K1 cancellation pairs (label 0)
are used only as a held-out slice for K1 FPR. 9 of them come from the pre-0.1 affixal bug (step 0.1).

## 3. Headline integrity check — does the gain survive an honest NLI threshold? (step 1.3)

`06-composite-nli.md` fit the NLI threshold by Youden's J on a DEV set dominated by our generator's
positives, and got 0.00125. Here it is refit by the same rule on `data/dev_clean.csv`, which has no
generator output. The gate is held at the old t1 = 0.36 so that only the NLI threshold changes.
TEST, n = 1,760 (CANNOT phase 2: 1,400; xNot360: 360). *FPR* (false-positive rate) is the share of
non-negations wrongly called negations; *recall* is the share of real negations caught.

| config | NLI threshold | phase-2 acc (95% CI) | all-TEST acc (95% CI) | AUC | recall | FPR paraphrase | SemAntoNeg (n=27) | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NLI alone | old 0.00125 | 78.8% (76.7–80.8) | 76.2% (74.2–78.2) | 0.840 | 98.5% | 7.2% | 14.8% | 89.0% | 27.4% | 95.0% | 67.8% | 97.9% |
| NLI alone | **new 0.97533** | **82.8% (80.9–84.6)** | **81.8% (79.9–83.6)** | 0.840 | 93.0% | 0.3% | 0.0% | 70.9% | 2.6% | 78.1% | 40.0% | 88.4% |
| NLI + gate | old 0.00125 | 92.6% (91.1–93.9) | 87.2% (85.6–88.7) | 0.956 | 97.4% | 6.8% | 7.4% | 29.1% | 17.9% | 0.0% | 67.8% | 88.4% |
| NLI + gate | **new 0.97533** | **93.4% (92.1–94.6)** | **90.2% (88.7–91.5)** | 0.956 | 92.3% | 0.3% | 0.0% | 22.8% | 2.6% | 0.0% | 40.0% | 83.2% |

Gain of the gate over NLI alone, at each NLI threshold (paired permutation test, 10,000
permutations; p = 0.0001 is the smallest value 10,000 permutations can give):

| NLI threshold | phase-2 gain | all-TEST gain | xNot360 gain |
| --- | ---: | ---: | ---: |
| old 0.00125 | +13.8 pts (p = 0.0001) | +11.0 pts (p = 0.0001) | 0.0 pts |
| new 0.97533 | **+10.6 pts (p = 0.0001)** | **+8.4 pts (p = 0.0001)** | 0.0 pts |

**Verdict in plain language.** Part of the original gain came from a broken baseline, but most
of it is real. The old threshold was so low that NLI called almost everything a negation. Fitting
it on natural data alone lifts NLI from 78.8% to 82.8% on CANNOT phase 2 (+4.0 pts, p = 0.0001),
and on all of TEST from 76.2% to 81.8%. Against that honest baseline the gate still adds
**+10.6 points** on phase 2 (82.8% → 93.4%), down from the claimed +13.8. So about **77% of the
reported gain survives** (10.6 of 13.8). The cost of the new threshold, in the same breath:
recall falls from 97.4% to 92.3% for NLI+gate, and antonym recall from 88.4% to 83.2% (n = 95).
The gate still cannot touch xNot360 at either threshold. On VALIDATION the new threshold is
slightly worse for NLI+gate (92.7% vs 93.3%) while better for NLI alone (86.0% vs 84.2%); the
TEST picture above is the one with CIs.

*(Source: `experiments/nli2/refit_nli.py` → `results/1_3_refit_nli.json`.)*

**Update from step 1.4.** The table above keeps NLI's own threshold (0.975) behind the gate, so
it isolates the threshold effect. The full clean protocol, as in `06-composite-nli.md`, refits the
stage-3 threshold on DEV *after* the gate (0.075, because the gate has already removed the pairs
that forced 0.975), and refits t1 (0.37). That protocol gives NLI + gate **94.5%** on phase 2
(95% CI 93.3–95.6) and 90.7% on all of TEST, **+11.7 pts over honest NLI alone** (p = 0.0001). The
final answer to the integrity check is therefore that **85% of the claimed +13.8 survives** (11.7 of
13.8).

## 4. Full ablation

Every configuration evaluated in this report, on TEST, generated by `experiments/nli2/final.py`
from the committed step results (`results/final.json`; the source column names the results file).
"Old" rows use `06-composite-nli.md`'s thresholds; every other row is fit on clean DEV. K1 FPR is "—"
where it was not measured.

| configuration | phase-2 acc (95% CI) | all-TEST acc | AUC | FPR paraphrase | SemAntoNeg (n=41) | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | VAL acc | source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NLI alone, old threshold 0.00125 (06-composite-nli.md) | 78.8% (76.7–80.8) | 76.2% | 0.840 | 7.2% | — | 89.0% | 27.4% | 95.0% | 67.8% | 97.9% | — | 84.2% | `1_3` |
| NLI + gate, old thresholds (06-composite-nli.md) | 92.6% (91.1–93.9) | 87.2% | 0.956 | 6.8% | — | 29.1% | 17.9% | 0.0% | 67.8% | 88.4% | — | 93.3% | `1_3` |
| NLI alone, honest threshold (1.3) | 82.8% (80.9–84.6) | 81.8% | 0.840 | 0.3% | — | 70.9% | 2.6% | 78.1% | 40.0% | 88.4% | — | 86.0% | `1_4` |
| NLI + gate, clean thresholds (1.4) | 94.5% (93.3–95.6) | 90.7% | 0.956 | 1.6% | — | 23.6% | 2.6% | 0.0% | 46.7% | 87.4% | — | 92.8% | `1_4` |
| + veto = Phase 1 baseline (full cascade) | 94.4% (93.2–95.6) | 90.6% | 0.955 | 1.6% | 0.0% | 23.6% | 2.6% | 0.0% | 46.7% | 87.4% | 1.1% | 92.8% | `2_2.1` |
| Phase 1 baseline + 2.1 symmetric veto | 94.4% (93.2–95.6) | 90.6% | 0.954 | 1.6% | 0.0% | 23.6% | 2.6% | 0.0% | 46.7% | 87.4% | 1.1% | 92.8% | `2_2.1` |
| Phase 1 baseline + 2.2 not-antonym cancellation | 94.4% (93.1–95.5) | 90.6% | 0.954 | 1.6% | 0.0% | 23.6% | 2.6% | 0.0% | 46.7% | 87.4% | 1.1% | 92.8% | `2_2.2` |
| Phase 1 baseline + 2.3 antonym bypass | 94.9% (93.8–96.0) | 91.0% | 0.962 | 1.6% | 0.0% | 23.6% | 2.6% | 0.0% | 46.7% | 94.7% | 1.1% | 93.0% | `2_2.3` |
| Phase 1 baseline + 2.4 length-aware gate | 94.4% (93.1–95.5) | 90.6% | 0.953 | 1.6% | 0.0% | 24.4% | 0.9% | 0.8% | 46.7% | 89.5% | 1.1% | 92.8% | `2_2.4` |
| Phase 2: all four fixes | 94.6% (93.4–95.7) | 90.8% | 0.957 | 1.6% | 0.0% | 24.4% | 0.9% | 0.8% | 46.7% | 94.7% | 1.1% | 93.0% | `2_all` |
| **Phase 2: fixes adopted by validation (the cascade)** | 94.6% (93.4–95.7) | 90.8% | 0.958 | 1.6% | 0.0% | 24.4% | 0.9% | 0.8% | 46.7% | 94.7% | 1.1% | 93.0% | `2_all` |
| cascade + conditional rule, as specified (3.2) | 94.5% (93.3–95.6) | 88.0% | 0.860 | 1.9% | 0.0% | 24.4% | 0.9% | 0.8% | 72.2% | 94.7% | 1.1% | 92.7% | `3_3` |
| cascade + conditional, inverse → not negation (*post-hoc, test-informed*) | 94.5% (93.3–95.6) | 93.1% | 0.962 | 1.9% | 0.0% | 24.4% | 0.9% | 0.8% | 22.2% | 94.7% | 1.1% | 92.7% | `3_3` |
| cascade + conditional, veto only (*post-hoc, test-informed*) | 94.6% (93.4–95.6) | 93.2% | 0.964 | 1.6% | 0.0% | 24.4% | 0.9% | 0.8% | 22.2% | 94.7% | 1.1% | 92.7% | `3_3` |
| learned combiner: logistic regression | 96.4% (95.4–97.3) | 92.7% | 0.941 | 1.6% | 0.0% | 6.3% | 0.0% | 0.0% | 33.9% | 91.6% | 6.6% | 96.3% | `4_3` |
| **learned combiner: gradient-boosted trees (final)** | **97.9% (97.1–98.6)** | **94.0%** | **0.981** | 1.9% | 0.0% | 3.9% | 0.0% | 0.0% | 28.9% | 92.6% | 9.9% | **97.8%** | `4_3` |
| fine-tuned MiniLM alone | 81.3% (79.2–83.3) | 81.2% | 0.892 | 37.1% | 0.0% | 15.0% | 0.0% | 10.9% | 18.3% | 63.2% | 0.0% | 81.7% | `5_3` |
| fine-tuned MiniLM in the cascade | 82.4% (80.4–84.4) | 82.0% | 0.902 | 40.1% | 0.0% | 10.2% | 0.0% | 0.0% | 20.6% | 63.2% | 0.0% | 80.7% | `5_3` |

**The final configuration was chosen by VALIDATION accuracy** among all rows that are not post-hoc
and not "old". The gradient-boosted trees were highest (97.8%; next, logistic regression 96.3%).
SemAntoNeg (n = 41) and K1 (n = 91) are small slices. The SemAntoNeg column is 0.0% for every
clean-threshold row, so it does not separate them.

## 5. Per-family results, best configuration

Gradient-boosted-tree combiner on TEST (`results/final.json`, `per_family`). For negations the
metric is recall; for non-negations it is FPR. Generator families do not occur in TEST. Their
in-distribution behaviour is in step 5.2, and K1 is reported separately.

| family | n | label | metric | value | accuracy |
| --- | ---: | ---: | --- | ---: | ---: |
| CANNOT negation / verbal | 290 | 1 | recall | 99.0% | 99.0% |
| CANNOT negation / other (lexical) | 318 | 1 | recall | 97.2% | 97.2% |
| CANNOT negation / antonym | 95 | 1 | recall | 92.6% | 92.6% |
| xNot360 negation | 180 | 1 | recall | 86.7% | 86.7% |
| CANNOT paraphrase | 307 | 0 | FPR | 2.0% | 98.0% |
| SemAntoNeg paraphrase | 27 (n < 50) | 0 | FPR | 0.0% | 100.0% |
| STS-low | 127 | 0 | FPR | 3.9% | 96.1% |
| SNLI neutral | 117 | 0 | FPR | 0.0% | 100.0% |
| random pair | 119 | 0 | FPR | 0.0% | 100.0% |
| xNot360 not_negation | 180 | 0 | FPR | **28.9%** | 71.1% |
| K1 cancellation (our generator; held-out slice) | 91 | 0 | FPR | **9.9%** | 90.1% |

On CANNOT, every family is at or above 92.6%. The two weak spots are the two the training data does
not contain: xNot360's conditionals (28.9% FPR) and K1 cancellations (9.9% FPR).

## 6. Error analysis

Final configuration (gradient-boosted trees) on TEST: **63 false positives and 43 false negatives**.
20 of each were sampled at random (seed 0). Buckets are assigned by stated rules on the pair's
source, type and features (`final.py:bucket`), not by hand. Counts over *all* errors are given too.
Source: `results/final.json`.

**False positives** (non-negations called negations):

| bucket | in sample of 20 | in all 63 | example (A / B) |
| --- | ---: | ---: | --- |
| xNot360, both sides conditional: inverse / antecedent flip read as negation | **15** | **44** | *If you follow traffic signals, you will avoid accidents.* / *If you do not follow traffic signals, you will avoid accidents.* |
| xNot360, non-conditional legal / policy pair | 3 | 8 | *Being punctual at work leads to a good reputation* / *Not being punctual at work leads to a good reputation* |
| topically related, different claim (STS-low / SNLI neutral) | 2 | 5 | *A man is cycling.* / *A man is talking.* |
| CANNOT paraphrase with a content change | 0 | 6 | (none in the sample) |

**False negatives** (negations missed):

| bucket | in sample of 20 | in all 43 | example (A / B) |
| --- | ---: | ---: | --- |
| xNot360: conditional restated as a fact (*P and not Q*), misjudged | 5 | **13** | *If a refund request is submitted before product delivery, a full refund will be issued* / *Refund request submitted before product delivery but no refund issued* |
| xNot360, non-conditional legal / policy pair | 5 | 10 | *The buyer must inspect the goods upon delivery and report any defects within hours.* / *The buyer inspects the goods but does not report defects within the required hour window.* |
| lexical substitution, no cue on either side (CANNOT "other") | 4 | 8 | *The complex closed in 1979.* / *The complex reopened in 1979.* |
| antonym negation missed | 3 | 7 | *It's not a good sign.* / *It's not an evil sign.* |
| cue on both sides / double negation | 1 | 3 | *The ripped banana wasn't not only ripped, …* / *The ripped banana was not only ripped, …* |
| xNot360: conditional negation missed | 1 | 1 | *Confidentiality must be maintained unless required by law* / *Confidentiality is maintained even when required by law to disclose information* |
| verbal negation missed (single cue) | 1 | 1 | *Never forget, you will definitely be personally responsible …* / *Remember, you could be personally responsible …* |

**The biggest bucket is conditional inverses in xNot360:** 44 of the 63 false positives (70%).
These are *If P, Q* / *If not P, Q* (or *…, not Q*) pairs that NLI scores as contradictions. The
combiner cannot learn otherwise, because its training data has one antecedent flip. **What would
fix it:** the post-hoc rule from step 3.3 (variant B: an antecedent-only flip or an inverse means
not a negation). On TEST it took this slice from 66.4% to 98.5%, but it must first be validated on
conditional pairs that are not TEST. The second-largest pattern is xNot360's *If P, Q* → *P and
not Q* negations (13 FNs). That is the real negation of a conditional, and its surface form differs
from A, so overlap-based features underweight it. Conditional pairs in the training data are the fix
for both.

Some sampled false negatives are arguably label errors rather than model errors. In *It's not a
good sign.* / *It's not an evil sign.*, gold says negation, but by the same logic as SemAntoNeg
the pair is close to meaning-preserving.

## 7. What did not work

Every change that failed to improve things, with its numbers.

- **2.1 Symmetric cancellation veto.** On TEST the veto fires 4 times instead of 2. One of the two
  new fires is right (*He's not unhappy.* / *He's happy.*, a SemAntoNeg pair NLI already got
  right) and one is wrong (*The ripped banana wasn't not only ripped …* / *… was not only ripped
  …*, a real negation). Phase-2 accuracy change: +0.0 pts (p = 1.0). VALIDATION is unchanged
  (92.8%) and clean-DEV accuracy drops 91.98% → 91.95%, so it is not adopted.
- **2.2 not + antonym cancellation, on TEST.** The detector is accurate (38 of 41 phase-2 SemAntoNeg
  pairs, 93%), but the Phase 1 baseline already rejects all 41 at the clean NLI threshold, so
  TEST phase-2 accuracy moves −0.1 pts (p = 1.0). One of its 20 new TEST vetoes hits a real
  negation that flips two things (*less toxic … not responsible* / *more toxic … responsible*).
  It was adopted anyway, by the pre-stated rule (VALIDATION tied at 92.8%; clean DEV rose 91.98% →
  92.03%). Its only measurable benefit is DEV SemAntoNeg FPR, 2.4% → 0.8% (n = 125).
- **2.4 Length-aware gate, on TEST.** Separate thresholds for short, medium and long pairs (0.31 /
  0.44 / 0.38) raise antonym recall 87.4% → 89.5% (n = 95). In the same configuration sts_low FPR
  rises 23.6% → 24.4% and random_pair 0.0% → 0.8%. Phase-2 accuracy: −0.1 pts (p = 1.0). Adopted by the
  pre-stated rule on the clean-DEV tie-break (VALIDATION tied at 92.8%, DEV 91.98% → 92.24%), but TEST
  shows no gain. The antonym bypass (2.3) recovers short antonym pairs more precisely.
- **Combining the Phase 2 fixes.** The set adopted by validation (2.2 + 2.3 + 2.4) gains only
  +0.2 pts on phase 2 (p = 0.68), *less* than 2.3 alone (+0.5 pts, p = 0.016). On TEST, 2.2 and 2.4
  cost a little more than they add. VALIDATION could not tell the two apart (93.0% each), and the
  clean-DEV tie-break preferred the combination (92.42% vs 92.21%). The selection rule was not
  overridden after seeing TEST.
- **3.2 Conditional rule as specified (both halves flipped → negation).** On TEST, all 90 pairs where
  both halves flip are xNot360 pairs gold-labelled *not a negation*. *If not P, then not Q* is the
  **inverse** of *If P, then Q*, which is neither its contrapositive nor its negation. Forcing them to
  "negation" takes that slice from 97.8% to 0.0%. xNot360 accuracy falls 75.8% → 62.8% (−13.1 pts,
  p = 0.0001) and all-TEST accuracy 90.8% → 88.0%, even though the antecedent-only branch on its own
  works (44 pairs, 4.5% → 95.5%). Not adopted (VALIDATION 93.0% → 92.7%).
- **4.4 Per-family NLI thresholds.** Only antonyms want a different threshold (0.910 vs the global
  0.975): +4.2 pts antonym recall (88.4% → 92.6%, n = 95) for +0.8 pts FPR on phase-2 negatives. Verbal
  (0.977) and other (0.759) change recall by 0.0 and +1.6 pts, at FPR +0.0 and +1.2 pts. Accuracy moves
  between −0.3 and +0.1 pts per family. It also needs the family known in advance, which a deployed
  detector does not have. The single threshold is not "wrong for both"; it is slightly wrong for
  antonyms.
- **3.1 conditional detection accepts *as if*.** spaCy parses *looks as if the phone isn't …* as an
  `advcl` marked by *if*, and the detector takes it for a conditional. This was found on VALIDATION
  after the Phase 3 TEST runs. It is not fixed, so the reported Phase 3 numbers stay as run. A fix
  would exclude *if* when preceded by *as*, and would treat *even if* / *only if* as distinct
  connectives.
- **5 Fine-tuned cross-encoder on generator output.** Alone it scores 81.3% on phase 2 against 82.8%
  for the off-the-shelf model (−1.5 pts, p = 0.30). Swapped into the cascade, it drops the cascade from
  94.6% to 82.4% (−12.2 pts, p = 0.0001). It calls **37.1% of human paraphrases negations** (off-the-shelf:
  0.3%), and antonym recall falls to 63.2% (n = 95). Its wins are on the generator's own ground: K1 FPR
  0.0% (n = 91) and sts_low FPR 15.0% (alone). See 5.3 for what this says about the generator.
- **The cascade itself, against a learned combiner.** With everything above, the cascade reaches 94.6%
  on phase 2; logistic regression and GBM over the same signals reach 96.4% and 97.9% (4.3). Hand-ordered
  hard rules with hand-picked thresholds lose to letting the data weight the signals.

## 8. Caveats

1. **TEST was read 14 times in this report** (`results/test_reads.json`: steps 0.2, 1.3, 1.4, 2.1–2.4,
   2-combined, 3.2, 3.3, 4.3, 4.4, 5.3, final). Before that it was read by experiments/polarity and
   experiments/composite. **No parameter, threshold, rule or model in this report was fit on TEST**;
   every fit is on clean DEV, and every choice of configuration was made on VALIDATION or clean DEV by
   a rule stated before the run. About 20 configurations are reported, and the p-values are **not
   corrected for multiple comparisons**. Read a p of 0.016 (step 2.3) as suggestive, not settled.
2. **The headline gain is in-distribution.** The combiners are trained on CANNOT phase 1 (+ SNLI /
   STS-B), and CANNOT phase 2 is the same distribution. On xNot360 (out of distribution) no adopted
   change beats honest NLI alone significantly.
3. **No fitting set contains conditionals in quantity.** xNot360 is entirely in TEST, clean DEV has
   36 conditional pairs and VALIDATION has 4. The Phase 3 rules could not be checked before TEST, and
   the 3.3 variants were **designed after seeing TEST** (labelled so everywhere).
4. **Slices under 50 pairs carrying claims:** SemAntoNeg (41 in phase 2, 27 in TEST); the Phase 3
   branches (n = 4, 10, 44); several fine-tune held-out families (step 5.2); VALIDATION's sts_low
   (40) and random_pair (47). The n is printed next to each.
5. **K1 is generator output** (our OS/CN dataset), and 9 of its 91 pairs come from the affixal bug fixed
   in 0.1. The K1 FPR claims (cascade 1.1%, combiner 9.9%, fine-tune 0.0%) carry that caveat.
6. **Weak adoptions.** 2.2 and 2.4 were adopted only on a clean-DEV tie-break (VALIDATION tied). On
   TEST, 2.3 alone beats the adopted set by 0.3 pts. The configuration was not changed after seeing
   that.
7. **Rules partly fit to DEV.** The 22 curated antonym pairs (2.3) were mined from clean-DEV failures.
   The 2.4 bucket edges were fixed in advance; the thresholds were fit on DEV.
8. **Hard negatives may hold a few mislabels.** Mid-score STS-B pairs can be antonym contradictions
   with no cue (*sitting* / *standing*). The cue filter removes only cue negations.
9. **The datasets' label conventions conflict.** CANNOT labels some antecedent or connective flips as
   negations (*even if* → *only if*); xNot360 labels the inverse as not a negation. No single rule
   can satisfy both.
10. **One seed (0).** Bootstrap CIs resample pairs; they do not capture variation from seeds, folds or
    the fine-tune's training order.
11. **Not tried:** per-family thresholds for the generator families (they are not in clean DEV);
    `deberta-v3-small` (MiniLM was used for CPU budget); more than one fine-tune epoch; a fine-tune
    mixing natural paraphrase negatives into generator data; giving the combiner the post-hoc
    conditional features; fixing the *as if* detection bug; calibration of any probability.
12. **Not regenerated after the 0.1 fix:** the saved variant files in `experiments/pattern` and the
    `negation_detection` refsets. Earlier experiments and every K1 number still use them.
13. **`06-composite-nli.md` line 7** still says "cheapest decision first". It was not edited (this brief
    forbids overwriting it); the correction is in step 0.3.
14. **Reproducibility gaps.** `experiments/polarity/` and `experiments/pattern/` are **not committed to
    git**, yet experiments/composite and nli2 import them. `data/dev_clean.csv` and the STS-B train /
    validation files are not committed (licence); they are rebuilt deterministically and checked by
    sha256. Model weights come from the Hugging Face hub (revisions in section 9).
15. **Hardware.** All runs are CPU-only (Ryzen 7 7730U, 8 torch threads). The first fine-tune run was
    killed by a tool time limit and restarted from scratch.

## 9. Reproduction

Seed 0 everywhere (splits, sampling, CV folds, GBM subsampling, fine-tune order, bootstrap,
permutation tests). Python environment: torch 2.14.0+cpu, transformers 5.17.0, scikit-learn 1.9.1,
spaCy 3.8.16 with `en_core_web_sm` 3.8.0.

| model | hub id | revision |
| --- | --- | --- |
| NLI cross-encoder | `cross-encoder/nli-deberta-v3-base` | `6c749ce3425cd33b46d187e45b92bbf96ee12ec7` |
| sentence embeddings | `sentence-transformers/all-mpnet-base-v2` | `e8c3b32edf5434bc2275fc9bab85f82640a19130` |
| fine-tune base | `cross-encoder/nli-MiniLM2-L6-H768` | `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d` (weights sha256 `ec9c5b6a…`) |

Prerequisites: the polarity and composite caches (see `06-composite-nli.md` §7), and the STS-B train /
validation files:

```sh
cd experiments/negation_detection/data/raw
for s in train validation; do curl -sfL -o stsb_$s.jsonl.gz "https://huggingface.co/datasets/mteb/stsbenchmark-sts/resolve/main/$s.jsonl.gz"; done
cd -
```

| step | commit | command |
| --- | --- | --- |
| 0.1 | `c1ac00f` | `./venv/bin/python -m pytest tests -q && ./venv/bin/python experiments/nli2/affixal_audit.py` |
| 0.2 | `1921457` | `./venv/bin/python experiments/nli2/reproduce_baseline.py` |
| 0.3 | `c190799` | (text change only) |
| 1.1 | `4ec66c9` | `./venv/bin/python experiments/nli2/build_dev_clean.py` |
| 1.2 | `690bd47` | `./venv/bin/python experiments/nli2/build_dev_clean.py --hard` |
| 1.3 | `5213db8` | `./venv/bin/python experiments/nli2/refit_nli.py` |
| 1.4 | `1b25bb0` | `./venv/bin/python experiments/nli2/refit_gate.py` |
| 2.1 | `ea91d29` | `./venv/bin/python experiments/nli2/phase2.py --step 2.1` |
| 2.2 | `2395beb` | `./venv/bin/python experiments/nli2/phase2.py --step 2.2` |
| 2.3 | `4bcf2ec` | `./venv/bin/python experiments/nli2/phase2.py --step 2.3` |
| 2.4 | `4782e3f` | `./venv/bin/python experiments/nli2/phase2.py --step 2.4` |
| 2 combined | `1a4e016` | `./venv/bin/python experiments/nli2/phase2.py --step all` |
| 3.1 | `7e389ee` | `./venv/bin/python -m pytest experiments/nli2 -q && ./venv/bin/python experiments/nli2/phase3.py --step 3.1` |
| 3.2 | `851a12e` | `./venv/bin/python experiments/nli2/phase3.py --step 3.2` |
| 3.3 | `38a6d3f` | `./venv/bin/python experiments/nli2/phase3.py --step 3.3` |
| 4.1 | `0d2877c` | `./venv/bin/python experiments/nli2/phase4.py --step 4.1` |
| 4.2 | `342f5fd` | `./venv/bin/python experiments/nli2/phase4.py --step 4.2` |
| 4.3 | `2106e22` | `./venv/bin/python experiments/nli2/phase4.py --step 4.3` |
| 4.4 | `4593291` | `./venv/bin/python experiments/nli2/phase4.py --step 4.4` |
| 5.1 | `fbf66b5` | `./venv/bin/python experiments/nli2/phase5.py --step data` |
| 5.2 | `a50b639` | `./venv/bin/python experiments/nli2/phase5.py --step train` (~28 min CPU) |
| 5.3 | `d404993` | `./venv/bin/python experiments/nli2/phase5.py --step eval` |
| summary | this commit (`docs: summary …`) | `./venv/bin/python experiments/nli2/final.py` |

Every step appends to `results/test_reads.json`; rerunning a step adds a line there.

## 10. Next steps

Ranked by expected gain per unit of work, from what the steps above found:

1. **Validate the conditional inverse rule off TEST.** Conditional inverses are 44 of the final model's
   63 false positives. The post-hoc rule (3.3 variant B) took that slice from 66.4% to 98.5%, but it
   was designed after seeing TEST. Build a conditional DEV set: split xNot360 going forward, or
   generate *If P, Q* pairs and their inverses and negations. Then confirm or reject it, and fix the
   *as if* / *even if* / *only if* detection bug first.
2. **Give the combiner the data its features need.** Clean DEV has 1 antecedent flip and about 2%
   cancellations, so the GBM learned near-zero weight for both, and its K1 FPR rose to 9.9%. Adding a
   few hundred natural or generated cancellation and conditional pairs to its training data should
   recover K1 without the hard veto. Alternatively, keep the cascade's veto as a hard override in front
   of the GBM, which costs one rule.
3. **Retrain the fine-tune with natural negatives.** Generator-only training produced a model that
   flags 37% of paraphrases. Mix CANNOT phase-1 paraphrases and SemAntoNeg into the generator data,
   and use the generator for what it teaches well (cancellation, morphology).
4. **Get a fresh test set.** This TEST has now been read by three experiments and 14 times here. A new
   held-out sample of CANNOT plus a conditional set would make the next headline number clean.
5. **Drop what did not earn its place.** 2.1 (symmetric veto), 2.4 (length buckets) and the 2.2
   veto added nothing measurable on TEST. On VALIDATION the simpler cascade (Phase 1 baseline + 2.3)
   ties the adopted set, so it is the one to keep if a cascade is kept at all.

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
earlier experiment (polarity, composite, `06-composite-nli.md`) and step 0.2 below still use the old
variants. 11 of the 91 K1 pairs used in evaluation come from the K1-affixal generator, and the
fixed rule rejects 9 of those 11, so the reported K1 FPRs include 9 defective pairs (9.9% of K1). Phase 5 regenerates training data with the fixed generator.

```sh
./venv/bin/python -m pytest tests -q                    # 385 passed
./venv/bin/python experiments/nli2/affixal_audit.py
```

### 0.2 Honest baseline — reproduced

`experiments/nli2/reproduce_baseline.py` → `results/0_2_reproduce.json`. It re-runs the
`06-composite-nli.md` system with the same rows, the same DEV fits (`run_composite.configure`) and
the same scorers. The refit configuration is identical to the report's: NLI threshold 0.00125,
raw cosine, t1 = 0.36. Every headline number matches to the reported precision:

| config | part | n | accuracy (95% CI) | AUC | 06-composite-nli.md |
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

The cascade is **not faster** than NLI alone. Measured in `06-composite-nli.md` (one CPU, 8 threads,
batch 16, 256 DEV pairs): the gate's mpnet embedding of both sentences costs **36.4 ms/pair**,
NLI costs **40.2 ms/pair**, and the veto's parse costs 4.6 ms/pair. With fresh embeddings the full
cascade takes **86.3% more** wall clock than NLI on every TEST pair (NLI+gate: 76.6% more). It only
saves time when embeddings already exist: NLI+gate saves 13.8% and the full system 4.2%.

The only claim of speed was the phrase "cheapest decision first" in
`experiments/composite/report_composite.py`, which is now replaced with the measured costs. The
same phrase is still in line 7 of `06-composite-nli.md`. That file was not edited, because this brief
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
- **The CSV is gitignored.** `06-composite-nli.md`'s sources table records STS-B's licence as unclear and
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

### 1.3 NLI threshold refit on clean DEV

See **section 3**, which holds the full table and verdict. In one line: the threshold moves from
0.00125 to 0.97533. Honest NLI alone gets 82.8% on phase 2 (was 78.8%). The gate's gain over it is
+10.6 pts (was +13.8), p = 0.0001; recall falls 5.1 pts. On clean DEV: NLI alone 77.7% → 83.8%,
NLI+gate 88.6% → 92.0% (old → new threshold).

```sh
./venv/bin/python experiments/nli2/refit_nli.py
```

### 1.4 Gate threshold refit on clean DEV — the Phase 1 baseline

`experiments/nli2/refit_gate.py` → `results/1_4_refit_gate.json`. Same procedure as
`06-composite-nli.md` (`composite.fit_gate`), with step 1.3's NLI threshold:

| | similarity | t1 | DEV J of gate → NLI | DEV positives the gate keeps | stage-3 threshold (DEV refit) |
| --- | --- | ---: | ---: | ---: | ---: |
| 06-composite-nli.md (old DEV) | cos | 0.36 | 0.8835 | 99.6% | 0.00125 |
| clean DEV | **cos** (c_resid: t1 0.37, J 0.8382) | **0.37** | 0.8388 | 99.1% | **0.07545** |

t1 hardly moves. The stage-3 threshold does, because the clean DEV's dissimilar negatives (sts_low,
snli_neutral, random) are what forced NLI-alone's threshold up to 0.975, and the gate removes them
first. On TEST (all three thresholds fit on clean DEV):

| config | phase-2 acc (95% CI) | all-TEST acc (95% CI) | AUC | recall | FPR paraphrase | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | VAL acc |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NLI alone (thr 0.975) | 82.8% (80.9–84.6) | 81.8% (79.9–83.6) | 0.840 | 93.0% | 0.3% | 70.9% | 2.6% | 78.1% | 40.0% | 88.4% | 86.0% |
| NLI + gate, old t1 0.36 | 94.6% (93.4–95.6) | 90.7% (89.4–92.0) | 0.956 | 95.4% | 1.6% | 23.6% | 2.6% | 0.0% | 46.7% | 87.4% | 92.7% |
| NLI + gate, t1 0.37 | 94.5% (93.3–95.6) | 90.7% (89.3–91.9) | 0.956 | 95.2% | 1.6% | 23.6% | 2.6% | 0.0% | 46.7% | 87.4% | 92.9% |
| **full (+ veto) — Phase 1 baseline** | **94.4% (93.2–95.6)** | **90.6% (89.3–91.9)** | 0.955 | 95.1% | 1.6% | 23.6% | 2.6% | 0.0% | 46.7% | 87.4% | 92.9% |

SemAntoNeg FPR is 0.0% (n = 27) in every row. Gains over honest NLI alone, on phase 2:
NLI + gate +11.7 pts (p = 0.0001); full +11.6 pts (p = 0.0001). Refitting t1 from 0.36 to 0.37 changes
phase-2 accuracy by −0.07 pts (p = 1.0): nothing. On TEST, the gate stops 247 pairs and the veto 2.

The Phase 1 baseline for Phase 2 is the **full** cascade at these thresholds: cos, t1 = 0.37,
stage-3 threshold 0.07545, veto on. The veto is included because 2.1 and 2.2 change it.

```sh
./venv/bin/python experiments/nli2/refit_gate.py
```

### 2.1 Symmetric cancellation veto — no effect, not adopted

`experiments/nli2/phase2.py --step 2.1` → `results/2_2.1.json`. The veto now also fires when *A*
carries the same-clause cancellation (and polarities match). The Phase 2 machinery
(`pipeline.py`, `rules.py`) reproduces the step-1.4 baseline exactly. From here on, K1 FPR is
measured on the 91 K1 pairs as a held-out slice, since they are no longer in any fitting set
(generator caveat applies). SemAntoNeg is pooled over all of phase 2 (validation + test, n = 41).

| config | phase-2 acc (95% CI) | all-TEST acc (95% CI) | AUC | recall | FPR paraphrase | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | SemAntoNeg FPR (n=41) | VAL acc | gain vs baseline, phase 2 (p) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 1 baseline | 94.4% (93.2–95.6) | 90.6% (89.3–91.9) | 0.955 | 95.1% | 1.6% | 23.6% | 2.6% | 0.0% | 46.7% | 87.4% | 1.1% | 0.0% | 92.8% | — |
| 2.1 symmetric veto | 94.4% (93.2–95.6) | 90.6% (89.3–91.9) | 0.954 | 95.1% | 1.6% | 23.6% | 2.6% | 0.0% | 46.7% | 87.4% | 1.1% | 0.0% | 92.8% | +0.0 pts (p = 1.0) |

How often the veto fires on TEST (n = 1,760): **B only: 2** (1 on a true negation); **symmetric: 4** (2 on true
negations). On VALIDATION: 1 → 2. The new fires:

| A | B | gold | veto right? |
| --- | --- | --- | --- |
| He's not unhappy. | He's happy. | not a negation (SemAntoNeg) | yes, but NLI already rejected it |
| The ripped banana wasn't not only ripped, but petrified and tasteless. | The ripped banana was not only ripped, but petrified and tasteless. | negation | **no** |

So the veto is too rare to matter in either direction on natural data. The baseline's K1 FPR is
already 1.1% (n = 91) at the clean stage-3 threshold, so there is little left for it to fix.

```sh
./venv/bin/python experiments/nli2/phase2.py --step 2.1
```

### 2.2 not + antonym cancellation — the detector works, the accuracy does not move

`experiments/nli2/phase2.py --step 2.2` → `results/2_2.2.json`; detector in
`rules.not_antonym_cancel`. One sentence adds a negation cue that the other lacks (*not*, *n't*,
*never*, *no*, …), and after alignment the only content change is one token pair that are WordNet
antonyms (*good* / *evil*). Since not(antonym(x)) ≈ x, the meaning is kept, so the veto fires. It
checks both directions. It does **not** fire when both sides carry the cue (*She's not out.* /
*She's not safe.* is a real negation).

**SemAntoNeg slice (phase 2, n = 41, all label 0):** the detector fires on **38 of 41 (92.7%)**.
It misses 3: *It's clear.* / *It's not ill-defined.* (hyphenated word, not a WordNet antonym of
*clear*), and *You are playful.* / *You are not serious.* and *You are not serious?* / *You are
playful?* (*playful*'s only WordNet antonym is *unplayful*). FPR on the slice: baseline **0.0%**,
with 2.2 **0.0%** (n = 41). NLI at the clean threshold already rejected every one of them, so
the veto has nothing left to remove on TEST.

| config | phase-2 acc (95% CI) | all-TEST acc (95% CI) | AUC | recall | FPR paraphrase | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | SemAntoNeg FPR (n=41) | VAL acc | gain vs baseline, phase 2 (p) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 1 baseline | 94.4% (93.2–95.6) | 90.6% (89.3–91.9) | 0.955 | 95.1% | 1.6% | 23.6% | 2.6% | 0.0% | 46.7% | 87.4% | 1.1% | 0.0% | 92.8% | — |
| 2.2 + not-antonym cancellation | 94.4% (93.1–95.5) | 90.6% (89.2–91.8) | 0.954 | 95.0% | 1.6% | 23.6% | 2.6% | 0.0% | 46.7% | 87.4% | 1.1% | 0.0% | 92.8% | −0.1 pts (p = 1.0) |

Veto fires on TEST: 2 → 22. Of the 20 new fires, 19 are on non-negations and 1 is on a real
negation: *Acetaldehyde is less toxic than alcohol and is not responsible for many hangover
symptoms.* / *… more toxic … and is responsible …*. B flips two things, *less → more* and drops
*not*, and the rule reads the pair as one antonym swap plus one removed cue. On clean DEV it fires
76 more times, none on a negation; DEV SemAntoNeg FPR goes 2.4% → 0.8% (n = 125), with antonym
recall unchanged (91.8%, n = 208).

**Adopted** under the pre-stated rule (VALIDATION accuracy rises, or ties and clean DEV breaks the
tie). VALIDATION is tied at 92.8% and clean DEV goes 91.98% → 92.03%. This is a weak adoption: the
TEST cost (one true negation lost, −0.1 pts, p = 1.0) is the same size as the DEV benefit.

```sh
./venv/bin/python experiments/nli2/phase2.py --step 2.2
```

### 2.3 Pair-level antonym bypass — works, adopted

`experiments/nli2/phase2.py --step 2.3` → `results/2_2.3.json`; rule in `rules.antonym_aligned`.
Before the gate runs, A and B are aligned token by token on lemmas, ignoring punctuation and
*a/an/the*. If there is **exactly one** differing aligned position, it holds one token on each side,
the two share a dependency slot, and they are antonyms, then the pair skips the gate and the veto
and goes straight to NLI. "Antonyms" means WordNet (any part of speech, surface or lemma, either
direction) or a curated list of 26 pairs. The curated list is the brief's four, which WordNet
already has, plus 22 that WordNet misses (*right/incorrect*, *hard/easy*, *serious/playful*,
*increase/decline*, …). All 22 were found among failing aligned antonym negations in **clean DEV
only**.

| config | phase-2 acc (95% CI) | all-TEST acc (95% CI) | AUC | recall | FPR paraphrase | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | SemAntoNeg FPR (n=41) | VAL acc | gain vs baseline, phase 2 (p) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 1 baseline | 94.4% (93.2–95.6) | 90.6% (89.3–91.9) | 0.955 | 95.1% | 1.6% | 23.6% | 2.6% | 0.0% | 46.7% | 87.4% | 1.1% | 0.0% | 92.8% | — |
| 2.3 antonym bypass | **94.9% (93.8–96.0)** | **91.0% (89.7–92.3)** | 0.962 | 95.9% | 1.6% | 23.6% | 2.6% | **0.0%** | 46.7% | **94.7%** | 1.1% | 0.0% | 93.0% | **+0.5 pts (p = 0.016)** |

**Antonym recall 87.4% → 94.7% (n = 95), with random_pair FPR unchanged at 0.0% (n = 119).** All-TEST
gain +0.4 pts (p = 0.015). No other FPR moves. The strict alignment does its job:

| split | bypass fires | on non-negations | what it caught |
| --- | ---: | ---: | --- |
| clean DEV | 137 | 1 (a hard negative) | 95 antonym, 37 other, 4 verbal negations |
| VALIDATION | 29 | 0 | 22 antonym, 5 other, 2 verbal |
| TEST | 48 | 0 | 39 antonym, 7 other, 2 verbal |

On TEST the gate now stops 239 pairs instead of 247; the 8 rescued pairs are negations NLI then
gets right. **Adopted** (VALIDATION 92.8% → 93.0%, clean DEV 91.98% → 92.21%). Cost: the curated list
was mined from DEV, so the rule is partly fit to DEV (caveats). How much the 22 curated pairs add
over WordNet alone was not measured separately.

```sh
./venv/bin/python experiments/nli2/phase2.py --step 2.3
```

### 2.4 Length-aware gate threshold — short pairs get a lower bar, TEST does not improve

`experiments/nli2/phase2.py --step 2.4` → `results/2_2.4.json`. Pairs are bucketed by
min(len(A), len(B)) in words. The bucket edges (≤ 6, 7–12, ≥ 13) were set in advance, not tuned. Within
each bucket t1 is fit by the same rule as before: max DEV J of gate → NLI, subject to the gate
keeping ≥ 95% of that bucket's DEV positives.

| min length bucket | DEV pairs | DEV positives | t1 (was 0.37 everywhere) | DEV J | DEV positives the gate keeps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–6 words | 849 | 255 | **0.31** | 0.8088 | 96.9% |
| 7–12 words | 1,669 | 625 | 0.44 | 0.8766 | 99.4% |
| ≥ 13 words | 884 | 621 | 0.38 | 0.8351 | 99.8% |

The hypothesis holds on DEV: short pairs want a lower threshold. On TEST:

| config | phase-2 acc (95% CI) | all-TEST acc (95% CI) | AUC | recall | FPR paraphrase | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | SemAntoNeg FPR (n=41) | VAL acc | gain vs baseline, phase 2 (p) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 1 baseline | 94.4% (93.2–95.6) | 90.6% (89.3–91.9) | 0.955 | 95.1% | 1.6% | 23.6% | 2.6% | 0.0% | 46.7% | 87.4% | 1.1% | 0.0% | 92.8% | — |
| 2.4 length-aware gate | 94.4% (93.1–95.5) | 90.6% (89.1–91.9) | 0.953 | 95.0% | 1.6% | 24.4% | 0.9% | 0.8% | 46.7% | 89.5% | 1.1% | 0.0% | 92.8% | −0.1 pts (p = 1.0) |

Recall recovered: antonym recall +2.1 pts (87.4% → 89.5%, n = 95, i.e. 2 pairs). Overall recall does not
move (95.1% → 95.0%), because the 7–12 bucket's higher threshold (0.44) costs elsewhere what the short
bucket wins. The gate still stops 247 TEST pairs, just not the same 247. snli_neutral FPR falls
2.6% → 0.9%, while sts_low rises 23.6% → 24.4% and random_pair 0.0% → 0.8%. **Adopted** by the pre-stated rule
(VALIDATION tied at 92.8%; clean DEV 91.98% → 92.24%), but TEST shows no gain (see section 7).

```sh
./venv/bin/python experiments/nli2/phase2.py --step 2.4
```

### 2 (combined) All Phase 2 fixes together

`experiments/nli2/phase2.py --step all` → `results/2_all.json`. Two combinations: all four fixes,
and the ones adopted by validation (2.2, 2.3, 2.4; 2.1 was not adopted). They differ only in the
veto direction.

| config | phase-2 acc (95% CI) | all-TEST acc (95% CI) | AUC | recall | FPR paraphrase | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | SemAntoNeg FPR (n=41) | VAL acc | gain vs baseline, phase 2 (p) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 1 baseline | 94.4% (93.2–95.6) | 90.6% (89.3–91.9) | 0.955 | 95.1% | 1.6% | 23.6% | 2.6% | 0.0% | 46.7% | 87.4% | 1.1% | 0.0% | 92.8% | — |
| 2.3 alone (for reference) | 94.9% (93.8–96.0) | 91.0% (89.7–92.3) | 0.962 | 95.9% | 1.6% | 23.6% | 2.6% | 0.0% | 46.7% | 94.7% | 1.1% | 0.0% | 93.0% | +0.5 pts (p = 0.016) |
| all four fixes | 94.6% (93.4–95.7) | 90.8% (89.4–92.1) | 0.957 | 95.5% | 1.6% | 24.4% | 0.9% | 0.8% | 46.7% | 94.7% | 1.1% | 0.0% | 93.0% | +0.2 pts (p = 0.68) |
| **adopted by validation (2.2 + 2.3 + 2.4)** | 94.6% (93.4–95.7) | 90.8% (89.4–92.1) | 0.958 | 95.5% | 1.6% | 24.4% | 0.9% | 0.8% | 46.7% | 94.7% | 1.1% | 0.0% | 93.0% | +0.2 pts (p = 0.68) |

Together, the fixes keep 2.3's antonym recall (94.7%) but take on 2.4's FPR costs (sts_low 24.4%,
random_pair 0.8%). Net: +0.2 pts on phase 2 and +0.2 pts on all TEST, neither significant (p = 0.68).
**The configuration carried into Phases 3 and 4 is the validation-adopted set**, because that is
what the pre-stated rule picks: VALIDATION 93.0% for both it and 2.3 alone, clean DEV 92.42% vs
92.21%. TEST prefers 2.3 alone by 0.3 pts; switching on that basis would be tuning on TEST.

```sh
./venv/bin/python experiments/nli2/phase2.py --step all
```

### 3.1 Conditionals detected and split

`experiments/nli2/conditionals.py`, tests in `experiments/nli2/test_nli2.py` (18 pass). A
conditional is an adverbial clause attached to its consequent's head and marked by *if*, *unless*,
*when*, *whenever*, *provided* or *assuming*. spaCy parses these markers three ways, and all three
are handled:

| shape | markers | antecedent head | example |
| --- | --- | --- | --- |
| `mark` child of an `advcl` | if, unless, whenever | the `advcl` verb | *If you sign the contract, you will receive benefits.* |
| `advmod` child of an `advcl` | when, whenever | the `advcl` verb | *When it rains, the street gets wet.* |
| the marker heads the clause | provided, assuming | the marker's `ccomp` | *Provided that you sign, …* |

Antecedent and consequent spans are `negation.nlp_core.clause_scope()` of their heads, so there is
no second scope implementation. Each span also includes the `clause_scope` of the head's `conj`
heads, since `clause_scope` stops at `conj` and would otherwise drop *and pay* from *If you sign and
pay*. Punctuation is dropped. Nesting (a conditional inside another's antecedent) is detected by
subtree containment, and the decision uses the outermost conditional.

Tests: preposed and inverted conditionals (*You will receive benefits if you sign*), *unless*,
*when*, *provided that*, *assuming*, a coordinated antecedent, sibling conditionals (two
antecedents, one consequent), a nested conditional, and *He asked if it was raining* (a `ccomp`,
correctly not a conditional).

How often pairs contain conditionals (`phase3.py --step 3.1` → `results/3_1_conditional_coverage.json`;
no labels read):

| split | both A and B conditional | A only | B only | neither |
| --- | ---: | ---: | ---: | ---: |
| clean DEV (n = 3,402) | 36 | 5 | 9 | 3,352 |
| VALIDATION (n = 601) | 4 | 0 | 1 | 596 |
| TEST — CANNOT phase 2 (n = 1,400) | 14 | 3 | 5 | 1,378 |
| TEST — xNot360 (n = 360) | **134** | **190** | 0 | 36 |

The pair rule (3.2) applies only when both sides are conditionals, so it can reach at most 134 of the
360 xNot360 pairs. In the 190 "A only" pairs, B restates the situation without *if* (*If P, Q* /
*P and not Q*). That is the logical negation of a conditional, and the cascade still decides those.
No nested conditional occurs in any split. **No fitting set contains more than 36 conditional pairs**
(VALIDATION: 4), so nothing about the rule can be tuned or checked before TEST (caveats).

```sh
./venv/bin/python -m pytest experiments/nli2 -q
./venv/bin/python experiments/nli2/phase3.py --step 3.1
```

### 3.2 Conditional rule, as specified — one branch works, one destroys xNot360

`experiments/nli2/phase3.py --step 3.2` → `results/3_2_conditionals.json`. The rule runs in front of
the Phase 2 adopted configuration. It applies only when **both** A and B are conditionals. Each
half's polarity is the parity of the v2 cues in that half's clauses, and *unless* starts the
antecedent negated:

| branch (as in the brief) | decision | TEST pairs | gold label 0 / 1 | accuracy before → after |
| --- | --- | ---: | --- | --- |
| consequent flipped only | negation | 4 | 2 / 2 | 75.0% → 50.0% (n = 4) |
| antecedent flipped only | **not** a negation | 44 | 42 / 2 | **4.5% → 95.5%** (n = 44, p = 0.0001) |
| both flipped, tagged `contraposition` | negation | 90 | **90 / 0** | **97.8% → 0.0%** (n = 90, p = 0.0001) |
| neither flipped | cascade decides | 10 | 5 / 5 | 90.0% → 90.0% (n = 10) |

| config | phase-2 acc (95% CI) | all-TEST acc (95% CI) | AUC | recall | FPR paraphrase | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | SemAntoNeg FPR (n=41) | VAL acc | gain vs baseline, phase 2 (p) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 2 adopted config | 94.6% (93.4–95.7) | 90.8% (89.4–92.1) | 0.958 | 95.5% | 1.6% | 24.4% | 0.9% | 0.8% | 46.7% | 94.7% | 1.1% | 0.0% | 93.0% | — |
| + conditional rule (as specified) | 94.5% (93.3–95.6) | 88.0% (86.5–89.5) | 0.860 | 95.2% | 1.9% | 24.4% | 0.9% | 0.8% | **72.2%** | 94.7% | 1.1% | 0.0% | 92.7% | −0.1 pts (p = 0.50) |

All 90 "both flipped" pairs are xNot360 pairs of the form *If P, Q* / *If not P, not Q*, and the dataset
labels every one of them *not a negation*. Logically that is right. *If not P, then not Q* is the
**inverse** of the conditional. The contrapositive would be *If not Q, then not P*, and the negation
is *P and not Q*. So the brief's "both flipped → negation (contraposition)" branch encodes a logical
error, and the data punishes it: xNot360 75.8% → 62.8% (−13.1 pts, p = 0.0001).

**Not adopted** (VALIDATION 93.0% → 92.7%; VALIDATION has only 4 conditional pairs, so it can barely see
this). Step 3.3 breaks the result down by slice and reports post-hoc variants.

```sh
./venv/bin/python experiments/nli2/phase3.py --step 3.2
```

### 3.3 Conditionals — slice report and post-hoc variants

`experiments/nli2/phase3.py --step 3.3` → `results/3_3_conditionals_posthoc.json` (slices for the
as-specified rule are in `results/3_2_conditionals.json`).

**Slices, rule as specified (3.2)** — TEST:

| slice | n | accuracy before → after | p |
| --- | ---: | --- | ---: |
| xNot360, all | 360 | 75.8% → **62.8%** | 0.0001 |
| xNot360, both sides conditional | 134 | 66.4% → 31.3% | 0.0002 |
| xNot360, the rest | 226 | 81.4% → 81.4% | 1.0 |
| every pair that is not both-conditional (CANNOT + xNot360) | 1,612 | 92.8% → 92.8% | 1.0 |
| CANNOT phase 2 | 1,400 | 94.6% → 94.5% | 0.50 |

The non-conditional slice does not move, so the rule broke nothing outside conditionals. The
xNot360 baseline here (75.8%) is higher than `06-composite-nli.md`'s 66.1% because of the clean thresholds
from Phase 1.

**Worked example of each branch** (TEST unless marked):

| branch | A | B | gold | rule says |
| --- | --- | --- | --- | --- |
| consequent flipped → negation | If the museum has a long-range conservation plan, it can be mentioned here. | If the museum has a long-range conservation plan, it cannot be written here. | negation | negation ✓ |
| antecedent flipped only → not a negation | If a person files for bankruptcy, their debts may be discharged. | If a person does not file for bankruptcy, their debts may be discharged. | not a negation | not a negation ✓ |
| both flipped → negation (`contraposition`) | If a person is found guilty of extortion, they can be sentenced to prison. | If a person is not found guilty of extortion, they cannot be sentenced to prison. | not a negation | negation ✗ (all 90 such pairs) |
| *unless* = *if not* (constructed, unit test) | Unless you sign, you will receive benefits. | If you do not sign, you will receive benefits. | same claim | no flip → cascade decides ✓ |
| *unless*, from TEST | Confidentiality must be maintained unless required by law | Confidentiality is maintained even when required by law to disclose information | negation | antecedent-only flip → not a negation ✗ |

The last row shows a limit: B re-expresses the exception as *even when* and turns it into a
claim. The rule reads polarities, not what the condition means.

**Post-hoc variants — designed after seeing the 3.2 TEST result, so these numbers are
test-informed and the variants are not adopted:**

| config | phase-2 acc (95% CI) | all-TEST acc (95% CI) | AUC | xNot360 acc | xNot360 conditional slice (n=134) | non-conditional pairs (n=1,612) | xNot360 not_neg FPR | VAL acc |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 2 adopted config | 94.6% (93.4–95.7) | 90.8% (89.4–92.1) | 0.958 | 75.8% | 66.4% | 92.8% | 46.7% | 93.0% |
| rule as specified (3.2) | 94.5% (93.3–95.6) | 88.0% (86.5–89.5) | 0.860 | 62.8% | 31.3% | 92.8% | 72.2% | 92.7% |
| A: both flipped → **not** a negation | 94.5% (93.3–95.6) | 93.1% (91.9–94.3) | 0.962 | 87.8% | 98.5% | 92.8% | 22.2% | 92.7% |
| B: veto only (never force a negation) | 94.6% (93.4–95.6) | 93.2% (92.0–94.3) | 0.964 | 87.8% | 98.5% | 92.8% | 22.2% | 92.7% |

Treating the inverse as what it is (not a negation) lifts xNot360 from 75.8% to 87.8% (+11.9 pts,
p = 0.0001) and all of TEST from 90.8% to 93.2% (variant B, +2.4 pts, p = 0.0001). The cost, in the same
sentence: VALIDATION falls 93.0% → 92.7% and CANNOT phase 2 moves −0.1 pts. **The two datasets
disagree.** CANNOT labels some antecedent flips as negations, e.g. *If you haven't choked in your
own vomit by the end … / If you have choked …* (label 1, TEST), and the rule rejects those. xNot360
labels the inverse as not a negation. The VALIDATION drop is 2 pairs, both real negations that the
rule wrongly stops, and neither is a clean antecedent flip. One is *… even if all formats do not meet
the chart rules* / *… only if all formats meet the chart rules*: the connective itself changes (*even
if* → *only if*), which the rule does not model. The other is *… looks as if the phone isn't all
lightweight cheap plastic* / *… is all …*, where *as if* is **not a conditional at all**. That is a
3.1 detection bug, found on VALIDATION after the TEST runs and left unfixed so the reported runs stay
as they were (see section 7). *(Corrected while writing the summary: an earlier version of this
paragraph said both were CANNOT antecedent flips.)*

**Why not adopted:** the pre-stated rule picks by VALIDATION accuracy, which falls; the variants
were designed after seeing TEST; and no fitting set has enough conditionals to check them (36 in
clean DEV, 4 in VALIDATION). Variant B is the best-supported next step, if it can be confirmed on a
conditional set that is not TEST (section 10).

```sh
./venv/bin/python experiments/nli2/phase3.py --step 3.3
```

### 4.1 Feature table

`experiments/nli2/phase4.py --step 4.1` → `results/4_1_features.csv` (one row per pair, features
and labels only, no sentence text; 5,854 rows: clean DEV 3,402, VALIDATION 601, TEST 1,760, K1 91) and
`results/4_1_features.json`. The features are exactly the brief's list, and every one comes from
machinery already in the pipeline:

| feature | definition | clean DEV mean | TEST mean |
| --- | --- | ---: | ---: |
| `cos_sim` | mpnet cosine(A, B) | 0.681 | 0.694 |
| `p_contradiction` | NLI P(contradiction) | 0.559 | 0.637 |
| `p_entailment` | NLI P(entailment) | 0.200 | 0.156 |
| `p_neutral` | NLI P(neutral) | 0.241 | 0.207 |
| `cue_count_a` | lexicon negation cues in A | 0.262 | 0.236 |
| `cue_count_b` | lexicon negation cues in B | 0.252 | 0.457 |
| `net_negation` | v2 polarity(B) − v2 polarity(A) | −0.019 | 0.161 |
| `cancellation_a` | A has ≥ 2 cues cancelling inside one clause | 0.010 | 0.010 |
| `cancellation_b` | B has ≥ 2 cues cancelling inside one clause | 0.017 | 0.011 |
| `antonym_aligned` | the 2.3 aligned antonym swap | 0.040 | 0.027 |
| `has_conditional_a` | A is a conditional (3.1) | 0.012 | 0.194 |
| `has_conditional_b` | B is a conditional (3.1) | 0.013 | 0.087 |
| `antecedent_only_flip` | only the antecedent's polarity flips (3.2) | 0.000 | 0.025 |
| `len_a`, `len_b` | words in A, B | 11.1, 10.6 | 12.0, 11.3 |
| `len_diff` | absolute difference of the two lengths | 2.259 | 2.461 |
| `token_overlap` | word-set Jaccard overlap | 0.540 | 0.570 |

**Distribution shift, stated before training:** clean DEV has almost no conditionals (1.2% of A
sentences vs 19.4% in TEST), and `antecedent_only_flip` fires on **1** clean DEV pair and 44 TEST
pairs. *(Corrected in the 4.2 commit: the 4.1 commit said 0.)* A combiner trained on clean DEV therefore cannot learn the conditional pattern from data,
whatever weight it might deserve. `net_negation` and `cue_count_b` also differ (TEST has more
explicit cue negations, mostly in xNot360).

```sh
./venv/bin/python experiments/nli2/phase4.py --step 4.1
```

### 4.2 Logistic regression and gradient-boosted trees, trained on clean DEV

`experiments/nli2/phase4.py --step 4.2` → `results/4_2_combiners.json`. Both models are trained on
the 3,402 clean DEV pairs only. Hyperparameters were chosen by 5-fold cross-validated AUC. The folds
are grouped by connected component of shared sentences, so **no sentence (as A or B) is in two
folds**; the code asserts this. That is stricter than the repository's existing `group_folds`, which
holds out base sentence A only. The decision threshold is Youden's J on the out-of-fold DEV
predictions. The final model is refit on all of clean DEV. Seed 0 throughout.

| model | grid (CV AUC) | chosen | threshold | out-of-fold DEV acc | VALIDATION acc (95% CI) | VALIDATION AUC |
| --- | --- | --- | ---: | ---: | --- | ---: |
| logistic regression (standardised features) | C = 0.01: 0.9810, 0.1: 0.9818, 1: 0.9818, 10: 0.9818 | C = 0.1 | 0.581 | 94.4% | 96.3% (94.8–97.7) | 0.989 |
| gradient-boosted trees (300 trees, subsample 0.8) | depth 2 / lr 0.03: 0.9915, 2 / 0.1: 0.9929, 3 / 0.03: 0.9927, 3 / 0.1: 0.9932 | depth 3, lr 0.1 | 0.428 | 96.6% | **97.8% (96.7–99.0)** | 0.997 |

For reference, the cascade (Phase 2 adopted config) scores 93.0% on VALIDATION.

**What the models use.** Logistic coefficients are per standard deviation of the feature, so
they can be compared directly. A positive coefficient pushes towards "negation". GBM importance is
the share of impurity reduction.

| feature | logistic coefficient | GBM importance |
| --- | ---: | ---: |
| `token_overlap` | +1.691 | 0.327 |
| `p_contradiction` | +1.442 | 0.537 |
| `p_entailment` | −1.046 | 0.011 |
| `len_diff` | −0.969 | 0.005 |
| `p_neutral` | −0.717 | 0.015 |
| `antonym_aligned` | +0.688 | 0.020 |
| `cos_sim` | +0.681 | 0.044 |
| `cue_count_b` | +0.385 | 0.004 |
| `cue_count_a` | +0.369 | 0.004 |
| `len_a` | +0.317 | 0.022 |
| `len_b` | +0.236 | 0.009 |
| `has_conditional_b` | −0.091 | 0.000 |
| `net_negation` | −0.080 | 0.002 |
| `has_conditional_a` | −0.080 | 0.001 |
| `cancellation_a` | +0.050 | 0.001 |
| `antecedent_only_flip` | +0.036 | 0.000 |
| `cancellation_b` | +0.006 | 0.000 |
| *(intercept)* | see JSON | — |

Reading it in plain terms:
- **A negation is a contradiction between near-identical sentences.** NLI's contradiction
  probability and word overlap do most of the work, and both models agree on that.
- **The other NLI probabilities matter in the linear model.** It uses entailment and neutral as
  separate evidence against a negation, information the cascade threw away by keeping only
  P(contradiction).
- **A big length difference argues against a negation**, since a negation rarely adds a clause.
- **The aligned antonym swap (2.3) gets real weight** in the linear model.
- **The conditional and cancellation features get almost nothing.** That is not evidence that they
  are useless: clean DEV contains 1 antecedent-only flip and about 1–2% cancellations, so there is
  nothing to learn them from.

```sh
./venv/bin/python experiments/nli2/phase4.py --step 4.2
```

### 4.3 Cascade vs logistic vs gradient-boosted trees, on TEST

`experiments/nli2/phase4.py --step 4.3` → `results/4_3_compare.json`. Every model's parameters and
threshold were fit on clean DEV only.

| model | phase-2 acc (95% CI) | xNot360 acc (95% CI) | all-TEST acc (95% CI) | AUC | recall | FPR paraphrase | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | SemAntoNeg FPR (n=41) | VAL acc |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NLI alone (honest threshold) | 82.8% (80.9–84.6) | 77.8% (73.3–82.2) | 81.8% (79.9–83.6) | 0.840 | 93.0% | 0.3% | 70.9% | 2.6% | 78.1% | 40.0% | 88.4% | 4.4% | 0.0% | 86.0% |
| cascade (Phase 2 adopted) | 94.6% (93.4–95.7) | 75.8% (71.4–80.3) | 90.8% (89.4–92.1) | 0.958 | 95.5% | 1.6% | 24.4% | 0.9% | 0.8% | 46.7% | 94.7% | **1.1%** | 0.0% | 93.0% |
| logistic regression | 96.4% (95.4–97.3) | 78.3% (74.2–82.5) | 92.7% (91.4–93.9) | 0.941 | 93.8% | 1.6% | 6.3% | 0.0% | 0.0% | 33.9% | 91.6% | 6.6% | 0.0% | 96.3% |
| **gradient-boosted trees** | **97.9% (97.1–98.6)** | 78.9% (74.7–83.3) | **94.0% (92.8–95.1)** | **0.981** | 95.1% | 1.9% | **3.9%** | 0.0% | 0.0% | 28.9% | 92.6% | 9.9% | 0.0% | **97.8%** |

Paired permutation tests (10,000 permutations):

| comparison | phase 2 | all TEST | xNot360 |
| --- | --- | --- | --- |
| cascade vs honest NLI | +11.9 pts (p = 0.0001) | +9.0 pts (p = 0.0001) | −1.9 pts (p = 0.14) |
| logistic vs cascade | +1.7 pts (p = 0.0023) | +1.9 pts (p = 0.0012) | +2.5 pts (p = 0.19) |
| **GBM vs cascade** | **+3.2 pts (p = 0.0001)** | **+3.2 pts (p = 0.0001)** | +3.1 pts (p = 0.18) |
| GBM vs honest NLI | +15.1 pts (p = 0.0001) | +12.2 pts (p = 0.0001) | +1.1 pts (p = 0.67) |

**The cascade does not win.** Gradient-boosted trees beat it on CANNOT phase 2 (97.9% vs 94.6%, +3.2 pts,
p = 0.0001), mostly by cutting sts_low false positives from 24.4% to 3.9%. The costs, in the same
sentence: **K1 FPR rises from 1.1% to 9.9% (n = 91)**, antonym recall falls 94.7% → 92.6% (n = 95), and
paraphrase FPR rises 1.6% → 1.9%. The hard-rule veto protected K1, and the learned model, which has
almost no cancellations to learn from in clean DEV, does not. On xNot360 no model beats honest NLI
alone significantly, and the cascade is actually 1.9 pts below it (p = 0.14). Logistic regression sits
between the cascade and GBM. Its AUC (0.941) is below the cascade's (0.958) even though its accuracy is higher, so
it ranks pairs less well overall but places the threshold better.

```sh
./venv/bin/python experiments/nli2/phase4.py --step 4.3
```

### 4.4 Per-family NLI thresholds

`experiments/nli2/phase4.py --step 4.4` → `results/4_4_per_family.json`. Families are CANNOT's
natural negation types; the generator families do not exist in clean DEV. Each family's threshold
is Youden's J on that family's clean-DEV positives against all clean-DEV negatives. On TEST it is
scored on that family's positives plus all 697 phase-2 negatives. *Balanced accuracy* is the mean
of recall and (1 − FPR), so it is not swayed by how many positives a family has.

| family | DEV positives | threshold | TEST positives | recall: family thr / global 0.975 | FPR: family / global | balanced acc: family / global | accuracy: family / global |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| verbal | 580 | 0.97749 | 290 | 94.5% / 94.5% | 26.7% / 26.8% | 83.9% / 83.8% | 79.5% / 79.4% |
| antonym | 208 | **0.91048** | 95 | **92.6% / 88.4%** | 27.6% / 26.8% | 82.5% / 80.8% | 74.9% / 75.0% |
| other | 713 | 0.75853 | 318 | 93.1% / 91.5% | 28.0% / 26.8% | 82.5% / 82.3% | 78.6% / 78.9% |

These are NLI-alone thresholds, so the FPRs include NLI's high sts_low / random_pair FPR (no gate).
Antonyms do want a lower threshold: +4.2 pts of recall for +0.8 pts of FPR. Verbal negation sits right
at the global threshold. "Other" wants a much lower threshold (0.759), but that buys only +1.6 pts
of recall for +1.2 pts of FPR. **It is a diagnostic, not a deployable fix:** applying a per-family
threshold needs the family known before the decision, and for a non-negation there is no family.
The GBM combiner (4.3) already gets 92.6% antonym recall without being told the family.

```sh
./venv/bin/python experiments/nli2/phase4.py --step 4.4
```

### 5.1 Training data for the fine-tune (generator output, fixed generator)

`experiments/nli2/phase5.py --step data` → `results/5_1_ft_data.json` (the pairs themselves are in
the gitignored `experiments/nli2/cache/ft_pairs.jsonl`, rebuilt deterministically, seed 0).

- **Bases: 1,695 CANNOT phase-1 A sentences from clean DEV.** None is in VALIDATION or TEST, so the
  generator never sees a TEST base sentence. The existing 70,451-variant refsets were **not** used,
  for two reasons. They were generated from all 3,924 CANNOT / xNot360 A sentences, 1,642 of them
  phase-2 (VALIDATION / TEST) bases and 180 xNot360 bases. And they predate fix 0.1, with up to 80%
  of their affixal variants corrupted.
- **Generator run after fix 0.1.** 30,304 variants were generated in 95 s. 42 bases got no variant
  and 3 raised a generator error. At most 8 variants per base were kept, sampled evenly across
  families so that C / H / I (the bulk of the generator's output) do not dominate. This cap keeps
  CPU training feasible.
- **Labels: the refsets rule.** A variant is a negation iff its `net_negation` parity differs from
  A's own cue parity. K1 cancellations and K2 compounds are therefore mostly label 0.
- **Random re-pairings** of two different bases were added as label 0, up to a 50 / 50 balance.

| family | label 1 (negation) | label 0 |
| --- | ---: | ---: |
| A_syntactic | 1,593 | 106 |
| B_quantifier | 136 | 20 |
| C_neg_adverb | 2,396 | 86 |
| D_affixal | 158 | 26 |
| E_antonym | 923 | 370 |
| F_implicit | 556 | 52 |
| G_prepositional | 165 | 7 |
| H_hedged | 1,810 | 148 |
| I_intensified | 1,627 | 131 |
| J_contrastive | 305 | 40 |
| K1_cancellation | 102 | 287 |
| K2_compound | 131 | 238 |
| L_scope_position | 11 | 2 |
| M_modal | 35 | 0 |
| random re-pairing | — | 8,435 |
| **total** | **9,948** | **9,948** |

**Limit, stated before training:** only 1,513 of the 9,948 negatives come from the generator. The other
8,435 are random pairs. The generator has no paraphrase family, so it can teach "a cancellation is not a
negation" and "an unrelated sentence is not a negation", but almost nothing about "a reworded
sentence is not a negation". That is exactly CANNOT's paraphrase class.

```sh
./venv/bin/python experiments/nli2/phase5.py --step data
```

### 5.2 Fine-tune, with base sentences held out

`experiments/nli2/phase5.py --step train` → `results/5_2_ft_train.json` (model saved to the
gitignored `cache/ft_model/`).

- **Setup.** Base model `cross-encoder/nli-MiniLM2-L6-H768` (82M parameters, revision `b95119ce`,
  weights sha256 `ec9c5b6a…`, checked at load), given a new 2-way head. 1 epoch, AdamW, lr 2e-5,
  batch 16, linear warm-up 10%, max length 128, seed 0. CPU, 8 threads, 1,674 s.
- **Held out by base sentence, not by pair.** 10% of bases (169) were held out with all their
  pairs: 1,246 held-out pairs and 17,100 training pairs. Random pairs whose two sentences fall on
  different sides were dropped, so no sentence crosses; the code asserts this.
- **A first attempt was killed** at step 900 of 1,069 by the tool's 30-minute background limit and
  saved nothing. The run reported here is a full restart with identical settings.

**In-distribution check (held-out bases, generator output, threshold 0.5): accuracy 95.1%, AUC 0.990.**

| family | held-out pairs | accuracy |
| --- | ---: | ---: |
| C_neg_adverb | 244 | 97.5% |
| H_hedged | 204 | 98.0% |
| I_intensified | 180 | 96.7% |
| A_syntactic | 164 | 94.5% |
| E_antonym | 136 | 86.8% |
| random re-pairing | 98 | 100.0% |
| F_implicit | 57 | 98.2% |
| K1_cancellation | 39 | 92.3% (n = 39) |
| K2_compound | 36 | 75.0% (n = 36) |
| J_contrastive | 35 | 97.1% (n = 35) |
| D_affixal | 18 | 83.3% (n = 18) |
| G_prepositional / B_quantifier / M_modal / L_scope_position | 15 / 11 / 7 / 2 | 93.3% / 100% / 100% / 100% (all n < 50) |

So the model learns the generator's task well on sentences it has not seen. Step 5.3 asks whether
that transfers to natural data.

```sh
./venv/bin/python experiments/nli2/phase5.py --step train     # ~28 min on 8 CPU threads
```

### 5.3 Fine-tuned vs off-the-shelf, alone and inside the cascade

`experiments/nli2/phase5.py --step eval` → `results/5_3_ft_eval.json`. The fine-tuned model's
P(negation) replaces NLI's P(contradiction). Every threshold is refit on clean DEV: Youden for the
model alone, and the cascade's stage-3 threshold for the cascade. The cascade is the Phase 2 adopted
configuration.

| system | phase-2 acc (95% CI) | xNot360 acc (95% CI) | all-TEST acc (95% CI) | AUC | recall | FPR paraphrase | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | SemAntoNeg FPR (n=41) | VAL acc |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off-the-shelf NLI (deberta-v3-base) alone | 82.8% (80.9–84.6) | 77.8% (73.3–82.2) | 81.8% (79.9–83.6) | 0.840 | 93.0% | 0.3% | 70.9% | 2.6% | 78.1% | 40.0% | 88.4% | 4.4% | 0.0% | 86.0% |
| off-the-shelf NLI in cascade | **94.6% (93.4–95.7)** | 75.8% (71.4–80.3) | **90.8% (89.4–92.1)** | **0.958** | 95.5% | 1.6% | 24.4% | 0.9% | 0.8% | 46.7% | **94.7%** | 1.1% | 0.0% | **93.0%** |
| fine-tuned MiniLM alone | 81.3% (79.2–83.3) | **80.8% (76.7–84.4)** | 81.2% (79.4–83.0) | 0.892 | 82.8% | 37.1% | 15.0% | 0.0% | 10.9% | **18.3%** | 63.2% | **0.0%** | 0.0% | 81.7% |
| fine-tuned MiniLM in cascade | 82.4% (80.4–84.4) | 80.3% (76.1–84.2) | 82.0% (80.2–83.8) | 0.902 | 83.7% | 40.1% | 10.2% | 0.0% | 0.0% | 20.6% | 63.2% | **0.0%** | 0.0% | 80.7% |

Paired permutation tests: fine-tuned vs off-the-shelf alone: phase 2 −1.5 pts (p = 0.30), xNot360 +3.1 pts
(p = 0.23). In the cascade: phase 2 **−12.2 pts (p = 0.0001)**, xNot360 +4.4 pts (p = 0.094).

**The fine-tune underperforms, and the way it fails says something about the generator.** Trained
only on generator output, it learned the generator's world: B is A with a negation-shaped edit,
unless the cues cancel, or B is unrelated. On natural data that becomes "any rewording is a
negation". It calls 37.1% of human-written CANNOT paraphrases negations (off-the-shelf: 0.3%). It
misses a third of antonym negations (recall 63.2%), because the generator's E_antonym pairs are
lexicon swaps on technical prose, not the short colloquial *It's natural. / It's artificial.* that
CANNOT has. What it does learn transfers where the generator is realistic:
- **Cancellation:** K1 FPR 0.0% (n = 91, generator caveat), and every SemAntoNeg pair is rejected
  (n = 41).
- **Unrelated pairs:** random-pair FPR 10.9% alone, against 78.1% for off-the-shelf NLI.
- **Conditional inverses:** xNot360 not_negation FPR 18.3%, against 40.0%. It never saw a
  conditional labelled, so this probably comes from its "same wording, small polarity edit" prior
  pointing the right way. The +3.1 pts on xNot360 is not significant (p = 0.23).

What it implies: **the generator teaches cancellation and polarity morphology well, but it does not
represent meaning-preserving rewording.** It has no paraphrase family and no natural antonymy, so a
model trained on it alone over-reads any edit as a negation. Mixing in natural paraphrase negatives,
or using generator output only for the cancellation and morphology slices, would be the next test
(section 10). The fine-tune is not adopted (VALIDATION 81.7% alone and 80.7% in the cascade, against
93.0%).

```sh
./venv/bin/python experiments/nli2/phase5.py --step eval
```
