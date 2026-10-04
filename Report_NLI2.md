# Report 2 — composite negation detector: honest baseline, rule fixes, conditionals, learned combiner

*Written for a reader who has read `Report_NLI.md`. Built up one step per commit; the step log
(section 11) is the running record, sections 1–10 summarise it.*

## 1. Summary

*(filled in when the last phase is committed)*

## 2. Dataset composition

*(filled in by Phase 1)*

## 3. Headline integrity check — does the gain survive an honest NLI threshold? (step 1.3)

`Report_NLI.md` fit the NLI threshold by Youden's J on a DEV set dominated by our generator's
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
it isolates the threshold effect. The full clean protocol, as in `Report_NLI.md`, refits the
stage-3 threshold on DEV *after* the gate (0.075, because the gate has already removed the pairs
that forced 0.975), and refits t1 (0.37). That protocol gives NLI + gate **94.5%** on phase 2
(95% CI 93.3–95.6) and 90.7% on all of TEST, **+11.7 pts over honest NLI alone** (p = 0.0001). The
final answer to the integrity check is therefore that **85% of the claimed +13.8 survives** (11.7 of
13.8).

## 4. Full ablation

*(filled in by Phases 2–4)*

## 5. Per-family results, best configuration

*(filled in by Phase 4)*

## 6. Error analysis

*(filled in at the end)*

## 7. What did not work

*(every change that failed to improve things, with its numbers; added to as we go)*

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
`Report_NLI.md` (`composite.fit_gate`), with step 1.3's NLI threshold:

| | similarity | t1 | DEV J of gate → NLI | DEV positives the gate keeps | stage-3 threshold (DEV refit) |
| --- | --- | ---: | ---: | ---: | ---: |
| Report_NLI.md (old DEV) | cos | 0.36 | 0.8835 | 99.6% | 0.00125 |
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
xNot360 baseline here (75.8%) is higher than `Report_NLI.md`'s 66.1% because of the clean thresholds
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
labels the inverse as not a negation. The VALIDATION drop comes from 2 such CANNOT pairs (the rule
fires on 2 VALIDATION pairs).

**Why not adopted:** the pre-stated rule picks by VALIDATION accuracy, which falls; the variants
were designed after seeing TEST; and no fitting set has enough conditionals to check them (36 in
clean DEV, 4 in VALIDATION). Variant B is the best-supported next step, if it can be confirmed on a
conditional set that is not TEST (section 10).

```sh
./venv/bin/python experiments/nli2/phase3.py --step 3.3
```
