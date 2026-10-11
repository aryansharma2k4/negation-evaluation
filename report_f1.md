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

### 0.2 Conditional detection fix: *as if*, *even if*, *only if*

`experiments/nli2/conditionals.py` (the shared detector, fixed in place so every caller gets it).
Recount: `experiments/f1/phase0.py --step 0.2` → `experiments/f1/results/0_2_conditional_fix.json`.

- ***as if*** is a comparison, not a condition. When the token before *if* is *as*, the clause is no
  longer taken for a conditional.
- ***even if*** and ***only if*** are now their own markers, not plain *if*. When A and B use
  different connectives and one of them is *even if* or *only if*, the polarity rule returns no
  decision. The connective changed, not the antecedent's polarity, so the combiner decides. A pair
  that keeps the same connective on both sides is still read by the polarity rule.
- The detector's cache file moves to `conditionals_v2.pkl`. The old cache is left in place, so the
  recount compares against the pre-fix output directly. **Rerunning nli2's Phase 3 / 4 scripts after
  this commit gives slightly different conditional features**; `Report_NLI2.md`'s numbers reproduce at
  its own commit hashes.

Unit tests (`experiments/nli2/test_nli2.py`, 25 pass, 7 new): two *as if* sentences are not
conditionals; *even if* / *only if* are recognised in preposed and postposed position, with plain *if*
unchanged; *even if → only if* gives no decision; *even if → even if* with the antecedent flipped is
still `not_negation`.

How many pairs change (no labels read):

| split | n | coverage class changes | marker-only changes | the changes |
| --- | ---: | ---: | ---: | --- |
| clean DEV | 3,402 | 0 | 2 | two *even if* pairs (both sides), previously *if* |
| VALIDATION | 601 | **1** | 1 | *… looks as if the phone isn't all lightweight cheap plastic* / *… is all …*: both → neither; *… even if all formats do not meet the chart rules* / *… only if all formats meet …*: if/if → even if/only if |
| TEST | 1,760 | 0 | 1 | one xNot360 *even if* pair (both sides) |

Coverage per split after the fix: clean DEV 36 both-conditional pairs (unchanged), VALIDATION 3 (was 4),
TEST unchanged (CANNOT 14, xNot360 134). These are exactly the two VALIDATION pairs `Report_NLI2.md`
step 3.3 traced its VALIDATION drop to. The fix touches 5 pairs in 5,763 and cannot move any headline
number by more than a pair or two. Its value is that the conditional features and rules now mean what
they say.

```sh
./venv/bin/python -m pytest experiments/nli2 -q
./venv/bin/python experiments/f1/phase0.py --step 0.2
```

### 0.3 The Phase 0 baseline

`experiments/f1/phase0.py --step 0.3` → `experiments/f1/results/0_3_baseline.json`. This is
nli2's final GBM, retrained with the fixed conditional features (0.2), with the cancellation override
(0.1) in front. **Every number in later phases is measured against this row.**

| configuration | phase-2 acc (95% CI) | all-TEST acc | AUC | precision | recall | F1 | FPR paraphrase | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | VAL acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Phase 0 baseline** | **97.7% (96.9–98.5)** | **93.9%** | 0.978 | 93.0% | 94.9% | 94.0% | 1.9% | 3.9% | 0.0% | 0.0% | 28.9% | 92.6% | 2.2% | 97.8% |

xNot360 accuracy: 78.9% (n = 360; 95% CI 74.7–83.3). SemAntoNeg FPR: 0.0% (n = 41). Out-of-fold clean-DEV
accuracy: 96.6%. GBM: depth 3, lr 0.1, threshold 0.428 (Youden on out-of-fold predictions).

It is identical to the 0.1 row at every digit. The model and threshold are bit-identical, and that is
expected. The two clean-DEV pairs that 0.2 touched changed marker only (*if* → *even if* on both
sides), so no training feature moved. The one VALIDATION pair whose features did change (*as if*) was
classified the same either way.

```sh
./venv/bin/python experiments/f1/phase0.py --step 0.3
```

### 1.1 Affirmation generator (`N_affirmation`)

**The brief's premise was wrong, and the fix is smaller for it.** The brief says the generator
"cannot take a sentence that is already negative and produce its positive form". It can.
`negation/generators/affirm.py` (634 lines, registered since stage 2) already removes one cue per
family: do-support collapse, aux / copula negator, quantifier flip, negative adverb, affixal strip,
and implicit trigger. Writing a second one would have duplicated it. So I ran the brief's table through the
existing code first. It handled most rows, and the probe exposed **five defects**. Four of them
are things the brief's warnings anticipate:

| defect found | example (before) | fix |
| --- | --- | --- |
| contracted modals lose their stem | *can't access* → "**ca** access", *won't respond* → "**wo** respond" | `affirm._delete_negator`: restore *can / will / shall* when *n't* is fused to *ca / wo / sha* (shared by every removal path, including `rescope`) |
| **never → always** (warning 1, live in the code) | *He never calls* → "He always calls" | `lexicons.NEG_ADVERB_AFFIRMATIONS["never"] = ("", "sometimes")`; plain removal is primary, *sometimes* secondary; a regression test pins "always" as wrong |
| litotes come out as negations, unflagged | *It is not uncommon …* → "It is uncommon …", "It is not common …" | litotes rule: *not un-X* → *X*, tagged `approximate: true`, `polarity_flip: false`; the two one-cue removals are kept as secondary readings (they are negations of the litotes) |
| relative clauses get affirmed | *A process that is not running is blocked* → "… that is running is blocked" | only cues inside `clause_scope(ROOT)` are removed |
| no *avoid*, and *deny* / *refuse* only promote the complement | *He denies taking the money* → "He takes the money" | curated swap lexicon is primary (*deny → admit*, *refuse → agree*, *fail → manage*, *neglect / forget → remember*, *cease → continue*); promotion kept as secondary; *avoid* promotes its gerund |

What was built (`negation/generators/n_affirmation.py`, 290 lines, imports
`negation.nlp_core.clause_scope`, no second scope implementation):

- `affirm(sentence)` returns `Affirmation` records: `text`, `kind` (cue type), `cue`, `primary`,
  `approximate`, `polarity_flip`, `rule`, `family = "N_affirmation"`.
- It runs the registered affirm generators and their filters. It keeps an output only if the removed
  cue is in the main clause's scope, and only if an affixal strip passes the step-0.1 eligibility rules:
  predicative adjective or whitelisted noun, never in a compound or hyphenated term.
- On top of those it adds the lexicon swaps, *avoid*, and litotes.
- `polarity_flip = False` marks outputs that are rewordings, not negations (litotes). Phase 3 uses
  only `polarity_flip = True` outputs as negations of T.
- **Affixal exception blocklist** (`negation/affixes.AFFIXAL_BLOCKLIST`) is enforced inside
  `is_negative_derivative`, the one gate both directions pass through. That means D_affixal also
  stops producing *valuable → invaluable*. Seeded with *invaluable, inflammable, disgruntled*; the
  clean-DEV scan is step 1.2.

**Choice: how to restore inflection.** Reverse do-support (*did not run* → *ran*) rebuilds a surface
verb from its lemma plus the auxiliary's Tense / Person / Number. Every inflected verb already in
clean DEV is a gold example of that operation, so each candidate was given (spaCy lemma, Penn tag)
and asked to reproduce the written form. Data: 5,788 tokens, 1,657 distinct (lemma, tag, form) types,
from 6,151 sentences (`experiments/f1/inflection_bench.py` → `results/1_1_inflection.json`).

| inflector | token accuracy | type accuracy | VBZ / VBD / VBN / VBG (tokens) | typical errors |
| --- | ---: | ---: | --- | --- |
| hand-rolled rules (`nlp_core.inflect_verb`, what `affirm.py` used) | 93.5% | 90.7% | 98.1 / 92.3 / 91.3 / 94.2 | *considerred*, *developped*, *beared* |
| **lemminflect** | 97.9% | **97.8%** | 98.1 / 98.7 / 98.3 / 96.9 | *rid → ridding* (spaCy lemma error) |
| pyinflect | **98.4%** | 96.5% | 98.6 / 98.2 / 98.1 / 98.6 | ***work → wrought*** |

**The selection rule was revised on DEV, after the first run, and this is the record of it.** As
first stated, the rule was token accuracy, and it picked pyinflect. The probe then produced *The
kernel did not work* → "The kernel **wrought**". *work* is a common verb in OS text, and pyinflect's
lexicon has only *wrought* for it. pyinflect's 0.5-pt token lead also comes mostly from one SNLI
word: spaCy lemmatises *riding* as *rid* (53 tokens), and pyinflect happens to map *rid* back to
*riding*. That is an upstream error cancelling out, not better inflection. Counting each distinct
verb form once removes that weighting, and on that count lemminflect leads by 1.3 pts (37 vs 58
type errors). **Adopted: lemminflect, with the rules as fallback** (`nlp_core.inflect_lexical`). It is
used only on the affirmation path; the negation generators keep the rules, because stage 1 output is
pinned by a regression test. The selection was made on DEV only, and no TEST was read.

Other choices:

- **Antonym lookup:** WordNet for affixal (the existing antonymy gate plus the blocklist), and a
  curated lexicon for implicit negatives. WordNet antonyms of *refuse* / *deny* include
  *accept* / *admit*, but also senses that do not fit the complement (*deny* → *allow*). Curated pairs
  keep the complement's syntax.
- **One or all affirmations per sentence:** all valid ones, each labelled `primary` or secondary, so
  callers choose.
- The method stays rule-based; no LLM is used anywhere.

Tests: `tests/test_affirmation.py`, 29 tests, one per table row:

- *is / were / does / did / do not*, *cannot*, *can't*, *will not*, *won't*, *has not*;
- *no → some* (and **not** *all*), affixal *un- / in-*, *-less → -ful*, *lack → have*;
- the four implicit swaps, *never* (with *always* pinned as wrong), the three blocklisted words;
- the term-position rule, litotes, the main-clause restriction, *did not work → worked* (not
  *wrought*), and an affirmative input.

Full suite: 414 pass (385 before).

```sh
./venv/bin/python -m pytest tests -q
./venv/bin/python experiments/f1/inflection_bench.py
```
