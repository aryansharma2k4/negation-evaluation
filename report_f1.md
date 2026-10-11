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

What was built (`negation/generators/n_affirmation.py`, 263 lines at this commit, imports
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

### 1.2 Affirmation generator measured on clean DEV

`experiments/f1/phase1.py --step 1.2` → `experiments/f1/results/1_2_affirmation.json`. Sentences
are every distinct A and B of clean DEV (6,151). A sentence counts as **negative** when it has an
odd number of cues by the generator's own classifier (`classify_input`): **960 sentences**. No TEST
data is read in Phase 1.

**Two more defects surfaced while measuring, both fixed in this commit:**

1. **A pre-existing filter bug blocked whole base sentences, for every family.** The shared
   blacklist (`filters.apply_blacklist`) rejected a variant whenever its text matched a pattern. That
   included patterns already in the base. *"I didn't think **that the** instructions …"* matched
   `doubled_determiner` (*that* is a complementiser here), so neither the affirmer nor any negation
   family could produce a variant of it. The fix counts only matches the edit added
   (`blacklist_hit(variant, base)`). **58 of the 6,151 clean-DEV sentences (0.9%) carry such a
   pattern** and previously got no variant at all. The stage-1 regression pin (448 variants on the
   20 frozen bases) is unchanged, because none of those bases is affected.
2. **The litotes rule over-fired.** *Unfortunately it will not recharge …* was turned into
   *Fortunately it will recharge …*. The affixal word is now required to be what the negator
   negates (the negator's head, or a dependent of that head), and adverbs are excluded.

Both have regression tests (`tests/test_affirmation.py`, now 31; full suite 416 pass).

**Coverage.**

| | count | share of the 960 |
| --- | ---: | ---: |
| negative sentences (odd cue parity) | 960 | 100% |
| **at least one affirmation** | **659** | **68.6%** |
| none | 301 | 31.4% |

**Why 301 got none** (named buckets, assigned by `n_affirmation.failure_reason`, a stated rule):

| failure bucket | count | example |
| --- | ---: | --- |
| every cue is outside the main clause (relative / subordinate / coordinated clause) | 214 | *Construction featured a mansion, …, but not kennels.* |
| affixal word in attributive / term position (step-0.1 rule) or no WordNet-attested stem | 38 | *… run the risk of developing an incomplete solution.* |
| *not* with no auxiliary to hang on (*not* + NP / adverb / fronted) | 21 | *Not often do we see her lose her cool like that.* |
| syntactic negator in a parse shape the removal rules do not cover | 13 | *This product is clearly not ready for prime time, I don't care how cool it looks, …* |
| *without* (G_prepositional): no affirmation rule exists | 9 | *Without a license, a practitioner is still permitted to bill …* |
| implicit trigger with no complement to promote / not in the swap lexicon | 5 | *The bank failed the stress test …* |
| negative quantifier in a form with no flip | 1 | *The cashier had no care what so ever …* |

The biggest bucket, 71% of failures, is **by design**. The brief restricts affirmation to cues in
the main clause's scope, and 214 of these sentences carry their only cue in a coordinated,
relative or subordinate clause. Coordination is most of it (*… but not kennels*). Extending scope to
coordinated main clauses (`conj` of ROOT) is the obvious next step. It was not done because, for
those sentences, "the positive form" means removing several cues, which breaks the
one-cue-per-variant contract of the existing generators.

**Litotes** (even parity, so not in the 960): **38 sentences** carry the *not un-X* shape. Every one
gets the approximate reading, e.g. *I'm not imperfect.* → *I'm perfect.*, *That's not incorrect.*
→ *That's correct.*. All are tagged `approximate: true, polarity_flip: false`.

**Exception blocklist** (`negation/affixes.AFFIXAL_BLOCKLIST`). I scanned all 75 affixal (word, stem)
pairs the stripper produces over clean DEV, listed with counts in the JSON, and checked them by hand
for lexicalised words that are not the negation of their stem:

| word | why it is not *not* + stem | source |
| --- | --- | --- |
| invaluable | means "very valuable" | brief |
| inflammable | means "flammable" | brief |
| disgruntled | no attested *gruntled* | brief |
| disorder | a condition (*a mental disorder*), not *not order* | clean-DEV scan (5 uses) |
| immeasurably | an intensifier (*immeasurably better*) | clean-DEV scan (2) |
| disarmament | a process, not the absence of armament | clean-DEV scan (1) |
| disappearance | an event; *appearance* also means "looks" | clean-DEV scan (1) |
| impersonal | "lacking warmth", not *not personal* | clean-DEV scan (1) |

Borderline pairs left in: *independent / dependent*, *infinite / finite*, *illicit / licit*. All three are
logically negations even where the stem is rare. The list is enforced in both directions, so
D_affixal no longer produces *valuable → invaluable* either.

**Spot-check: 25 of 30 correct (83%).** Thirty primary affirmations were sampled at random (seed 0), one
per sentence. Verdicts are in `experiments/f1/data/affirmation_spotcheck.json`, each with a reason,
judged against criteria stated before sampling:

1. grammatical, with no leftover fragment of the cue;
2. the removed cue is the main clause's negation, and nothing else changed;
3. the output is the logical negation of the input;
4. secondary / approximate tags sit on the readings that need them.

**The verdicts were written by the same model that built the generator.** This is a machine judgement
of machine output, and a spot-check, not a gold label. The 5 failures:

| input → output | why it is wrong |
| --- | --- |
| *It is not known anywhere else.* → *It is known anywhere else.* | negative-polarity *anywhere* left in place (needs *somewhere*) |
| *So anyone near you won't hear part of your conversation.* → *So anyone near you will hear …* | *anyone* under negation means *no one*; the affirmation is *someone will hear*, not *anyone* |
| *I didn't realize how wonderful the short really is until the last two scenes.* → *I realized … until the last two scenes.* | *not … until* is a construction; *until* has to become *only in* |
| *Now this dish wasn't quite flavourful.* → *Now this dish was quite flavourful.* | idiom *not quite X* (almost X); dropping *not* shifts *quite* to "fairly" |
| *Chiya Khere-5 (CK-5) well failed to be completed in Q3 of 2015.* → *… well managed to be completed …* | *manage* presupposes an agent; unnatural with an inanimate passive subject |

**Three of the five are negative-polarity items** (*any-*, *until*). A small NPI table (*any → some*,
*anyone → someone*, *anywhere → somewhere*, *ever → at some point*, *yet → already*) is the next fix. It
was not added after the spot-check, so that the 25 / 30 describes the code as committed.

**Round trip** (`affirm(negate(S))` recovers S). I took 600 affirmative clean-DEV sentences (0 cues; seed 0),
negated each with every stage-1 family the affirmer can undo (A, B, C, D, F; `net_negation = 1`), then
affirmed the result. Only `polarity_flip` outputs count, and the one closest to S in word edits is
kept. One base raised the generator's depth assertion and was skipped: *disappeared* is not read as a
cue but *disappear* is, a classifier quirk.

| family | pairs | exact S | one word off | differs | no affirmation |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_syntactic | 536 | 522 (97.4%) | 8 | 2 | 4 |
| B_quantifier | 38 (n < 50) | 20 (52.6%) | 10 | 6 | 2 |
| C_neg_adverb | 3,202 | 3,188 (99.6%) | 0 | 0 | 14 |
| D_affixal | 28 (n < 50) | 22 (78.6%) | 0 | 0 | 6 |
| F_implicit | 1,178 | 1,021 (86.7%) | 57 | 24 | 76 |
| **all** | **4,982** | | | | **102** |

**Of the 4,880 negations that got an affirmation, 97.8% recover S exactly and 99.3% recover it to
within one word.** What the misses are:

- **B_quantifier is not supposed to round-trip.** *All human beings …* → *None of the human beings
  …* → ***Some** of the human beings …*. The affirmation of *none* is *some* (warning 2), so *all*
  cannot come back. The B_quantifier rate measures the quantifier logic, not an error.
- **Several misses start as junk on the negation side.** On verbless SNLI captions the negator
  writes *Black and white dog **do not run** through water* and *dog **deny openning***. On doubled
  phrases it writes *None of the of the graduates*. The affirmer faithfully undoes the junk.
- **F_implicit's primary reading is the lexicon swap,** so *X denies producing* comes back as *X
  admits producing* when the promotion reading (*X produces*) is not available. 9 of the 40 listed
  non-exact examples are this.

```sh
./venv/bin/python -m pytest tests/test_affirmation.py -q
./venv/bin/python experiments/f1/phase1.py --step 1.2
```

### 1.3 Affirmation inventory

`experiments/f1/phase1.py --step 1.3` → `experiments/f1/results/1_3_inventory.json`. The table counts
the 960 negative clean-DEV sentences by the type of their first main-clause cue (first cue
overall, if none is in the main clause). Spot-check column: verdicts from step 1.2 (30 items in all,
machine-judged), so most rows have a tiny or zero sample. **Every spot-check cell is n < 50.**

| cue type | count in clean DEV | affirmation produced | spot-check correct |
| --- | ---: | ---: | ---: |
| be + not (copula) | 238 | 200 (84%) | 9 / 10 |
| affixal prefix (un- / in- / non- / dis- …) | 171 | 90 (53%) | 3 / 3 |
| implicit (refuse / deny / fail / lack / unable …) | 112 | 93 (83%) | 4 / 5 |
| negative quantifier (no / none / nobody / nothing / neither / nor) | 97 | 72 (74%) | 2 / 2 |
| *not*, other position (not + NP / adverb / fronted) | 58 | 1 (2%) | — (none sampled) |
| be + not (aux / passive / progressive) | 47 | 38 (81%) | 1 / 2 |
| did not V | 39 | 32 (82%) | 0 / 1 |
| do not V | 37 | 24 (65%) | — |
| never | 33 | 24 (73%) | 3 / 3 |
| does not V | 26 | 24 (92%) | 3 / 3 |
| has / have / had not (perfect) | 23 | 15 (65%) | — |
| other modal + not (would / should / could / must / may …) | 22 | 16 (73%) | — |
| can + not / can't / cannot | 18 | 12 (67%) | — |
| without | 12 | 0 (0%) | — |
| will not / won't | 10 | 9 (90%) | 0 / 1 |
| rarely / seldom / hardly / barely | 9 | 6 (67%) | — |
| -less suffix | 4 | 2 (50%) | — |
| other (*absence*, curly-quote *n’t*, *nt*) | 4 | 1 (25%) | — |
| **total** | **960** | **659 (68.6%)** | **25 / 30** |

The *did not V* failure in the sample is the *not … until* NPI case and the *will not* failure is
*anyone* (step 1.2), so neither row's 0 / 1 is a morphology error. Where coverage is weakest, the
cue is outside what a one-cue removal can express:

- *not* in another position: 2% (*Not often do we …*);
- *without*: 0%, because no rule exists;
- affixal: 53%, because most remaining affixal cues are attributive (*an incomplete solution*) and the
  step-0.1 rule refuses them on purpose.

The failure buckets and the blocklist are in step 1.2.

```sh
./venv/bin/python experiments/f1/phase1.py --step 1.3
```

### 2.1 Five NLI cross-encoders scored, revision-pinned, cached

`experiments/f1/nli_models.py` scores every pair of clean DEV (3,402), VALIDATION (601), TEST (1,760)
and K1 (91), 5,854 rows in all. Premise is A, hypothesis is B, as in every earlier number.
Probabilities are mapped by each model's own `id2label`, because label order differs between models.
Caches: `experiments/f1/cache/nli_<key>.pkl` (gitignored). The reference model's probabilities are
the exact values the baseline features already use.

| model | size | revision | ms / pair (isolated) | minutes for 5,854 rows | kept for 2.2 |
| --- | ---: | --- | ---: | ---: | --- |
| `cross-encoder/nli-deberta-v3-base` (reference) | 184M | `6c749ce3` | 45.2 | 4.4 | yes |
| `cross-encoder/nli-roberta-base` | 125M | `1be05674` | 36.1 | 3.5 | yes |
| `cross-encoder/nli-distilroberta-base` | 82M | `b14d131f` | 18.5 | 1.8 | yes |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | 184M | `6f5cf0a2` | 48.0 | 4.7 | yes |
| `facebook/bart-large-mnli` | 407M | `d7645e12` | 145.7 | 14.2 | yes |

Timing setup: Ryzen 7 7730U, 8 torch threads, batch 16, max length 256, 64 clean-DEV pairs after one
warm-up batch, nothing else running (`results/2_1_timing.json`). bart-large is projected at 14.2
minutes, under the brief's 40-minute limit, so **`deberta-v3-small` was not substituted**.

What went wrong on the way, all fixed:

1. **Download bandwidth was the binding constraint.** The Hugging Face CDN gave ~65 KB/s per
   connection on this machine, so bart-large over one connection was ~7 hours.
   `experiments/f1/fetch_models.py` downloads each pinned revision with many parallel range
   requests, persists chunks so a killed run resumes (one did), and **verifies every weight file
   against the hub's LFS sha256** for that exact revision.
2. **fp16 checkpoint slowed CPU inference 5×.** The MoritzLaurer checkpoint is stored in fp16, and
   transformers 5 keeps the stored dtype, so it ran at 238 ms/pair. Every model is now loaded in
   float32 (48.0 ms/pair). Its scores were deleted and recomputed in fp32. The other checkpoints were
   already fp32, so their scores are unchanged.
3. **Extensions to the list were considered and not run.** I planned two larger adversarial-NLI models
   (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, 435M; `ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli`,
   355M) and pinned their revisions in `MODELS`. They were dropped: together they would have added
   about 3.2 GB of downloads, an estimated 1.5 hours at the measured rate. This is listed under
   "approaches considered and not tried".

```sh
./venv/bin/python experiments/f1/fetch_models.py roberta_base distilroberta ml_deberta_base_anli bart_large
./venv/bin/python experiments/f1/nli_models.py --score roberta_base distilroberta ml_deberta_base_anli bart_large
./venv/bin/python experiments/f1/nli_models.py --time deberta_base roberta_base distilroberta ml_deberta_base_anli bart_large
```

### 2.2 Each NLI model standalone

`experiments/f1/ensemble.py --step 2.2` → `results/2_2_standalone.json`. Each model gets its own Youden
threshold on clean DEV for P(contradiction) and is then scored alone, with no gate, features or
override. TEST, n = 1,760; antonym n = 95; K1 n = 91 (machine-generated); SemAntoNeg n = 41.

| model | threshold | phase-2 acc (95% CI) | all-TEST acc | xNot360 acc (n=360) | AUC | precision | recall | F1 | FPR paraphrase | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall | K1 FPR | VAL acc |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deberta-v3-base (reference) | 0.975 | 82.8% (80.9–84.6) | 81.8% | 77.8% | 0.840 | 76.0% | 93.0% | 83.7% | 0.3% | 70.9% | 2.6% | 78.1% | 40.0% | 88.4% | 4.4% | 86.0% |
| roberta-base | 0.472 | 81.1% (79.1–83.1) | 81.3% | 81.9% | 0.843 | 75.3% | 93.3% | 83.4% | 1.3% | 74.0% | 6.0% | 84.9% | 35.0% | 84.2% | 7.7% | 84.7% |
| distilroberta-base | 0.314 | 78.1% (75.9–80.1) | 74.0% | 58.1% | 0.796 | 68.2% | 90.4% | 77.7% | 2.9% | 78.7% | 6.0% | 86.6% | 82.2% | 80.0% | 19.8% | 81.0% |
| **DeBERTa-v3-base mnli-fever-anli** | 0.931 | **84.8% (82.9–86.6)** | **85.2%** | **86.7%** | **0.894** | 81.2% | 91.7% | **86.1%** | 0.3% | 66.9% | 5.1% | **45.4%** | **22.2%** | 85.3% | **1.1%** | **88.8%** |
| bart-large-mnli | 0.689 | 82.9% (81.1–84.8) | 81.8% | 77.2% | 0.863 | 75.5% | 94.2% | 83.8% | 0.3% | 74.8% | 4.3% | 73.1% | 45.0% | 89.5% | 6.6% | 85.7% |

**The model trained on adversarial NLI data (ANLI, FEVER) is the best single model on every
summary column.** Against the reference:

- VALIDATION +2.8 pts; phase 2 +2.0 pts; all TEST +3.4 pts;
- xNot360 +8.9 pts, with conditional-inverse FPR 40.0% → 22.2%;
- K1 FPR 4.4% → 1.1% (machine-generated, n = 91).

The cost, in the same sentence: antonym recall falls 88.4% → 85.3% (n = 95, 3 pairs) and SemAntoNeg FPR
rises 0.0% → 7.4% (n = 27 on TEST, 2 pairs). bart-large, the biggest and slowest model, is level with the
reference, so 3.2× the compute buys nothing standalone. distilroberta is the weakest by a wide margin.
It calls 82.2% of xNot360 conditional inverses negations. Every model shares the same structural
failure without the gate: 67–79% of STS-low pairs are called negations.

```sh
./venv/bin/python experiments/f1/ensemble.py --step 2.2
```

### 2.3 Model disagreement

`experiments/f1/ensemble.py --step 2.3` → `results/2_3_disagreement.json`. A cell is the share of pairs
where two models' binary decisions differ, each model at its own clean-DEV threshold. TEST, n = 1,760:

| | deberta-base | roberta-base | distilroberta | DeBERTa-anli | bart-large |
| --- | ---: | ---: | ---: | ---: | ---: |
| deberta-base | — | 6.6% | 14.1% | 8.4% | 7.0% |
| roberta-base | 6.6% | — | 11.3% | 10.4% | 7.2% |
| distilroberta | 14.1% | 11.3% | — | 17.2% | 13.9% |
| DeBERTa-anli | 8.4% | 10.4% | 17.2% | — | 8.9% |
| bart-large | 7.0% | 7.2% | 13.9% | 8.9% | — |

Pearson correlation of P(contradiction) on TEST: **mean over pairs 0.84**. It ranges from 0.71
(distilroberta / DeBERTa-anli) to 0.90 (deberta-base / roberta-base 0.902, DeBERTa-anli / bart 0.900,
deberta-base / DeBERTa-anli 0.899). On clean DEV the mean is 0.87 and disagreement is 6.3–14.0%.

**No pair agrees on more than 95% of pairs**, on TEST or on clean DEV, so none is redundant by the
brief's test. The closest pair, deberta-base / roberta-base, differs on 6.6% of TEST pairs (116 pairs).
That is still room for an ensemble to gain, though the models are clearly correlated (r ≈ 0.9 for most pairs).

**Correction to the brief.** The brief asks for this matrix on TEST *and* says to choose the model set
on this evidence, which would be choosing on TEST. The TEST matrix is reported above as asked. The
pruning rule was stated in `ensemble.py` before the run and applied to clean DEV and VALIDATION only:

1. drop models whose standalone VALIDATION accuracy is > 3 pts below the reference's;
2. then drop the weaker of any pair agreeing on > 95% of clean-DEV pairs.

It drops **distilroberta** (VALIDATION 81.0% vs 86.0%); no pair triggers step 2. Pruned set: deberta-base,
roberta-base, DeBERTa-anli, bart-large (4 models). Step 2.4 evaluates every method on both the full and the
pruned set.

```sh
./venv/bin/python experiments/f1/ensemble.py --step 2.3
```

### 2.4 Ensembling methods

`experiments/f1/ensemble.py --step 2.4` → `results/2_4_ensembles.json`. Every configuration keeps the
Phase 0 cancellation override in front. Methods, each on the full set (5 models) and the pruned
set (4 models; step 2.3):

- **(a) mean P(contradiction)**, one Youden threshold (the brief's baseline method);
  **(a') median**, a robustness variant.
- **(b) GBM on the 17 features plus each extra model's three probabilities** (17 + 3 × 4 = 29 features for
  5 models; 26 for 4). Retrained exactly as nli2 step 4.2: grouped 5-fold CV over connected components,
  grid by CV AUC, Youden on out-of-fold predictions, refit on all clean DEV.
- **(c) calibrated mean.** Each model's P(contradiction) is Platt-calibrated on clean DEV (logistic on
  its logit), then averaged. This is the first calibration of any probability in this project.
- **(d) logistic stacker** over all models' probabilities alone (no other features), with threshold from
  out-of-fold predictions.
- **(e) GBM on the same 17 features**, with the three NLI features replaced by their mean across models
  (no extra features).

**Rule, stated in the script before the run:** highest VALIDATION accuracy among the Phase 0 baseline
and every (method, set); tie → higher out-of-fold clean-DEV accuracy.

| configuration | phase-2 acc (95% CI) | all-TEST acc | AUC | precision | recall | F1 | FPR paraphrase | sts_low | snli_neutral | random_pair | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | VAL acc | out-of-fold DEV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 0 baseline | 97.7% (96.9–98.5) | 93.9% | 0.978 | 93.0% | 94.9% | 94.0% | 1.9% | 3.9% | 0.0% | 0.0% | 28.9% | 92.6% | 2.2% | 97.8% | 96.6% |
| (a) mean, 5 models | 82.8% (80.8–84.6) | 83.9% | 0.874 | 80.3% | 89.9% | 84.8% | 0.0% | 64.6% | 0.9% | 62.2% | 21.1% | 77.9% | 0.0% | 84.7% | 84.5%* |
| (a') median, 5 models | 82.6% (80.6–84.4) | 82.0% | 0.865 | 75.9% | 93.9% | 84.0% | 0.0% | 72.4% | 2.6% | 82.3% | 38.9% | 89.5% | 0.0% | 86.4% | 83.2%* |
| (c) calibrated mean, 5 models | 82.8% (80.8–84.6) | 84.2% | 0.871 | 80.3% | 90.7% | 85.2% | 0.0% | 63.8% | 0.9% | 69.8% | 17.8% | 79.0% | 0.0% | 84.5% | 84.2%* |
| (d) logistic stacker, 5 models | 85.1% (83.3–86.9) | 84.3% | 0.897 | 80.1% | 91.5% | 85.4% | 0.3% | 62.2% | 2.6% | 48.7% | 33.3% | 90.5% | 0.0% | 88.7% | 88.0% |
| **(b) GBM 29 features, 5 models ← VALIDATION pick** | **98.6% (97.9–99.1)** | 92.6% | 0.982 | 90.1% | 95.8% | 92.9% | 0.7% | 0.8% | 0.0% | 0.0% | **50.0%** | 94.7% | 2.2% | **98.5%** | 96.9% |
| (e) GBM 17, averaged probs, 5 models | 98.1% (97.4–98.8) | 90.6% | 0.984 | 86.0% | 97.1% | 91.2% | 1.6% | 7.1% | 0.0% | 0.0% | 69.4% | 95.8% | 1.1% | 97.7% | 96.2% |
| (a) mean, 4 models | 83.5% (81.6–85.4) | 84.2% | 0.887 | 80.9% | 89.5% | 85.0% | 0.0% | 62.2% | 0.9% | 54.6% | 22.8% | 80.0% | 0.0% | 85.7% | 86.0%* |
| (a') median, 4 models | 83.7% (81.8–85.6) | 84.9% | 0.874 | 81.9% | 89.9% | 85.7% | 0.0% | 61.4% | 0.9% | 57.1% | 16.1% | 74.7% | 0.0% | 85.4% | 85.2%* |
| (c) calibrated mean, 4 models | 83.3% (81.4–85.1) | 84.7% | 0.880 | 81.5% | 90.0% | 85.5% | 0.0% | 63.0% | 0.9% | 59.7% | 16.1% | 76.8% | 0.0% | 85.5% | 85.3%* |
| (d) logistic stacker, 4 models | 84.9% (83.1–86.7) | 84.4% | 0.903 | 79.4% | 93.0% | 85.7% | 0.3% | 68.5% | 1.7% | 54.6% | 32.2% | 92.6% | 0.0% | 88.7% | 87.5% |
| (b) GBM 26 features, 4 models | 98.4% (97.6–99.0) | 94.5% | 0.983 | 93.6% | 95.7% | 94.6% | 1.0% | 0.8% | 0.0% | 0.0% | 30.0% | 92.6% | 2.2% | 98.2% | 97.0% |
| (e) GBM 17, averaged probs, 4 models | 98.1% (97.3–98.7) | 94.3% | 0.984 | 93.0% | 95.9% | 94.4% | 1.3% | 3.9% | 0.0% | 0.0% | 30.6% | 96.8% | 1.1% | 98.0% | 96.6% |

\* threshold-only methods: clean-DEV accuracy at the DEV-fit threshold (one fitted parameter), not out-of-fold.

**The VALIDATION rule selects (b) with all 5 models** (98.5%; next, (b) on 4 models at 98.2%). Its gains
over the Phase 0 baseline, paired permutation, uncorrected:

- CANNOT phase 2: **+0.9 pts (p = 0.018)**, mostly sts_low FPR 3.9% → 0.8% and paraphrase FPR 1.9% → 0.7%.
- **The cost, in the same sentence:** xNot360 −9.4 pts (78.9% → 69.4%, p = 0.0001), because conditional-inverse
  FPR rises 28.9% → 50.0%. All of TEST therefore falls **−1.25 pts (93.9% → 92.6%, p = 0.012)**.

The gain is in-distribution and the loss is out-of-distribution, the same pattern as `Report_NLI2.md`'s
main caveat, now larger.

**Ensembling NLI scores without the other features is far worse.** Methods (a), (a'), (c) and (d) land at
82–85% on phase 2. Calibration (c) changes almost nothing (84.5% VALIDATION vs 84.7% uncalibrated), because
Youden's threshold is invariant to monotone rescaling of one model, and averaging calibrated
probabilities barely reorders pairs. Every one of these methods is at least 12.6 pts below the
baseline on phase 2 (p = 0.0001). The GBM's other features carry the decision.

Not reported as a selection, because TEST must not choose: on TEST the pruned 4-model (b) is better than the
5-model one, at all-TEST 94.5% (+0.7 pts, p = 0.042) and xNot360 79.7%. VALIDATION (98.2% vs 98.5%) could
not see that, because VALIDATION contains no xNot360-like data.

```sh
./venv/bin/python experiments/f1/ensemble.py --step 2.4
```

### 2.5 Overfitting check — the ensemble is flagged; the 17-feature model is carried forward

`experiments/f1/ensemble.py --step 2.5` → `results/2_5_overfit.json`.

| configuration | features | out-of-fold DEV | VALIDATION | TEST (all) | phase 2 | out-of-fold DEV − TEST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 0 baseline | 17 | 96.6% | 97.8% | 93.9% | 97.7% | +2.7 pts |
| **(b) GBM, 5 models (VALIDATION pick)** | 29 | 96.9% | 98.5% | 92.6% | 98.6% | **+4.3 pts** |
| (e) GBM 17, averaged, 5 models | 17 | 96.2% | 97.7% | 90.6% | 98.1% | +5.6 pts |
| (b) GBM, 4 models | 26 | 97.0% | 98.2% | 94.5% | 98.4% | +2.4 pts |
| (e) GBM 17, averaged, 4 models | 17 | 96.6% | 98.0% | 94.3% | 98.1% | +2.3 pts |

**Said plainly: the VALIDATION-selected ensemble's out-of-fold DEV accuracy exceeds its TEST accuracy by
4.3 points, more than the brief's 2-point bar, and by more than the 17-feature baseline's own 2.7.**
The excess is entirely xNot360 (the out-of-distribution set). On CANNOT phase 2 the ensemble is
*better* than its out-of-fold DEV figure. VALIDATION cannot see the problem, because it is CANNOT-only.

**Decision.** The brief stated in advance that in this case the 17-feature model should be reported as
the safer choice. **The Phase 0 17-feature model is therefore carried forward to Phases 3–6.**
This is a TEST-based rule that overrides the VALIDATION pick. The brief pre-specified it, so it is
followed as stated rather than invented after the fact, but it is a use of TEST in a decision, and it is
recorded as such in the caveats. Without it, the adopted system would be 1.25 pts worse on all of TEST
(p = 0.012) for a 0.9-pt in-distribution gain.

**Feature importances: the ensemble is really one model swap.** In (b), one feature dominates: the
DeBERTa-anli model's P(contradiction), at importance **0.618**. The reference model's
P(contradiction) drops from 0.537 to 0.013. Every roberta-base, distilroberta and bart-large
probability is at or below 0.006. So **three of the four added models earn near-zero importance and do
not pay for their cost**: an extra ~200 ms per pair, of which bart alone is 146 ms. In (e), averaging
keeps P(contradiction) on top (0.569). Top 6 features of (b), 5 models:

| feature | importance |
| --- | ---: |
| DeBERTa-anli P(contradiction) | 0.618 |
| `token_overlap` | 0.229 |
| `len_b` | 0.019 |
| `cos_sim` | 0.015 |
| `antonym_aligned` | 0.015 |
| reference P(contradiction) | 0.013 |

**Post-hoc check, designed after the 2.4 TEST read and not eligible for adoption.** If the ensemble is
just a model swap, the simplest version is the 17 features with DeBERTa-anli's three probabilities in
place of the reference model's. It gets phase 2 98.0% (+0.3 pts, p = 0.54), but **xNot360 collapses to
57.2%** (−21.7 pts, p = 0.0001), with conditional-inverse FPR at 77.2%; all TEST 89.7% (−4.2 pts, p = 0.0001).
**The best standalone model on xNot360 (86.7%, step 2.2) becomes the worst inside the combiner.** The GBM
learns from CANNOT-like DEV how this model's probabilities interact with `token_overlap`, and on
xNot360's conditional pairs, which have high overlap and high contradiction, that interaction points
the wrong way. A stronger NLI model does not help if the combiner has never seen the data it is
strong on.

```sh
./venv/bin/python experiments/f1/ensemble.py --step 2.5
```

### 3.1–3.2 Generate-and-match guardrail: the margin rule, downgrade-only

`experiments/f1/variants.py` and `experiments/f1/guardrail.py`, tests in `experiments/f1/test_f1.py`
(5 pass).

- **N(T)**, the negations of the teacher sentence: every stage-1 generator variant whose negation
  parity differs from T's own cue parity (nli2 step 5.1's labelling rule, so K1 cancellations and
  double negations are excluded). When T is itself negative, the Phase 1 affirmations with
  `polarity_flip = True` are added. Litotes rewordings are therefore excluded, because they are not
  negations of T. Precomputed for every A sentence of clean DEV, VALIDATION, TEST and K1: **4,693
  teacher sentences, 73,208 variants, median 18 per sentence**, cached with their mpnet embeddings
  (`cache/variants.pkl`, `cache/mpnet.pkl`).
- **The margin rule, implemented exactly as specified.** s_orig = sim(S, T); s_neg = aggregate over n in N
  of sim(S, n); margin = s_neg − s_orig. The pair stays a negation only if margin > δ. No raw maximum is
  used anywhere.
- **Downgrade-only (3.2).** The guardrail runs after the combiner, only on pairs it called a negation, and
  can only lower them to "not a negation". `test_guardrail_only_downgrades_and_abstains_without_variants`
  pins this.
- **Abstain when the generator produced nothing for T.** Such pairs are not downgraded. The
  alternative, downgrading them, would turn every generator gap into a lost negation.
- **Variants that step 3.4 compares** (the margin comparison is fixed; everything else is a choice):
  - similarity: mpnet cosine, word-set Jaccard, or their mean;
  - aggregation: max, or mean of the top 3;
  - δ fit by Youden's J either on all clean DEV (the brief) or only on the clean-DEV pairs the combiner's
    out-of-fold predictions call negations (the pairs the guardrail actually sees);
  - plus the margin as an extra combiner feature instead of a veto, an architecture variant that is
    reported against 3.2's downgrade-only constraint, which it does not satisfy.

  **Rule:** highest VALIDATION accuracy among "no guardrail" and every variant; tie → clean-DEV accuracy.

### 3.3 Generator coverage on clean DEV: the guardrail's precondition

`experiments/f1/guardrail.py --step 3.3` → `results/3_3_coverage.json`. For each of the 1,501 real
negation pairs (A, B) in clean DEV, all negations of A are generated, and the best mpnet cosine between B
and any of them is recorded. No TEST is read.

| gold negations | n | best sim ≥ 0.9 | ≥ 0.8 | ≥ 0.7 | median best sim | exact string match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **all** | **1,501** | **53.5%** | **77.3%** | **86.8%** | 0.923 | 221 (14.7%) |
| verbal | 580 | 66.2% | 82.1% | 86.4% | 0.972 | |
| antonym | 208 | 54.3% | 61.5% | 69.7% | 1.000 | |
| other (lexical) | 713 | 42.9% | 78.0% | 92.2% | 0.888 | |

104 of the 1,501 (6.9%) have **no** generated variant at all, which caps coverage at 93.1%. For
comparison, sim(B, A), B against the original sentence, has a median of 0.808. So the generator
moves the best match from 0.81 to 0.92 at the median.

By generator family: the share of gold negations whose best variant *from that family* reaches ≥ 0.9,
among bases where the family produced anything.

| family | bases with the family | best ≥ 0.9 | family is the overall argmax |
| --- | ---: | ---: | ---: |
| **N_affirmation (Phase 1)** | 315 | **71.1%** | 302 |
| L_scope_position | 17 (n < 50) | 58.8% | 9 |
| E_antonym | 648 | 43.5% | 310 |
| D_affixal | 65 | 41.5% | 20 |
| K2_compound | 45 (n < 50) | 35.6% | 28 |
| J_contrastive | 255 | 32.6% | 30 |
| F_implicit | 343 | 32.4% | 78 |
| K1_cancellation | 89 | 31.5% | 27 |
| H_hedged | 988 | 27.9% | 200 |
| G_prepositional | 112 | 27.7% | 18 |
| A_syntactic | 988 | 27.6% | 177 |
| C_neg_adverb | 977 | 27.1% | 119 |
| B_quantifier | 68 | 25.0% | 11 |
| I_intensified | 988 | 24.3% | 66 |

**Phase 1's affirmation is the best-covering family.** When A is negative, the affirmation recovers the
real B at cosine ≥ 0.9 for 71% of those pairs. *In 1973, Schilling was not a part of the … Commission* →
affirmation *… was a part of …* against B *… was a member of …*: cosine 0.996.

**The precondition, and the discriminative test.** Coverage says the generator can often produce
something close to B. The guardrail also needs the *margin* (max sim to a negation minus sim to T) to
separate negations from non-negations. On clean DEV:

| similarity | AUC of the margin (n = 3,124 pairs with variants) | margin > 0: negations | paraphrase | SemAntoNeg | hard negative | sts_low | snli_neutral | random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mpnet cosine | 0.776 | 73.9% | 14.0% | 45.7% | 22.5% | 67.6% | 49.4% | 80.9% |
| word Jaccard | 0.545 | 36.6% | 12.1% | 94.0% | 4.7% | 13.9% | 31.2% | 19.6% |
| mean of both | 0.715 | 62.7% | 8.0% | 77.6% | 12.0% | 50.5% | 48.5% | 78.2% |

**Verdict, in plain language: coverage is moderate, and the margin is a weak signal.**

- Coverage: about half of real negations have a generated twin at ≥ 0.9, and three-quarters at ≥ 0.8.
- The worst-covered type is antonym negation (61.5% at ≥ 0.8), where the student's antonym is often not
  the one WordNet offers.
- With cosine, the margin is positive for 74% of negations and 14% of paraphrases. That is a useful
  separation on the pairs that matter, because the guardrail only sees pairs the combiner already
  called negations, and those are paraphrase-like.
- The margin cannot tell a negation from an unrelated sentence (random pairs: 81% positive). The
  combiner already rejects those, so it does not need to.
- Jaccard is almost useless (AUC 0.545) and flags SemAntoNeg in the wrong direction.

The precondition is met well enough to test the guardrail. But an AUC of 0.776 sits far below the
combiner's own ~0.98, which predicts the guardrail will remove more true negations than false
positives. Step 3.4 measures that.

```sh
./venv/bin/python experiments/f1/variants.py
./venv/bin/python experiments/f1/guardrail.py --step 3.3
```

### 3.4 Guardrail on TEST: it raises precision by destroying recall

`experiments/f1/guardrail.py --step 3.4` → `results/3_4_guardrail.json`. Base system: the Phase 0 model,
carried forward by step 2.5. δ is fit by Youden's J on clean DEV, on all pairs or on the
out-of-fold-called-negation pairs. "Downgrades" counts pairs the combiner called negations that the
guardrail turned into non-negations; "wrong" counts how many of those were real negations. The
xNot360 conditional slice is the 134 xNot360 pairs where both sides are conditionals (step 0.2
detector).

| configuration | phase-2 acc (95% CI) | all-TEST acc | precision | recall | F1 | FPR paraphrase | sts_low | xNot360 not_neg | antonym recall (n=95) | K1 FPR (n=91) | VAL acc | TEST downgrades (wrong) | xNot360 conditional slice (n=134) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **no guardrail (Phase 0)** | **97.7% (96.9–98.5)** | **93.9%** | 93.0% | **94.9%** | **94.0%** | 1.9% | 3.9% | 28.9% | 92.6% | 2.2% | **97.8%** | — | 66.4% |
| cos, max, δ on all DEV (the brief's default) | 81.6% (79.6–83.7) | 81.7% | **97.8%** | 64.9% | 78.0% | 0.0% | 2.4% | **5.6%** | 64.2% | 2.2% | 82.2% | 315 (265) | **94.0%** |
| cos, max, δ on called pairs | 85.4% (83.5–87.1) | 85.1% | 96.7% | 72.8% | 83.1% | 0.0% | 3.1% | 10.0% | 66.3% | 2.2% | 86.7% | 236 (195) | 90.3% |
| cos, top-3, δ on all DEV | 80.9% (78.7–82.9) | 81.1% | 97.4% | 64.1% | 77.3% | 0.0% | 2.4% | 6.7% | 63.2% | 2.2% | 81.7% | 320 (272) | 93.3% |
| cos, top-3, δ on called pairs | 86.3% (84.5–88.1) | 85.9% | 95.8% | 75.1% | 84.2% | 0.0% | 3.1% | 13.9% | 68.4% | 2.2% | 87.2% | 209 (175) | 85.8% |
| Jaccard, max, δ on all DEV | 69.3% (66.7–71.6) | 71.0% | 91.2% | 46.8% | 61.8% | 0.0% | 0.8% | 21.7% | 55.8% | 2.2% | 70.7% | 448 (425) | 72.4% |
| Jaccard, max, δ on called pairs | 89.3% (87.6–90.9) | 87.3% | 93.3% | 80.5% | 86.4% | 1.6% | 0.8% | 25.0% | 62.1% | 2.2% | 89.3% | 139 (127) | 68.7% |
| Jaccard, top-3, δ on all DEV | 67.2% (64.6–69.7) | 65.3% | 88.7% | 35.4% | 50.6% | 0.0% | 0.8% | 21.7% | 55.8% | 2.2% | 67.4% | 548 (525) | 72.4% |
| Jaccard, top-3, δ on called pairs | 88.9% (87.1–90.5) | 86.9% | 93.2% | 79.7% | 86.0% | 1.6% | 0.8% | 25.0% | 60.0% | 2.2% | 89.0% | 146 (134) | 68.7% |
| mix, max, δ on all DEV | 79.6% (77.4–81.7) | 80.5% | 96.7% | 63.2% | 76.4% | 0.0% | 0.8% | 10.0% | 60.0% | 2.2% | 79.9% | 324 (280) | 88.8% |
| mix, max, δ on called pairs | 88.1% (86.4–89.9) | 86.4% | 93.3% | 78.5% | 85.2% | 0.3% | 2.4% | 25.6% | 63.2% | 2.2% | 87.5% | 158 (145) | 68.7% |
| mix, top-3, δ on all DEV | 76.9% (74.6–79.1) | 78.0% | 96.8% | 58.0% | 72.5% | 0.0% | 0.8% | 8.9% | 59.0% | 2.2% | 76.9% | 372 (326) | 90.3% |
| mix, top-3, δ on called pairs | 86.6% (84.7–88.4) | 85.3% | 93.1% | 76.3% | 83.9% | 0.7% | 2.4% | 25.0% | 62.1% | 2.2% | 86.5% | 177 (164) | 69.4% |
| margin (cos, max) as a combiner feature (not downgrade-only) | 97.9% (97.1–98.6) | 94.0% | 92.9% | 95.4% | 94.1% | 1.9% | 3.9% | 28.9% | **97.9%** | 2.2% | 97.3% | — | 65.7% |

snli_neutral and random_pair FPR are 0.0% in every row except the feature variant (random_pair 0.8%).

**Every veto variant is a large net loss.**

- The brief's default (cosine, max, δ on all DEV) raises precision 93.0% → 97.8%. In the same sentence
  it cuts **recall 94.9% → 64.9%**, F1 94.0% → 78.0%, and **phase-2 accuracy by 16.1 pts** (p = 0.0001).
- **265 of its 315 TEST downgrades were real negations.** The precision filter is wrong 84% of the time
  it acts.
- The least damaging veto (Jaccard, δ on called pairs) still costs 8.4 pts on phase 2 and 14.4 pts of
  recall.
- Fitting δ on the pairs the guardrail actually sees halves the damage, but cannot remove it.

**The accidental help on conditionals is real but does not pay.** The default variant takes the
xNot360 conditional slice from 66.4% to 94.0% (n = 134) and xNot360 conditional-inverse FPR from 28.9%
to 5.6%. Inverses are generated by the generator's antecedent negation, so S sits closer to T than to
any generated negation of T. All of xNot360 rises only +2.8 pts (p = 0.29), because the same veto drops
xNot360's real negations too.

**The margin as a combiner feature** is the only variant near the baseline: +0.2 pts on phase 2 (p = 0.63),
antonym recall 92.6% → 97.9% (n = 95), random_pair FPR 0.0% → 0.8%, margin importance 0.028. It does
not satisfy 3.2 (it can upgrade), and VALIDATION falls 97.8% → 97.3%.

**Why it fails** is what 3.3 predicted. The margin's AUC on DEV is 0.776; the combiner's is 0.98. A
second opinion that is much weaker than the first can only veto decisions the first got right more
often than ones it got wrong. Three causes:

- the embedding is nearly blind to negation, so a reworded negation (*closed* / *reopened*) lands about
  as close to T as to any generated negation;
- generator coverage is 53.5% at ≥ 0.9;
- the student's negation is often not one the generator writes.

```sh
./venv/bin/python experiments/f1/guardrail.py --step 3.4
```

### 3.5 Guardrail: rejected

**Rule, stated in `guardrail.py` before the 3.4 run:** highest VALIDATION accuracy among "no guardrail" and
every variant, with clean DEV as the tie-break. **No guardrail** wins at 97.8%. The best alternative is the margin as a
combiner feature (97.3%), and the best veto reaches 89.3%. The guardrail is **not adopted**.

The adopted configuration for Phases 4–6 is the Phase 0 model with no guardrail. It is recorded in
`experiments/f1/results/final_config.json` and read by `experiments/f1/detector.py` and the dashboard.
The generated negations stay visible in the dashboard as an explanation aid. There they are not a
decision stage.

What would make the idea work, from 3.3 / 3.4:

- use the margin as one more feature rather than a veto (already near-neutral, and it lifts antonym recall);
- or score S against N with the NLI model instead of an embedding, so the comparison can see negation.

The second costs about 18 NLI calls per pair at the median and was not run (see "approaches considered").

### 4.1 Paragraph evaluation set — **machine-authored**

`experiments/f1/make_paragraphs.py` → `data/paragraph_eval.jsonl`. **All 28 items were written by the
model (Claude) for this report, not by teachers or students**, and are labelled `authorship: machine`
in the file. The gold labels were written by the same author. The script asserts that every teacher
sentence has a verdict, and that every student sentence is either matched to exactly one teacher
sentence or listed as extra. Student sentences are deliberately shuffled out of teacher order.

| property | count |
| --- | ---: |
| teacher / student paragraph pairs | **28** (OS 10, CN 10, DBMS 8) |
| teacher sentences | 93 (3–6 per paragraph) |
| student sentences | 91 |
| fully correct but reworded (`reworded`: all covered, nothing missing or extra) | 9 |
| exactly one contradicted teacher sentence (`one_contradiction`) | 11 |
| one or more teacher sentences missing (`missing`) | 5 (10 missing teacher sentences) |
| extra student sentences (`extra`) | 6 (8 extra sentences) |
| a teacher sentence that is itself negative (`negative_teacher`; tests Phase 1) | 8 |
| a cancellation form that must not be flagged (`cancellation`: *not uncommon*, *not impossible*, *not untrue*, *not rare*) | 4 |
| a conditional misstatement (`conditional`) / a quantifier-scope error (`scope`) | 1 / 1 |

Gold per teacher sentence: 72 covered, 11 contradicted, 10 missing. Tags overlap. The contradictions
cover the main shapes:

- verbal negation, in either direction, including a negative teacher sentence contradicted by a
  positive student one (*UDP does not guarantee delivery* / *UDP guarantees that every datagram is
  delivered*);
- antonym (*simple / difficult*, *declarative / procedural*, *distributed / central*, *reduces /
  increases*);
- quantifier scope (*All … are waiting* / *Not all … are waiting*);
- *never → sometimes*;
- a conditional consequent flip (*must stop sending* / *keeps sending even when …*).

`data/paragraph_eval_human.jsonl` is an **empty stub with the same schema**. `data/README.md` documents the
schema and asks Aryan, Clovis and Arush to add human-written items. Human and machine items are always
reported separately.

```sh
./venv/bin/python experiments/f1/make_paragraphs.py
```

### 4.2 Matching: optimal vs greedy, and what goes in the cells (machine-authored items, n = 28 paragraphs)

`experiments/f1/paragraphs.py --step 4.2` → `results/4_2_matching.json`. For each paragraph pair, an
m × n matrix (93 teacher × 91 student sentences over all 28 items) holds the similarity and the adopted
detector's verdict for each (teacher, student) cell. Assignment is then made two ways:

- **greedy:** repeatedly take the highest unassigned cell;
- **optimal:** `scipy.optimize.linear_sum_assignment`, the Hungarian method, in one call.

Alternatives compared:

- cell content: mpnet cosine, word-set Jaccard, or their mean;
- the coverage threshold τ applied before assignment (cells below τ cannot match) or after it.

τ is fit on **clean DEV**, never on the paragraph set (`results/4_2_tau.json`). It is Youden's J of the
similarity, separating "same claim or its negation" pairs (CANNOT negations, paraphrases, SemAntoNeg)
from "different claim" pairs (random, STS-low, SNLI neutral, hard negatives). τ is cosine 0.661,
Jaccard 0.563, mixed 0.597.

**Selection:** there is no DEV paragraph set, so nothing could be chosen on held-out data. The default
(optimal, cosine, threshold before) was fixed before the run, and the alternatives are reported beside
it, not chosen from. **Machine-authored data; n printed throughout.**

| assignment, cells, threshold | teacher-verdict accuracy (n = 93) | all-unit accuracy (n = 184) | contradictions caught / false alarms (gold 11) | extra: TP / FP / FN (gold 8) |
| --- | ---: | ---: | --- | --- |
| **optimal, cosine, before (default)** | **88.2%** | **90.8%** | 6 / 2 | 8 / 6 / 0 |
| optimal, cosine, after | 88.2% | 90.8% | 6 / 2 | 8 / 6 / 0 |
| greedy, cosine, before | 87.1% | 89.7% | 6 / 2 | 8 / 7 / 0 |
| greedy, cosine, after | 87.1% | 89.7% | 6 / 2 | 8 / 7 / 0 |
| optimal or greedy, Jaccard, either | 22.6% | 21.7% | 5 / 0 | — |
| optimal or greedy, mixed, either | 53.8% | 53.8% | 6 / 1 | — |

*All-unit accuracy* counts teacher sentences (verdict right) plus student sentences (extra / not extra right).

- **Greedy and optimal disagree on 2 of 28 items (p04, p10) and 4 of 93 teacher sentences.** Optimal is
  right on one more sentence (88.2% vs 87.1%) and makes one fewer false "extra". It stays the default:
  same cost, and slightly better.
- **Thresholding before or after assignment** gives identical verdicts at this τ. Before is kept,
  because it never forces a weak match.
- **Jaccard and mixed cells collapse**, and that is a finding about τ, not about the cells. τ was fit on
  clean DEV, where same-claim pairs share most of their words (CANNOT negations are near copies). Reworded
  student paragraphs share far fewer words, so with Jaccard τ = 0.563 almost every teacher sentence becomes
  "missing". A DEV built from near copies cannot set a similarity bar for rewording. Cosine survives
  because mpnet is robust to rewording.
- **Not tried:** many-to-one matching (one student sentence covering two teacher sentences). The data has
  no gold case of it, so it could not be measured.

```sh
./venv/bin/python experiments/f1/paragraphs.py --step 4.2
```

### 4.3–4.4 Per-sentence verdicts and the paragraph score

Implemented in `experiments/f1/paragraphs.py` (`verdicts`, `score`), tested in `test_f1.py`. After optimal
assignment with cells below τ = 0.661 (cosine, fit on clean DEV) excluded:

| condition (teacher sentence i) | verdict |
| --- | --- |
| matched to student j, cos ≥ τ, detector says not a negation | **covered** |
| matched to student j, cos ≥ τ, detector says negation (any stage) | **contradicted** |
| no student sentence at cos ≥ τ | **missing** |
| student sentence matched to no teacher sentence | **extra** |

A contradiction must clear the similarity bar too. A student sentence too far from the teacher's is
treated as not addressing it, rather than as a contradiction of it.

**Paragraph score: coverage = covered ÷ teacher sentences, and contradiction count, reported
separately and never combined into one grade.** A student who contradicts a point has made a different
mistake from one who leaves it out, and one number would hide which. The dashboard and
`results/4_5_paragraphs.json` (`paragraph_scores`) show both, beside the gold values.

### 4.5 Paragraph case study — results (**machine-authored items; a case study, not a benchmark**)

`experiments/f1/paragraphs.py --step 4.5` → `results/4_5_paragraphs.json`. Adopted detector (Phase 0 GBM
+ override), optimal assignment, cosine cells, τ = 0.661 fit on clean DEV. **Every number below is on
the 28 machine-authored items** (93 teacher sentences, 91 student sentences) **written by the same
model that built the system**. The human subset is empty: 0 items so far, so no human result exists.

**Per-sentence verdict accuracy: 88.2% of 93 teacher sentences; 90.8% of 184 units** (teacher sentences
plus student extra / not-extra). Assignment matches the gold student sentence for 90.4% of the 83
teacher sentences that have one.

Confusion over teacher sentences (rows gold, columns predicted; n = 93):

| gold \ predicted | covered | contradicted | missing |
| --- | ---: | ---: | ---: |
| covered (72) | **66** | 2 | 4 |
| contradicted (11) | 3 | **6** | 2 |
| missing (10) | 0 | 0 | **10** |

Extra student sentences (n = 91 student sentences, 8 gold extras): 8 found, 0 missed, **6 false extras**.
These come from the 4 covered → missing and 2 contradicted → missing cases, whose student sentence was
then left unmatched. In the four-verdict view, then: covered 66 / 72, contradicted 6 / 11, missing 10 / 10,
extra 8 / 8 with 6 false alarms.

**Paragraph scores (coverage and contradiction count, reported separately):**

- mean absolute coverage error 0.098 (n = 28 paragraphs);
- contradiction count exactly right for 21 of 28 paragraphs.

By tag (teacher sentences; every slice n < 50):

| tag | items | teacher sentences | verdict accuracy | contradictions caught / gold | false alarms |
| --- | ---: | ---: | ---: | ---: | ---: |
| reworded | 9 | 32 | 90.6% | — | 1 |
| one_contradiction | 11 | 33 | 81.8% | 6 / 11 | 0 |
| missing | 5 | 19 | 94.7% | — | 1 |
| extra | 6 | 19 | 89.5% | 0 / 1 | 0 |
| negative_teacher | 8 | 24 | 83.3% | **2 / 5** | 0 |
| cancellation | 4 | 12 | 83.3% | — | **0** |
| conditional / scope | 1 / 1 | 3 / 3 | 100% / 100% | 1 / 1, 1 / 1 | 0 |

- **No cancellation form was flagged** (*not uncommon*, *not impossible*, *not untrue*, *not rare*; 0
  false alarms in 4 items). The override and the combiner both hold.
- **Contradictions are where it fails: 6 of 11 caught.** Only 2 of 5 contradictions of a
  *negative* teacher sentence are caught.

**The finding: the combiner, not NLI, loses the reworded contradictions.** In every one of the 7 pairs
behind the errors below, the reference NLI model alone gives the right answer: P(contradiction) ≥ 0.999
for the 5 missed or mis-matched contradictions, and ≤ 0.001 for the 2 false alarms. The GBM gets all 7
wrong. On CANNOT, a negation is nearly always a minimal edit of A, so `token_overlap` became the
second-largest feature, and a contradiction that rewords the teacher's sentence is overruled. Student
answers reword. As a side-by-side check, not a selection (there is no paragraph DEV set), the same
pipeline with **NLI alone** as the verdict (P ≥ 0.975, its clean-DEV threshold) scores **92.5%
teacher-verdict accuracy, catches 9 of 11 contradictions, and raises 1 false alarm**. The false alarm is
*… makes round-robin behave like first-come first-served* / *… round robin degenerates into FCFS*,
P = 0.997. Its two misses are matching failures: the student sentence is below τ (*FCFS* vs *first-come
first-served*, cosine 0.245; *UDP does not guarantee delivery* / *UDP guarantees that every datagram is
delivered*, cosine 0.591).

**Every paragraph where the system got a contradiction wrong, quoted in full** (✗ marks the wrong
verdict; n = 7 of 28):

**p04** (OS; tags: one_contradiction, negative_teacher)

| | teacher | gold | predicted |
| --- | --- | --- | --- |
| T1 | A user-level thread is managed without kernel support. | covered | covered |
| T2 | The kernel does not know about user-level threads. | contradicted | covered ✗ |
| T3 | If one user-level thread blocks, the whole process can block. | covered | covered |

Student: (S1) The kernel is aware of every user-level thread. (S2) User threads are handled by a library rather than by the kernel. (S3) A blocking call in one user thread may block the entire process.

**p08** (OS; tags: one_contradiction, extra)

| | teacher | gold | predicted |
| --- | --- | --- | --- |
| T1 | The FCFS scheduling algorithm is simple to implement. | contradicted | missing ✗ |
| T2 | FCFS can cause the convoy effect. | covered | covered |
| T3 | Short processes may wait a long time behind a long process under FCFS. | covered | covered |

Student: (S1) First-come first-served is difficult to implement. (S2) FCFS suffers from the convoy effect. (S3) Under FCFS a short job can wait behind a long one for a long time. (S4) SJF gives the minimum average waiting time.

**p11** (CN; tags: one_contradiction, negative_teacher)

| | teacher | gold | predicted |
| --- | --- | --- | --- |
| T1 | UDP is a connectionless protocol. | covered | covered |
| T2 | UDP does not guarantee delivery. | contradicted | missing ✗ |
| T3 | UDP has lower overhead than TCP. | covered | covered |

Student: (S1) UDP guarantees that every datagram is delivered. (S2) No connection is set up before UDP sends a datagram. (S3) UDP headers are smaller, so UDP has less overhead than TCP.

**p13** (CN; tags: one_contradiction)

| | teacher | gold | predicted |
| --- | --- | --- | --- |
| T1 | Every host on the internet needs an IP address. | covered | covered |
| T2 | DNS translates domain names into IP addresses. | covered | covered |
| T3 | DNS uses a hierarchical, distributed database. | contradicted | covered ✗ |

Student: (S1) Domain names are converted to IP addresses by DNS. (S2) DNS data is stored in one central database. (S3) An internet host must have an IP address.

**p21** (DBMS; tags: missing)

| | teacher | gold | predicted |
| --- | --- | --- | --- |
| T1 | Normalization reduces redundancy in a database. | covered | covered |
| T2 | A table is in first normal form if every attribute is atomic. | missing | missing |
| T3 | Second normal form removes partial dependencies. | missing | missing |
| T4 | Third normal form removes transitive dependencies. | covered | contradicted ✗ |

Student: (S1) Normalizing a database cuts down on redundant data. (S2) In 3NF there are no transitive dependencies.

**p25** (DBMS; tags: one_contradiction, negative_teacher)

| | teacher | gold | predicted |
| --- | --- | --- | --- |
| T1 | Every relation in a relational database has a schema. | covered | covered |
| T2 | A view is a virtual table defined by a query. | covered | covered |
| T3 | Views do not store data themselves. | contradicted | covered ✗ |

Student: (S1) A view is a virtual table built from a query. (S2) Views store their own copy of the data. (S3) Each relation has a schema.

**p28** (CN; tags: reworded)

| | teacher | gold | predicted |
| --- | --- | --- | --- |
| T1 | The OSI model has seven layers. | covered | contradicted ✗ |
| T2 | The physical layer transmits raw bits. | covered | covered |
| T3 | The network layer handles routing. | covered | covered |
| T4 | The transport layer provides end-to-end communication. | covered | covered |
| T5 | The application layer is closest to the user. | covered | covered |
| T6 | Each layer provides services to the layer above it. | covered | covered |

Student: (S1) Services flow upward: each layer serves the one above. (S2) Routing is the job of the network layer. (S3) Seven layers make up the OSI reference model. (S4) Raw bits are sent over the medium by the physical layer. (S5) End-to-end delivery between processes is provided by the transport layer. (S6) The layer nearest to the user is the application layer.

Why each one went wrong (P = combiner raw output, threshold 0.428):

- **p04:** contradiction of a negative teacher sentence (*does not know* / *is aware of*), P = 0.16.
- **p08:** antonym (*simple* / *difficult*) but cosine 0.245 because of *FCFS* vs *first-come first-served*.
  Below τ, so the sentence is called missing.
- **p11:** *does not guarantee* / *guarantees that every datagram is delivered*, cosine 0.591 < τ: missing.
- **p13:** antonym *distributed* / *one central*, P = 0.075.
- **p21:** false alarm, *removes transitive dependencies* / *there are no transitive dependencies*, P = 0.873.
- **p25:** *do not store data* / *store their own copy*, P = 0.093.
- **p28:** false alarm on a pure reordering, *The OSI model has seven layers* / *Seven layers make up the OSI
  reference model*, P = 0.85.

```sh
./venv/bin/python experiments/f1/paragraphs.py --step 4.5
```

### 5.1 500 definitions from a real source

`experiments/f1/fetch_definitions.py` → `data/definitions.jsonl` and `results/5_1_definitions.json`.
**All 500 come from English Wikipedia lead sentences. None was written for this project**, so the
self-written column is empty, and every row has `authorship: real source`.

The fixed concept categories per subject (listed in the JSON) were:

- **OS:** OS technology, process, memory management, scheduling, file systems, virtual memory, IPC, kernels,
  concurrency control;
- **CN:** routing, transport / network / link-layer protocols, topology, error detection, network
  architecture, internet and network protocols;
- **DBMS:** database theory, relational model, transaction processing, SQL, database algorithms,
  concurrency-control algorithms, index techniques.

For each article, the first lead sentence was kept only if it passed all of these filters:

- definition-shaped (*X is / are / refers to …*);
- 8–45 words, and not a disambiguation page;
- not a company / product / venue page (stated regex);
- no year in the title.

Then 167 OS, 167 CN and 166 DBMS were sampled at random (seed 0).

| subject | articles in the categories | with a lead sentence | definition-shaped after filters | sampled |
| --- | ---: | ---: | ---: | ---: |
| OS | 532 | 517 | 441 | 167 |
| CN | 1,019 | 1,005 | 839 | 167 |
| DBMS | 306 | 296 | 236 | 166 |

The only change to the source text is that parenthetical asides (pronunciations, abbreviations, dates)
were removed from 268 of the 500. Each row keeps the raw lead sentence, the URL and the revision id
(CC BY-SA 4.0 attribution).

**Limits, stated now:**

- The source is one encyclopedia, so the register is encyclopedic, not lecture-note.
- Some product pages survive the filter. 50 of the 500 mention a vendor or "software"; examples are
  *Tuxedo is a middleware platform …* and *IBM CICS is a family of mixed-language application servers …*.
- A first run without the product filter and with broader DBMS categories pulled in *Interfolio* (a
  company) and *facility condition index*; it was discarded and the filter added. This is reported here,
  not hidden.
- The 100 OS / CN sentences in `experiments/pattern` were checked first and not used. They are
  "hand-written, textbook-style" with no recorded author, and contain no DBMS.

```sh
./venv/bin/python experiments/f1/fetch_definitions.py
```

### 5.2 Student answers: **machine-generated**, five categories plus one

`experiments/f1/make_student_answers.py` → `data/student_answers.jsonl` (2,645 items) and
`results/5_2_answers.json`. **Every answer is machine-generated**, by the generator or a template. Labels
follow from the category by construction, and nobody checked them.

| category | label | how | items (OS / CN / DBMS) | not buildable |
| --- | ---: | --- | ---: | ---: |
| correct paraphrase | 0 | one of four templates (*X can be described as …*, *The term X refers to …*, *Put simply, X is …*, *…: that is what X means*), seeded draw | 484 (166 / 161 / 157) | 16 (no copula split) |
| negation of the definition | 1 | generator, one stage-1 variant with flipped parity from the cue families (A 126, C 114, I 106, H 98, B 5, F 5, D 4) | 458 (147 / 160 / 151) | 42 (no variant) |
| partially correct | 0 | template: the definition cut back to its first predicate (*TCP is a transport protocol.*) | 411 (145 / 139 / 127) | 89 (no cut point) |
| wrong but topically related | 0 | another same-subject definition's predicate, with this term as subject | 484 (166 / 161 / 157) | 16 |
| cancellation form | 0 | generator K1_cancellation; **template *It is not untrue that …* for 489 of 500**, where K1 produced nothing | 500 (167 / 167 / 166) | 0 |
| *extra:* antonym negation | 1 | generator E_antonym (a content word swapped for its antonym) | 308 (105 / 113 / 90) | 192 |

**The generator could not make cancellations of definitions.** K1 needs a predicative adjective it can
prefix (*not un-X*), and definitions are *X is a NOUN that …*, so only 11 of 500 got a K1 variant. The
other 489 use one template, flagged `template_fallback: true`. This is a coverage limit of the
generator, found here.

**The extra category was added because the five miss a realistic failure:** a student who uses the
wrong one of a pair of opposites (*an open standard* → *a closed standard*). It is reported separately.
**Its labels are noisier than any other category's.** The antonym swap often hits a name or a fixed
phrase rather than the claim: *Stop-and-wait ARQ* → ***Start**-and-wait ARQ*, *also known as* → *also
**ignored** as*, *anonymous pipe* → *onymous pipe*. It also has generator grammar errors (*an closed
standard*). Such items are labelled negations by construction, but many are really nonsense or renamings.

```sh
./venv/bin/python experiments/f1/make_student_answers.py
```
