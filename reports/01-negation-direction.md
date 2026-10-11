# Is there a consistent negation direction in sentence-embedding space?

*Paths in the reproduction section refer to the repository layout at the commit hashes listed. See `docs/PATHS.md` for the current locations.*

Exploratory measurement, run on CPU, no LLM calls. Everything below comes from
`results.json`, which `experiment.py` writes; the tables are generated from it
by `report_tables.py`, not typed by hand.

**Short answer.** There is a direction and it is reproducible — on every model,
an axis fit only on explicit negation ranks the negated variant above its own
base in 100% of held-out explicit pairs, and paraphrases barely move along it.
But it is *mostly a cue detector*: it transfers to implicit negation in
direction while losing 10–40% of its magnitude and missing antonyms almost
entirely; it gets cancellation wrong on two of three models (MiniLM pushes
"never fails to sort the array" *further* along the negation axis than a plain
"does not sort the array", though mpnet largely resists this); and it carries
essentially no intensity information (hedged vs intensified AUC 0.53–0.61). On
four of five groups it does no better than a word lookup against the project's
own lexicons; implicit negation is the one place it wins.

---

## 0. What was measured, and on what data

### Data provenance — read this first

The `variants.jsonl` checked in at the repo root is **not** the stage-1 corpus.
It is a 5-sentence smoke-test artefact: 61 records, six families
(A/C/E/H/I/J) out of fourteen, no K1, no K2, no D/F/G, and three of its five
inputs are debug strings (*"Not that and surely not i will pass."*). Running the
experiment on it would have measured nothing.

The corpus used here is the stage-1 corpus as `docs/status.md` describes it:
455 records over the 20 base sentences in `tests/data/base_sentences_v1.txt`,
regenerated with the unmodified generator:

```bash
./run.sh tests/data/base_sentences_v1.txt \
         experiments/negation_direction/data/variants.jsonl --quiet
```

No generator code was touched. The regenerated file matches the documented
stage-1 numbers (455 records / 20 bases / 14 families).

### Pairs and groups

| group | families | pairs |
| --- | --- | --- |
| explicit | A_syntactic, B_quantifier, C_neg_adverb, H_hedged, I_intensified | 301 |
| implicit | D_affixal, E_antonym, F_implicit, G_prepositional, J_contrastive | 100 |
| cancellation | K1_cancellation (net_negation = 0) | 42 |
| compound | K2_compound (net_negation = 2) | 5 |
| other | L_scope_position, M_modal — in no group the brief defines, reported separately | 6 |
| control | paraphrases, no negation | 21 |

One record is a *de-negation* ("The task is impossible." → "The task is
possible.", `operation = affirm`). Its expected sign is the opposite of every
other pair, so it is held out of all group statistics and reported on its own
at the end.

### The paraphrase control, and how defensible it is

Controls were generated mechanically (`make_controls.py`): a WordNet synonym
swap on one content word that is not negation-bearing, re-inflected with the
generator's own morphology helpers, plus clause reordering for coordinated and
subordinated sentences. WordNet's synsets produce a lot of nonsense — *"The
parser scats with a cache."*, *"Everyone surpassed the exam."* — so the 54
candidates were hand-filtered once, on English grammaticality and meaning
preservation alone, **before any embedding was computed**, leaving 21 control
pairs over 14 of the 20 base sentences (`data/control_keep.txt` records the
selection; rerunning `make_controls.py` reproduces both files).

This is the weakest part of the design and it should be treated that way: 21
pairs, hand-selected, and four base sentences (including *"The function sorts
the array."*) get no control at all because WordNet offered nothing usable for
them. Every claim resting on the control is a claim resting on 21 pairs.

### Models

| key | model | pooling |
| --- | --- | --- |
| minilm | sentence-transformers/all-MiniLM-L6-v2 | mean over tokens, L2-normalised |
| mpnet | sentence-transformers/all-mpnet-base-v2 | mean over tokens, L2-normalised |
| bert | bert-base-uncased | mean over tokens, L2-normalised |

Embeddings come from `transformers` directly rather than `sentence-transformers`:
for both SBERT models the published pipeline is Transformer → mean pool →
normalize with no dense layer, so the extra dependency buys nothing. Vectors are
cached under `cache/` keyed by (model, sentence); a pytest asserts every cached
vector has unit norm.

### Splits

Split is **by base sentence**, 70/30, seed 0 → 14 train bases / 6 test bases.
That is 6 held-out base sentences, so every test-split number is anecdotal.
Because of that, the headline numbers are the **leave-one-base-sentence-out**
ones: 20 folds, each pair scored by an axis fit without ever seeing its own base
sentence. The fixed test split is reported too, and agrees.

### Sign convention

Both axes point base → negated variant, so **a higher projection means more
negated**, and a correct pair has `delta = score(variant) - score(base) > 0`.
The brief phrases this as the negated sentence scoring *lower*; that is the same
statement with the axis multiplied by −1, and no number changes.
---

## The four questions, answered

### Is there a consistent negation direction?

**Partly. There is a reproducible direction, but it is not a single direction.**

Fit on training explicit pairs only and evaluated leave-one-base-sentence-out,
the axis puts the negated variant on the negated side of its own base in
**100% of the 301 explicit pairs, for all three models**. That is the strongest
result in the experiment and it does not depend on the split.

But the individual difference vectors are only loosely aligned with each other.
Mean pairwise cosine between explicit difference vectors is 0.39 (MiniLM),
0.27 (mpnet), 0.19 (BERT). For reference, random vectors in these spaces have
cosines of ±0.04–0.05, and the paraphrase controls sit at 0.03–0.04 — i.e. no
direction at all. So "negation" is a broad cone, not a line: reliably positive
projection onto the group mean, but individual differences that mostly disagree
with each other. The mean difference vector keeps only 45–63% of the average
member's length (0.28–0.32 for controls).

The two ways of fitting the axis agree almost exactly (mean-difference vs
logistic probe: cosine 0.975–0.988 depending on model, and every group number is
within a few points), so nothing here turns on the fitting method.

### Does it generalise to implicit negation?

**In direction yes, in magnitude no — and one family breaks it.**

An axis that has only ever seen *not*/*never*/*no*-style negation still ranks
the implicit variant above its base 95–99% of the time (LOOCV). The shift is
smaller than for explicit negation: 67% of the explicit magnitude for MiniLM,
88% for mpnet, 61% for BERT. Under the fixed midpoint, only 66–90% of implicit
variants actually land on the negated *side* (vs 90–100% for explicit), so the
transfer is partial rather than clean.

The failure is concentrated in **E_antonym** (*"Everyone failed the exam."*,
*"He always arrives late."*): pairwise accuracy 0.80–0.93 and a shift of a quarter
to a half of the explicit mean, with within-family difference-vector
alignment of 0.08–0.11, i.e. no shared direction at all. Antonym substitution is
negation that leaves no trace the axis recognises. *lacks*/*fails to* (F_implicit,
66 pairs) transfers well — 0.98–1.00 accuracy — but those are precisely the
cases the project's own lexicon already lists as cues.

### Does it handle cancellation correctly?

**No for two models out of three, and the failure is the informative kind.**

K1 pairs are *"The function sorts the array."* → *"The function never fails to
sort the array."* — two cues that cancel, `net_negation = 0`, so the variant
means what the base means and should barely move.

| model | K1 mean shift as a fraction of the explicit shift | K1 pairs pushed to the negated side |
| --- | --- | --- |
| MiniLM | 1.05 | 98% |
| BERT | 0.79 | 76% |
| mpnet | 0.28 | 36% |

MiniLM moves a cancelled sentence *further* along the negation axis than a plain
negation does. That is a cue counter, not a meaning reader — the mean K1
difference vector even sits at cosine 0.79 to the explicit one. BERT is barely
better. mpnet is the only model that partially resists: its K1 shift is a
quarter of the explicit shift, only 36% of K1 variants cross the midpoint, and
the projection shift separates explicit from cancelled pairs with AUC 0.89
(MiniLM 0.45, i.e. worse than chance; BERT 0.66).

One model behaving differently on 42 pairs from 20 base sentences is a lead, not
a finding. But it is the single most interesting number here: cancellation
sensitivity is not uniformly absent from sentence embeddings.

### Does it encode intensity?

**No.** Hedged (*"probably not"*) vs intensified (*"certainly not"*) projection
shifts: 0.333 vs 0.364 (MiniLM), 0.318 vs 0.341 (mpnet), 0.228 vs 0.254 (BERT).
The direction is the expected one every time, but the distributions sit on top of
each other — ROC-AUC 0.53–0.61 with 90 and 84 pairs. Nothing usable for grading.
Note also that C_neg_adverb's `partial` bucket (*hardly*, *barely*, *rarely*),
which is semantically *weaker* than plain negation, gets a shift as large as
either (0.37 MiniLM), which is the wrong ordering.

### And the control?

**Paraphrases stay put.** Mean projection shift as a fraction of the explicit
shift: 8.9% (MiniLM), 3.2% (mpnet), 0.3% (BERT). Pair by pair, the shift
separates negation pairs from paraphrase pairs with ROC-AUC 0.95–0.99.

This is the clearest positive result after the explicit-pair accuracy, and it is
worth contrasting with analysis 1: **raw cosine cannot do this at all**
(AUC 0.50 for MiniLM, 0.70 mpnet, 0.71 BERT; best balanced accuracy 0.64–0.72).
A paraphrase moves an embedding about as far as a negation does — it just moves
it somewhere else. The direction carries the signal; the distance does not.

The caveat is the control set itself: 21 pairs, hand-filtered, 14 base sentences.

### Against the trivial baseline

A word lookup against the project's own lexicons (plus *-less* / *un-* / *non-*
/ *im-* / *ir-* / *il-* affixes) scores a pair as negated when the variant
contains more cues than the base:

| group | word lookup | best embedding axis |
| --- | --- | --- |
| explicit | **1.00** | 1.00 |
| implicit | 0.78 | **0.99** |
| cancellation | 1.00 *(wrong by construction — it counts both cues)* | 0.74 (mpnet) |
| compound | 1.00 | 1.00 |
| control (false positives) | **0.05** | 0.43–0.71 by sign, but only 0.3–9% of the explicit magnitude |

Stated plainly: **on four of five groups the embedding axis does no better than
a word lookup, and on the control it is noisier by sign.** It earns its keep in
exactly one place — implicit negation, +0.17 to +0.21 pairwise accuracy — driven
by the 22% of implicit pairs (antonyms, contrastive frames) that contain no
lexicon cue at all. Everything else the lexicon already knew.

The cancellation row deserves care: the lookup gets 1.00 only because K1 variants
genuinely contain more cue words; by the semantics it should score ≈ 0. Both
methods fail cancellation; the lookup fails it by design, MiniLM and BERT fail it
while looking like they are reading meaning.

---

## Sample-size caveats — read before quoting any number

- **20 base sentences.** The fixed 70/30 split leaves **6 test base sentences**,
  so every number in §3 is anecdotal. The LOOCV tables (§3b) are the ones to
  quote, and even they resample 20 items.
- **Pairs are not independent.** 475 pairs come from 20 bases; pairs sharing a
  base share half their content. Grouping by base (both the split and the LOOCV)
  is why the accuracies are not inflated, but the effective n is nearer 20 than
  475.
- **Families are wildly unbalanced.** C_neg_adverb (102), H_hedged (90),
  I_intensified (84) and F_implicit (66) are 74% of all pairs; D_affixal has 4,
  B_quantifier 4, K2_compound 5, G_prepositional 5, M_modal 2. Any per-family
  number below n = 15 is an anecdote, and "explicit" as a group is mostly
  *hardly/probably not/certainly not*, not plain *not*.
- **The control is 21 hand-picked pairs** over 14 bases, described in §0.
- **The base sentences are short, simple, and mostly about the same few topics**
  (functions, arrays, tests, exams). Nothing here says anything about real prose.
- One de-negation record (n = 1) behaves correctly — it moves *backwards* along
  the axis (−0.45 MiniLM, −0.47 mpnet, −0.27 BERT). One record.

---

## What would have to be true for this to become the scoring mechanism

In rough order of how badly each is currently violated:

1. **Cancellation must cost, not add.** A `net_negation = 0` variant has to land
   near its base. Today: MiniLM 1.05× the explicit shift, BERT 0.79×, mpnet
   0.28×. Target: ≤ 0.2× with the explicit-vs-cancelled AUC above ~0.9 on more
   than 42 pairs from more than 20 bases. mpnet is the only starting point.
2. **Intensity must be ordered.** intensified > neutral > hedged > partial in
   projection magnitude, with separation AUC well above 0.7. Today 0.53–0.61,
   and *partial* is misordered. Without this the axis can say "negated", never
   "how negated" — which is the entire point of a graded score.
3. **Implicit negation must not be a second-class case**, in magnitude as well
   as sign, and E_antonym in particular has to stop being invisible. Until then
   the axis is a cue detector with a small semantic margin.
4. **It must beat the word lookup somewhere that matters.** Right now the honest
   summary is "equal to `grep` on four groups of five". A scoring mechanism that
   the project's existing word lists match is not worth the dependency.
5. **The evaluation has to be an order of magnitude larger**: ≥ 200 base
   sentences across genres, balanced families (≥ 30 pairs each), and a control
   set built the same way — ideally human-validated paraphrases, not WordNet
   swaps. Everything above is measured on 20 sentences.
6. **The variants have to be verified.** Every record here carries
   `verified = null`; stage 2 has never passed its own gate
   (`reports/verifier_validation.md`). Some fraction of these 455 "negations" are
   generator noise, and that fraction is unknown.
7. **A calibrated zero point.** Scoring needs "this pair is 0.3 negated", not
   "variant > base". The midpoint used here is fit on training means and drifts
   per model: 90–100% of explicit variants land on the correct side of it, but
   only 66–90% of implicit ones and 20–76% of cancelled ones. The pairwise result is robust; the absolute scale is not.

If 1 and 2 cannot be fixed, this direction can still be useful as a *feature*
(it is nearly free and it does beat the lexicon on implicit negation), but not
as the score itself.

---

## Reproducing

```bash
# corpus (unmodified generator; writes data/variants.jsonl)
./run.sh tests/data/base_sentences_v1.txt \
         experiments/negation_direction/data/variants.jsonl --quiet

# controls (candidates + the hand-filtered frozen set)
./venv/bin/python experiments/negation_direction/make_controls.py

# the experiment: ~3 min on CPU with embeddings cached, ~15 min cold
./venv/bin/python experiments/negation_direction/experiment.py
./venv/bin/python experiments/negation_direction/experiment.py --models minilm mpnet

# tables below, regenerated from results.json
./venv/bin/python experiments/negation_direction/report_tables.py

# guardrails: no base-sentence leakage across the split, embeddings normalised
./venv/bin/python -m pytest experiments/negation_direction -q
```

Plots in `plots/`: `cosine_<model>.png` (per-group cosine distributions),
`projection_<model>.png` (base vs variant projection histograms per group, with
the train midpoint), `pca_<model>.png` (2D PCA with base → variant arrows; note
that PC1/PC2 encode topic, not polarity — variants cluster around their own base
sentence, which is itself a finding about where negation lives in these spaces).

---

## All numbers

`results.json` holds every value, including the raw per-pair arrays. The tables
below are printed from it by `report_tables.py`.

### 1. Raw cosine(base, variant)

| model | group | n | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| minilm | explicit | 301 | 0.8138 | 0.0848 | 0.5525 | 0.8323 | 0.9518 |
| minilm | implicit | 100 | 0.8049 | 0.0892 | 0.4673 | 0.8108 | 0.9605 |
| minilm | cancellation | 42 | 0.7193 | 0.1321 | 0.3974 | 0.7558 | 0.9516 |
| minilm | compound | 5 | 0.8355 | 0.1186 | 0.6532 | 0.8796 | 0.9611 |
| minilm | other | 6 | 0.8561 | 0.041 | 0.8092 | 0.8589 | 0.9063 |
| minilm | control | 21 | 0.808 | 0.1205 | 0.5915 | 0.7891 | 0.9813 |
| mpnet | explicit | 301 | 0.7311 | 0.1086 | 0.4344 | 0.7457 | 0.9403 |
| mpnet | implicit | 100 | 0.7034 | 0.1256 | 0.3158 | 0.7246 | 0.9744 |
| mpnet | cancellation | 42 | 0.7299 | 0.0958 | 0.4476 | 0.7346 | 0.9171 |
| mpnet | compound | 5 | 0.7832 | 0.1822 | 0.5655 | 0.8341 | 0.9778 |
| mpnet | other | 6 | 0.6809 | 0.108 | 0.5596 | 0.6714 | 0.813 |
| mpnet | control | 21 | 0.8178 | 0.1207 | 0.5903 | 0.833 | 0.9815 |
| bert | explicit | 301 | 0.8597 | 0.0636 | 0.6359 | 0.8701 | 0.976 |
| bert | implicit | 100 | 0.8734 | 0.0661 | 0.678 | 0.8853 | 0.9796 |
| bert | cancellation | 42 | 0.8397 | 0.0545 | 0.7462 | 0.828 | 0.9546 |
| bert | compound | 5 | 0.9054 | 0.056 | 0.8244 | 0.8989 | 0.9692 |
| bert | other | 6 | 0.9048 | 0.0294 | 0.8466 | 0.9181 | 0.9224 |
| bert | control | 21 | 0.906 | 0.0576 | 0.7679 | 0.9204 | 0.9748 |

**Negation pairs vs paraphrase controls, using cosine alone:**

| model | ROC-AUC | best-threshold acc | best-threshold balanced acc | majority-class acc |
| --- | --- | --- | --- | --- |
| minilm | 0.4984 | 0.9621 | 0.6391 | 0.9558 |
| mpnet | 0.7015 | 0.96 | 0.7018 | 0.9558 |
| bert | 0.7138 | 0.9537 | 0.7219 | 0.9558 |

### 1b. Cosine by family

| family | minilm | mpnet | bert | n |
| --- | --- | --- | --- | --- |
| A_syntactic | 0.8401 | 0.7768 | 0.906 | 21 |
| B_quantifier | 0.6904 | 0.6811 | 0.9075 | 4 |
| CONTROL_paraphrase | 0.808 | 0.8178 | 0.906 | 21 |
| C_neg_adverb | 0.7813 | 0.7174 | 0.8854 | 102 |
| D_affixal | 0.7446 | 0.6545 | 0.8325 | 4 |
| E_antonym | 0.7609 | 0.6766 | 0.8886 | 15 |
| F_implicit | 0.799 | 0.7199 | 0.8707 | 66 |
| G_prepositional | 0.8814 | 0.7519 | 0.9452 | 5 |
| H_hedged | 0.8352 | 0.7588 | 0.8358 | 90 |
| I_intensified | 0.8297 | 0.7091 | 0.8402 | 84 |
| J_contrastive | 0.8952 | 0.6298 | 0.8484 | 10 |
| K1_cancellation | 0.7193 | 0.7299 | 0.8397 | 42 |
| K2_compound | 0.8355 | 0.7832 | 0.9054 | 5 |
| L_scope_position | 0.8547 | 0.619 | 0.8969 | 4 |
| M_modal | 0.8589 | 0.8048 | 0.9206 | 2 |

### 2. Difference vectors

Mean pairwise cosine between difference vectors within a group (1.0 = one single direction, 0.0 = unrelated):

| model | group/family | n pairs of pairs | mean | std | median |
| --- | --- | --- | --- | --- | --- |
| minilm | explicit | 45150 | 0.3898 | 0.1725 | 0.3687 |
| minilm | implicit | 4950 | 0.2508 | 0.2073 | 0.2152 |
| minilm | cancellation | 861 | 0.4369 | 0.2204 | 0.4203 |
| minilm | compound | 10 | 0.3089 | 0.3036 | 0.2661 |
| minilm | other | 15 | 0.5037 | 0.1932 | 0.4487 |
| minilm | control | 210 | 0.0359 | 0.1345 | 0.0179 |
| minilm | family:A_syntactic | 210 | 0.458 | 0.1809 | 0.4405 |
| minilm | family:B_quantifier | 6 | 0.6371 | 0.0746 | 0.616 |
| minilm | family:CONTROL_paraphrase | 210 | 0.0359 | 0.1345 | 0.0179 |
| minilm | family:C_neg_adverb | 5151 | 0.4573 | 0.1718 | 0.4489 |
| minilm | family:D_affixal | 6 | 0.2197 | 0.287 | 0.1632 |
| minilm | family:E_antonym | 105 | 0.081 | 0.1762 | 0.0296 |
| minilm | family:F_implicit | 2145 | 0.3657 | 0.2069 | 0.3222 |
| minilm | family:G_prepositional | 10 | 0.4699 | 0.1795 | 0.4822 |
| minilm | family:H_hedged | 4005 | 0.4607 | 0.1808 | 0.4294 |
| minilm | family:I_intensified | 3486 | 0.4693 | 0.1681 | 0.4392 |
| minilm | family:J_contrastive | 45 | 0.4263 | 0.2208 | 0.3901 |
| minilm | family:K1_cancellation | 861 | 0.4369 | 0.2204 | 0.4203 |
| minilm | family:K2_compound | 10 | 0.3089 | 0.3036 | 0.2661 |
| minilm | family:L_scope_position | 6 | 0.6287 | 0.1847 | 0.6145 |
| minilm | family:M_modal | 1 | 0.7749 | 0.0 | 0.7749 |
| mpnet | explicit | 45150 | 0.2672 | 0.2264 | 0.2627 |
| mpnet | implicit | 4950 | 0.24 | 0.2332 | 0.2178 |
| mpnet | cancellation | 861 | 0.3112 | 0.2456 | 0.2916 |
| mpnet | compound | 10 | 0.1896 | 0.3779 | 0.0888 |
| mpnet | other | 15 | 0.4554 | 0.2388 | 0.4538 |
| mpnet | control | 210 | 0.0393 | 0.1123 | 0.0336 |
| mpnet | family:A_syntactic | 210 | 0.2544 | 0.2466 | 0.2641 |
| mpnet | family:B_quantifier | 6 | 0.3311 | 0.2276 | 0.2591 |
| mpnet | family:CONTROL_paraphrase | 210 | 0.0393 | 0.1123 | 0.0336 |
| mpnet | family:C_neg_adverb | 5151 | 0.3052 | 0.2276 | 0.3046 |
| mpnet | family:D_affixal | 6 | 0.2175 | 0.3997 | 0.2278 |
| mpnet | family:E_antonym | 105 | 0.1137 | 0.2084 | 0.0992 |
| mpnet | family:F_implicit | 2145 | 0.3417 | 0.214 | 0.3133 |
| mpnet | family:G_prepositional | 10 | 0.5959 | 0.2791 | 0.5204 |
| mpnet | family:H_hedged | 4005 | 0.3453 | 0.2056 | 0.3319 |
| mpnet | family:I_intensified | 3486 | 0.3054 | 0.2477 | 0.2966 |
| mpnet | family:J_contrastive | 45 | 0.3214 | 0.3667 | 0.3018 |
| mpnet | family:K1_cancellation | 861 | 0.3112 | 0.2456 | 0.2916 |
| mpnet | family:K2_compound | 10 | 0.1896 | 0.3779 | 0.0888 |
| mpnet | family:L_scope_position | 6 | 0.6479 | 0.1419 | 0.6525 |
| mpnet | family:M_modal | 1 | 0.7843 | 0.0 | 0.7843 |
| bert | explicit | 45150 | 0.1938 | 0.1594 | 0.1708 |
| bert | implicit | 4950 | 0.1612 | 0.1637 | 0.1383 |
| bert | cancellation | 861 | 0.2698 | 0.1905 | 0.2421 |
| bert | compound | 10 | 0.3156 | 0.2573 | 0.2461 |
| bert | other | 15 | 0.2056 | 0.2094 | 0.1616 |
| bert | control | 210 | 0.028 | 0.1207 | 0.0183 |
| bert | family:A_syntactic | 210 | 0.2133 | 0.137 | 0.1971 |
| bert | family:B_quantifier | 6 | 0.2443 | 0.1082 | 0.2288 |
| bert | family:CONTROL_paraphrase | 210 | 0.028 | 0.1207 | 0.0183 |
| bert | family:C_neg_adverb | 5151 | 0.2314 | 0.166 | 0.1991 |
| bert | family:D_affixal | 6 | 0.2378 | 0.2356 | 0.1137 |
| bert | family:E_antonym | 105 | 0.0772 | 0.1384 | 0.0668 |
| bert | family:F_implicit | 2145 | 0.2226 | 0.1794 | 0.1896 |
| bert | family:G_prepositional | 10 | 0.3093 | 0.2466 | 0.2047 |
| bert | family:H_hedged | 4005 | 0.2693 | 0.1572 | 0.2545 |
| bert | family:I_intensified | 3486 | 0.3099 | 0.1605 | 0.2869 |
| bert | family:J_contrastive | 45 | 0.4631 | 0.1696 | 0.4483 |
| bert | family:K1_cancellation | 861 | 0.2698 | 0.1905 | 0.2421 |
| bert | family:K2_compound | 10 | 0.3156 | 0.2573 | 0.2461 |
| bert | family:L_scope_position | 6 | 0.342 | 0.2029 | 0.3649 |
| bert | family:M_modal | 1 | 0.4809 | 0.0 | 0.4809 |

Cosine between group **mean** difference vectors:


*minilm*

|  | explicit | implicit | cancellation | compound | other | control |
| --- | --- | --- | --- | --- | --- | --- |
| explicit | 1.0 | 0.8188 | 0.785 | 0.7065 | 0.7927 | 0.196 |
| implicit | 0.8188 | 1.0 | 0.6864 | 0.6609 | 0.704 | 0.2829 |
| cancellation | 0.785 | 0.6864 | 1.0 | 0.5058 | 0.5244 | 0.2174 |
| compound | 0.7065 | 0.6609 | 0.5058 | 1.0 | 0.5688 | 0.2187 |
| other | 0.7927 | 0.704 | 0.5244 | 0.5688 | 1.0 | 0.1689 |
| control | 0.196 | 0.2829 | 0.2174 | 0.2187 | 0.1689 | 1.0 |

*mpnet*

|  | explicit | implicit | cancellation | compound | other | control |
| --- | --- | --- | --- | --- | --- | --- |
| explicit | 1.0 | 0.8704 | 0.2838 | 0.706 | 0.6295 | 0.1174 |
| implicit | 0.8704 | 1.0 | 0.3048 | 0.6858 | 0.6041 | 0.1458 |
| cancellation | 0.2838 | 0.3048 | 1.0 | 0.1158 | -0.0837 | -0.0422 |
| compound | 0.706 | 0.6858 | 0.1158 | 1.0 | 0.2946 | 0.0807 |
| other | 0.6295 | 0.6041 | -0.0837 | 0.2946 | 1.0 | 0.1454 |
| control | 0.1174 | 0.1458 | -0.0422 | 0.0807 | 0.1454 | 1.0 |

*bert*

|  | explicit | implicit | cancellation | compound | other | control |
| --- | --- | --- | --- | --- | --- | --- |
| explicit | 1.0 | 0.7126 | 0.628 | 0.4072 | 0.6053 | 0.0851 |
| implicit | 0.7126 | 1.0 | 0.6208 | 0.4398 | 0.578 | 0.141 |
| cancellation | 0.628 | 0.6208 | 1.0 | 0.2421 | 0.5206 | 0.0026 |
| compound | 0.4072 | 0.4398 | 0.2421 | 1.0 | 0.3454 | 0.0161 |
| other | 0.6053 | 0.578 | 0.5206 | 0.3454 | 1.0 | -0.0558 |
| control | 0.0851 | 0.141 | 0.0026 | 0.0161 | -0.0558 | 1.0 |

Norm of the group mean difference vector vs the mean norm of its members (equal norms = perfectly aligned; a much smaller mean = cancellation):

| model | group | ||mean d|| | mean ||d|| | ratio |
| --- | --- | --- | --- | --- |
| minilm | explicit | 0.3756 | 0.5953 | 0.631 |
| minilm | implicit | 0.311 | 0.6084 | 0.511 |
| minilm | cancellation | 0.5054 | 0.7284 | 0.694 |
| minilm | compound | 0.3773 | 0.5424 | 0.696 |
| minilm | other | 0.4103 | 0.5318 | 0.772 |
| minilm | control | 0.1857 | 0.5803 | 0.32 |
| mpnet | explicit | 0.3701 | 0.7181 | 0.515 |
| mpnet | implicit | 0.3742 | 0.7509 | 0.498 |
| mpnet | cancellation | 0.4205 | 0.7238 | 0.581 |
| mpnet | compound | 0.4165 | 0.5988 | 0.696 |
| mpnet | other | 0.5954 | 0.7886 | 0.755 |
| mpnet | control | 0.176 | 0.5631 | 0.313 |
| bert | explicit | 0.2306 | 0.5157 | 0.447 |
| bert | implicit | 0.2016 | 0.4857 | 0.415 |
| bert | cancellation | 0.3 | 0.557 | 0.539 |
| bert | compound | 0.286 | 0.4184 | 0.684 |
| bert | other | 0.2531 | 0.4328 | 0.585 |
| bert | control | 0.1158 | 0.4157 | 0.279 |

### 3. Negation axis — held-out test split (fit on TRAIN explicit only)


*axis: mean_diff*

| model | group | n | pairwise acc | mean delta | median delta | frac variant on negated side | frac opposite sides of midpoint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| minilm | explicit | 86 | 1.0 | 0.3392 | 0.3048 | 0.9884 | 0.9884 |
| minilm | implicit | 34 | 0.9706 | 0.2362 | 0.2393 | 0.7059 | 0.7059 |
| minilm | cancellation | 11 | 1.0 | 0.4242 | 0.4098 | 1.0 | 1.0 |
| minilm | compound | 5 | 1.0 | 0.2588 | 0.2242 | 1.0 | 1.0 |
| minilm | control | 8 | 0.875 | 0.0464 | 0.0332 | 0.25 | 0.25 |
| mpnet | explicit | 86 | 0.9884 | 0.2821 | 0.2671 | 0.8256 | 0.8256 |
| mpnet | implicit | 34 | 0.8529 | 0.2597 | 0.2864 | 0.7647 | 0.7353 |
| mpnet | cancellation | 11 | 0.4545 | 0.0229 | -0.0 | 0.1818 | 0.0909 |
| mpnet | compound | 5 | 0.8 | 0.2638 | 0.3015 | 1.0 | 0.6 |
| mpnet | control | 8 | 0.375 | 0.0218 | -0.0026 | 0.25 | 0.0 |
| bert | explicit | 86 | 1.0 | 0.1779 | 0.1506 | 0.9186 | 0.7326 |
| bert | implicit | 34 | 0.9706 | 0.112 | 0.0888 | 0.6765 | 0.3529 |
| bert | cancellation | 11 | 1.0 | 0.1606 | 0.0863 | 0.9091 | 0.4545 |
| bert | compound | 5 | 1.0 | 0.1059 | 0.1044 | 1.0 | 0.6 |
| bert | control | 8 | 0.375 | -0.0025 | -0.0023 | 0.375 | 0.125 |

*axis: probe*

| model | group | n | pairwise acc | mean delta | median delta | frac variant on negated side | frac opposite sides of midpoint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| minilm | explicit | 86 | 1.0 | 0.3391 | 0.3114 | 0.9884 | 0.9884 |
| minilm | implicit | 34 | 0.9706 | 0.2345 | 0.2434 | 0.7353 | 0.7353 |
| minilm | cancellation | 11 | 1.0 | 0.3975 | 0.3721 | 1.0 | 1.0 |
| minilm | compound | 5 | 1.0 | 0.258 | 0.2463 | 1.0 | 1.0 |
| minilm | control | 8 | 0.75 | 0.0426 | 0.0277 | 0.25 | 0.25 |
| mpnet | explicit | 86 | 1.0 | 0.288 | 0.2725 | 0.8488 | 0.8488 |
| mpnet | implicit | 34 | 0.8824 | 0.2593 | 0.2883 | 0.7647 | 0.7353 |
| mpnet | cancellation | 11 | 0.5455 | 0.0242 | 0.0129 | 0.1818 | 0.0909 |
| mpnet | compound | 5 | 0.8 | 0.2599 | 0.3205 | 1.0 | 0.6 |
| mpnet | control | 8 | 0.5 | 0.0209 | 0.0026 | 0.25 | 0.0 |
| bert | explicit | 86 | 1.0 | 0.1739 | 0.1488 | 0.9535 | 0.7674 |
| bert | implicit | 34 | 0.9706 | 0.1148 | 0.0857 | 0.8235 | 0.5 |
| bert | cancellation | 11 | 1.0 | 0.1666 | 0.0995 | 0.9091 | 0.4545 |
| bert | compound | 5 | 1.0 | 0.1115 | 0.1158 | 1.0 | 0.6 |
| bert | control | 8 | 0.375 | -0.0057 | -0.0036 | 0.375 | 0.125 |

### 3b. Leave-one-base-sentence-out cross-validation (all 20 bases)


*axis: mean_diff*

| model | group | n | pairwise acc | mean delta | std | frac variant on negated side | frac opposite sides |
| --- | --- | --- | --- | --- | --- | --- | --- |
| minilm | explicit | 301 | 1.0 | 0.3586 | 0.1149 | 0.9967 | 0.9967 |
| minilm | implicit | 100 | 0.98 | 0.2404 | 0.1146 | 0.83 | 0.83 |
| minilm | cancellation | 42 | 1.0 | 0.3777 | 0.1418 | 0.9762 | 0.9524 |
| minilm | compound | 5 | 1.0 | 0.2559 | 0.1431 | 1.0 | 1.0 |
| minilm | other | 6 | 1.0 | 0.3096 | 0.0914 | 1.0 | 1.0 |
| minilm | control | 21 | 0.7143 | 0.0318 | 0.0598 | 0.1905 | 0.0952 |
| mpnet | explicit | 301 | 1.0 | 0.3275 | 0.1316 | 0.9568 | 0.7874 |
| mpnet | implicit | 100 | 0.95 | 0.2881 | 0.1626 | 0.9 | 0.69 |
| mpnet | cancellation | 42 | 0.7381 | 0.093 | 0.1517 | 0.3571 | 0.2619 |
| mpnet | compound | 5 | 0.8 | 0.2696 | 0.2765 | 1.0 | 0.6 |
| mpnet | other | 6 | 1.0 | 0.3181 | 0.1169 | 1.0 | 0.6667 |
| mpnet | control | 21 | 0.5714 | 0.0106 | 0.0771 | 0.1905 | 0.0 |
| bert | explicit | 301 | 1.0 | 0.2017 | 0.0754 | 0.9003 | 0.7475 |
| bert | implicit | 100 | 0.99 | 0.1226 | 0.0662 | 0.66 | 0.42 |
| bert | cancellation | 42 | 0.9762 | 0.1589 | 0.0702 | 0.7619 | 0.5952 |
| bert | compound | 5 | 1.0 | 0.1048 | 0.0407 | 1.0 | 0.2 |
| bert | other | 6 | 1.0 | 0.1289 | 0.0193 | 1.0 | 1.0 |
| bert | control | 21 | 0.4286 | 0.0007 | 0.0415 | 0.381 | 0.0476 |

*axis: probe*

| model | group | n | pairwise acc | mean delta | std | frac variant on negated side | frac opposite sides |
| --- | --- | --- | --- | --- | --- | --- | --- |
| minilm | explicit | 301 | 1.0 | 0.3528 | 0.0977 | 0.99 | 0.99 |
| minilm | implicit | 100 | 0.98 | 0.2391 | 0.11 | 0.82 | 0.82 |
| minilm | cancellation | 42 | 1.0 | 0.3544 | 0.1243 | 0.9762 | 0.9524 |
| minilm | compound | 5 | 1.0 | 0.2554 | 0.1425 | 1.0 | 1.0 |
| minilm | other | 6 | 1.0 | 0.3037 | 0.0801 | 1.0 | 1.0 |
| minilm | control | 21 | 0.6667 | 0.0266 | 0.0559 | 0.1905 | 0.0952 |
| mpnet | explicit | 301 | 1.0 | 0.3214 | 0.1223 | 0.9668 | 0.9136 |
| mpnet | implicit | 100 | 0.96 | 0.2727 | 0.1527 | 0.89 | 0.79 |
| mpnet | cancellation | 42 | 0.7381 | 0.0955 | 0.1474 | 0.4048 | 0.3333 |
| mpnet | compound | 5 | 0.8 | 0.2556 | 0.2526 | 1.0 | 0.6 |
| mpnet | other | 6 | 1.0 | 0.32 | 0.1236 | 1.0 | 1.0 |
| mpnet | control | 21 | 0.619 | 0.008 | 0.0673 | 0.1905 | 0.0 |
| bert | explicit | 301 | 1.0 | 0.1947 | 0.0658 | 0.9502 | 0.8605 |
| bert | implicit | 100 | 0.99 | 0.124 | 0.0644 | 0.81 | 0.62 |
| bert | cancellation | 42 | 0.9762 | 0.1588 | 0.0667 | 0.9048 | 0.7619 |
| bert | compound | 5 | 1.0 | 0.1108 | 0.0409 | 1.0 | 0.6 |
| bert | other | 6 | 1.0 | 0.1288 | 0.0242 | 0.8333 | 0.8333 |
| bert | control | 21 | 0.5714 | 0.0044 | 0.0399 | 0.381 | 0.1429 |

**Can the projection shift tell a negation from a paraphrase, pair by pair?** (ROC-AUC of delta; 0.5 = chance)

| model | axis | negation vs control AUC | explicit vs cancellation AUC |
| --- | --- | --- | --- |
| minilm | mean_diff | 0.984 | 0.4483 |
| minilm | probe | 0.9848 | 0.4862 |
| mpnet | mean_diff | 0.9488 | 0.8866 |
| mpnet | probe | 0.9542 | 0.8829 |
| bert | mean_diff | 0.9877 | 0.6595 |
| bert | probe | 0.9876 | 0.6567 |

Per family (LOOCV, mean-difference axis):

| family | minilm acc | mpnet acc | bert acc | minilm mean delta | mpnet mean delta | bert mean delta |
| --- | --- | --- | --- | --- | --- | --- |
| A_syntactic | 1.0 | 1.0 | 1.0 | 0.3405 | 0.287 | 0.1275 |
| B_quantifier | 1.0 | 1.0 | 1.0 | 0.4551 | 0.3693 | 0.1027 |
| CONTROL_paraphrase | 0.7143 | 0.5714 | 0.4286 | 0.0318 | 0.0106 | 0.0007 |
| C_neg_adverb | 1.0 | 1.0 | 1.0 | 0.3767 | 0.3318 | 0.1545 |
| D_affixal | 1.0 | 1.0 | 1.0 | 0.2368 | 0.258 | 0.1322 |
| E_antonym | 0.8667 | 0.8 | 0.9333 | 0.129 | 0.1659 | 0.0561 |
| F_implicit | 1.0 | 0.9848 | 1.0 | 0.2787 | 0.3266 | 0.129 |
| G_prepositional | 1.0 | 0.8 | 1.0 | 0.2151 | 0.0406 | 0.0647 |
| H_hedged | 1.0 | 1.0 | 1.0 | 0.3333 | 0.3181 | 0.2283 |
| I_intensified | 1.0 | 1.0 | 1.0 | 0.3638 | 0.3406 | 0.2538 |
| J_contrastive | 1.0 | 1.0 | 1.0 | 0.1682 | 0.3535 | 0.205 |
| K1_cancellation | 1.0 | 0.7381 | 0.9762 | 0.3777 | 0.093 | 0.1589 |
| K2_compound | 1.0 | 0.8 | 1.0 | 0.2559 | 0.2696 | 0.1048 |
| L_scope_position | 1.0 | 1.0 | 1.0 | 0.3166 | 0.3009 | 0.1262 |
| M_modal | 1.0 | 1.0 | 1.0 | 0.2955 | 0.3525 | 0.1345 |

### 4. Intensity: hedged vs intensified (LOOCV projections)

| model | axis | bucket | n | mean delta | std | median |
| --- | --- | --- | --- | --- | --- | --- |
| minilm | mean_diff | hedged | 90 | 0.3333 | 0.1073 | 0.2991 |
| minilm | mean_diff | intensified | 84 | 0.3638 | 0.118 | 0.3228 |
| minilm | mean_diff | neutral | 216 | 0.2763 | 0.1562 | 0.2642 |
| minilm | mean_diff | partial | 85 | 0.3696 | 0.1117 | 0.3606 |
| minilm | mean_diff | family:H_hedged | 90 | 0.3333 | 0.1073 | 0.2991 |
| minilm | mean_diff | family:I_intensified | 84 | 0.3638 | 0.118 | 0.3228 |
| minilm | probe | hedged | 90 | 0.3386 | 0.0927 | 0.3118 |
| minilm | probe | intensified | 84 | 0.3595 | 0.1043 | 0.3224 |
| minilm | probe | neutral | 216 | 0.2668 | 0.1436 | 0.2703 |
| minilm | probe | partial | 85 | 0.3568 | 0.0936 | 0.3606 |
| minilm | probe | family:H_hedged | 90 | 0.3386 | 0.0927 | 0.3118 |
| minilm | probe | family:I_intensified | 84 | 0.3595 | 0.1043 | 0.3224 |
| mpnet | mean_diff | hedged | 90 | 0.3181 | 0.1057 | 0.3258 |
| mpnet | mean_diff | intensified | 84 | 0.3406 | 0.1374 | 0.3231 |
| mpnet | mean_diff | neutral | 216 | 0.2306 | 0.1867 | 0.243 |
| mpnet | mean_diff | partial | 85 | 0.3263 | 0.1436 | 0.3218 |
| mpnet | mean_diff | family:H_hedged | 90 | 0.3181 | 0.1057 | 0.3258 |
| mpnet | mean_diff | family:I_intensified | 84 | 0.3406 | 0.1374 | 0.3231 |
| mpnet | probe | hedged | 90 | 0.3179 | 0.1005 | 0.3307 |
| mpnet | probe | intensified | 84 | 0.3304 | 0.1262 | 0.3091 |
| mpnet | probe | neutral | 216 | 0.2208 | 0.1759 | 0.2314 |
| mpnet | probe | partial | 85 | 0.3215 | 0.1337 | 0.3178 |
| mpnet | probe | family:H_hedged | 90 | 0.3179 | 0.1005 | 0.3307 |
| mpnet | probe | family:I_intensified | 84 | 0.3304 | 0.1262 | 0.3091 |
| bert | mean_diff | hedged | 90 | 0.2283 | 0.0677 | 0.2347 |
| bert | mean_diff | intensified | 84 | 0.2538 | 0.0645 | 0.2518 |
| bert | mean_diff | neutral | 216 | 0.1195 | 0.0727 | 0.1209 |
| bert | mean_diff | partial | 85 | 0.1561 | 0.0488 | 0.1495 |
| bert | mean_diff | family:H_hedged | 90 | 0.2283 | 0.0677 | 0.2347 |
| bert | mean_diff | family:I_intensified | 84 | 0.2538 | 0.0645 | 0.2518 |
| bert | probe | hedged | 90 | 0.2178 | 0.0599 | 0.2199 |
| bert | probe | intensified | 84 | 0.236 | 0.0578 | 0.2352 |
| bert | probe | neutral | 216 | 0.1212 | 0.0703 | 0.1212 |
| bert | probe | partial | 85 | 0.1588 | 0.0467 | 0.1514 |
| bert | probe | family:H_hedged | 90 | 0.2178 | 0.0599 | 0.2199 |
| bert | probe | family:I_intensified | 84 | 0.236 | 0.0578 | 0.2352 |

Intensified vs hedged, ROC-AUC of the projection shift (0.5 = indistinguishable):

| model | axis | AUC |
| --- | --- | --- |
| minilm | mean_diff | 0.5944 |
| minilm | probe | 0.5651 |
| mpnet | mean_diff | 0.5307 |
| mpnet | probe | 0.516 |
| bert | mean_diff | 0.6086 |
| bert | probe | 0.5919 |

### 5. Lexical cue-lookup baseline (no embeddings at all)

| group | n | pairwise acc (more cues in variant) | frac tied | mean cue delta |
| --- | --- | --- | --- | --- |
| explicit | 301 | 1.0 | 0.0 | 1.0 |
| implicit | 100 | 0.78 | 0.22 | 0.78 |
| cancellation | 42 | 1.0 | 0.0 | 1.9048 |
| compound | 5 | 1.0 | 0.0 | 1.6 |
| other | 6 | 1.0 | 0.0 | 1.0 |
| control | 21 | 0.0476 | 0.8571 | -0.0476 |

Embedding axis vs word lookup, same pairwise task (LOOCV for the axis):

| group | lexical | minilm mean_diff | mpnet mean_diff | bert mean_diff | minilm probe | mpnet probe | bert probe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| explicit | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| implicit | 0.78 | 0.98 | 0.95 | 0.99 | 0.98 | 0.96 | 0.99 |
| cancellation | 1.0 | 1.0 | 0.7381 | 0.9762 | 1.0 | 0.7381 | 0.9762 |
| compound | 1.0 | 1.0 | 0.8 | 1.0 | 1.0 | 0.8 | 1.0 |
| other | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| control | 0.0476 | 0.7143 | 0.5714 | 0.4286 | 0.6667 | 0.619 | 0.5714 |

### Held-out de-negation records (operation = affirm, n = 1)

| model | base | variant | cosine | delta on negation axis |
| --- | --- | --- | --- | --- |
| minilm | The task is impossible. | The task is possible. | 0.6776 | -0.4511 |
| mpnet | The task is impossible. | The task is possible. | 0.4276 | -0.4661 |
| bert | The task is impossible. | The task is possible. | 0.7505 | -0.2727 |
