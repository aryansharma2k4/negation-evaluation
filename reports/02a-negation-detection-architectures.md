# Is B the negation of A? Two architectures on real data

*Paths in the reproduction section refer to the repository layout at the commit hashes listed. See `docs/PATHS.md` for the current locations.*

Every number below comes from `results.json`, which `detect.py` writes; the
tables are printed from it by `report_tables.py`. CPU only; the one expensive
component is an NLI cross-encoder (~13 pairs/s on 16 threads).

**Short answer.** An off-the-shelf NLI cross-encoder wins, and it is not close:
CV ROC-AUC **0.993** against the next best **0.744**, and a paraphrase
false-positive rate of **3.0%** against the next best **36%**. Of the two
architectures asked for, the reference-set comparison (arch 2) beats the feature
scorer (arch 1) on the discrimination that matters — negation vs paraphrase,
0.744 vs 0.550 — but both are far behind the cross-encoder, and both would fail
a student for paraphrasing between one time in three and nine times in ten. The
hybrid helps on the mixed phase-2 set and *hurts* on phase 1. The negation axis
from `experiments/negation_direction` carries almost no signal on this data
(per-feature AUC 0.57 at best), and the lexical cue baseline carries none at all
(0.49) — the opposite of what the first experiment found on generator output.

---

## 0. Data: what exists, what it is licensed as, and what it is worth

Every source was checked before use. Nothing is vendored into the repo;
`data/raw/` is gitignored and `data.py` documents the download commands.

| source | used for | n used | licence |
| --- | --- | --- | --- |
| [CANNOT v1.1](https://github.com/dmlls/cannot-dataset) | phase 1 and most of phase 2 | 5,003 of 77,376 pairs | **CC BY-SA 4.0** (stated in its README) |
| [STS-Benchmark](https://huggingface.co/datasets/mteb/stsbenchmark-sts) test, score < 2 | dissimilar positives | 167 pairs | **unclear** — the HF mirror declares `license: unknown`; the original SemEval data carries per-source terms. Used for evaluation only, not redistributed. |
| [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) validation, label *neutral* | dissimilar positives | 167 pairs | **CC BY-SA 4.0** |
| random re-pairings of STS-B / SNLI sentences | dissimilar positives | 166 pairs | derived from the two above |
| [xNot360](https://huggingface.co/datasets/nguyenthanhasia/xNot360) | external test, run once | all 360 | **Apache-2.0 per the dataset card body**; the card's metadata block declares no licence, so this is a stated-but-not-machine-readable licence |

Two honest caveats about the data, both of which bound every number below:

1. **CANNOT's negations are largely machine-made.** It is a merge of five
   corpora, and the two that dominate what we sampled — WikiFactCheck-English
   (77% of pairs) and SemAntoNeg — were built by rule-based negation and antonym
   substitution. Results on it will overstate performance on real student
   writing, which negates by rephrasing, hedging and omission.
2. **The NLI cross-encoder has home-field advantage.** It was trained on
   SNLI+MNLI. Our SNLI-neutral negatives come from SNLI's validation split —
   same distribution as its training data — and WikiFactCheck-style pairs are
   close to MNLI's genre. Its 0.993 on phase 1 should be read as "a model
   trained on exactly this kind of contradiction detection does well at
   contradiction detection", not as a measurement of a general advantage.

### Source attribution inside CANNOT

The released `.tsv` has no provenance column, but the repository ships the
per-source processed files it was merged from, so the source of each pair is
recoverable by exact text match — 99.99% of rows; 11 are tagged `unattributed`.
Phase-1 and phase-2 sampling is stratified by that source, proportionally:

| source | share of CANNOT | phase-1 negated | phase-1 paraphrase |
| --- | --- | --- | --- |
| wikifactcheck-english | 59,669 | 1,119 | 1,195 |
| antonym-substitution (SemAntoNeg) | 8,599 | 208 | 125 |
| sentiment-labelled-sentences | 8,353 | 158 | 166 |
| glue-diagnostic | 432 | 6 | 11 |
| nan-nli | 316 | 9 | 3 |

### Negation type

CANNOT does not label how a pair negates, so `neg_type` is **derived**: SemAntoNeg
pairs are `antonym`; a pair whose lexicon cue count changes is `verbal`; a pair
where a replaced word is a WordNet antonym of the word it replaced is `antonym`;
the rest are `other`. On phase 1 that gives verbal 580, other 713, antonym 208 —
i.e. **only 39% of CANNOT's "negations" involve a negation cue at all.** That
single fact explains most of what follows.

### Splits and leakage

Phase 1 and phase 2 are disjoint at the level of *sentences*, not just of A:
pairs are grouped by normalised A, each group is assigned to one phase whole,
and a pair is dropped if either of its sentences was already claimed by the other
phase. Measured overlap is 0 sentences, and a pytest asserts it. Cross-validation
is 5-fold grouped by A, and every threshold, axis, scaler and logistic regression
is fit on training rows only. xNot360 was scored once, at the end.

---

## 1. The questions, answered

### Which architecture wins on phase 2?

**Architecture 3 (hybrid), among the three asked for — and all three lose to the
NLI baseline.**

| | phase-1 CV AUC | phase-2 AUC | phase-2 paraphrase FPR | xNot360 AUC |
| --- | --- | --- | --- | --- |
| arch 1 — feature scorer | 0.550 | 0.734 | 0.934 | 0.364 |
| arch 2 — refset margin (cosine) | **0.744** | 0.686 | 0.395 | 0.642 |
| arch 2 — refset margin (resid+polarity) | 0.737 | 0.677 | **0.361** | 0.624 |
| arch 3 — hybrid | 0.645 | **0.777** | 0.717 | 0.371 |
| baseline — NLI P(contradiction) | **0.993** | **0.834** | **0.030** | **0.940** |

Read the two AUC columns together, because they measure different things.
Phase 1 is negation vs *paraphrase* — both sentences are about the same thing,
which is the hard case and the one grading actually faces. Phase 2 adds 500
unrelated pairs, which are easy, so phase-2 AUC rewards a model for noticing
relatedness. Architecture 1 scores 0.734 on phase 2 while scoring 0.550 on
phase 1: it is detecting *topical relatedness*, not negation. Its single most
useful feature on phase 2 is `c_raw` (AUC 0.738) — plain cosine similarity.

So the honest ranking on the question asked ("is B the negation of A") is
architecture 2, then architecture 3, then architecture 1 — and the NLI
cross-encoder above all of them by a distance no amount of feature engineering
here closed.

### Does the reference-set comparison fail on paraphrased student answers?

**Yes, but less badly than the alternative — and badly enough to be unusable.**

Architecture 2 marks 36–40% of correct-but-paraphrased answers as negations.
Architecture 1 marks 93%. The NLI cross-encoder marks 3%.

Two different failures produce that number, and the margins plot separates them.

For **paraphrases the margin mostly gets the sign right**: 81% of paraphrase
pairs score a negative margin (`s_same` wins) against 65% of negations scoring
positive. What converts that into a 40% false-positive rate is *where the
threshold lands*. Fit to maximise F1 on a 50/50 phase-1 training set, it sits at
−0.017, just below zero, so every paraphrase whose margin is a small negative
number — and there are many, because B and A are near-identical — is called a
negation. At a threshold of exactly 0 the paraphrase false-positive rate would
be 17.8%, with negation recall falling to 64.6%. Either way the two
distributions overlap too much for grading; the point is that this is a margin
*calibration* problem, not an absent signal.

For **unrelated pairs the margin has no signal at all**, and there the
generator's set-size asymmetry does bite: 75% of random re-pairings score a
*positive* margin. The NEG set averages 16 sentences, the SAME set 2, so when B
matches nothing, the maximum over 16 candidates beats the maximum over 2 by
sampling alone. That is a property of the generator, not of the answer.

The residual-cosine-plus-polarity similarity helps slightly and consistently
(paraphrase FPR 0.361 vs 0.395, and a small loss of recall), which is the
opposite of what happens on the unrelated pairs, where both similarities fail
badly (FPR 0.78–0.89 on random re-pairings, because max-similarity margins are
noise when B matches nothing). The three-way decision was added for exactly this
and mostly fixes it: with the "unrelated" gate, 462 of 500 unrelated pairs are
classified correctly, and overall three-way accuracy is 0.754 — but it barely
changes the paraphrase problem, which is a margin problem, not a
similarity-floor problem.

### How often did generator coverage fail?

**3.1% of phase-1 pairs and 4.0% of phase-2 pairs have no NEG variant at all**
(94 and 80 pairs), and rather more — 73 and 67 — got no variants whatsoever,
usually because the sentence is a fragment or the parse produced no clause the
generators could attach to. On xNot360 coverage was perfect (0 failures), which
is unsurprising: its sentences are short, templated and well-formed.

Where the generator did fire it fired generously: mean NEG set 16.3 sentences,
mean SAME set 2.1. That imbalance is the bias described above; a fairer
architecture 2 would need the SAME set filled out with paraphrases, which the
stage-1 generator does not produce.

### Does anything beat the NLI cross-encoder?

**No.** On phase-1 CV it scores 0.993 AUC and 0.969 accuracy; the best thing we
built scores 0.744 and 0.688. On phase 2 it takes 0.834 AUC at a 3% paraphrase
false-positive rate. On xNot360 it takes 0.940 AUC — though only 0.744 accuracy,
because the threshold carried over from CANNOT is badly calibrated for
xNot360's logical near-misses.

Feeding the generator's reference sets *into* the cross-encoder (architecture 2
with NLI as the similarity function, on a 300-pair subsample with reference sets
capped at the 8 nearest by cosine) makes things worse, not better:

| subsample | arch 2 + NLI | plain NLI(A, B) on the same 300 pairs |
| --- | --- | --- |
| phase 1 | AUC 0.905, F1 0.865 | **AUC 0.997, F1 0.981** |
| phase 2 | AUC 0.842, F1 0.719 | **AUC 0.854, F1 0.860** |

Asking "does B entail one of my generated negations of A?" is strictly worse
than asking "does B contradict A?". The generator adds a lossy indirection.

One caveat already noted: the cross-encoder's training data covers this
territory. The result to trust is the *relative* one — nothing we built with
embeddings and lexicons came close — rather than the absolute 0.993.

### Does the negation axis from experiment 1 transfer to real data?

**No.** Per-feature ROC-AUC on phase 1, where the axis was fit:

| feature | phase 1 | phase 2 | xNot360 |
| --- | --- | --- | --- |
| delta_p (shift along the axis) | 0.566 | 0.509 | 0.443 |
| c_resid | 0.524 | 0.738 | 0.313 |
| c_raw | 0.523 | 0.738 | 0.315 |
| cue_delta (lexical) | 0.492 | 0.491 | 0.280 |

Mean `delta_p` is 0.0038 for paraphrases and 0.0130 for negations — a real
difference in the right direction, and far too small to separate anything. The
first experiment found a direction that ranked generator-made negations
perfectly; on CANNOT that same construction is worth 0.57 AUC. The reason is in
§0: only 39% of CANNOT's negations involve a cue at all, and the axis was fit on
pairs where the "negation" is often an antonym swap or a de-negation pointing
the other way, so the mean difference vector largely cancels.

`cue_delta` collapsing to 0.49 is the sharpest contrast with the first
experiment, where the same lexicon scored 1.00 on explicit pairs. The lexicon
was not good at negation; it was good at the generator, which builds negations
out of that lexicon.

### Where each approach's errors live

Recall on phase-2 negations by derived type:

| | verbal | antonym | other |
| --- | --- | --- | --- |
| arch 2 (cosine margin) | 0.876 | 0.640 | 0.705 |
| arch 3 hybrid | 0.890 | 0.669 | 0.891 |
| NLI baseline | 0.981 | 0.978 | 0.967 |

Architecture 2 does best exactly where the generator does: cue-based verbal
negation. It loses a third of the antonym pairs, which the generator's E family
can produce only when WordNet has the antonym. The cross-encoder is flat across
all three types.

---

## 2. Caveats that bound every number

- **CANNOT is rule-generated.** Its negations come from automatic negation and
  antonym substitution over Wikipedia-derived claims. Real student answers
  paraphrase, hedge, omit and misspell. Expect every number here to be optimistic
  for the grading use case, most of all the NLI baseline's.
- **The NLI model trained on this kind of data** (SNLI + MNLI), and 167 of our
  500 dissimilar positives are SNLI validation pairs. Its margin over the others
  is real but partly a home-ground effect.
- **Architecture 2 under the NLI similarity is a 300-pair subsample** per set
  with reference sets capped at 8, because the full version is ~110k
  cross-encoder passes (hours on this CPU). Its numbers carry the corresponding
  uncertainty; the plain-NLI column in that table is computed on the *same* 300
  pairs so the comparison is fair.
- **One embedding model.** all-MiniLM-L6-v2, chosen for CPU budget
  (`--model mpnet` is wired up and was not run). The first experiment found
  mpnet behaves differently on cancellation, so architecture 1's numbers might
  shift somewhat; nothing suggests they would shift by the 0.25 AUC needed to
  matter.
- **Thresholds maximise F1 on training rows.** For grading, the right objective
  is a bound on the paraphrase false-positive rate, which would move every
  threshold and lower every recall.
- **`neg_type` is derived, not annotated** (§0), so the per-type breakdown is a
  heuristic partition, not ground truth.

---

## 3. What this means for the grading mechanism

1. **Don't build the scorer out of a negation direction.** Two experiments now
   say the same thing from opposite directions: the axis is excellent on
   sentences the stage-1 generator made and worth ~0.57 AUC on sentences it
   didn't.
2. **A cross-encoder is the default, not the fallback.** 3% paraphrase
   false-positive rate against 36% for the best thing we built. If a scoring
   mechanism ships, it should start from an NLI cross-encoder and be judged
   against it, not against cosine.
3. **The generator's place is as a data source, not a runtime oracle.** Its
   reference sets are lexically lopsided (16 near-copies vs 2), which is what
   produces the paraphrase false positives, and it cannot cover 3–4% of real
   inputs at all. It is, on the other hand, exactly the right tool for making
   training data for a small cross-encoder that would be cheaper to run.
4. **A usable grader needs the three-way decision**, or something like it: 500
   of our 2,001 phase-2 pairs are neither a negation nor a paraphrase, and a
   binary scorer has to call those something. The max-similarity gate handles
   them at 0.754 three-way accuracy; that is a start, not a solution.

---

## 4. Reproducing

```bash
# data: downloads CANNOT, xNot360, STS-B, SNLI into data/raw (gitignored)
#   see data.py's docstring for the exact URLs and licences
./venv/bin/python experiments/negation_detection/data.py

# reference sets: runs the stage-1 generator over every A (~4 min, unmodified)
./venv/bin/python experiments/negation_detection/refsets.py

# the experiment (~25 min cold on CPU, most of it the cross-encoder; ~8 min warm)
./venv/bin/python experiments/negation_detection/detect.py
./venv/bin/python experiments/negation_detection/detect.py --model mpnet --no-nli

# tables below
./venv/bin/python experiments/negation_detection/report_tables.py

# guardrails: phase leakage, fold grouping, NEG/SAME polarity rule, cache norms
./venv/bin/python -m pytest experiments/negation_detection -q
```

Plots in `plots/`: `scores_<model>.png` (phase-2 score histograms per subtype for
every scorer, with the threshold carried over from phase 1),
`resid_vs_deltap_<model>.png` (the two embedding features against each other —
the negation and paraphrase clouds sit on top of one another, which is the whole
story of architecture 1), `margins_<model>.png` (NEG-minus-SAME margins by
subtype: the paraphrase mass piled up just left of zero, with the fitted
threshold at −0.017 cutting through it, is architecture 2's false-positive
problem).

---

## 5. All numbers

`results.json` holds every value. The tables below are printed from it by
`report_tables.py`.

### Data

| set | n | composition |
| --- | --- | --- |
| phase1 | 3002 | negation: 1501, paraphrase: 1501 |
| phase2 | 2001 | negation: 1000, paraphrase: 501, random_pair: 166, snli_neutral: 167, sts_low: 167 |
| xnot360 | 360 | negation: 180, not_negation: 180 |

Phase-1 negated pairs by derived negation type: antonym: 208, other: 713, verbal: 580.

### Generator coverage (architecture 2's dependency)

| set | pairs | unique A | A with no NEG variant | rate | no variants at all | mean NEG set | mean SAME set |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase1 | 3002 | 2102 | 94 | 0.0313 | 73 | 16.26 | 2.1 |
| phase2 | 2001 | 1642 | 80 | 0.04 | 67 | 15.94 | 1.73 |
| xnot360 | 360 | 180 | 0 | 0.0 | 0 | 19.37 | 2.49 |

### Phase 1: 5-fold CV, grouped by sentence A

| scorer | AUC (fold mean ± sd) | accuracy | precision | recall | F1 |
| --- | --- | --- | --- | --- | --- |
| arch 1 — feature scorer | 0.55 ± 0.0394 | 0.5093 | 0.5048 | 0.9587 | 0.6613 |
| arch 2 — refset margin (cosine) | 0.7444 ± 0.0013 | 0.6879 | 0.6679 | 0.7477 | 0.7054 |
| arch 2 — refset margin (resid+polarity) | 0.7374 ± 0.0047 | 0.6797 | 0.6617 | 0.7365 | 0.6968 |
| arch 3 — hybrid | 0.6448 ± 0.0348 | 0.5525 | 0.5364 | 0.8666 | 0.6597 |
| baseline — raw cosine | 0.4781 ± 0.0207 | 0.5026 | 0.5013 | 0.9993 | 0.6676 |
| baseline — lexical cue delta | 0.4916 ± 0.0205 | 0.5003 | 0.5001 | 0.9979 | 0.6663 |
| baseline — NLI P(contradiction) | 0.9926 ± 0.0028 | 0.9688 | 0.9681 | 0.9696 | 0.9688 |

Logistic-regression coefficients (standardised features, mean over folds):


*arch 1 — feature scorer*

| p_a | p_b | delta_p | c_raw | c_resid | cue_delta |
| --- | --- | --- | --- | --- | --- |
| -0.121 | 0.0342 | 0.2156 | -0.9997 | 1.2333 | 0.003 |

*arch 3 — hybrid*

| p_a | p_b | delta_p | c_raw | c_resid | cue_delta | margin_cos | margin_res |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.1373 | 0.0361 | 0.2413 | -1.0259 | 1.3521 | 0.0061 | 3.6972 | -3.591 |

### Phase 2: trained on all of phase 1, evaluated once

| scorer | AUC | accuracy | precision | recall | F1 | **paraphrase FPR** |
| --- | --- | --- | --- | --- | --- | --- |
| arch 1 — feature scorer | 0.7341 | 0.6737 | 0.6127 | 0.943 | 0.7428 | **0.9341** |
| arch 2 — refset margin (cosine) | 0.6861 | 0.5922 | 0.5682 | 0.766 | 0.6525 | **0.3952** |
| arch 2 — refset margin (resid+polarity) | 0.6765 | 0.6082 | 0.5856 | 0.739 | 0.6534 | **0.3613** |
| arch 3 — hybrid | 0.7773 | 0.7131 | 0.6646 | 0.86 | 0.7498 | **0.7166** |
| baseline — raw cosine | 0.2619 | 0.5022 | 0.501 | 0.998 | 0.6671 | **0.986** |
| baseline — lexical cue delta | 0.4912 | 0.4998 | 0.4997 | 0.995 | 0.6653 | **0.992** |
| baseline — NLI P(contradiction) | 0.8336 | 0.8271 | 0.7527 | 0.974 | 0.8492 | **0.0299** |

False-positive rate by negative subtype (lower is better; paraphrase is the one that matters for grading):

| scorer | paraphrase | sts_low | snli_neutral | random_pair | recall on negations |
| --- | --- | --- | --- | --- | --- |
| arch 1 — feature scorer | 0.9341 | 0.2455 | 0.479 | 0.0422 | 0.943 |
| arch 2 — refset margin (cosine) | 0.3952 | 0.6946 | 0.7246 | 0.8855 | 0.766 |
| arch 2 — refset margin (resid+polarity) | 0.3613 | 0.6287 | 0.6407 | 0.7831 | 0.739 |
| arch 3 — hybrid | 0.7166 | 0.1916 | 0.1916 | 0.0663 | 0.86 |
| baseline — raw cosine | 0.986 | 1.0 | 1.0 | 1.0 | 0.998 |
| baseline — lexical cue delta | 0.992 | 0.994 | 1.0 | 1.0 | 0.995 |
| baseline — NLI P(contradiction) | 0.0299 | 0.8144 | 0.0958 | 0.9217 | 0.974 |

Recall on negated pairs by derived negation type:

| scorer | verbal | antonym | other |
| --- | --- | --- | --- |
| arch 1 — feature scorer | 0.939 | 0.8345 | 0.98 |
| arch 2 — refset margin (cosine) | 0.8756 | 0.6403 | 0.7051 |
| arch 2 — refset margin (resid+polarity) | 0.8634 | 0.6475 | 0.6541 |
| arch 3 — hybrid | 0.8902 | 0.6691 | 0.8914 |
| baseline — raw cosine | 0.9976 | 1.0 | 0.9978 |
| baseline — lexical cue delta | 0.9878 | 1.0 | 1.0 |
| baseline — NLI P(contradiction) | 0.9805 | 0.9784 | 0.9667 |

### xNot360 (external, evaluated once at the end)

| scorer | AUC | accuracy | precision | recall | F1 |
| --- | --- | --- | --- | --- | --- |
| arch 1 — feature scorer | 0.3644 | 0.5 | 0.5 | 0.9944 | 0.6654 |
| arch 2 — refset margin (cosine) | 0.6417 | 0.5167 | 0.5086 | 0.9889 | 0.6717 |
| arch 2 — refset margin (resid+polarity) | 0.6238 | 0.5417 | 0.5225 | 0.9667 | 0.6784 |
| arch 3 — hybrid | 0.3708 | 0.4917 | 0.4956 | 0.9444 | 0.6501 |
| baseline — raw cosine | 0.6854 | 0.5 | 0.5 | 1.0 | 0.6667 |
| baseline — lexical cue delta | 0.2797 | 0.5 | 0.5 | 1.0 | 0.6667 |
| baseline — NLI P(contradiction) | 0.9395 | 0.7444 | 0.663 | 0.9944 | 0.7956 |

### Architecture 2 with an NLI similarity function (subsample)

| set | n | arch 2 + NLI: AUC | F1 | plain NLI(A,B) on the same pairs: AUC | F1 |
| --- | --- | --- | --- | --- | --- |
| phase1 | 300 | 0.9054 | 0.865 | 0.9972 | 0.981 |
| phase2 | 300 | 0.8422 | 0.7193 | 0.8537 | 0.8597 |

Mean max-similarity to the NEG set vs the SAME set: phase1: 0.5333 vs 0.4295; phase2: 0.4446 vs 0.2874.

### Three-way decision (architecture 2, cosine margin)

Thresholds: unrelated below max-similarity 0.6685 (5th percentile of training pairs), negation above margin -0.0169. Overall accuracy **0.7541**.

| gold \ predicted | negation | agrees | unrelated |
| --- | --- | --- | --- |
| negation | 757 | 225 | 18 |
| agrees | 181 | 290 | 30 |
| unrelated | 20 | 18 | 462 |

### Where the architecture-2 margins fall (phase 2, cosine margin)

| subtype | n | margin > 0 | margin = 0 | margin < 0 | above the fitted threshold |
| --- | --- | --- | --- | --- | --- |
| negation | 1000 | 0.646 | 0.01 | 0.344 | 0.766 (thr -0.0169) |
| paraphrase | 501 | 0.1776 | 0.008 | 0.8144 | 0.3952 (thr -0.0169) |
| random_pair | 166 | 0.747 | 0.0 | 0.253 | 0.8855 (thr -0.0169) |
| snli_neutral | 167 | 0.5449 | 0.0 | 0.4551 | 0.7246 (thr -0.0169) |
| sts_low | 167 | 0.509 | 0.0 | 0.491 | 0.6946 (thr -0.0169) |

### What each feature is worth on its own (ROC-AUC)

| set | p_a | p_b | delta_p | c_raw | c_resid | cue_delta |
| --- | --- | --- | --- | --- | --- | --- |
| phase1 | 0.4581 | 0.4975 | 0.5661 | 0.5231 | 0.5237 | 0.4918 |
| phase2 | 0.4321 | 0.4308 | 0.5085 | 0.7381 | 0.7382 | 0.4912 |
| xnot360 | 0.5 | 0.467 | 0.4433 | 0.3146 | 0.3132 | 0.2797 |

Mean feature value by class:

| set | class | p_a | p_b | delta_p | c_raw | c_resid | cue_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase1 | not negation | -0.0393 | -0.0354 | 0.0038 | 0.8402 | 0.8401 | 0.0027 |
| phase1 | negation | -0.0495 | -0.0365 | 0.013 | 0.8587 | 0.8588 | -0.0167 |
| phase2 | not negation | -0.0268 | -0.0246 | 0.0022 | 0.5697 | 0.5693 | -0.002 |
| phase2 | negation | -0.0426 | -0.0409 | 0.0017 | 0.851 | 0.851 | -0.02 |
| xnot360 | not negation | -0.0516 | -0.0744 | -0.0228 | 0.9274 | 0.9278 | 1.3667 |
| xnot360 | negation | -0.0516 | -0.0824 | -0.0308 | 0.9117 | 0.912 | 0.9222 |
