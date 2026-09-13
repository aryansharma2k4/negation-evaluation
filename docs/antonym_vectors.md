# Predicting an antonym's embedding

Can a learned map over word vectors produce the opposite of a word, for the
words WordNet does not cover?

**No.** The best method reaches **0.198 precision@1** against WordNet's
**0.999**, and roughly three quarters of its top-1 answers are neither the
antonym nor anything close to it. It does beat WordNet on *coverage* — it fires
on 100% of frequent words against WordNet's 16.6% — but at that precision the
extra coverage is not worth having, and the module is wired in disabled.

This was the expected outcome; the brief said to expect it. The useful output is
a measured account of *how* it fails, and a check that the failure is the
phenomenon rather than the implementation.

Reproduce:

```bash
./venv/bin/python -m negation.antonym_vec.fetch                    # GloVe 6B, 822 MB
./venv/bin/python scripts/antonym_experiment.py --examples         # headline
./venv/bin/python scripts/antonym_experiment.py --pool lemmas      # comparable pool
./venv/bin/python scripts/antonym_experiment.py --source contextual --pool lemmas
```

---

## 1. Why this is hard, as a measurement

Distributional embeddings are built from context, and antonyms share contexts
almost perfectly — *hot* and *cold* both modify *water*, *day*, *weather*. The
space places them **close together**:

```
cos(hot, cold)  = 0.509    GloVe 6B 300d
                = 0.715    BERT base, last-4 layers, template sentence
cos(good, bad)  = 0.727    BERT
cos(happy, unhappy) = 0.816  BERT
```

Two consequences, both visible in the results below.

`f(v) = -v` cannot work: word vectors occupy a narrow cone, so `-v` points into
a region empty of real words. And *any* nearest-neighbour retrieval is fighting
the geometry — the true antonym is close to the query, but so is every synonym,
and the synonyms are usually closer. That is why contamination is reported
beside precision throughout rather than precision alone.

Note that the contextual source makes this **worse**, not better. `bert-base`
puts *happy* and *unhappy* at 0.816 cosine. Part of that is the shared template
sentence, which is a confound of the type-vector construction rather than a
property of the model, but it is not a promising starting point either way.

## 2. The gap this would have to fill

The fallback only fires where WordNet has no antonym. Over the adjective and
verb tokens `E_antonym` considers eligible, across the project's base corpus
plus a sample of technical sentences:

```
eligible tokens                     36
  WordNet has an antonym            18   (50.0%)
  WordNet returns nothing           18   (50.0%)
```

And over 1,000 frequent vocabulary words queried as adjectives, WordNet supplies
an antonym for only **16.6%**.

The words it draws a blank on are exactly the interesting ones: *sonorous*,
*ornate*, *sluggish*, *deterministic*, *idiomatic*, *crimson*, *negligible*, and
verbs such as *sort*, *handle*, *solve*, *return*. A method that worked would
roughly double `E_antonym`'s reach. That is the prize, and it is why the negative
result is worth establishing properly rather than assuming.

## 3. Setup

**Data.** WordNet adjective, verb and adverb antonym links, both directions,
single-word lemmas only.

```
all pairs                4,976        lemmas   4,558
  train                  3,976        lemmas   3,645
  test                   1,000        lemmas     913
  lemma leakage              0
after filtering to GloVe's vocabulary:  2,954 train / 742 test pairs
evaluated queries (one per query word):   676
```

The split is **by lemma, not by pair**, assigning whole connected components of
the antonymy graph. Antonymy is many-to-one — *good* is antonymous with *bad*,
*evil* and *ill* — so a per-pair split would put *good/bad* in training and
*good/evil* in test, and a model that merely memorised where *good* points would
score without generalising. Component assignment also costs nothing: no pair
straddles the boundary, where a per-lemma random split would discard about a
third of the data and leave only ~4% as test.

**Scoring.** One row per query word, not per pair, so a word with three antonyms
does not count three times. Credit is given for *any* gold antonym of the query,
so *active → inactive* is correct even when the row displays *dormant*.

**Controls.** 14,145 synonym pairs (contamination) and 4,554 within-POS random
pairs (the floor).

**Candidate pool.** 51,356 words — the 50k most frequent plus every dataset
lemma. This is the realistic setting. Restricting candidates to the ~900 test
lemmas would multiply every precision several-fold and measure nothing a
deployed module could rely on.

**Tuning.** Ridge's `alpha` was selected on a split of the *training* pairs
(chosen: 10.0). Nothing was selected on test, and no number below was tuned
toward.

## 4. The implementations are correct

The results are poor, so the learnable methods were checked against a synthetic
task that genuinely *is* a linear reflection — vectors paired with their
reflections across a fixed hyperplane, same dimensionality, same training size:

| method | P@1 on the synthetic task |
| --- | ---: |
| `linear_map` | 40/40 |
| `mlp_map` | 40/40 |
| `reflection` | 40/40 |

All three recover it perfectly. Whatever fails on real embeddings is not a
gradient bug, a normalisation slip, or a broken retrieval loop. Pinned in
`tests/test_antonym_vec.py`.

## 5. Results

GloVe 6B 300d, 51,356 candidates, 676 queries.

| method | P@1 | P@5 | P@10 | synonym@1 | synonym@10 | coverage | median rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_wordnet` | **0.999** | 0.999 | 0.999 | 0.000 | 0.000 | 0.999 | 1 |
| `baseline_negate` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | >10,000 |
| `linear_map` | 0.123 | 0.253 | 0.306 | 0.022 | 0.059 | 1.000 | 73 |
| `mlp_map` | 0.170 | 0.302 | 0.351 | 0.022 | 0.068 | 1.000 | 54 |
| `reflection` | **0.198** | 0.346 | 0.398 | 0.050 | 0.172 | 1.000 | 39 |
| `counterfit_style` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | >10,000 |
| `counterfit_transductive` † | 0.166 | 0.222 | 0.247 | 0.000 | 0.000 | 1.000 | — |

† not a generalisation result; see §8.

**`-v` scores exactly zero.** Not "weak" — zero, at every k, with the gold
antonym beyond rank 10,000. It is included as a baseline to be beaten and it is
beaten by everything, which is the one unambiguous finding here.

**The most constrained model wins.** `reflection` (one hyperplane per POS,
`d + 1` parameters) beats the MLP and beats ridge's full `d × d` map. That
ordering is what you expect when the binding constraint is data: 90,000
parameters fitted from 2,954 examples is a bad ratio, and the extra capacity
buys nothing. It also has the right inductive bias — a reflection is an
involution, so the antonym of the antonym is the original word, which is what
antonymy actually does.

### By part of speech

| | `linear_map` | `mlp_map` | `reflection` | `reflection` synonym@1 |
| --- | ---: | ---: | ---: | ---: |
| adjective (n=505) | 0.139 | 0.192 | **0.216** | 0.032 |
| adverb (n=77) | 0.104 | 0.143 | 0.169 | 0.104 |
| verb (n=94) | 0.053 | 0.074 | 0.128 | 0.106 |

Verbs are hardest and most contaminated. Verbal antonymy is frequently
directional rather than polar (*rise/fall*, *buy/sell*), and the direction is
exactly what a context-distributional vector does not encode — *buy* and *sell*
occur in near-identical contexts.

### By formation: the expected result reverses

| | `linear_map` | `mlp_map` | `reflection` |
| --- | ---: | ---: | ---: |
| morphological (n=362) — *happy/unhappy* | 0.124 | 0.171 | 0.157 |
| suppletive (n=328) — *hot/cold* | 0.128 | 0.171 | **0.247** |

The split was added expecting morphological pairs to be the easy ones, since
they are recoverable by string edit. **The opposite holds**: the best method
scores 0.247 on suppletive pairs against 0.157 on morphological ones.

Frequency dominates formation. *hot* and *cold* are common words with
well-estimated vectors; *unabridged*, *nonadsorbent* and *adsorbent* are rare
and their vectors are noise. Morphological antonymy is over-represented among
rare technical adjectives, so the slice that should have been easy is the slice
with the worst embeddings.

This matters for the deployment case, and not in a good way: the morphological
cases are the ones `D_affixal` already handles by string rule, and they are the
ones the vector method is *worse* at. There is no division of labour in which
the vector method covers what the rule cannot.

## 6. Synonym contamination

The headline contamination rate is modest — `reflection` returns an outright
synonym as its top answer 5.0% of the time, and 17.2% of its top-10 lists
contain one. But read it against precision:

| method | P@1 | synonym@1 | neither (%) |
| --- | ---: | ---: | ---: |
| `linear_map` | 0.123 | 0.022 | 85.5 |
| `mlp_map` | 0.170 | 0.022 | 80.8 |
| `reflection` | 0.198 | 0.050 | **75.2** |

Three quarters of top-1 answers are *neither* the antonym nor a synonym of the
query. The dominant failure is not confusing opposites with near-synonyms; it is
returning something unrelated.

**Contamination rises with precision, which is the informative part.** `ridge`
has the lowest contamination *and* the lowest precision; `reflection` has the
highest of both. That is not a coincidence: heavy ridge regularisation shrinks
predictions toward the mean of the space, which is the high-frequency function
word region — safely far from the query's synonyms, and equally far from its
antonym. Avoiding contamination by not landing near the query's meaning at all
is not a virtue.

Contamination is also concentrated where meaning is hardest: verbs and adverbs
run 10.4–10.6% synonym@1 against adjectives' 3.2%.

## 7. What the failures look like

Asking the deployed configuration for opposites of words WordNet cannot handle:

```
ornate     -> elegant, adorned, sleek, spacious, shiny
crimson    -> purple, pink, dark, blue, yellow
sluggish   -> enough, so, very, even, less
sonorous   -> (out of vocabulary)
```

and from the test set, where the gold answer is known:

```
active     -> inactive, activity, actively, involved      (hit, then relatives)
inactive   -> active, deactivated, activated, reactivated (hit, then relatives)
passive    -> rather, manner, very, behavior              (linear_map: drift)
dormant    -> moribund, revived, frozen, lain             (miss)
abridged   -> faced, sea, headquarters, nearby            (-v: noise)
```

Four distinct failure modes, none of them near-misses:

1. **Synonym return** — *ornate → elegant*. Right neighbourhood, wrong pole.
   Spliced into a sentence this produces a variant that asserts roughly what the
   original did, while the record claims it is a negation. The dangerous one,
   and the reason for the synonym veto in §10.
2. **Co-hyponym return** — *crimson → purple*. Another member of the same
   contrast set. *crimson* has no opposite, and a method with no way to represent
   "this word has no antonym" returns a sibling instead.
3. **Morphological-relative return** — *active → activity, actively*;
   *inactive → deactivated, reactivated*. The retrieval finds the right *stem*
   and the wrong *word*, because inflected and derived forms of a stem cluster
   tightly. Prominent in the top-5 even when the top-1 is correct.
4. **Function-word drift** — *sluggish → enough, so, very*; *passive → rather,
   manner, very*. The prediction landed near the high-frequency centre of the
   space, which is what a regularised map does when it has no signal.

## 8. Counter-fitting cannot help here, structurally

`counterfit_style` scores **0.000** — identical to `-v`, which is what it
reduces to.

This is not an implementation failure. Counter-fitting moves the vectors of
words that appear in its constraints. The split is lemma-disjoint, so **no test
lemma appears in any training constraint**, so their vectors move only under the
space-preservation term that pulls every vector back toward its origin.
Measured: unconstrained words stay within 0.99 cosine of where they started
(asserted in `tests/test_antonym_vec.py`).

So it cannot improve generalisation to unseen words — by construction, not by
accident. The transductive variant, which is *given* the test antonym
constraints, reaches 0.166, confirming the machinery works and the problem is
that it has nothing to apply to an unseen word. **That row is not a
generalisation result and must not be read as one.**

This is worth stating carefully because counter-fitting is genuinely effective
at what Mrkšić et al. built it for: improving *similarity judgements* among
words you have constraints for. It is not, and was never claimed to be, a method
for *generating* antonyms of words you do not.

## 9. Static vs contextual

{{CONTEXTUAL}}

## 10. Confidence carries signal but does not make answers usable

Bucketing predictions by the cosine margin between the top candidate and the
runner-up — the quantity `api.py` turns into its confidence score — precision@1
rises monotonically:

| method | Q1 (lowest margin) | Q2 | Q3 | Q4 | Q5 (highest) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linear_map` | 0.037 | 0.037 | 0.111 | 0.141 | 0.287 |
| `mlp_map` | 0.052 | 0.082 | 0.147 | 0.178 | 0.390 |
| `reflection` | 0.067 | 0.067 | 0.200 | 0.207 | **0.449** |
| `baseline_negate` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

So the score is not noise — it rank-orders correctness, by a factor of about
seven from bottom quintile to top. A threshold on it is therefore worth having.

What it cannot do is make a suggestion trustworthy. **Even the most confident
fifth of predictions is wrong more often than right** (0.449). There is no
operating point on this curve that yields a usable generator: the threshold
selects the band where the model is least bad, not a band where it is good.

`MIN_CONFIDENCE = 0.15` is set on that basis, and deliberately high enough that
in practice the fallback declines almost everything it is asked.

## 11. Verdict

**Against WordNet on accuracy: no, decisively.** 0.198 against 0.999 P@1. Not
close, on a test set drawn from WordNet where WordNet is by construction
near-perfect.

**Against WordNet on coverage: yes, by 6×.** 100% versus 16.6% on frequent
vocabulary. This is the one axis where the module wins, and it is the axis the
brief hoped for.

**Net: neither, because coverage at this precision is negative value.** Adding
records that are wrong four times in five to a corpus whose other twelve
families are rule-exact would degrade it. The `E_antonym` fallback is wired in
but **disabled by default** (`NEGATION_ANTONYM_VEC=1` to enable), and every
record it can produce is tagged `generator="antonym_vec_v1"` with a confidence
so it can be filtered or weighted downstream rather than silently mixed in.

**Against the `-v` baseline: yes, decisively** — but that says more about `-v`
than about the learned maps. `-v` scores exactly zero.

The one genuinely positive finding is structural rather than practical: the most
constrained model (`reflection`, `d+1` parameters per POS) beats both the
unconstrained linear map and the MLP. If this were to be pushed further, the
direction indicated is *more* inductive bias, not more capacity.

### What is actually shipped

The module is complete, tested and off. With it enabled, a synonym veto runs on
every suggestion — any candidate WordNet lists as a synonym of the query, or
that shares a synset with it, is discarded outright rather than downweighted.
WordNet having no *antonym* for a word does not mean it has no *synonyms*, so
the veto is available exactly when the fallback fires. Between the veto and the
confidence floor, the deployed configuration declines nearly everything:
*ornate*, *sluggish*, *crimson*, *deterministic*, *idiomatic* and *sonorous* all
return `None`.

That is the correct behaviour. A module that cannot tell a real suggestion from
a filler one should produce nothing rather than fill the corpus with
unverifiable records.

## 12. What would change the answer

Being explicit, since a negative result is only useful if its scope is clear:

- **Vectors specialised for lexical contrast.** These experiments use GloVe, the
  classic "antonyms are close" case. Paragram-SL999 or published counter-fitted
  vectors start from a space where the problem is partly solved — but that would
  not help the deployment case, because the words needing opposites are exactly
  the ones absent from those resources' constraint sets. It would raise the
  benchmark number without raising the useful number.
- **An order of magnitude more training pairs.** 2,954 pairs for a 300×300 map
  is the binding constraint, visible in the function-word drift and in the
  constrained model winning. But WordNet *is* the source of antonym pairs, so
  there is nowhere obvious to get ten times more.
- **A different objective.** Every method here minimises `||f(v) - v_ant||`, as
  specified. A contrastive or ranking objective that explicitly pushes synonyms
  down would attack contamination directly instead of hoping regression avoids
  it. Given that contamination is only 5% of the error, this would help less
  than it sounds.
- **Not framing it as retrieval at all.** The honest division of labour is
  affixes by morphology (`D_affixal`, already exact), suppletive pairs by
  lexicon (`E_antonym` via WordNet, already exact), and nothing by vector
  arithmetic. The 50% of tokens WordNet cannot answer are, on this evidence, not
  answerable this way.
