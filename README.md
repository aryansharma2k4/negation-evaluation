# Negation-variant generation

Given a base sentence of **any** type — affirmative or already negated,
declarative, interrogative, imperative or exclamative, active or passive — apply
exactly one negation-transforming operation and tag each result with the
metadata a graded negation-sensitivity study needs: which cue realises the
negation, where it sits, what it scopes over, and how many negations survive.

No LLM is involved at this stage. Verification is stage 2; every record carries
a `verified: Optional[bool]` field that is `None` until then.

## Install

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python -c "import nltk; nltk.download('wordnet')"
```

## Run

```bash
./run.sh
```

With no arguments this runs on `data/sample_sentences.txt` and writes
`variants.jsonl`. To use your own sentences (one per line):

```bash
./run.sh my_sentences.txt -o my_variants.jsonl
```

Either way it writes one JSON record per line and prints counts per family, per
family/subtype, per `net_negation`, and the reason every filtered-out candidate
was dropped.

```python
from negation import generate_all
records = generate_all(["The function sorts the array."])
```

Tests: `./venv/bin/python -m pytest tests -q` (206 tests)

`run.sh` is a one-line wrapper around `./venv/bin/python -m negation.cli`.

## Record schema

`negation/schema.py`. One `NegationVariant` per generated sentence:

| field | meaning |
| --- | --- |
| `base_id` / `base_sentence` | content-derived id, stable across reruns |
| `variant` | the generated sentence |
| `family` / `subtype` | `A_syntactic` … `M_modal`, and the rule within it |
| `cue_tokens` / `cue_char_spans` / `cue_count` | the negation cues, one span per token, offsets into `variant` |
| `net_negation` | cancellation-aware: `0` for K1, `2` for K2, `1` otherwise |
| `scope_target` | `subject` \| `predicate` \| `clause` |
| `intensity_hint` | `neutral` \| `hedged` \| `intensified` \| `partial` |
| `generator` | e.g. `syn_do_support_v1` |
| `operation` | `negate` \| `affirm` \| `rescope` |
| `input_polarity` / `input_cue_count` | what the *input* was: the `N` in the depth-1 rule |
| `clause_type` / `voice` / `modality` | of the input clause |
| `target_clause_idx` | which clause this record negated, as an index into the clause heads |
| `depth` | operations stacked from an affirmative baseline (hard cap 2) |
| `op_depth` | operations applied *to the input* (1 for the whole arbitrary-input layer) |
| `verified` | stage-2 hook, always `None` here |

### `depth` vs `op_depth`

They count different things and both are wanted:

```
"The task is impossible."  ->  "The task is not impossible."
    depth    = 2   two cues away from a fully affirmative sentence
    op_depth = 1   one operation applied to the sentence we were handed
```

`depth` is the composition cap from stage 1 and stays as it was. `op_depth` is
the depth-1 contract: every generator in the arbitrary-input layer declares
`op_depth = 1`, enforced in `Generator.__init_subclass__`.

The operative form of the rule is **an input carrying `N` cues yields a variant
carrying at most `N + op_depth`**, checked in `filters.check_operation_depth`
against the *re-parsed* variant using the same `classify_input` that counted the
input — so a multiword cue such as *by no means* counts once on both sides
rather than as its three `cue_char_spans`. It **raises `DepthViolation`** rather
than filtering: a generator that miscounts its own operations is a bug, and a
silent drop would hide it.

`cue_tokens` and `cue_char_spans` are read straight off the splice, so
`variant[start:end] == token` holds for every record — asserted in the tests.

## Architecture

```
negation/
  nlp_core.py    one spaCy pipeline, cached WordNet, morphology, dep helpers
  lexicons.py    every word list, frozen at import time
  affixes.py     affixal derivation, in both directions, WordNet-gated
  splice.py      character-offset editing; the only way variants are built
  frames.py      clause frames — the shared substrate for clausal negation
  polarity.py    what negative material a clause already carries
  classify.py    InputProfile: what kind of sentence are we being handed?
  scope.py       the K1-vs-K2 containment decision
  base.py        Generator ABC + registry + record construction
  generators/    one module per family, plus the arbitrary-input layer
  filters.py     dedup → blacklist → batched re-parse → depth guardrail
  driver.py      generate_all
  cli.py         JSONL + summary
  data/          third-party corpora: verified sources, loaders, common schema
  antonym_vec/   predicting an antonym's embedding — see the verdict below
```

## Input classification

`classify.classify_input(doc)` returns an `InputProfile` describing the sentence
before anything is generated. It decides three things: which operation is even
available, where a cue may land, and what `N` is in the depth-1 rule.

| field | values |
| --- | --- |
| `clause_type` | `declarative` \| `interrogative` \| `imperative` \| `exclamative` |
| `voice` | `active` \| `passive` |
| `polarity` | `affirmative` \| `negated` |
| `existing_cues` | `(token_idx, cue_text, family_guess)` per cue, **all families** |
| `modality` | `none` \| `epistemic` \| `deontic` \| `ability` |
| `clause_count` / `has_subordinate` | clause structure |
| `is_existential` / `is_comparative` / `is_conditional` | constructions with their own rules |
| `copula_type` | `none` \| `be` \| `become/seem/appear` |

Cue detection reads the generators' own lexicons rather than keeping a second
word list, so the two directions cannot drift. It spans every cue-bearing
family, and detection order matters: *never* is a negative adverb before it is a
quantifier, *no* is a quantifier even when the parser labels it `neg`, *unable*
is an implicit trigger before it is an affixal negative. The affixal reader runs
the same WordNet antonymy gate as the derivation, so *income* is not read as
`in` + *come*. Multiword cues (*no one*) count once.

Clause type is decided structurally, not by punctuation: `"Sort the array!"` is
an imperative, not an exclamative.

The profile is cached on the `Doc`, so the driver's `applies` sweep computes it
once per sentence.

## The three operations

| operation | direction | example |
| --- | --- | --- |
| `negate` | affirmative → negated, or a negated input gaining a second cue | *sorts* → *does not sort* |
| `affirm` | negated → affirmative | *does not sort* → *sorts* |
| `rescope` | negated → same cue count, different scope | *Not everyone passed* ↔ *Everyone did not pass* |

### Affirmation

One generator per cue-bearing family, each declining unless `classify_input`
reports a cue of its own family. Every one removes **exactly one** cue, so a
two-cue input affirms to a one-cue sentence rather than straight to a bare
positive, and the surviving cue is re-spliced and flagged so `cue_char_spans`
keeps pointing at the negation that is still there:

```
The function does not fail to sort the array.
  ->  The function fails to sort the array.    net 1, cue "fails"
  ->  The function does not sort the array.    net 1, cue "not"
```

The work is in what the cue was holding up, not in deleting it. Do-support
exists only to carry a negator, so removing the negator collapses the auxiliary
and hands its tense, person and number back to the lexical verb — `frames.py`'s
rule run in reverse:

```
does not sort   ->  sorts          not "does sort", not "sort"
did not sort    ->  sorted
fails to sort   ->  sorts          the trigger was carrying the finiteness
lacks X         ->  has X
```

Except in a question, where the auxiliary carries the *inversion* as well as the
negator, so it stays: *"Does it not compile?"* → *"Does it compile?"*.

### Rescoping

When the input is already *"Not everyone passed"* or *"Everyone did not pass"*,
neither of the other operations fits: negating stacks a second cue, affirming
throws the negation away. What is wanted is the *other reading* of the negation
already there — same cue count in, same cue count out, a different
`scope_target`. Both directions are implemented, and the removal half reuses the
affirmation machinery, so collapsing a predicate-scope negator gives the verb
its tense back rather than stranding the auxiliary:

```
Everyone did not pass the exam.  ->  Not everyone passed the exam.
                                     not "*Not everyone did pass the exam."
```

Gated on a scope-bearing quantified subject: without one the two positions are
paraphrases rather than distinct readings.

## Sentence types

Stage 1's clausal families all put the negator straight after the finite slot.
That is right for a declarative and for an imperative, and wrong for everything
else — in an inverted question the finite slot is the *fronted* auxiliary, so
the rule yields `*"Does not it compile?"`. `Generator.clause_types` declares what
each rule was written for and the driver checks it before `applies`.

| input | handling |
| --- | --- |
| interrogative | negator after the subject, or contracted onto the fronted aux; inversion and `?` preserved |
| imperative | bare do-support — *Do not sort the array.* / *Don't sort the array.* |
| passive | attaches to the passive auxiliary — *was sorted* → *was not sorted* |
| existential | quantifier negation, at the NP's left edge — *There is no solution* |
| modal | both readings as separate records in `M_modal`, each with its own `modality` |
| comparative | *not faster than X* (family A) and *no faster than X* (family B) — different claims |
| conditional | main clause and *if*-clause as separate records, never both at once |
| multi-clause | one record per clause targeted, recorded in `target_clause_idx` |

```
Does it compile?              ->  Does it not compile?  /  Doesn't it compile?
There is a solution.          ->  There is no solution.
The parser must handle noise. ->  must not handle    (prohibition)
                              ->  need not handle    (absence of obligation)
If the test passes, …         ->  If the test does not pass, the build succeeds.
                              ->  If the test passes, the build does not succeed.
```

Negating both halves of a conditional at once is two operations, so no depth-1
generator does it. `K2_compound` does, and is allowed to — it declares
`op_depth = 2`.

A question that already carries a cue gains a *second* one, which is still one
operation; only the label changes, and it is decided by the same
`clause_scope` containment test used everywhere else (K1 vs K2 below).

When the copula itself is fronted, only the contracted form is offered: the
parser cannot find the subject/predicate boundary in *"Is the task simple?"*
(it analyses it as `[the task simple]`), and the contraction needs no boundary.

**One parse per sentence.** `driver.parse_bases` runs `nlp.pipe(batch_size=256)`
once; every generator reads that same `Doc`. Nothing re-parses a base sentence.
The only other model call is the filters' single batched re-parse of surviving
candidates.

**Registry + cheap `applies`.** Generators register themselves with a class
decorator. `applies(doc)` does dependency-label and POS checks only — no string
building — and the driver runs `generate` only for those that return `True`.

**Splicing, never regex.** Every variant is a list of `Edit`s naming character
ranges of the base at spaCy token offsets. Overlapping edits raise. Edits carry
`(text, is_cue)` pieces, so the splicer reports exactly where each cue landed in
the output — which is why the cue spans cannot drift out of sync with the
tokens no matter how many edits compose.

## Families

| family | rule | example |
| --- | --- | --- |
| `A_syntactic` | copula / existing aux / do-support | *does not sort*, *cannot handle*, *has not finished* |
| `B_quantifier` | positive → negative quantifier | *some* → *no*, *all* → *none of the* |
| `C_neg_adverb` | `absolute` *never* / `partial` *hardly, barely, …* | *rarely sorts* |
| `D_affixal` | negative prefix or `-ful`→`-less` | *unimportant*, *useless* |
| `E_antonym` | WordNet antonym, inflection matched | *increased* → *decreased* |
| `F_implicit` | NegEx verbal triggers | *fails to sort*, *is unable to sort*, *lacks* |
| `G_prepositional` | privatives | *without a cache*, *is devoid of redundancy* |
| `H_hedged` | epistemic hedge over A | *probably does not sort*, *might not sort* |
| `I_intensified` | emphatic modifier over A | *definitely does not sort*, *is by no means simple* |
| `J_contrastive` | contrastive framing of a copular complement | *is anything but simple* |
| `K1_cancellation` | two cues, one scope → `net_negation = 0` | *is not impossible*, *never fails to sort* |
| `K2_compound` | two cues, two clauses → `net_negation = 2` | *does not sort … and does not return …* |
| `L_scope_position` | both scope readings of a quantified subject | *Not everyone passed* vs *Everyone did not pass* |
| `M_modal` | the readings a negated modal splits into | *must not* (prohibition) vs *need not* (no obligation) |

### A_syntactic: getting do-support right

The auxiliary inherits exactly the features the lexical verb loses, read off
`token.morph` rather than guessed from the surface:

```
The function sorts the array.   ->  The function does not sort the array.
The function sorted the array.  ->  The function did not sort the array.
The functions sort the array.   ->  The functions do not sort the array.
He always arrives early.        ->  He does not always arrive early.
Sort the array.                 ->  Do not sort the array.
```

The last two are the ones that go wrong most often: the introduced auxiliary has
to land *left* of a preverbal adverb, and a subjectless imperative still takes
do-support. `can` is the one auxiliary with a lexicalised negative (`cannot`).

The same three frames are reused by C, F, H, I, K and L, so there is exactly one
place in the codebase that knows where an English negator can go.

### K1 vs K2: scope containment, never cue counting

Both families put two cues in one sentence. Counting them cannot tell the cases
apart — the containment relation can:

```
The result is not unimportant.                              K1, net 0
The result is unimportant and the method does not work.     K2, net 2
```

Identical cue counts, opposite meanings. `scope.classify_double` asks whether
the second cue lies inside the clause the first negator commands, using
`nlp_core.clause_scope`, which descends the dependency tree but stops at
clause-boundary edges (`conj`, `advcl`, `ccomp`, `relcl`, …).

A plain `token.subtree` test is not good enough, and the failure is silent: in
*"the function sorts the array and returns the result"* the conjunct verb
*returns* **is** inside `sorts.subtree`, so a subtree test reports containment
and labels a K2 sentence as a cancellation. `tests/test_scope_classification.py`
asserts both halves of that distinction explicitly.

### Composition

`depth` counts stacked negation operations and is capped at 2, enforced in
`Generator.__init_subclass__` and again in `NegationVariant.__post_init__`.
H and I compose over A; K1/K2 compose A with D, E or F. Nothing reaches depth 3.
These are the only generators that apply two operations to their input, and they
say so with `op_depth = 2`; everything in the arbitrary-input layer is
`op_depth = 1`. One case shows why both numbers are needed: `k1_lexical_cancel`
is `depth = 2` (two cues from affirmative) but `op_depth = 1`, because the input
*"The task is impossible"* already supplied one of them.

H and I do not concatenate onto A's output string — `*"does maybe not sort"` is
not English. Each modifier declares the slot it actually occupies: an `adverb`
goes after the first auxiliary but *before* an introduced `do`; a `modal`
(*might*, *may*) replaces the finite slot outright; a `post_aux` phrase
(*by no means*) needs a real auxiliary and is skipped under do-support.

## Correctness stance

The brief was that a wrong record is worse than a missing one, so several rules
refuse cases they could only guess at:

- **D_affixal** requires a WordNet *antonymy* link between stem and derivative,
  not just attestation. `dis`+`array` and `in`+`tense` are real words that are
  not negations of their stems; only the antonymy gate rejects them. Bare
  `noun + -less` is not attempted at all — the derivative is an adjective and
  would not fit the noun's slot.
- **E_antonym** skips comparatives/superlatives and already-negative words (whose
  antonym is an *affirmation*, not a negation).
- **A** and **C** stand aside when the clause already contains a negative content
  word, so *"The task is not impossible"* is emitted once, by K1, with
  `net_negation = 0` — not by A with `net_negation = 1`.
- **F_implicit** restructures only bare finite lexical verbs; threading a trigger
  through an existing auxiliary chain is left undone rather than guessed.
- Clauses that are already overtly negated are not negated again — including in
  a question, so `*"Hasn't she not finished?"` is never produced. A *lexical*
  negative in the clause is a different matter: that is the K1 route, and it is
  still one operation.
- **Interrogatives** only get post-subject placement when the fronted element is
  a real auxiliary over a lexical verb. With a fronted copula the parser cannot
  locate the subject/predicate boundary, so only the contraction is emitted
  rather than guessing where *not* goes.
- **Affirmation** validates the stem it restores with the same WordNet antonymy
  gate the derivation used, so *income* does not affirm to *come*.
- `G_prepositional` has no affirmation route: *"runs without a cache"* is
  classified as negated and can be negated further, but is not turned back into
  *"runs with a cache"*.

## Filters

Cheapest first, so the only expensive stage sees the fewest candidates:

1. **dedup** on a normalised hash (lowercase, punctuation stripped, whitespace
   collapsed); variants identical to their base are dropped here too.
2. **blacklist** of precompiled regexes: *not not*, stacked auxiliaries, doubled
   determiners, dangling *to*. Tuned to leave well-formed output alone —
   *"does not have"* and *"did not do"* are grammatical and pass.
3. **parse validity**: survivors are re-parsed in one batched pipe pass and
   rejected for more than one ROOT, an unreachable head (cycle), or a tree depth
   more than 3 away from the base's. The same pass carries the **operation-depth
   guardrail**, which raises rather than drops — see `depth` vs `op_depth` above.

Dedup keys on `(base_id, normalised, family)` by default: when two families
derive the same string they are two different analyses of it and both labels are
worth keeping. `--global-dedup` collapses on surface form alone.

## Stages and the regression baseline

Generators declare a `stage`: `1` is the original affirmative-declarative
battery, `2` the arbitrary-input layer. `generate_all(sentences, stages=(1,))`
runs stage 1 alone, which is what the regression test pins — the 20 base
sentences in `tests/data/base_sentences_v1.txt` still produce exactly **448**
variants, and every one of them is still present in the full corpus. The
baseline lives with the tests rather than reading `data/sample_sentences.txt`,
which is a scratch file meant to be edited.

## Stage 2: verification

`negation/verification/` reads stage 1's JSONL (every record carrying
`verified = null`) and writes it back with a verdict and the evidence behind it.
Generator code is untouched — records are handled as dicts, so a re-run of
verification cannot invalidate a stage-1 artefact.

```bash
./venv/bin/python -m negation.verification.cli \
    --input variants.jsonl --output variants_verified.jsonl \
    --model qwen2.5:7b --cola-threshold 0.3 --batch-size 32 --resume
```

**Two tiers, cheap first.** A CoLA classifier scores acceptability on everything
(seconds for the corpus); only survivors reach the LLM at ~9s/record on CPU. The
threshold is reported rather than guessed — every run prints what each candidate
threshold would have cut — and defaults low at 0.30, because tier 1 is the one
irreversible step.

**Three narrow questions, never one vague one.** `grammatical`,
`semantically_valid` and `category_correct`, each answered separately, with
`suggested_family` when the third is false. Strict JSON via Ollama's
`format=json`, three few-shot examples spanning pass / miscategorised /
ungrammatical, and explicit instructions not to rewrite anything and not to judge
the base sentence.

**Local only.** Ollama by default, llama.cpp fallback; `LlamaCppBackend` refuses
a non-local URL and there is no code path to a hosted API.

**Never crashes.** Malformed JSON is retried with backoff, then the batch is
split to one record per request, then the record is marked `verified = null`
with `llm_parse_failure`. `null` means *not judged* and is kept distinct from
`false` (*judged and rejected*) throughout — collapsing them would launder
infrastructure flakiness into training signal.

**Resumable.** Verdicts cache to SQLite keyed by
`sha256(base + variant + family)`, with the model checked on read since verdicts
are not interchangeable across models. Output is written per batch.

Two reports come out of a run:

- [`reports/verification_disagreements.md`](reports/verification_disagreements.md)
  — records judged miscategorised, clustered by `(labelled → suggested)`. A
  cluster is a generator bug with a return address, not a set of bad rows.
- [`reports/verifier_validation.md`](reports/verifier_validation.md) — agreement
  between the verifier and two reference sets, gated on Cohen's kappa ≥ 0.6.
  **Run this first**; the full pass is ~45 minutes of CPU inference.

## Antonym vectors (negative result)

`negation/antonym_vec/` asks whether a learned map over word embeddings can
produce a word's opposite, to cover the ~50% of `E_antonym`-eligible tokens
WordNet has no antonym for. Six approaches, lemma-disjoint split, GloVe and
BERT.

**It does not work.** Best method 0.198 precision@1 against WordNet's 0.999;
three quarters of its top-1 answers are neither the antonym nor a synonym.
It wins on coverage alone (100% vs 16.6% of frequent words), which at that
precision is not worth having.

```
ornate    -> elegant, adorned, sleek      (synonyms — the dangerous failure)
crimson   -> purple, pink, blue           (co-hyponyms)
sluggish  -> enough, so, very             (drift to the frequency centre)
f(v) = -v -> 0.000 P@1                    (the baseline; exactly zero)
```

The module is complete and tested but **disabled by default** — set
`NEGATION_ANTONYM_VEC=1` to enable. It runs only where WordNet returns nothing,
tags its records `generator="antonym_vec_v1"` with a `confidence`, and vetoes
any candidate WordNet calls a synonym. With the veto and the confidence floor it
declines nearly everything, which is the right behaviour.

Full analysis, including why counter-fitting cannot help here and why
morphological pairs turned out *harder* than suppletive ones:
[`docs/antonym_vectors.md`](docs/antonym_vectors.md).

## Notes

`neg.py` and `result.txt` are the earlier itertools prototype and are not used
by this pipeline.
