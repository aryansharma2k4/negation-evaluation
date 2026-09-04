# Negation-variant generation (stage 1)

Given an affirmative base sentence, generate every linguistically valid negated
variant and tag each one with the metadata a graded negation-sensitivity study
needs: which cue realises the negation, where it sits, what it scopes over, and
how many negations survive.

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

Tests: `./venv/bin/python -m pytest tests -q`

`run.sh` is a one-line wrapper around `./venv/bin/python -m negation.cli`.

## Record schema

`negation/schema.py`. One `NegationVariant` per generated sentence:

| field | meaning |
| --- | --- |
| `base_id` / `base_sentence` | content-derived id, stable across reruns |
| `variant` | the generated sentence |
| `family` / `subtype` | `A_syntactic` … `L_scope_position`, and the rule within it |
| `cue_tokens` / `cue_char_spans` / `cue_count` | the negation cues, one span per token, offsets into `variant` |
| `net_negation` | cancellation-aware: `0` for K1, `2` for K2, `1` otherwise |
| `scope_target` | `subject` \| `predicate` \| `clause` |
| `intensity_hint` | `neutral` \| `hedged` \| `intensified` \| `partial` |
| `generator` | e.g. `syn_do_support_v1` |
| `depth` | negation operations stacked (hard cap 2) |
| `verified` | stage-2 hook, always `None` here |

`cue_tokens` and `cue_char_spans` are read straight off the splice, so
`variant[start:end] == token` holds for every record — asserted in the tests.

## Architecture

```
negation/
  nlp_core.py    one spaCy pipeline, cached WordNet, morphology, dep helpers
  lexicons.py    every word list, frozen at import time
  splice.py      character-offset editing; the only way variants are built
  frames.py      clause frames — the shared substrate for clausal negation
  polarity.py    what negative material a clause already carries
  scope.py       the K1-vs-K2 containment decision
  base.py        Generator ABC + registry + record construction
  generators/    one module per family
  filters.py     dedup → blacklist → batched re-parse
  driver.py      generate_all
  cli.py         JSONL + summary
```

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
- Clauses that are already overtly negated are not negated again.

## Filters

Cheapest first, so the only expensive stage sees the fewest candidates:

1. **dedup** on a normalised hash (lowercase, punctuation stripped, whitespace
   collapsed); variants identical to their base are dropped here too.
2. **blacklist** of precompiled regexes: *not not*, stacked auxiliaries, doubled
   determiners, dangling *to*. Tuned to leave well-formed output alone —
   *"does not have"* and *"did not do"* are grammatical and pass.
3. **parse validity**: survivors are re-parsed in one batched pipe pass and
   rejected for more than one ROOT, an unreachable head (cycle), or a tree depth
   more than 3 away from the base's.

Dedup keys on `(base_id, normalised, family)` by default: when two families
derive the same string they are two different analyses of it and both labels are
worth keeping. `--global-dedup` collapses on surface form alone.

## Notes

`neg.py` and `result.txt` are the earlier itertools prototype and are not used
by this pipeline.
