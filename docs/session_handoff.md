# Session handoff

For an assistant that has read the spec but never the code. Written 2026-09-13.
Numbers here are from actual runs on this machine, not estimates. Where something
is broken or unverified, it says so.

---

## 1. What was asked vs what was delivered

Five tasks were attempted this session, on top of a pre-existing stage-1
generator (commit `6801210`, 20 generators / 13 families / 448 variants / 75
tests).

| # | Task | Status |
| --- | --- | --- |
| 1 | Scale generators to arbitrary input sentence types, depth 1 | **Fully done** |
| 2 | Evaluate public datasets, write loaders | **Fully done** |
| 3 | Antonym vector generation | **Fully done — negative result** |
| 4 | Local LLM verification (stage 2) | **Partially done — blocked at the gate** |
| 5 | `run.sh` / `run_test.sh`, sample sentences | **Fully done** |

Detail on the two that are not clean:

**Task 3 produced a negative result, not a working component.** The module is
complete, tested and wired in, but it does not work well enough to use and ships
disabled. This was an anticipated outcome — the brief said to expect it.

**Task 4 is genuinely incomplete.** Everything is built (two tiers, caching,
retries, reports, CLI) and 56 tests pass, but the verifier **failed its own
validation gate twice**, so the full 455-record verification run was never
executed. `reports/verification_disagreements.md` — the output the task called
"the valuable output" — **does not exist**, because generating it from an
unvalidated verifier would produce misleading data. See §10.

One further gap in task 4: the hand-labelled half of the validation never
completed. Ollama evicted the 7B model partway through (memory pressure from
concurrent test runs on a 14 GB machine) and the remaining requests failed
instantly. The code handled it correctly — marked them `verified = null` rather
than guessing — but it means only the constructed reference set has numbers.

---

## 2. Repository map

```
negation/                       stage-1 generator package (pre-existing, extended)
  schema.py                     NegationVariant dataclass; family/operation constants; DepthViolation
  base.py                       Generator ABC, registry, build() -- the single place records are made
  driver.py                     generate_all(): parse once, run licensed generators, filter
  cli.py                        JSONL writer + summary (python -m negation.cli)
  nlp_core.py                   one spaCy pipeline, cached WordNet, morphology, clause_scope()
  lexicons.py                   every word list, frozen at import
  splice.py                     character-offset Edit application; the only way variants are built
  frames.py                     the three clause frames (copula / existing_aux / do_support)
  polarity.py                   what negative material a clause already carries
  scope.py                      classify_double(): the K1-vs-K2 containment decision
  filters.py                    dedup -> blacklist -> batched re-parse -> depth guardrail
  affixes.py                    NEW: affixal derivation in both directions, WordNet-gated
  classify.py                   NEW: classify_input(doc) -> InputProfile
  generators/
    a_syntactic.py              A: not/n't in the verb complex
    b_quantifier.py             B: some -> no, all -> none of the
    c_neg_adverb.py             C: never / hardly / barely / rarely / seldom / scarcely
    d_affixal.py                D: thin wrapper over affixes.py
    e_antonym.py                E: WordNet antonym substitution (+ vector fallback hook)
    f_implicit.py               F: fails to / refuses to / is unable to / lacks
    g_prepositional.py          G: without / devoid of / free of
    h_hedged.py                 H: probably not / might not
    i_intensified.py            I: definitely not / by no means
    j_contrastive.py            J: anything but / far from
    k_double.py                 K1 + K2: five double-negation routes
    l_scope_position.py         L: both scope readings of a quantified subject
    _modified.py                shared base for H and I
    m_sentence_type.py          NEW: interrogative, imperative, existential, modal, comparative
    n_clause_target.py          NEW: one clause per record in multi-clause input
    affirm.py                   NEW: 7 affirmation generators (negated -> affirmative)
    rescope.py                  NEW: 2 rescope generators
  data/                         NEW: third-party corpus loaders
    sources.py                  every URL, verified with the status observed on 2026-09-13
    loaders.py                  CD-SCO / SFU / BioScope / CANNOT / NegNLI / NaN-NLI
    family_map.py               maps a corpus's gold cue onto our 13 families
    schema.py                   AnnotatedSentence: the common record
    fetch.py                    downloader (stdlib only)
  antonym_vec/                  NEW: antonym-from-embedding experiment
    data.py                     WordNet pairs, lemma-disjoint split, morphological/suppletive tag
    embeddings.py               StaticEmbeddings (GloVe/fastText) + ContextualEmbeddings (BERT)
    methods.py                  the six approaches
    evaluate.py                 P@k, synonym contamination, kappa-free metrics
    api.py                      generate_antonym() -- the fallback E_antonym calls
    fetch.py                    vector downloader
  verification/                 NEW: stage 2
    schema.py                   Verdict, RunStats, record_key(), reason constants
    cola.py                     tier 1: CoLA acceptability + threshold sweep
    prompts.py                  the three-question prompt, few-shot, PROMPT_VERSION
    llm.py                      Ollama + llama.cpp backends, retry/fallback, tolerant parser
    cache.py                    SQLite verdict cache
    driver.py                   orchestration
    cli.py                      python -m negation.verification.cli
    validate.py                 constructed reference set, Cohen's kappa
    validate_report.py          renders reports/verifier_validation.md
    report.py                   renders reports/verification_disagreements.md

scripts/
  dataset_stats.py              per-corpus stats under our taxonomy (the K1/K2 evidence)
  antonym_experiment.py         runs the six-method comparison
  verify_validate.py            validates the verifier before it is trusted

tests/                          376 tests across 10 files
  data/base_sentences_v1.txt    the frozen 20-sentence regression corpus

docs/
  RUNBOOK.md                    how to run everything, ordered by how long it takes
  dataset_evaluation.md         task 2 report
  antonym_vectors.md            task 3 report
  status.md, diagrams.md        pre-existing mermaid status docs
reports/
  verifier_validation.md        task 4: the gate result (FAILED)
  verifier_validation_p1_baseline.txt   the first prompt's numbers, kept as evidence
run.sh, run_test.sh             entry points
data/sample_sentences.txt       5 sentences supplied by the researcher
```

14,581 lines of Python across tracked files.

### Divergences from the spec's structure

1. **`negation/` not `src/negation/`.** The spec for tasks 2–4 said
   `src/negation/...`. This repository has no `src/` directory; the pre-existing
   package is at `negation/`. New packages were put alongside it rather than
   creating a second, parallel tree. Purely a path difference; no behaviour
   depends on it.
2. **`negation/verification/validate_report.py` is an extra file** not named in
   the spec, split out so the kappa gate logic has exactly one home.
3. **`negation/affixes.py` is an extra file.** Affixal derivation was extracted
   out of `generators/d_affixal.py` so that `classify.py` could run it backwards
   without a generator→classifier import cycle.

---

## 3. Data schema — as actually implemented

`negation/schema.py`, class `NegationVariant`. Fields in declaration order.
**Bold** = added this session.

| field | type | default | meaning |
| --- | --- | --- | --- |
| `base_id` | `str` | — | content-derived id, stable across reruns |
| `base_sentence` | `str` | — | the input sentence |
| `variant` | `str` | — | the generated sentence |
| `family` | `str` | — | one of 14 family constants (13 original + `M_modal`) |
| `subtype` | `str` | — | the rule within the family |
| `cue_tokens` | `list[str]` | — | one entry per whitespace-delimited cue token |
| `cue_char_spans` | `list[tuple[int,int]]` | — | offsets into `variant`; `variant[s:e] == token` holds |
| `cue_count` | `int` | — | derived from spans in `__post_init__`, never trusted from the caller |
| `net_negation` | `int` | — | 0 for K1, 2 for K2, 1 otherwise |
| `scope_target` | `str` | — | `subject` / `predicate` / `clause` |
| `intensity_hint` | `str` | — | `neutral` / `hedged` / `intensified` / `partial` |
| `generator` | `str` | — | e.g. `syn_do_support_v1` |
| `depth` | `int` | `1` | operations from a fully affirmative baseline; hard cap 2 |
| **`operation`** | `str` | `"negate"` | `negate` / `affirm` / `rescope` |
| **`input_polarity`** | `str` | `"affirmative"` | polarity of the *input* |
| **`input_cue_count`** | `int` | `0` | the `N` in the depth-1 rule |
| **`clause_type`** | `str` | `"declarative"` | declarative / interrogative / imperative / exclamative |
| **`voice`** | `str` | `"active"` | active / passive |
| **`modality`** | `str` | `"none"` | none / epistemic / deontic / ability |
| **`target_clause_idx`** | `int` | `0` | which clause this record negated |
| **`op_depth`** | `int` | `1` | operations applied *to the input* |
| **`confidence`** | `Optional[float]` | `None` | only set by the antonym-vector fallback |
| `verified` | `Optional[bool]` | `None` | stage-2 hook |

**`depth` vs `op_depth` is the one subtlety.** They count different things and
both are wanted. Negating *"The task is impossible"* is `depth=2` (two cues away
from a fully affirmative sentence) but `op_depth=1` (one operation applied to the
sentence we were handed). The spec's "Schema additions" listed a field called
`depth` with "assert == 1 for every emitted record", but `depth` already existed
with the other meaning and a pre-existing test asserts `depth == 2` for hedged
records. Renaming it would have broken the no-regression requirement, so the new
field is `op_depth`. **This is a deliberate deviation from the spec's field
name.**

Stage 2 adds nine more fields to the *output JSONL only*, not to this dataclass:
`verify_grammatical`, `verify_semantic`, `verify_category`, `suggested_family`,
`verify_reason`, `verify_model`, `verify_confidence`, `cola_score` (plus setting
`verified`). Kept out of the dataclass so a re-run of verification cannot
invalidate a stage-1 artefact.

### Three real records

Copied verbatim from a run of
`./venv/bin/python -m negation.cli tests/data/base_sentences_v1.txt -o /tmp/full.jsonl`.

```json
{"base_id": "s00000_8ac2ef2e5f", "base_sentence": "The function sorts the array.", "variant": "The function never fails to sort the array.", "family": "K1_cancellation", "subtype": "never_fails_to", "cue_tokens": ["never", "fails"], "cue_char_spans": [[13, 18], [19, 24]], "cue_count": 2, "net_negation": 0, "scope_target": "clause", "intensity_hint": "neutral", "generator": "k1_trigger_cancel_v1", "depth": 2, "operation": "negate", "input_polarity": "affirmative", "input_cue_count": 0, "clause_type": "declarative", "voice": "active", "modality": "none", "target_clause_idx": 0, "op_depth": 2, "confidence": null, "verified": null}
```

```json
{"base_id": "s00003_c0d4cac8f4", "base_sentence": "The model can handle noise.", "variant": "The model cannot handle noise.", "family": "M_modal", "subtype": "cannot_ability_denied", "cue_tokens": ["cannot"], "cue_char_spans": [[10, 16]], "cue_count": 1, "net_negation": 1, "scope_target": "clause", "intensity_hint": "neutral", "generator": "modal_reading_v1", "depth": 1, "operation": "negate", "input_polarity": "affirmative", "input_cue_count": 0, "clause_type": "declarative", "voice": "active", "modality": "ability", "target_clause_idx": 0, "op_depth": 1, "confidence": null, "verified": null}
```

```json
{"base_id": "s00006_bf6a293a4c", "base_sentence": "Everyone passed the exam.", "variant": "Not everyone passed the exam.", "family": "L_scope_position", "subtype": "subject_scope", "cue_tokens": ["Not"], "cue_char_spans": [[0, 3]], "cue_count": 1, "net_negation": 1, "scope_target": "subject", "intensity_hint": "neutral", "generator": "scope_position_v1", "depth": 1, "operation": "negate", "input_polarity": "affirmative", "input_cue_count": 0, "clause_type": "declarative", "voice": "active", "modality": "none", "target_clause_idx": 0, "op_depth": 1, "confidence": null, "verified": null}
```

---

## 4. Generators — the real inventory

**35 registered generators** — 20 from stage 1, 15 added this session. Counts are
from the last full run: 20 base sentences (`tests/data/base_sentences_v1.txt`) →
**455 variants**, 12.2 s wall. `generate_all(stages=(1,))` on the same input
reproduces stage 1 exactly at **448 variants**, which is the regression pin.

| generator | family | source file | records |
| --- | --- | --- | ---: |
| `syn_copula_v1` | A_syntactic | `a_syntactic.py` | 5 |
| `syn_existing_aux_v1` | A_syntactic | `a_syntactic.py` | 2 |
| `syn_do_support_v1` | A_syntactic | `a_syntactic.py` | 11 |
| `quant_swap_v1` | B_quantifier | `b_quantifier.py` | 4 |
| `negadv_insert_v1` | C_neg_adverb | `c_neg_adverb.py` | 102 |
| `affix_derive_v1` | D_affixal | `d_affixal.py` | 4 |
| `antonym_wordnet_v1` | E_antonym | `e_antonym.py` | 15 |
| `implicit_trigger_v1` | F_implicit | `f_implicit.py` | 65 |
| `implicit_lack_v1` | F_implicit | `f_implicit.py` | 1 |
| `prep_without_v1` | G_prepositional | `g_prepositional.py` | 2 |
| `prep_privative_copula_v1` | G_prepositional | `g_prepositional.py` | 3 |
| `hedge_wrap_v1` | H_hedged | `h_hedged.py` | 90 |
| `intensify_wrap_v1` | I_intensified | `i_intensified.py` | 84 |
| `contrastive_frame_v1` | J_contrastive | `j_contrastive.py` | 10 |
| `k1_lexical_cancel_v1` | K1 | `k_double.py` | 2 |
| `k1_affixal_cancel_v1` | K1 | `k_double.py` | 4 |
| `k1_trigger_cancel_v1` | K1 | `k_double.py` | 36 |
| `k2_clause_pair_v1` | K2 | `k_double.py` | 3 |
| `k2_cross_clause_v1` | K2 | `k_double.py` | 1 |
| `scope_position_v1` | L_scope_position | `l_scope_position.py` | 4 |
| `modal_reading_v1` | M_modal | `m_sentence_type.py` | 2 |
| `interrogative_negation_v1` | A_syntactic / K1 / K2 | `m_sentence_type.py` | **0** |
| `imperative_contracted_v1` | A_syntactic | `m_sentence_type.py` | 1 |
| `existential_no_v1` | B_quantifier | `m_sentence_type.py` | **0** |
| `comparative_no_v1` | B_quantifier | `m_sentence_type.py` | **0** |
| `clause_target_negation_v1` | A_syntactic / K1 / K2 | `n_clause_target.py` | 3 |
| `affirm_affixal_strip_v1` | D_affixal | `affirm.py` | 1 |
| `affirm_do_support_v1` | A_syntactic | `affirm.py` | **0** |
| `affirm_aux_neg_v1` | A_syntactic | `affirm.py` | **0** |
| `affirm_copula_neg_v1` | A_syntactic | `affirm.py` | **0** |
| `affirm_quantifier_flip_v1` | B_quantifier | `affirm.py` | **0** |
| `affirm_neg_adverb_v1` | C_neg_adverb | `affirm.py` | **0** |
| `affirm_implicit_strip_v1` | F_implicit | `affirm.py` | **0** |
| `rescope_subject_to_predicate_v1` | L_scope_position | `rescope.py` | **0** |
| `rescope_predicate_to_subject_v1` | L_scope_position | `rescope.py` | **0** |

**All 13 original families are implemented**, plus `M_modal` added this session
(14 total). Nothing from the taxonomy is missing.

**11 of the 35 generators produced nothing on this corpus.** They are not
failures — they are generators whose licensing conditions the 20-sentence
regression corpus does not meet. That corpus contains no questions,
no existentials, no comparatives and only one already-negated sentence — so the
six affirmation generators, both rescope generators, and the interrogative,
existential and comparative generators have nothing to fire on. They do fire when
given suitable input; each is verified individually in
`tests/test_arbitrary_input.py`. For example `"Does it compile?"` →
`"Does it not compile?"` / `"Doesn't it compile?"`, and
`"The function does not sort the array."` → `"The function sorts the array."`.

**This matters for anyone reading the 455-variant distribution as representative:
it is not.** It is what one affirmative-declarative corpus produces. Roughly a
third of the generator inventory is invisible in it.

**Coverage is very uneven.** `negadv_insert_v1` alone produces 102 of 455 records
(22%) because it emits six adverbs per licensed clause. H and I together produce
174 (38%). B, D, G, L and M produce 4, 5, 5, 4 and 2 respectively. Any downstream
training on this corpus will be dominated by adverbial and modified negation
unless rebalanced.

---

## 5. The hard parts — how they were actually solved

### do-support

`negation/frames.py`, `_do_form_for(head)`. The auxiliary is chosen from the
lexical verb's own morphology, read off spaCy's `token.morph`, **not** guessed
from the surface string:

- `Tense=Past` in `morph` → `did`
- `Tense=Pres`, or tag `VBZ`/`VBP` → `does` if tag is `VBZ` or
  (`Person=3` and `Number=Sing`), else `do`
- fallback when the tagger leaves a finite verb featureless: tag `VBD` → `did`;
  tag `VB` with `VerbForm=Inf` → `do` (only reachable for imperatives)
- returns `None` otherwise, and the frame is declined

The lexical verb is then rewritten to its lemma by a separate `Edit`. Placement
is handled by `do_support_insertion_point()`: the inserted auxiliary must land
*left* of any preverbal adverbial, so `"He always arrives early"` →
`"He does not always arrive early"`, never `"*He always does not arrive"`.

`can` is the one auxiliary with a lexicalised negative; `CONTRACTED_NEGATIVE_AUX`
maps it to `cannot`.

**Known-correct on**: 3rd-singular present, plural present, past, imperative,
preverbal adverbs. There is a dedicated tense table in `tests/test_do_support.py`
(6 tests).

### `clause_scope()`

`negation/nlp_core.py`. Iterative DFS from a clause head, descending through
children but **refusing to cross** any edge in:

```python
CLAUSE_BOUNDARY_DEPS = {"conj", "advcl", "ccomp", "csubj",
                        "csubjpass", "parataxis", "relcl", "acl"}
```

Returns a `frozenset[int]` of token indices. Implemented with an explicit stack
and a `seen` set, so a malformed parse with a cycle terminates rather than
recursing forever.

The reason it is not `token.subtree`: in *"the function sorts the array **and**
returns the result"* the conjunct verb *returns* **is** inside `sorts.subtree`,
so a subtree test reports containment and mislabels a K2 sentence as a
cancellation. That failure is silent, which is why it has its own test file.

### K1 vs K2

`negation/scope.py`, four lines:

```python
def classify_double(outer_anchor, inner_anchor):
    if inner_anchor.i in clause_scope(outer_anchor):
        return FAM_CANCELLATION      # K1, net_negation = 0
    return FAM_COMPOUND              # K2, net_negation = 2
```

`outer_anchor` is the clause head the new negator attaches to; `inner_anchor` is
the token bearing the other cue. **Cue counting is never used** — both cases have
exactly two cues. `net_negation_for(family)` then maps K1→0, K2→2 and raises on
anything else.

`double_labelling(anchor, existing)` wraps this for the arbitrary-input
generators: returns `None` when the input carried no cue (caller keeps its own
family and `net_negation=1`), otherwise the K1/K2 pair.

### depth-1 enforcement

Two places, doing different jobs.

**Declaration time** — `negation/base.py:84`, in `Generator.__init_subclass__`:

```python
if cls.stage == 2 and cls.op_depth != MAX_OP_DEPTH:
    raise ValueError(...)
```

Any arbitrary-input generator that declares `op_depth != 1` fails at import.

**Emit time** — `negation/filters.py:146`, `check_operation_depth()`, inside the
batched re-parse stage. It counts cues in the **re-parsed variant** using the
same `classify_input()` that counted the input, and raises
`DepthViolation` (a subclass of `AssertionError`) if
`observed > input_cue_count + op_depth`.

It **raises, it does not filter**. A generator that miscounts its own operations
is a bug, and silently dropping the record would hide it. The whole run dies with
the generator name, the variant and both counts.

Counting on the re-parsed variant matters: a multiword cue like *by no means*
occupies three `cue_char_spans` but is one cue, and comparing spans to cues would
false-positive on every `I_intensified` record.

All 455 records of the last full run satisfy the rule.

### Known deviations and known-wrong cases

- **`op_depth` instead of `depth`** — see §3.
- **The spec asked for 3 few-shot examples in the verification prompt; there are
  4.** Added after measurement, not taste; see §9.
- **Fronted-copula questions get only the contracted form.** `en_core_web_sm`
  parses *"Is the task simple?"* as `[the task simple]` — one NP — so there is no
  trustworthy subject/predicate boundary to insert *not* after. The generator
  emits `"Isn't the task simple?"` and declines the post-subject form rather than
  guess. `"Is the task impossible?"` does work because the parse differs.
- **`"Does the parser that caches compile?"` produces nothing.** The parse is
  mangled (`compile` tagged PROPN, `relcl`), `_fronted_aux` finds no auxiliary,
  and the generator declines. Silent under-generation, not bad output.
- **`B_quantifier` `both_to_neither` does not fix noun number**, so
  *"Both tests passed"* → *"Neither tests passed"* (ungrammatical). The filters do
  not catch it. The *affirming* direction was fixed this session
  (*"Neither test passed"* → *"Both tests passed"*); the negating direction was
  not. Pre-existing, documented in `docs/status.md`.

---

## 6. Numbers from the last full run

Command: `./venv/bin/python -m negation.cli tests/data/base_sentences_v1.txt -o /tmp/full.jsonl`

```
base sentences : 20
variants kept  : 455
wall time      : 12.2 s
```

Per family:

| family | records |
| --- | ---: |
| C_neg_adverb | 102 |
| H_hedged | 90 |
| I_intensified | 84 |
| F_implicit | 66 |
| K1_cancellation | 42 |
| A_syntactic | 21 |
| E_antonym | 15 |
| J_contrastive | 10 |
| D_affixal | 5 |
| G_prepositional | 5 |
| K2_compound | 5 |
| B_quantifier | 4 |
| L_scope_position | 4 |
| M_modal | 2 |

By operation: `negate` 454, `affirm` 1, `rescope` 0.
By input type: declarative/affirmative 423, declarative/negated 6,
imperative/affirmative 26.
By `net_negation`: 0 → 43, 1 → 407, 2 → 5.

**Filter rejections: `duplicate` 3. That is all.** No blacklist hits, no parse
rejections, no depth violations on this corpus. That is a meaningfully clean
result and also a warning — the filters are barely exercised by these 20
sentences, so their true rejection behaviour is largely untested in practice.

**Stage 2 has never been run to completion on the full corpus.** See §10.

---

## 7. Tests

**376 tests, all passing, 52.5 s.** Last run this session, `./run_test.sh`.
Zero failures, zero skips on this machine (skips do occur on a clean checkout
where the GloVe vectors or the third-party corpora are absent — those tests skip
with a fetch hint rather than fail).

| file | tests | covers |
| --- | ---: | --- |
| `test_verification.py` | 56 | stage 2: parsing, cache, tiering, failure paths, kappa |
| `test_arbitrary_input.py` | 34 | clause_type × polarity, sentence types, depth-1, 448 regression |
| `test_antonym_vec.py` | 29 | split hygiene, method correctness, integration |
| `test_data_loaders.py` | 28 | corpus loaders, family mapping, source registry |
| `test_families.py` | 22 | one test per generator family |
| `test_pipeline.py` | 22 | schema, splicing, filters, driver, CLI |
| `test_classify.py` | 20 | InputProfile across all fields |
| `test_lexicons.py` | 10 | derived-lexicon invariants |
| `test_scope_classification.py` | 8 | K1 vs K2 containment |
| `test_do_support.py` | 6 | tense/person/number table |

Run with `./run_test.sh` (all), `./run_test.sh fast` (282 tests, no downloads,
6 s), `./run_test.sh verify` (56 stage-2 tests, no model, 0.2 s).

### Paths with NO test coverage

- **`negation/verification/llm.py` against a real model.** All 56 stage-2 tests
  use a scripted fake backend. Nothing tests that the real prompt gets a real
  model to produce parseable output — that is only known empirically, from the
  runs described in §10.
- **`ContextualEmbeddings`** (BERT source in `antonym_vec/embeddings.py`) has no
  test; it needs torch + transformers + a 440 MB download. It was exercised
  manually in the task-3 experiment.
- **`LlamaCppBackend.complete()`** — only its URL refusal is tested. The actual
  llama.cpp request path has never been executed; no llama.cpp server was
  available on this machine. **Treat that backend as unverified code.**
- **`negation/verification/report.py` against real disagreement data** — tested
  only on synthetic records, because no full verification run has happened.
- `neg.py` and `result.txt` are a pre-existing itertools prototype, unused and
  untested.

---

## 8. Known bugs, rough edges, shortcuts

Ranked by how much they would hurt a downstream consumer.

1. **Stage 2 is unusable as a quality signal right now.** The verifier fails its
   gate: grammaticality kappa −0.030 (worse than chance), catching 2 of 20
   deliberately corrupted sentences. Anything filtered on its `verified` field
   would be filtered on noise. It is disabled by default for this reason.
2. **The corpus is severely imbalanced.** 38% of records come from two families
   (H, I) that are mechanical wrappers over A. A model trained on the raw
   distribution will learn "negation = adverb insertion".
3. **`Both tests passed` → `Neither tests passed`** — ungrammatical output that
   reaches the corpus. One known bad string per occurrence of `both`.
4. **`data/sample_sentences.txt` is not the regression corpus.** It holds 5
   researcher-supplied sentences and produces 61 variants. The 448/455 numbers
   come from `tests/data/base_sentences_v1.txt`. Confusing these is easy and I
   did it once mid-session — a `./run.sh` overwrote `variants.jsonl` while a
   validation run was reading it.
5. **`TOKENS_PER_RECORD = 60` in `llm.py` is a hardcoded guess.** If a model
   becomes chattier, batches truncate and fall back to one-request-per-record,
   which is ~10× slower. No adaptive sizing.
6. **The `~15 s/record` figure in `run.sh`'s time estimate is hardcoded** and
   measured on this CPU with this model. It will be wrong on other hardware.
7. **`antonym_vec/api.py` hardcodes `RIDGE_ALPHA = 10.0`**, the value
   `tune_ridge_alpha` happened to select. It is not re-tuned at load time; if the
   training pairs change it becomes stale silently.
8. **BioScope's clinical subcorpus cannot be loaded at all** — what ships is an
   annotation skeleton where every token is `*`, and the text source
   (`computationalmedicine.org`) is dead. The loader refuses it explicitly.
9. **CD-SCO survives only on one third-party GitHub mirror.** The official \*SEM
   2012 page is 404. If that mirror disappears the corpus is effectively gone.
   Cache the four files.
10. **`_is_local()` in `llm.py` is a regex over the URL.** It correctly refuses
    `https://api.openai.com`, but it is a string check, not a network-level
    guarantee.

---

## 9. Decisions made that were not in the spec

Each of these was a judgement call. The reasoning matters because it cannot be
inferred from the code.

1. **`op_depth` rather than renaming `depth`.** The spec's depth-1 assertion
   could not be applied to the existing `depth` field without breaking
   `tests/test_families.py:110`, which asserts `depth == 2` for hedged records.
   Two fields, two meanings, both kept.

2. **The depth-1 rule is enforced on the *arbitrary-input layer*, not the whole
   corpus.** The spec said "assert depth == 1 across the entire generated
   corpus", but stage 1's `K2_compound` legitimately applies two operations and
   predates the requirement. Generators carry a `stage` attribute; stage 2
   generators are held to `op_depth == 1`, stage 1 keeps its `depth ≤ 2` cap.
   `generate_all(stages=(1,))` reproduces stage 1 exactly, which is what the
   448-variant regression test pins.

3. **A fourth few-shot example was added to the verification prompt.** The spec
   named three (clean pass / miscategorised / ungrammatical). Measurement showed
   `qwen2.5:7b` collapsing all three questions into one global judgement — it
   marked *"None of the students passed."* ungrammatical and returned identical
   values on all three fields for 20/20 records. The missing case was
   *grammatical but semantically implausible*; a fourth example covers it. This
   did not fix the problem (see §10) but the reasoning stands.

4. **`PROMPT_VERSION` is part of the cache key.** Not in the spec. Without it, a
   prompt rewrite silently replays the old prompt's cached verdicts. The
   backend's `name` becomes `ollama/qwen2.5:7b+p2`.

5. **Tier 1's default threshold is 0.30, not the spec's 0.5 example.** Tier 1 is
   the only irreversible step — a rejected record never reaches the LLM — so the
   default is permissive. At 0.30 it rejects 2.4% of the corpus; at 0.5 it would
   reject 6.4%. The sweep is printed on every run so this can be re-decided.

6. **A "constructed" validation reference set was invented.** The spec asked for
   100 hand-labelled records. Labelling them revealed that a random sample from
   this corpus is **100/100 grammatical and 100/100 correctly categorised** — so
   kappa is undefined on two of the three questions. The constructed set
   (deliberate word-swaps and family-swaps, ground truth by construction) is the
   only thing that makes those measurable.

7. **The 100 labels are the assistant's, not a human's.** This is a real
   weakness — an LLM validating an LLM shares blind spots. Stated plainly in
   `reports/verifier_validation.md` and `docs/RUNBOOK.md`, with instructions for
   re-labelling.

8. **The full verification run was withheld when the gate failed.** The spec said
   "the verifier must be validated before its output is used as training signal".
   Taken literally. The disagreement report does not exist as a result.

9. **Stage-2 fields live on the output dict, not on `NegationVariant`.** The
   spec said "no generator code changes"; extending the generator's dataclass
   would have been one.

10. **Datasets are fetched, never vendored.** Licences differ (SFU is GPL-3.0;
    CD-SCO and NegNLI state none at all) and redistribution is not ours to make.
    `negation/data/sources.py` records every URL with the status observed.

11. **`M_modal` was added as a 14th family.** Modal dual readings
    (*must not* = prohibition vs *need not* = no obligation) are not paraphrases,
    and putting both under `A_syntactic` meant per-family dedup collapsed them
    into one record.

---

## 10. State of the four extension tasks

### (1) Sentence-type scaling to depth 1 — **complete**

`classify.py` returns an `InputProfile` (clause_type, voice, polarity,
existing_cues across all 7 cue-bearing families, modality, clause_count,
has_subordinate, is_existential, is_comparative, is_conditional, copula_type).
Three operations implemented: `negate`, `affirm` (7 generators), `rescope`
(2 generators). Interrogative, imperative, passive, existential, modal,
comparative, conditional and multi-clause all handled.

Stage 1 regression intact: **448 variants** from `stages=(1,)`, 455 with the full
registry.

Next action: none required. Possible follow-up is rebalancing the family
distribution (§8 item 2).

### (2) Dataset evaluation — **complete**

`docs/dataset_evaluation.md`. Six corpora fetched and verified on 2026-09-13;
loaders normalise all of them to one schema.

| corpus | gold spans | speculation cues | multi-cue sentences | K1 | K2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CD-SCO | 1,421 | 0 | 169 | 63 | 106 |
| SFU | 8,891 | 5,363 | 369 | 89 | 280 |
| BioScope | 5,511 | 3,376 | 191 | 66 | 125 |
| CANNOT | 0 | 0 | — | — | — |
| NegNLI | 0 | 0 | — | — | — |
| NaN-NLI | 0 | 0 | — | — | — |

**The citable result**: 729 sentences across the three span corpora carry 2+
negation cues; our `clause_scope` logic splits them 218 K1 / 511 K2. **None of
the three corpora records that distinction** — every cue is annotated
independently. Roughly 30% of multiply-negated sentences in existing corpora are
cancellations and no corpus labels them as such.

Recommendation: train the detector on CD-SCO + SFU; build the intensity
regressor on SFU (5,363 hedge cues) with BioScope secondary; hold out BioScope
full papers, NegNLI, NaN-NLI. Reject CANNOT/SNLI/MultiNLI — no gold spans, and
deriving silver ones with our own classifier would be circular.

Next action: none required.

### (3) Antonym vector generation — **complete, negative result**

`docs/antonym_vectors.md`. Six methods, lemma-disjoint split (3,976 train /
1,000 test pairs, zero leakage), GloVe + fastText + BERT.

| method | P@1 | synonym@1 |
| --- | ---: | ---: |
| `baseline_wordnet` | 0.999 | 0.000 |
| `reflection` | **0.198** | 0.050 |
| `mlp_map` | 0.170 | 0.022 |
| `counterfit_transductive` | 0.166 | 0.000 |
| `linear_map` | 0.123 | 0.022 |
| `baseline_negate` (`-v`) | **0.000** | 0.000 |
| `counterfit_style` | **0.000** | 0.000 |

It beats WordNet on coverage (100% vs 16.6% of frequent words) and loses on
accuracy by 5×. Three quarters of top-1 answers are neither the antonym nor a
synonym. The root cause is measured: in GloVe, **antonyms average 0.601 cosine
and synonyms 0.566** — antonyms are *closer* than synonyms, so cosine does not
encode the distinction at all.

Wired into `E_antonym` as a fallback, **disabled by default**
(`NEGATION_ANTONYM_VEC=1`), tagged `generator="antonym_vec_v1"` with a
`confidence`, and vetoing any candidate WordNet calls a synonym.

Next action: none, unless the researcher wants to pursue the one positive signal
— the most *constrained* model won, suggesting more inductive bias rather than
more capacity.

### (4) Local LLM verification — **built, blocked at the gate**

**Exists and works**: two-tier pipeline; CoLA tier 1 with threshold sweep; three
separate questions; Ollama + llama.cpp backends, localhost-only; tolerant JSON
parser; retry → per-record fallback → `verified=null`; SQLite cache keyed by
`sha256(base+variant+family)` + model + prompt version; per-batch checkpointing
and `--resume`; CLI with the specified summary; 56 tests.

**Does not exist**: `reports/verification_disagreements.md`. No full run.

**The gate failed twice.** Constructed reference set, `qwen2.5:7b`:

| prompt | grammaticality kappa | caught corrupted | category detection | named a family |
| --- | ---: | ---: | ---: | ---: |
| p1 | +0.080 | — | 55% | 0/20 |
| p2 (after fix) | **−0.030** | **2/20** | 70% | 2/20 |

p1 over-rejected; p2 over-accepts. Accuracy stayed at 0.617 both times — which is
exactly why the gate is on kappa.

The one real signal: it flagged **14 of 20** deliberately mislabelled families.
But it named a replacement only 2/20 times, so the disagreement report would be
mostly unactionable.

The hand-labelled half never ran — ollama evicted the model mid-run.

**Immediate next action**, in order of expected value:

1. Re-run the labelled half (`scripts/verify_validate.py`) with nothing else
   competing for RAM. ~25 min. This is the only missing measurement.
2. Try **one question per request** instead of three-in-one. The observed failure
   is specifically the collapse of three answers into one; isolating them
   directly addresses it. Costs ~3× the time.
3. Or try a larger model. `qwen2.5:7b` at this quantisation may simply lack the
   discrimination.
4. Or narrow the scope: drop the grammaticality question to tier 1 entirely
   (CoLA already does it) and keep the LLM only for category detection, which is
   the one thing it showed skill at.

---

## 11. How to run it

### Install

```bash
cd /home/aryan/wspace/dev/negation
python -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python -m spacy download en_core_web_sm
./venv/bin/python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

Already installed in this environment beyond `requirements.txt`:
`torch 2.14.0+cpu`, `transformers 5.17.0` (needed for tier 1 and for the BERT
embedding source).

### Generate

```bash
./run.sh                                  # data/sample_sentences.txt -> variants.jsonl
./run.sh tests/data/base_sentences_v1.txt # the 455-variant regression corpus
./run.sh my.txt out.jsonl --quiet
./run.sh --verify                         # also runs stage 2 (slow, opt-in)
```

Generation never touches the LLM. Every record comes out `verified = null`.

### Test

```bash
./run_test.sh          # 376 tests, ~52 s
./run_test.sh fast     # 282, no downloads, 6 s
./run_test.sh verify   # 56, no model, 0.2 s
```

### Stage 2

```bash
ollama serve &
ollama pull qwen2.5:7b                    # 4.7 GB

./venv/bin/python -u scripts/verify_validate.py --sample 100 --constructed 60
# -> reports/verifier_validation.md ; CHECK THE GATE BEFORE THE NEXT COMMAND

./venv/bin/python -u -m negation.verification.cli \
    --input variants.jsonl --output variants_verified.jsonl \
    --model qwen2.5:7b --cola-threshold 0.3 --batch-size 40 --resume
```

Tier 1 downloads `textattack/roberta-base-CoLA` (~500 MB) on first use.

### Optional downloads

```bash
./venv/bin/python -m negation.data.fetch          # 6 corpora into data/external/
./venv/bin/python -m negation.antonym_vec.fetch   # GloVe 6B, 822 MB
```

### Environment assumptions

- Linux, CPU-only. No GPU present; `torch.cuda.is_available()` is `False`.
- 14 GB RAM. The 7B model uses ~5 GB. **Do not run the test suite concurrently
  with a verification run** — that is what evicted the model mid-session.
- Ollama 0.23.3 at `localhost:11434`. Models present: `qwen2.5:7b`,
  `qwen2.5:3b-instruct`, `qwen2.5:1.5b`, `qwen2:1.5b`.
- Python 3.14.7.

---

## 12. Open questions for the researcher

1. **Does the corpus need rebalancing before it becomes training data?** 38% of
   records come from H and I, which are mechanical wrappers over A. Should the
   generators be capped per family, or is the imbalance handled downstream by
   sampling?

2. **Is `net_negation` or `op_depth` the intended graded-intensity target?**
   Neither is a graded scalar. `intensity_hint` is the only ordinal-ish field
   (`neutral`/`hedged`/`intensified`/`partial`) and it is a four-way categorical,
   not a number. Nothing in the code currently produces the continuous adjustment
   the HSS work needs — that mapping has not been designed.

3. **How should K1 cancellations be scored?** They carry `net_negation = 0`,
   meaning "roughly equivalent to the affirmative". If the downstream penalty is
   a function of `net_negation`, K1 records will be treated as non-negated —
   which is arguably right semantically but means 42 of 455 records contribute
   nothing to a negation-sensitivity signal.

4. **Should the verifier be fixed, replaced, or scoped down?** See §10(4) for
   four options. This needs a decision before more compute is spent.

5. **Is an LLM-produced validation label set acceptable, or must the 100 records
   be labelled by a human?** The current labels are mine. If a human gold
   standard is required, that is the blocking item for stage 2, not the prompt.

6. **Which corpus should the detector actually train on?** The recommendation is
   CD-SCO + SFU, but SFU is **GPL-3.0** — copyleft. If a derived corpus or model
   is to be released, that needs a licensing decision, and SFU's terms may not be
   compatible with CANNOT's CC-BY-SA if they are ever combined.

7. **Should `data/sample_sentences.txt` stay as the 5 researcher-supplied
   sentences?** It is now the default input to `run.sh`, but the regression
   numbers everywhere in the docs come from `tests/data/base_sentences_v1.txt`.
   Two "default" corpora is a trap.
