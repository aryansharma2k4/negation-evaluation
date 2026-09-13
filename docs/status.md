# Project status

Mermaid sources tracking what is built. Same style as `docs/diagrams.md`.
Last updated after stage 1.5 (arbitrary input) completion.

## 1. Where the project stands

```mermaid
flowchart LR
    S1["STAGE 1 - Variant generation<br/>COMPLETE<br/>20 generators, 13 families<br/>448 variants from 20 base sentences<br/>75 tests passing"]

    S1 --> S15["STAGE 1.5 - Arbitrary input<br/>COMPLETE<br/>+13 generators, +1 family (M_modal)<br/>negate / affirm / rescope, all clause types<br/>every new record op_depth = 1<br/>stage 1 still 448; 206 tests passing"]

    S15 --> HOOK["Handoff: variants.jsonl<br/>every record carries verified = null"]

    HOOK --> S2["STAGE 2 - Verification<br/>NOT STARTED<br/>LLM checks each variant<br/>sets verified = true or false"]

    S2 --> S3["LATER STAGES<br/>NOT SPECIFIED YET<br/>similarity modelling and evaluation"]

    classDef done fill:#ecfdf5,stroke:#059669,color:#064e3b
    classDef todo fill:#f1f5f9,stroke:#94a3b8,color:#334155
    classDef hand fill:#eef2ff,stroke:#4f46e5,color:#1e1b4b
    class S1,S15 done
    class HOOK hand
    class S2,S3 todo
```

## 2. Stage 1 build status, module by module

```mermaid
flowchart TD
    ENV["Environment<br/>venv, spaCy 3.8 + en_core_web_sm,<br/>NLTK WordNet, pytest"]

    ENV --> CORE["CORE INFRASTRUCTURE - 12 modules"]
    ENV --> GENS["GENERATORS - 14 modules"]
    ENV --> PIPE["PIPELINE"]
    ENV --> QA["TESTS AND DOCS"]

    CORE --> C1["schema.py<br/>NegationVariant, family constants,<br/>MAX_DEPTH cap"]
    CORE --> C2["nlp_core.py<br/>one spaCy pipeline, lru_cache WordNet,<br/>morphology, clause_scope"]
    CORE --> C3["lexicons.py<br/>all word lists frozen at import"]
    CORE --> C4["splice.py<br/>character-offset Edit, cue tracking"]
    CORE --> C5["frames.py<br/>the three clause frames"]
    CORE --> C6["polarity.py<br/>explicit vs lexical negation"]
    CORE --> C7["scope.py<br/>K1 vs K2 containment decision"]
    CORE --> C8["base.py<br/>Generator ABC, registry, record build,<br/>clause_types licensing"]
    CORE --> C9["classify.py<br/>InputProfile: clause type, voice, polarity,<br/>existing cues across all families"]
    CORE --> C10["affixes.py<br/>affixal derivation both ways,<br/>WordNet antonymy gated"]

    GENS --> G1["A_syntactic - 3 generators<br/>copula, existing_aux, do_support"]
    GENS --> G2["B, C, D, E - 4 generators<br/>lexical families"]
    GENS --> G3["F, G, J - 5 generators<br/>constructional families"]
    GENS --> G4["H, I - 2 generators<br/>composed over A, depth 2"]
    GENS --> G5["K1, K2 - 5 generators<br/>double negation, depth 2"]
    GENS --> G6["L - 1 generator<br/>scope alternation"]
    GENS --> G7["affirm.py - 7 generators<br/>one per cue family, run backwards"]
    GENS --> G8["rescope.py - 2 generators<br/>subject scope and predicate scope"]
    GENS --> G9["m_sentence_type.py - 5 generators<br/>interrogative, imperative, existential,<br/>modal readings, comparative"]
    GENS --> G10["n_clause_target.py - 1 generator<br/>one clause per record"]

    PIPE --> P1["filters.py<br/>dedup, blacklist, batched re-parse,<br/>op_depth guardrail (raises)"]
    PIPE --> P2["driver.py<br/>generate_all, parse once per sentence"]
    PIPE --> P3["cli.py + run.sh<br/>JSONL output and summary"]

    QA --> Q1["206 tests across 7 files<br/>clause_type x polarity fixtures,<br/>affirm/negate round trip,<br/>depth-1 over whole corpora,<br/>448 stage-1 regression"]
    QA --> Q2["README.md<br/>docs/diagrams.md<br/>data/sample_sentences.txt"]

    classDef done fill:#ecfdf5,stroke:#059669,color:#064e3b
    classDef grp fill:#eef2ff,stroke:#4f46e5,color:#1e1b4b
    class ENV,CORE,GENS,PIPE,QA grp
    class C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,G1,G2,G3,G4,G5,G6,G7,G8,G9,G10,P1,P2,P3,Q1,Q2 done
```

## 3. Requirements coverage

```mermaid
flowchart LR
    REQ["Stage 1 brief"]

    REQ --> R1["Setup rules<br/>model loaded once at module level,<br/>nlp.pipe batch_size 256,<br/>lru_cache on WordNet,<br/>lexicons frozen at import<br/>MET"]
    REQ --> R2["One parse per base sentence,<br/>every generator reads that Doc<br/>MET"]
    REQ --> R3["Generator ABC with cheap applies,<br/>registry pattern,<br/>splicing at token offsets<br/>MET"]
    REQ --> R4["All 13 families implemented<br/>MET"]
    REQ --> R5["Composition capped at depth 2,<br/>enforced in two places<br/>MET"]
    REQ --> R6["Three filters in cost order<br/>MET"]
    REQ --> R7["generate_all + CLI + JSONL + summary<br/>MET"]
    REQ --> R8["Type hints, docstring per generator,<br/>tests per family, do_support tense,<br/>K1 vs K2 scope<br/>MET"]
    REQ --> R9["No LLM calls, verified hook left<br/>MET"]

    classDef done fill:#ecfdf5,stroke:#059669,color:#064e3b
    classDef grp fill:#eef2ff,stroke:#4f46e5,color:#1e1b4b
    class REQ grp
    class R1,R2,R3,R4,R5,R6,R7,R8,R9 done
```

## 3b. Stage 1.5 requirements coverage

```mermaid
flowchart LR
    REQ2["Stage 1.5 brief - arbitrary input"]

    REQ2 --> S1R1["classify.py with InputProfile,<br/>cues detected across ALL families,<br/>existing lexicons reused not duplicated<br/>MET"]
    REQ2 --> S1R2["Three operations: negate, affirm, rescope,<br/>operation field on every record<br/>MET"]
    REQ2 --> S1R3["7 affirmation generators<br/>incl. do-support collapse and<br/>WordNet-validated affixal strip<br/>MET"]
    REQ2 --> S1R4["Rescope for quantified negated input,<br/>both directions<br/>MET"]
    REQ2 --> S1R5["Sentence types: interrogative, imperative,<br/>passive, existential, modal, comparative,<br/>conditional, multi-clause<br/>MET"]
    REQ2 --> S1R6["Schema additions incl. target_clause_idx,<br/>input_polarity, input_cue_count<br/>MET"]
    REQ2 --> S1R7["DEPTH 1 asserted before emit,<br/>raises DepthViolation, never silent<br/>MET"]
    REQ2 --> S1R8["Applicability gating per generator<br/>plus clause_types licensing<br/>MET"]
    REQ2 --> S1R9["No regression: stage 1 still 448,<br/>all 75 original tests pass<br/>MET"]
    REQ2 --> S1R10["depth field named op_depth<br/>DEVIATION - depth was already taken,<br/>with different semantics, by stage 1"]

    classDef done fill:#ecfdf5,stroke:#059669,color:#064e3b
    classDef note fill:#fffbeb,stroke:#d97706,color:#78350f
    classDef grp fill:#eef2ff,stroke:#4f46e5,color:#1e1b4b
    class REQ2 grp
    class S1R1,S1R2,S1R3,S1R4,S1R5,S1R6,S1R7,S1R8,S1R9 done
    class S1R10 note
```

## 4. Open items carried into stage 2

```mermaid
flowchart TD
    OPEN["Known gaps - none block stage 2"]

    OPEN --> BUG["DEFECT<br/>B_quantifier both_to_neither<br/>does not fix noun number.<br/>Both tests passed produces<br/>Neither tests passed - ungrammatical.<br/>Filters do not catch it.<br/>The affirming direction now does<br/>fix number; the negating one still does not."]

    OPEN --> AFFIRMGAP["SCOPE GAP<br/>G_prepositional has no affirmation.<br/>runs without a cache is classified<br/>negated and can be negated further,<br/>but is not turned back into<br/>runs with a cache."]

    OPEN --> QPARSE["PARSER LIMIT<br/>Fronted-copula questions get only the<br/>contracted form: en_core_web_sm cannot<br/>place the subject/predicate boundary in<br/>Is the task simple? - it reads<br/>the task simple as one NP."]

    OPEN --> PARSE["PARSER LIMIT<br/>en_core_web_sm misparses some<br/>VP coordinations - prints tagged NOUN.<br/>K2 silently under-fires.<br/>en_core_web_trf would fix it."]

    OPEN --> RECALL["DELIBERATE RECALL LOSS<br/>D_affixal requires a WordNet antonymy edge,<br/>so unsafe and helpless are missed.<br/>Chosen over emitting disarray and intense."]

    OPEN --> SCOPE["SCOPE LIMIT<br/>F_implicit only restructures bare finite verbs.<br/>I_intensified not_at_all and by_no_means<br/>skip do-support frames."]

    OPEN --> LEGACY["HOUSEKEEPING<br/>neg.py and result.txt are the old<br/>itertools prototype, unused by the pipeline."]

    classDef bug fill:#fef2f2,stroke:#dc2626,color:#7f1d1d
    classDef warn fill:#fffbeb,stroke:#d97706,color:#78350f
    classDef ok fill:#f1f5f9,stroke:#94a3b8,color:#334155
    class OPEN warn
    class BUG bug
    class PARSE,SCOPE,QPARSE,AFFIRMGAP warn
    class RECALL,LEGACY ok
```

## 5. What runs today

```mermaid
flowchart LR
    CMD1["./run.sh"] --> OUT1["variants.jsonl plus summary by family,<br/>subtype, operation and input type.<br/>Stage 1 alone still yields 448<br/>on the frozen 20-sentence corpus."]
    CMD2["./run.sh my_sentences.txt -o out.jsonl"] --> OUT2["same, on your own sentences"]
    CMD3["./venv/bin/python -m pytest tests -q"] --> OUT3["206 passed"]

    classDef cmd fill:#eef2ff,stroke:#4f46e5,color:#1e1b4b
    classDef done fill:#ecfdf5,stroke:#059669,color:#064e3b
    class CMD1,CMD2,CMD3 cmd
    class OUT1,OUT2,OUT3 done
```
