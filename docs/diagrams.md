# Diagrams

Mermaid sources for the negation-variant pipeline. Renders as-is on GitHub,
Obsidian, Notion, VS Code preview, and mermaid.live.

## 1. Family taxonomy

```mermaid
flowchart LR
    ROOT["13 negation families"]

    ROOT --> SYN["Syntactic<br/>negator in the verb complex"]
    ROOT --> LEX["Lexical<br/>a cue word replaces a word"]
    ROOT --> CON["Constructional<br/>the predicate is rebuilt"]
    ROOT --> MOD["Force modulation<br/>composed over A"]
    ROOT --> DBL["Double negation<br/>two cues in one sentence"]
    ROOT --> SCP["Scope alternation"]

    SYN --> A["A_syntactic<br/>The function does not sort the array"]

    LEX --> B["B_quantifier<br/>No students solved the problem"]
    LEX --> C["C_neg_adverb<br/>The function never sorts the array"]
    LEX --> D["D_affixal<br/>The result is unimportant"]
    LEX --> E["E_antonym<br/>Everyone failed the exam"]

    CON --> F["F_implicit<br/>The function fails to sort the array"]
    CON --> G["G_prepositional<br/>The system is devoid of redundancy"]
    CON --> J["J_contrastive<br/>The result is anything but important"]

    MOD --> H["H_hedged<br/>probably does not sort<br/>intensity_hint = hedged"]
    MOD --> I["I_intensified<br/>definitely does not sort<br/>intensity_hint = intensified"]

    DBL --> K1["K1_cancellation<br/>The task is not impossible<br/>net_negation = 0"]
    DBL --> K2["K2_compound<br/>does not sort and does not return<br/>net_negation = 2"]

    SCP --> L["L_scope_position<br/>Not everyone passed<br/>vs Everyone did not pass"]

    classDef grp fill:#eef2ff,stroke:#4f46e5,color:#1e1b4b
    classDef fam fill:#ffffff,stroke:#94a3b8,color:#0f172a
    classDef zero fill:#ecfdf5,stroke:#059669,color:#064e3b
    classDef two fill:#fef2f2,stroke:#dc2626,color:#7f1d1d

    class ROOT,SYN,LEX,CON,MOD,DBL,SCP grp
    class A,B,C,D,E,F,G,H,I,J,L fam
    class K1 zero
    class K2 two
```

## 2. Families by subtype

```mermaid
flowchart TD
    A["A_syntactic"] --> A1["copula<br/>The result is not important"]
    A --> A2["existing_aux<br/>The model cannot handle noise"]
    A --> A3["do_support<br/>The function does not sort the array"]

    B["B_quantifier"] --> B1["some_to_no"]
    B --> B2["all_to_none_of"]
    B --> B3["every_to_no"]
    B --> B4["both_to_neither"]
    B --> B5["always_to_never"]
    B --> B6["someone_to_nobody<br/>something_to_nothing<br/>somewhere_to_nowhere"]

    C["C_neg_adverb"] --> C1["absolute<br/>never"]
    C --> C2["partial<br/>hardly, barely, rarely,<br/>seldom, scarcely"]

    D["D_affixal"] --> D1["prefix_un, prefix_in, prefix_im,<br/>prefix_ir, prefix_il, prefix_non, prefix_dis"]
    D --> D2["suffix_less<br/>useful to useless"]

    E["E_antonym"] --> E1["antonym_adj"]
    E --> E2["antonym_verb"]
    E --> E3["antonym_adv"]

    F["F_implicit"] --> F1["fails_to, refuses_to,<br/>neglects_to, ceases_to"]
    F --> F2["is_unable_to"]
    F --> F3["denies"]
    F --> F4["lacks"]

    G["G_prepositional"] --> G1["without"]
    G --> G2["devoid_of"]
    G --> G3["free_of"]
    G --> G4["in_the_absence_of"]

    H["H_hedged"] --> H1["maybe_not, possibly_not,<br/>probably_not"]
    H --> H2["might_not, may_not"]

    I["I_intensified"] --> I1["surely_not, definitely_not,<br/>certainly_not, absolutely_not"]
    I --> I2["not_at_all, by_no_means"]

    J["J_contrastive"] --> J1["anything_but"]
    J --> J2["far_from"]
    J --> J3["the_opposite_of"]

    K1["K1_cancellation"] --> KA["cancel_impossible<br/>negative word already in base"]
    K1 --> KB["not_prefix_un<br/>negator plus fresh derivation"]
    K1 --> KC["never_fails_to<br/>adverb over a trigger"]

    K2["K2_compound"] --> KD["do_support+do_support"]
    K2 --> KE["copula+do_support"]
    K2 --> KF["not+prefix_un"]

    L["L_scope_position"] --> L1["subject_scope<br/>Not everyone passed"]
    L --> L2["predicate_scope<br/>Everyone did not pass"]
```

## 3. A_syntactic: the three clause frames

```mermaid
flowchart TD
    IN["Clause head"] --> Q1{"Does the head govern<br/>an aux or auxpass?"}

    Q1 -->|yes| AUX["existing_aux"]
    Q1 -->|no| Q2{"Is the head lemma be?"}
    Q2 -->|yes| COP["copula"]
    Q2 -->|no| DO["do_support"]

    AUX --> AUX2["Insert not after the auxiliary<br/>She has not finished the report"]
    AUX2 --> AUX3["can is lexicalised<br/>can handle to cannot handle"]

    COP --> COP2["Insert not after be<br/>The result is not important"]

    DO --> DO2["Read Tense, Person, Number<br/>from token.morph"]
    DO2 --> DO3["sorts to does not sort"]
    DO2 --> DO4["sorted to did not sort"]
    DO2 --> DO5["sort to do not sort"]
    DO3 --> DO6["Lexical verb reverts to its lemma;<br/>the auxiliary lands left of<br/>any preverbal adverb"]
    DO4 --> DO6
    DO5 --> DO6

    classDef frame fill:#eef2ff,stroke:#4f46e5,color:#1e1b4b
    class AUX,COP,DO frame
```

## 4. K1 vs K2: the scope-containment decision

```mermaid
flowchart TD
    IN["Variant carries two negation cues"] --> Q{"Is cue 2 inside<br/>clause_scope of cue 1's head?"}

    Q -->|contained| K1["K1_cancellation<br/>the cues cancel<br/>net_negation = 0"]
    Q -->|separate| K2["K2_compound<br/>both cues survive<br/>net_negation = 2"]

    K1 --> K1E["The result is not unimportant"]
    K2 --> K2E["The result is unimportant<br/>and the method does not work"]

    Q -.-> NOTE["clause_scope descends the parse<br/>but stops at conj, advcl, ccomp, relcl.<br/>A plain subtree test answers 'contained'<br/>for BOTH cases and silently mislabels K2."]

    CNT["Counting cues cannot separate these:<br/>both have exactly 2"] -.-> Q

    classDef zero fill:#ecfdf5,stroke:#059669,color:#064e3b
    classDef two fill:#fef2f2,stroke:#dc2626,color:#7f1d1d
    classDef note fill:#fffbeb,stroke:#d97706,color:#78350f
    class K1,K1E zero
    class K2,K2E two
    class NOTE,CNT note
```

## 5. Why a subtree test is not enough

```mermaid
flowchart TD
    subgraph SG["The function sorts the array and returns the result"]
        R["sorts - ROOT"]
        R --> S["function - nsubj"]
        R --> O["array - dobj"]
        R -->|"conj = CLAUSE BOUNDARY"| RET["returns"]
        RET --> O2["result - dobj"]
    end

    C1["cue 1 attaches to 'sorts'"] -.-> R
    C2["cue 2 attaches to 'returns'"] -.-> RET

    SUB["subtree of 'sorts'<br/>includes 'returns'<br/>so a subtree test says CONTAINED to K1"]
    SCO["clause_scope of 'sorts'<br/>stops at the conj edge<br/>so it says SEPARATE to K2 - correct"]

    R -.-> SUB
    R -.-> SCO

    classDef bad fill:#fef2f2,stroke:#dc2626,color:#7f1d1d
    classDef good fill:#ecfdf5,stroke:#059669,color:#064e3b
    class SUB bad
    class SCO good
```

## 6. Pipeline

```mermaid
flowchart TD
    IN["data/sample_sentences.txt"] --> P["nlp.pipe batch_size=256<br/>each base parsed exactly ONCE"]
    P --> REG["Registry: 20 generators<br/>across 13 families"]

    REG --> AP{"applies(doc)<br/>dependency and POS checks only"}
    AP -->|false| SKIP["skipped, no string work"]
    AP -->|true| GEN["generate(doc)<br/>character-offset splicing<br/>at spaCy token boundaries"]

    GEN --> F1["Filter 1: dedup<br/>normalised hash"]
    F1 --> F2["Filter 2: regex blacklist<br/>not not, stacked aux,<br/>doubled determiners"]
    F2 --> F3["Filter 3: batched re-parse<br/>one ROOT, no cycles,<br/>depth delta at most 3"]
    F3 --> OUT["variants.jsonl<br/>one NegationVariant per line"]
    OUT --> V["verified = null<br/>stage 2 hook"]

    classDef cheap fill:#ecfdf5,stroke:#059669,color:#064e3b
    classDef costly fill:#fef2f2,stroke:#dc2626,color:#7f1d1d
    class F1,F2 cheap
    class F3,P costly
```

## 7. Metadata fields

```mermaid
flowchart LR
    V["NegationVariant"] --> FAM["family<br/>what kind of negation"]
    V --> SUB["subtype<br/>which rule fired"]
    V --> SCT["scope_target<br/>subject / predicate / clause"]
    V --> INT["intensity_hint<br/>neutral / hedged /<br/>intensified / partial"]
    V --> GEN["generator<br/>versioned code path"]
    V --> DEP["depth<br/>1 atomic, 2 composed, cap 2"]
    V --> NET["net_negation<br/>0 cancelled, 1 plain, 2 compound"]
    V --> CUE["cue_tokens + cue_char_spans<br/>variant[start:end] == token"]

    DEP -.->|"not the same thing"| NET
```

## Mindmap alternative

Compact version of diagram 1 for slides. Needs Mermaid 10 or newer.

```mermaid
mindmap
  root((Negation families))
    Syntactic
      A_syntactic
    Lexical
      B_quantifier
      C_neg_adverb
      D_affixal
      E_antonym
    Constructional
      F_implicit
      G_prepositional
      J_contrastive
    Force
      H_hedged
      I_intensified
    Double
      K1_cancellation
      K2_compound
    Scope
      L_scope_position
```
