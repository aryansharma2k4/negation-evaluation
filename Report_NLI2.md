# Report 2 — composite negation detector: honest baseline, rule fixes, conditionals, learned combiner

*Written for a reader who has read `Report_NLI.md`. Built up one step per commit; the step log
(section 11) is the running record, sections 1–10 summarise it.*

## 1. Summary

*(filled in when the last phase is committed)*

## 2. Dataset composition

*(filled in by Phase 1)*

## 3. Headline integrity check — does the gain survive an honest NLI threshold? (step 1.3)

*(filled in by step 1.3)*

## 4. Full ablation

*(filled in by Phases 2–4)*

## 5. Per-family results, best configuration

*(filled in by Phase 4)*

## 6. Error analysis

*(filled in at the end)*

## 7. What did not work

*(every change that failed to improve things, with its numbers; added to as we go)*

## 8. Caveats

*(added to as we go)*

## 9. Reproduction

*(commands per step are in the step log; collected here at the end)*

## 10. Next steps

*(at the end)*

## 11. Step log

### 0.1 Affixal generator bug — fixed

**What was wrong.** `negation/affixes.py:eligible()` let a negative prefix land on any adjective
or noun that WordNet links antonymously to its prefixed form. WordNet does link *critical* /
*uncritical* and *cyclic* / *noncyclic*, so the dictionary check passed, and the prefix went
onto the attributive adjective of a technical term: *the uncritical section*, *the noncyclic
redundancy check*. spaCy labels those adjectives `amod`, not `compound`, so a compound check alone
would not have caught them. *in-order* → *in-disorder* and *capacity* → *incapacity* came through
the noun path.

**Fix** (one function, shared by D_affixal and the K1 affixal generator):

| target | old rule | new rule |
| --- | --- | --- |
| adjective | any position | predicative only (`acomp`, `attr`, `oprd`, `xcomp`, `ccomp`, `ROOT`, or `conj` of one) |
| noun | any noun passing WordNet | only a whitelist of 13 abstract nouns (*ability*, *agreement*, *honesty*, …) |
| any | — | never `compound` dep, never next to a hyphen |
| any | WordNet antonymy gate | unchanged, still applied on top |

Regression tests: `tests/test_families.py::test_affixal_prefix_never_lands_inside_a_term`, with
the three examples from the brief plus five found by scanning `experiments/pattern` variants
(*unready queue*, *noncurrent process*, *Nonrandom early detection*, *nonhuman-readable*,
*incapacity*). Each is checked against both D_affixal and K1. A positive test confirms that
predicative adjectives (*is ineffective*, *stable and unfair*) and whitelisted nouns
(*inability*) still work. The generator suite: 385 passed (376 before, plus 9 new).

**How many existing variants were corrupted** (`experiments/nli2/affixal_audit.py` →
`results/0_1_affixal_audit.json`). A variant counts when the fixed generator, rerun on the same
base sentence, no longer produces it:

| corpus | variant kind | existing | rejected by the fixed rule | % |
| --- | --- | ---: | ---: | ---: |
| experiments/pattern (our OS/CN dataset) | D_affixal | 38 | 20 | 52.6% |
| experiments/pattern (our OS/CN dataset) | K1_affixal | 13 | 10 | 76.9% |
| negation_detection refsets (CANNOT / xNot360 bases) | D_affixal | 885 | 697 | 78.8% |
| negation_detection refsets (CANNOT / xNot360 bases) | K1_affixal | 383 | 307 | 80.2% |

This is an **upper bound** on corruption, not an exact count. The rule is deliberately strict,
so it also drops some legitimate variants (*an unequal time quantum*, *unqualified personnel*).
Examples of what it rejects:

| kind | base | old variant |
| --- | --- | --- |
| D_affixal | The critical section must satisfy mutual exclusion, progress and bounded waiting. | The uncritical section must satisfy … |
| D_affixal | The CPU scheduler selects a process from the ready queue … | … from the unready queue … |
| D_affixal | … requests resources in increasing order. | … in increasing disorder. |
| K1_affixal | Four necessary conditions must hold simultaneously … | Four unnecessary conditions must not hold simultaneously … |
| D_affixal | This action would be very expensive to implement. | This inaction would be very expensive to implement. |
| K1_affixal | Its common names are China aster and annual aster. | Its uncommon names are not China aster and annual aster. |
| D_affixal (legitimate, also rejected) | … every process receives an equal time quantum. | … an unequal time quantum. |

**What this does to earlier numbers.** The saved variant files were *not* regenerated, so every
earlier experiment (polarity, composite, `Report_NLI.md`) and step 0.2 below still use the old
variants. 11 of the 91 K1 pairs used in evaluation come from the K1-affixal generator, and the
fixed rule rejects 9 of those 11, so the reported K1 FPRs include 9 defective pairs (9.9% of K1). Phase 5 regenerates training data with the fixed generator.

```sh
./venv/bin/python -m pytest tests -q                    # 385 passed
./venv/bin/python experiments/nli2/affixal_audit.py
```
