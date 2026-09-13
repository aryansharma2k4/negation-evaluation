# Verifier validation

The verifier's output is only worth using as training signal if the verifier itself has been checked. This measures agreement between it and two reference sets, per question.

- verifier under test: `ollama/qwen2.5:7b+p2`
- gate: Cohen's kappa >= 0.6 on every question

**Kappa, not accuracy, is the number to read.** These labels are heavily skewed -- most generated variants really are grammatical -- and on a set that is 90% one class a verifier answering "true" unconditionally scores 0.90 accuracy while being worthless. Kappa corrects for that.

## 1. Constructed reference set (ground truth by construction)

The stronger of the two. Records were perturbed so that the correct answer follows from *how they were made* rather than from anyone's judgement: a variant with two adjacent words swapped is ungrammatical whatever a reader thinks, and a record whose family label has been replaced with an incompatible one is miscategorised by definition. No annotator is involved, so no annotator bias can leak in.

Composition:

- `clean`: 20
- `family_swapped`: 20
- `grammar_corrupted`: 20

The untouched group matters as much as the perturbed ones: without it, a verifier that rejects everything would score perfectly.

Records matched: 60

| question | n | accuracy | Cohen's kappa | strength | caught known-bad | gate (k >= 0.6) |
| --- | ---: | ---: | ---: | --- | ---: | :---: |
| `grammatical` | 60 | 0.617 | -0.030 | worse than chance | 10% | **FAIL** |
| `semantic` | 0 | — | — | not measured | — | n/a |
| `category` | 20 | 0.700 | undefined | reference has no variance | 70% | n/a |

**`grammatical`** — reference true: 40/60, verifier true: 53/60

| | verifier says true | verifier says false |
| --- | ---: | ---: |
| **reference true** | 35 | 5 |
| **reference false** | 18 | 2 |

**`category`** — reference true: 0/20, verifier true: 6/20

| | verifier says true | verifier says false |
| --- | ---: | ---: |
| **reference true** | 0 | 0 |
| **reference false** | 6 | 14 |

## Verdict

Not every question could be measured. Where the reference set used a single class throughout, kappa is undefined rather than zero, and the table says so instead of recording a failure: `semantic` (n=0), `category` (n=20). For those, read the *caught known-bad* column instead.

**The verifier does not clear the gate.** Questions below kappa 0.6: `grammatical` (k = -0.030).

Its output should not be used as training signal on those questions until the prompt is improved and this is re-run.

## Notes and limitations

- The labelled reference set did not run: ollama evicted the model partway through and the remaining requests failed instantly. The code marked them verified = null rather than guessing, which is correct, but it means only the constructed set is measured here.
- The constructed set can only assert the labels its perturbations determine. A clean record is asserted grammatical but not semantically valid or correctly categorised, because asserting those would assume the generator is right, which is the thing under test.
- Agreement is not correctness. Both reference sets could be wrong in the same direction as the verifier, and a high kappa would not reveal it.
