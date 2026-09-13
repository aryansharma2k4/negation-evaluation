# Dataset evaluation for the cue/scope detector and the intensity regressor

Everything below was checked by fetching it on **2026-09-13**. Sizes are counts
taken from the files themselves, not from papers or dataset cards; where a card
and the artefact disagree, the artefact wins and the disagreement is noted.
Where something could not be located, it says so rather than guessing.

Reproduce with:

```bash
./venv/bin/python -m negation.data.fetch          # pulls into data/external/
./venv/bin/python scripts/dataset_stats.py        # the numbers in this report
```

---

## Summary

| corpus | found | licence | gold spans | hedges separate | verdict |
| --- | --- | --- | --- | --- | --- |
| ConanDoyle-neg / CD-SCO | yes, via mirror | **none stated** | 1,421 | **no** | detector training |
| SFU Review (English) | yes | **GPL-3.0-or-later** | 8,891 | **yes** | detector + intensity |
| BioScope | partly | CC-BY-2.0 (asserted) | 5,511 | **yes** | intensity, held out |
| CANNOT | yes | CC-BY-SA-4.0 | 0 | no | rejected for detector |
| NegNLI | yes | **none stated** | 0 | no | held-out eval only |
| NaN-NLI | yes | CC-BY-SA-4.0 | 0 | no | held-out eval only |
| SNLI / MultiNLI | yes | see below | 0 | no | rejected |
| xNLI-neg | **does not exist under that name** | — | — | — | see §8 |

**Total usable gold-span annotation: 15,823 across three corpora, of which
8,739 are hedge/speculation cues.**

---

## 1. ConanDoyle-neg / CD-SCO (Morante & Daelemans, 2012)

**Availability — recovered, but not from an official source.** The canonical
\*SEM 2012 shared-task distribution page,
`https://www.clips.uantwerpen.be/sem2012-st-neg/data.html`, returns **HTTP 404**;
the whole `sem2012-st-neg` directory is gone. The corpus is not on HuggingFace in
its complete form (see §9). The four original CoNLL files were recovered from a
third-party research repository:

```
https://raw.githubusercontent.com/csenge-szabo/Negation_Scope_Detection/main/data/raw_data/
  SEM-2012-SharedTask-CD-SCO-training-09032012.txt      2,898,575 B
  SEM-2012-SharedTask-CD-SCO-dev-09032012.txt             547,140 B
  SEM-2012-SharedTask-CD-SCO-test-cardboard-GOLD.txt      401,244 B
  SEM-2012-SharedTask-CD-SCO-test-circle-GOLD.txt         342,076 B
```

These are byte-for-byte the shared-task format, with the cue/scope/event columns
intact. That they are reachable only through an unaffiliated mirror is a
**supply risk**, not a quality problem: cache a copy.

**Licence — none found anywhere.** The mirror states no terms, and the original
distribution page that might have is gone. The paper describes the corpus as
freely available for research. Treat as *research use, do not redistribute*.

**Size (counted):**

| split | sentences | negation instances |
| --- | --- | --- |
| training | 3,644 | 984 |
| dev | 787 | 173 |
| test-cardboard | 496 | 133 |
| test-circle | 593 | 131 |
| **total** | **5,520** | **1,421** (over 1,227 sentences) |

**Schema.** CoNLL: seven fixed columns, then a *triple* of columns (cue, scope,
negated event) appended per negation in the sentence — a sentence with three
negations has sixteen columns. Cue and scope are token-level and may be
discontinuous. Crucially, the cue column holds the **affix alone** for affixal
negation: `in` on *infrequent*, not the whole word. It also annotates the negated
event, which neither of the other corpora do.

**Domain.** Two Conan Doyle stories. Late-Victorian literary narrative — long
sentences, inverted and archaic constructions (*save upon those occasions*).

**Hedges/speculation: NO.** Negation only. Cannot train the intensity regressor.

**Cancellation vs compound: NO — confirmed.** 169 sentences carry 2+ negation
cues; each cue is annotated as an independent instance with no relation recorded
between them. See §7.

**Overlap with our generators: low, and useful where it is high.** Family
distribution over our taxonomy:

```
A_syntactic       594  41.8%
B_quantifier      446  31.4%
D_affixal         223  15.7%
C_neg_adverb       91   6.4%
G_prepositional    39   2.7%
unmapped           13   0.9%
L_scope_position    7   0.5%
I_intensified       4   0.3%
F_implicit          3   0.2%
J_contrastive       1   0.1%
```

Our generated corpus is weighted the opposite way (C 22%, H 20%, I 19%, F 15%,
A 5%, B 1%). The two are complementary rather than redundant: CD-SCO is rich in
exactly the families our generators produce least of (A, B, D), and the register
shares no vocabulary with our synthetic technical sentences. The 13 unmapped
cues are exceptives — *save*, *except*, *rather than*, *on the contrary* — which
our taxonomy has no family for. That is a genuine gap, not a mapping bug.

---

## 2. SFU Review Corpus, negation **and** speculation (Konstantinova et al., 2012)

**Availability — yes, direct from the author.**

```
https://www.sfu.ca/~mtaboada/docs/research/SFU_Review_Corpus_Negation_Speculation.zip
  1,540,440 B — 412 files: 400 annotated XML reviews, annotation guidelines, LREC paper
```

**Licence — GPL-3.0-or-later.** Stated verbatim in `README.txt` inside the zip:

> This corpus is free software: you can redistribute it and/or modify it under
> the terms of the GNU General Public License … either version 3 of the License,
> or (at your option) any later version.

The web page carries no licence statement; the README is the authority. **This
is copyleft** — redistributing a derived corpus means releasing it under GPL-3.0
as well. Whether model weights trained on it are a derived work is unsettled;
flag it before publishing anything built on this.

> **Do not confuse with `SINAI/SFU-Review-SP-Neg` on HuggingFace.** That is the
> **Spanish** corpus (`language:es`) under **CC-BY-NC-SA-4.0** — a different
> resource, with a non-commercial clause this one does not have.

**Size (counted):** 400 documents, 17,263 sentences, **8,891 cues** over 5,948
annotated sentences — **3,528 negation** and **5,363 speculation**. 1,083
sentences carry both.

**Schema.** Inline XML. `<cue type="negation"|"speculation" ID=n>` wraps the cue;
`<xcope ID=m>` wraps the scope and contains a `<ref SRC=n>` pointing back at the
cue it belongs to. Token-level, scope excludes cue, both may be discontinuous.

**Domain.** Consumer reviews in eight categories (books, cars, computers,
cookware, hotels, movies, music, phones). Informal register with contractions,
which is why the corpus tokenises them apart (`do n't`, `ca n't`) — and why 67 of
the files carry raw Windows-1252 apostrophes that survive XML parsing as U+0092
and have to be normalised.

**Hedges/speculation: YES — and it is the best source found.** 5,363 speculation
cues, separately typed, in the same documents and the same schema as the
negation cues. This is the corpus the intensity regressor should be built on.

**Cancellation vs compound: NO — confirmed.** 369 multi-cue sentences, all
annotated as independent instances.

**Overlap with our generators: low.** A 26%, B 7%, C 3%, G 2%, H 60%. Our
generated H_hedged records are built from a five-modifier lexicon
(*maybe/possibly/probably/might/may*); SFU's 5,363 speculation cues span a far
wider inventory. It extends our coverage rather than duplicating it.

---

## 3. BioScope (Vincze et al., 2008)

**Availability — two of three subcorpora, and the HuggingFace loader is broken.**

The URL that `bigbio/bioscope` on HuggingFace still points at,
`https://rgai.sed.hu/sites/rgai.sed.hu/files/bioscope.zip`, **does not resolve**
(connection failure — the `rgai.sed.hu` host is gone). That loader cannot run.
The corpus has moved:

```
https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/abstracts_pmid.xml   2,955,559 B
https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/full_papers.xml        591,638 B
https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/bioscope.zip         1,078,189 B
```

**The clinical subcorpus is unobtainable.** What ships is
`clinical_records_anon.xml` — 6,383 sentences of annotation in which *every token
is `*`*. The words come from the 2007 CMC Medical NLP Challenge data, which the
corpus README says to download from `computationalmedicine.org/catalog`. **That
host no longer resolves.** There is no documented route to reconstructing it, so
its 872 negation and 1,137 speculation cues are not available. `load_bioscope`
refuses that subcorpus with an explanation rather than returning 2,009 sentences
reading `* * *`.

**Licence — asserted, not stated.** The corpus homepage says only that it is
publicly available for research; there is no licence document. The CC-BY-2.0
claim comes from the `bigbio` dataset card, i.e. from a third party, not from the
authors. Treat as *research use*, and do not rely on the CC-BY claim for
redistribution.

**Size (counted, abstracts + full papers only):** 14,541 sentences (11,871 +
2,670), 1,282 documents, **5,511 cues** over 4,155 annotated sentences —
**2,135 negation** and **3,376 speculation**. 401 sentences carry both.

**Schema.** Inline XML, linked the *opposite* way from SFU: `<xcope id="X1.2">`
wraps the scope and the `<cue ref="X1.2">` sits *inside* it, pointing outward. So
the cue tokens are a subset of the scope tokens and must be subtracted to match
the other corpora's convention.

**Domain.** Biomedical: GENIA abstracts and full articles. Highly specialised
register — this is the reason to hold it out rather than mix it in (§10).

**Hedges/speculation: YES.** 3,376 speculation cues, separately typed.

**Cancellation vs compound: NO — confirmed.** 191 multi-cue sentences.

**Overlap with our generators: low, different failure mode.** F_implicit is 4.3%
here against 0.2% in CD-SCO — biomedical text uses *fails to*, *absence of*,
*lacks* heavily, which is the family our generators over-produce. Useful as a
check that our F records are not merely synthetic-looking.

---

## 4. CANNOT (Anschütz et al., 2023)

**Availability — yes.** `tum-nlp/cannot-dataset`, file
`cannot_dataset_v1.1.tsv`, 77,376 rows. Homepage `github.com/dmlls/cannot-dataset`.

**Licence — CC-BY-SA-4.0** (declared on the dataset card). Share-alike: a derived
corpus must carry the same licence.

**Size (counted):** 77,376 premise/hypothesis pairs — 39,989 labelled `1`, 37,387
labelled `0`.

**Schema.** Three columns: `premise`, `hypothesis`, `label`. **Sentence-pair
only.** There are no cue spans and no scope spans, at any granularity.

**Domain.** Mixed — assembled from other NLI and paraphrase resources, so the
register is heterogeneous (Wikipedia, image captions, more).

**Hedges/speculation: NO.**

**Cancellation vs compound: N/A** — no cue annotation at all, so the question
does not arise.

**Verdict: rejected for detector training.** A cue/scope detector needs spans and
this has none. The tempting move — run our own classifier over the hypotheses to
produce silver spans — is **circular**: it would train the detector on labels
produced by the taxonomy the detector is supposed to be evaluated against, and
any systematic error in our lexicons would be learned rather than exposed. It
loads with `provenance="pair-level-only"` and empty spans precisely so that this
shows up as a zero in any count of usable training material.

Its real use is as a **negation-aware sentence-pair resource for the downstream
similarity model**, which is a different component from the two under evaluation
here.

---

## 5. NegNLI (Hossain et al., 2020)

**Availability — yes, once located.** It is not under the name "NegNLI" anywhere;
it is the "new benchmarks" of *An Analysis of Natural Language Inference
Benchmarks through the Lens of Negation* (EMNLP 2020):

```
https://raw.githubusercontent.com/mosharafhossain/negation-and-nli/master/data/new_benchmarks/clean_data/
  RTE.txt    308,027 B    1,500 pairs
  SNLI.txt   172,716 B    1,500 pairs
  MNLI.txt   239,931 B    1,500 pairs
```

**Licence — none.** The repository has no `LICENSE` file (checked: HTTP 404).
Do not redistribute.

**Size (counted):** 4,500 pairs, matching the paper. The files are **Latin-1**,
not UTF-8, and fail to decode otherwise.

**Schema.** `index`, `Text`, `Hypothesis`, `gold_label`. Sentence-pair only, no
spans.

**Domain.** RTE / SNLI / MNLI source material with negation added **by hand** to
the main verb of the premise, the hypothesis, or both.

**Hedges/speculation: NO.**

**Verdict: held-out evaluation only.** No spans, so it cannot train the detector;
and because the negation was inserted manually into existing pairs, the
sentences are not naturally occurring. That makes it a good *test* of whether a
negation-sensitive model actually changed its prediction, which is the question
the project ultimately asks.

---

## 6. NaN-NLI (Truong et al., 2022)

**Availability — yes.** `joey234/nan-nli`, file `nan.csv`, 258 rows.

**Licence — CC-BY-SA-4.0** (declared on the dataset card).

**Size (counted):** 258 pairs. Tiny.

**Schema.** Sentence-pair, no spans — but it is the only resource found that
labels each pair with the **construction** of its negation, along Pullum &
Huddleston's axes: verbal/non-verbal, analytic/synthetic, clausal/sub-clausal,
ordinary/meta-linguistic, plus a `Construction` and `Construction Subtype`
column. The loader carries these through in `extra`.

**Hedges/speculation: NO.**

**Overlap with our generators: the highest of any corpus here, and that is the
point.** Its first rows are:

```
Not all people have had the opportunities you have had.
  -> Some people have not had the opportunities you have had.   (entailment)
```

That is exactly the alternation our `L_scope_position` family and our `rescope`
operation produce. **Do not train on it.** Held out, it is a direct external
check that our rescope records carry the right entailment relation — 258 pairs
of independent ground truth on the one phenomenon our generators are most likely
to be self-consistently wrong about.

---

## 7. The multi-cue finding

This is the number the paper can cite.

| corpus | sentences with 2+ negation cues | our logic: K1 (cancellation) | our logic: K2 (compound) |
| --- | --- | --- | --- |
| CD-SCO | 169 | 63 | 106 |
| SFU | 369 | 89 | 280 |
| BioScope | 191 | 66 | 125 |
| **total** | **729** | **218** | **511** |

**None of the three corpora records this distinction.** All three annotate each
cue as an independent instance with its own scope and no relation to any other
cue in the sentence. Whether two negations in one sentence cancel or compound is
recoverable from the annotations only by doing what `scripts/dataset_stats.py`
does — re-parsing and testing scope containment — which is to say it is not in
the annotation.

CD-SCO's own first training sentence is the example:

> Mr. Sherlock Holmes, who was usually very late in the mornings, save upon those
> **not** **in**frequent occasions when he was up all night, was seated at the
> breakfast table.

`not` (syntactic) scopes over `in-` (affixal): a cancellation, and the sentence
asserts that such occasions were *frequent*. CD-SCO stores this as two unrelated
negation instances. A model trained on the corpus as annotated has no signal
that one negation is inside the other's scope, and a cue-counting evaluation
scores it identically to a sentence with two independent negations in separate
clauses.

The 218/511 split also shows the two cases are both well attested — this is not
a rare-phenomenon argument. Roughly **30% of multiply-negated sentences in
existing corpora are cancellations**, and no corpus labels them as such.

---

## 8. Could not be located or verified

**xNLI-neg — no dataset of that name exists.** Searched HuggingFace (full-text
and by name) and the web; nothing. The closest real resource is Hartmann et al.
(2021), CoNLL, *A Multilingual Benchmark for Probing Negation-Awareness with
Minimal Pairs*, released at `github.com/mahartmann/negationminpairs` (repository
reachable, HTTP 200; `/data/minimal_pairs` and `/data/negation_cues`; **no
licence stated**). It is XNLI-derived minimal pairs at sentence-pair level with
no cue or scope spans, and it is multilingual where this project is
English-only. **Not recommended**, and not loaded.

**BioScope clinical subcorpus — unobtainable** (§3). Its source host is dead.

**\*SEM 2012 official distribution — gone** (§1). Recovered via mirror.

---

## 9. Rejected, with reasons

| candidate | why |
| --- | --- |
| **SNLI** | Verified available (`stanfordnlp/snli`, CC-BY-SA-4.0, 550,152 train). No negation annotation of any kind. Mining contradiction pairs whose hypothesis adds a cue means labelling them with our own detector — circular, as with CANNOT (§4). Usable only as a source of raw sentences. |
| **MultiNLI** | Verified available (`nyu-mll/multi_nli`, 392,702 train). **Licence is per-genre**: the card declares `cc-by-3.0`, `cc-by-sa-3.0`, `mit` *and* `other` together, so blanket redistribution of a derived corpus is not safe without resolving genre by genre. No negation annotation. Same circularity. |
| **`SINAI/SFU-Review-SP-Neg`** | Spanish, CC-BY-**NC**-SA-4.0. Not the Konstantinova English corpus; the non-commercial clause is an additional restriction. |
| **`joey234/conandoyle_cue_scope`** | 235 rows, no licence, no dataset card. A partial re-release of CD-SCO; the canonical CoNLL files carry the same content plus the negated-event layer. |
| **`MisterStino/negation-scope-cue-given`** | *Hound of the Baskervilles* with scope tags but **no cue column**, and no licence. Strictly less information than CD-SCO itself. |
| **`dannashao/sem2012forNegbert`** | CC0-1.0 and the split sizes are right (3,779/815/1,116), but cue and scope are collapsed into one tag sequence. Useful as a cross-check on our CoNLL parse, not as the primary source. |
| **`hapaxlegomenon/NegNLI-BR`** | Brazilian Portuguese. Out of scope for an English-only pipeline. |
| **`MilyaShams/multi_nli_negated`** | Automatically negated MultiNLI, no licence, no spans, and the negation is machine-generated — the same failure mode our own generators would introduce. |

---

## 10. Recommendation

**Cue/scope detector — train on CD-SCO + SFU.**
Together: **10,312 gold annotations over 7,175 annotated sentences** (of the
15,823 across all three corpora), two very different registers — Victorian
fiction and consumer reviews — and complementary family coverage. CD-SCO contributes the affixal and quantifier cues our
generators under-produce and the only negated-event layer available; SFU
contributes scale, informality and contractions. Add our generated corpus as
augmentation for the families neither covers well (E_antonym, J_contrastive,
I_intensified, M_modal have **zero to four** instances across all three real
corpora combined).

**Intensity / hedging regressor — SFU primary, BioScope secondary.**
These are the only two corpora found that annotate speculation separately from
negation, which is the requirement. SFU gives 5,363 hedge cues in general
register; BioScope adds 3,376 in biomedical. The 1,484 sentences carrying both a
negation and a speculation cue (1,083 SFU + 401 BioScope) are the most valuable
subset — they are where graded intensity is directly observable rather than
inferred.

**Held out for evaluation — BioScope full papers, NegNLI, NaN-NLI.**
- *BioScope full papers* (2,670 sentences): a domain-shift test. Hold the whole
  subcorpus out; do not train on any of it if it is the domain-transfer measure.
- *NegNLI* (4,500 pairs): tests whether negation actually changes the model's
  prediction.
- *NaN-NLI* (258 pairs): the sub-clausal and scope-alternation check. **Highest
  overlap with our generators and therefore the most informative** — it is
  independent ground truth on exactly the *not all* / *some not* alternation our
  `rescope` operation produces.

**Rejected for both tasks — CANNOT, SNLI, MultiNLI**, and anything else with no
gold spans (§4, §9). Deriving silver spans with our own classifier would train
the detector on its own assumptions.

### Licensing actions before publishing

1. **SFU is GPL-3.0.** Decide, before release, whether the trained detector or
   any derived corpus is a derived work. If a released corpus mixes SFU with
   CANNOT (CC-BY-SA-4.0), those two share-alike terms are **not** obviously
   compatible — resolve it or keep the corpora separate and release them
   separately.
2. **CD-SCO and NegNLI state no licence.** Use for research, do not redistribute,
   and cite the papers.
3. **BioScope's CC-BY-2.0 is a third-party claim**, not the authors'. Do not rely
   on it for redistribution.
4. **MultiNLI's licence is per-genre** if it is ever used for anything.

### Standing risks

- **CD-SCO has no official home.** Its only working source is one person's
  research repository. Cache the four files; if that repository disappears the
  corpus is, for practical purposes, gone.
- **Two of the URLs in this evaluation died before we looked** (\*SEM's
  distribution page, BioScope's clinical source). `negation/data/sources.py`
  records every URL with the status observed on 2026-09-13, dead ones included,
  so the next person can tell what was checked from what was assumed.
