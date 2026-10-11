# datasets/

One line per file. Report numbers (01–09) refer to `reports/`. sha256 is of the file as committed (for
`dev_clean.csv`, of the rebuilt file, which must match the recorded digest).

| file | what it is | how it is built | licence | authorship | sha256 |
| --- | --- | --- | --- | --- | --- |
| `splits/dev_clean.csv` | clean DEV set, 3,402 natural pairs, no generator output (07 step 1.1–1.2). **Gitignored**: STS-B text not redistributed | `scripts/nli2/build_dev_clean.py --hard`; sha256 checked against `reports/results/nli2/1_2_dev_clean.json` at load | CANNOT CC BY-SA 4.0; STS-B unclear; SNLI CC BY-SA 4.0 | natural (published dataset) | `93fe7846d19c2faa1f9d99b2e545b68edfc0438c0bde42c7d05d1704730aad03` |
| `splits/val_ood.csv` | out-of-distribution validation slice: 800 MNLI pairs (overlap < 0.6) + 528 synthetic conditionals, fit/select halves (09 steps 0.3, 5.1) | `scripts/f2/val_ood.py --step 0.3`, then `scripts/f2/conditional_val.py --step 5.1` | MNLI: mixed per genre (OANC/fiction etc., see GLUE); conditionals: ours | MNLI natural; conditionals **machine-generated** | `8b16e5ee138f59c242c7ddb894e3c5a843c6b7049a77977cf364fd0b0b8b2797` |
| `splits/phase1.jsonl` | CANNOT phase 1 (DEV / VALIDATION source), 3,002 pairs | `scripts/negation_detection/data.py` | CC BY-SA 4.0 | natural (published dataset) | `e854986ac73193be392054f1c05cc7081393869856655d14f026351d574900c0` |
| `splits/phase2.jsonl` | CANNOT phase 2: VALIDATION and TEST (1,400 TEST pairs), 2,001 pairs | `scripts/negation_detection/data.py` | CC BY-SA 4.0 (+ STS-B / SNLI dissimilar positives) | natural (published dataset) | `c55c3dc52e4328b197d9c7239aa0dd1effbcfdcbf6e2dd23ae5575e344c59ff2` |
| `splits/xnot360.jsonl` | xNot360, external TEST, 360 pairs | `scripts/negation_detection/data.py` | Apache-2.0 (dataset card body) | natural (published dataset) | `335130383d51404dae0700d052bb6d24e1dd7138bfd94223b8fd0a5782c5dc6f` |
| `generated/student_answers.jsonl` | 2,645 domain student answers in labelled categories, built from `raw/definitions.jsonl` (08 Phase 5) | `scripts/f1/make_student_answers.py` | ours | **machine-generated** | `88de857ed490bbeecb6b4ec4642aa8d271b8221d0746b23e6c376def52002d18` |
| `generated/negation_direction/variants.jsonl` | stage-1 generator corpus, 455 records over 20 bases (01) | `scripts/run.sh tests/data/base_sentences_v1.txt … --quiet` | ours | **machine-generated** | `c01fed70f2600e92c6e1b1845844362f9f9eaea1f0dd13a36b67108db254b250` |
| `generated/negation_direction/control_candidates.jsonl` | paraphrase-control candidates (01) | `scripts/negation_direction/make_controls.py` | ours | **machine-generated** | `af59bfc3997b5f91f890d77831c25f5293e6a97b04f8ba3e4b8c9bf82c031395` |
| `generated/negation_direction/control_keep.txt` | the candidates kept as grammatical, meaning-preserving controls | judged by hand, once | ours | hand-written (selection) | `3678965a351e613e658a5a05bb0b070fb2b77ee6acb37d192b563477ecb2eec8` |
| `generated/negation_direction/control_pairs.jsonl` | the 21 kept control pairs (01) | from the two files above | ours | **machine-generated** | `6218f7a14234a838736f71d3e09eda858e8a4aa96616b8f971b91cbf21d4fd80` |
| `generated/pattern/base_sentences.txt` | 100 textbook statements, 10 per OS / CN subtopic: the pattern dataset's bases (03) | written by hand for 03 | ours | hand-written | `897579e59e0e6353f37135730c3d837912c7936fab8332edc2d88b7bc40b3977` |
| `generated/pattern/paraphrase_candidates.tsv` | 90 paraphrase anchors, reviewed (`keep=0/1`) (03) | written by hand for 03 | ours | hand-written | `40dd08c5f269a5a7c39561e5d0b22bb809ccad70cd231a8365dd9124dd03b8bc` |
| `generated/pattern/to_label.jsonl` | 571 items to label (03) | `scripts/pattern/pattern.py` | ours | **machine-generated** | `710a07bd9d6aaa3c35d663bca4a2a3f4981257fb551b04483e890be6e3434a54` |
| `generated/pattern/metadata.jsonl` | generator metadata for those items | `scripts/pattern/pattern.py` | ours | **machine-generated** | `700d9cb2136ac44dc3df398d397904aba92f731e77b1542c96720599e5f603da` |
| `generated/pattern/build_report.json` | build statistics | `scripts/pattern/pattern.py` | ours | **machine-generated** | `664cc11a160737d173e150abaa65ef974d2e1f11fd6fa74fdd88f292e8d21b19` |
| `generated/pattern/labels_pass{1,2,3}.jsonl` | three LLM labelling passes; **failed their validation gate** (alpha 0.586) | `scripts/pattern/label.py` | ours | **machine-generated** labels | ac180e0e6e33daea…<br>b34bbdc7cb445933…<br>6de9fcbef2cb38c4… |
| `generated/pattern/labels_config.json` | labeller configuration | `scripts/pattern/label.py` | ours | **machine-generated** | `0c1b8d8cd663139e1ff417385edb724b115177402c94d26eac04de094aa5263c` |
| `generated/pattern/label_run.log` | labelling run log | `scripts/pattern/label.py` | ours | **machine-generated** | `320c514e0f8865c8af05baeb5dc3fcd52f27e33f333764aba2058a4a2bd0fe19` |
| `eval/paragraph_eval.jsonl` | 28 teacher / student paragraph pairs, OS / CN / DBMS, per-sentence gold (08 Phase 4). Matching and coverage only | `scripts/f1/make_paragraphs.py` | ours | **machine-authored** (written by the model) | `e869a33ba36e5f48147fd406e1ccdbc845d9bf9c311f9672f82242ae91b4ceee` |
| `eval/paragraph_eval_human.jsonl` | same schema, **empty stub** for human-written items | `scripts/f1/make_paragraphs.py` (stub) | ours | human (to be filled) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `eval/student_answers_human.jsonl` | 50 student answers (10 definitions × 5 categories), same schema (08 Phase 5.4) | written by the model | ours | **a model-written placeholder**, not human | `832df1ff3119251ee8c51d30017a558e24d96f3b92cfec4533950c89ed3de2e8` |
| `eval/human_seed.csv` | 60 pattern items for three human raters; rater columns empty (03) | `scripts/pattern/analyse.py` | ours | **machine-generated** items, no labels yet | `64a0ed388ab0c34254ef213eea1d53ba9d84255809b2aa75810cdffeada231a0` |
| `eval/affirmation_spotcheck.json` | 30-item spot-check of the affirmation generator against written criteria (08 step 1.2) | verdicts written by the model | ours | model judgement | `7fbcf28ec2bb1d8d65b8602c29f0128b08d0e11805f8778350b1b738ec735e04` |
| `raw/definitions.jsonl` | 500 OS / CN / DBMS definitions from English Wikipedia lead sentences, each with URL and revision id (08 Phase 5) | `scripts/f1/fetch_definitions.py` | CC BY-SA 4.0 | natural (Wikipedia) | `59bb8a290b4fe9038df0c8ee3c1d839f76e40df715bcd6b810467af4e1d7d20c` |
| `sample_sentences.txt` | 5 sentences, the default input of `negation.cli` / `scripts/run.sh` | hand-written | ours | hand-written | `26d72d423d93e05116b2180cbb577c523396bb45d997559c264b1605eefa9927` |

**Gitignored, fetched at pinned revisions** (not redistributed):

| directory | contents | fetched by | manifest |
| --- | --- | --- | --- |
| `raw/negation_detection/` | CANNOT v1.1 (tsv + repo tarball), xNot360, STS-B train/validation/test, SNLI validation (60 MB) | `scripts/negation_detection/data.py` documents the download commands | sources and licences in `reports/02a-negation-detection-architectures.md` |
| `raw/nli/` | MNLI train + validation matched/mismatched, SNLI train (parquet, 71 MB) | `scripts/f2/fetch_data.py` | `reports/results/f2/raw_manifest.json` (revision + sha256 per file) |
| `raw/external/` | cue/scope corpora (CD-SCO, SFU, BioScope, CANNOT, NegNLI, NaN-NLI; 25 MB) | `python -m negation.data.fetch` | `negation/data/sources.py` |
| `raw/embeddings/` | static word vectors (6.1 GB) | `python -m negation.antonym_vec.fetch` | `negation/antonym_vec/fetch.py` |
| `generated/negation_detection/refsets.jsonl` | per-sentence reference variant sets (12 MB) | `scripts/negation_detection/refsets.py` (~4 min) | — |
| `generated/pattern/variants.jsonl` | full pattern variant pool (1.6 MB) | `scripts/pattern/pattern.py` | — |

The paragraph-set schema, and the standing request for human-written items, are in `eval/README.md`.
