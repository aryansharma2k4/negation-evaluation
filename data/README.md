# data/

| file | what it is | authorship |
| --- | --- | --- |
| `dev_clean.csv` | clean DEV set (Report_NLI2.md step 1.1–1.2); gitignored, rebuilt by `experiments/nli2/build_dev_clean.py --hard` | natural datasets |
| `paragraph_eval.jsonl` | 28 teacher / student paragraph pairs, OS / CN / DBMS, per-sentence gold (report_f1.md Phase 4) | **machine-authored** (written by the model for report_f1.md) |
| `paragraph_eval_human.jsonl` | same schema, **empty stub** for human-written items | human (to be filled) |
| `definitions.jsonl` | 500 OS / CN / DBMS definitions from English Wikipedia lead sentences, each with URL and revision id (CC BY-SA 4.0) (Phase 5) | real source |
| `student_answers.jsonl` | 2,500+ student answers generated from those definitions in labelled categories (Phase 5) | **machine-generated** |
| `student_answers_human.jsonl` | 50 student answers (10 definitions × 5 categories), same schema (Phase 5.4) | written by the model as a placeholder — to be replaced |

**Request to Aryan, Clovis and Arush:** please add human-written items to `paragraph_eval_human.jsonl`
(any number; a teacher paragraph from your own notes and a student paragraph written the way a
student would, with the gold verdicts filled in), and replace the 50 placeholder items in
`student_answers_human.jsonl` with answers you write yourselves, keeping the `definition_id` and
`category` fields. report_f1.md reports human and machine items separately and never merges them.

`paragraph_eval*.jsonl` schema — one JSON object per line:

```json
{"id": "h01", "subject": "OS", "authorship": "human", "author_note": "who wrote it",
 "tags": ["one_contradiction"],
 "teacher": ["sentence 1", "sentence 2", "sentence 3"],
 "student": ["sentence a", "sentence b", "sentence c"],
 "gold_teacher": [{"verdict": "covered", "student": 1}, {"verdict": "contradicted", "student": 0},
                  {"verdict": "missing", "student": null}],
 "gold_extra": [2]}
```

`verdict` is `covered`, `contradicted` or `missing`; `student` is the index of the student sentence that
covers or contradicts that teacher sentence (`null` when missing); `gold_extra` lists student sentences
that match no teacher sentence. Tags used: `reworded`, `one_contradiction`, `missing`, `extra`,
`negative_teacher`, `cancellation`, `conditional`, `scope`.
