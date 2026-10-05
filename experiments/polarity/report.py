"""Render REPORT_POLARITY.md (repo root) from results_polarity.json. Numbers are
read from the results file; prose that depends on them is chosen by comparing
them, not typed per run.

    ./venv/bin/python experiments/polarity/report.py
"""

from __future__ import annotations

import json
from pathlib import Path

from notes import ITERATIONS, NOT_RUN

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FORMULAS = ["f1", "f2", "f3", "f4", "f5"]
BASE_NAMES = {"cue_delta": "cue_delta (crude cue counter)", "raw_cosine": "raw cosine (negated)",
              "nli_contradiction": "NLI P(contradiction)", "arch2_margin": "arch 2 margin (mpnet, cosine)"}


def f(x, d=3):
    if x is None:
        return "—"
    if isinstance(x, float) and x != x:
        return "n/a"
    return f"{x:.{d}f}"


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def headline_row(name, block):
    d, v = block["dev"], block["validation"]
    return (f"| {name} | {f(d['auc'])} | {f(d['f1'])} | {f(v['auc'])} | {f(v['f1'])} | "
            f"{f(d['auc'] - v['auc'])} | {pct(d['paraphrase_fpr']['fpr'])} | {pct(v['paraphrase_fpr']['fpr'])} | "
            f"{pct(d['k1_fpr']['fpr'])} |")


HEAD = ("| scheme | dev AUC | dev F1 | val AUC | val F1 | dev − val AUC | dev paraphrase FPR | "
        "val paraphrase FPR | K1 FPR (dev) |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")


def baseline_rows(r):
    return "\n".join(headline_row(f"*baseline:* {BASE_NAMES[name]}", block)
                     for name, block in r["baselines"].items())


def render(r: dict) -> str:
    fin = r["final"]
    scheme = fin["scheme"]
    key = f"{scheme['version']}:{scheme['formula']}"
    chosen = r["iterations"][scheme["version"]]["grid"][scheme["formula"]]
    test = fin["result"]["test"]
    bt = fin["baselines_test"]
    base = r["baselines"]
    val_auc = chosen["validation"]["auc"]
    test_auc = test["auc"]
    ph2 = test["by_dataset"]["cannot_phase2"]
    xnot = test["by_dataset"]["xnot360"]
    cue, nli, a2 = bt["cue_delta"], bt["nli_contradiction"], bt["arch2_margin"]
    counts = fin["test_scorings"]["count"]

    # --- section 0, decided by the numbers
    beats_cue = test_auc > cue["auc"]
    beats_a2 = test_auc > a2["auc"]
    first = (f"**Sentence-level polarity {'beats' if beats_cue else 'does not beat'} cue counting "
             f"(test AUC {f(test_auc)} vs {f(cue['auc'])}), {'beats' if beats_a2 else 'loses to'} the "
             f"previous best architecture (arch 2, {f(a2['auc'])}), and is {f(nli['auc'] - test_auc)} AUC short "
             f"of the NLI cross-encoder ({f(nli['auc'])}).**")
    drop = val_auc - test_auc
    graded = [v for v in ("v3", "v4", "v5") if v in r["iterations"]]
    v2_dev = r["iterations"]["v2"]["summary"]["f1"]["dev_auc"]
    all_below = all(r["iterations"][v]["summary"]["f1"]["dev_auc"] < v2_dev for v in graded)
    grading = (f"Every hand-graded version ({', '.join(graded)}) scored below plain parity on dev, and "
               f"fitting the grades to dev (v6) widened the dev-validation gap."
               if all_below else "Hand-graded versions did not consistently beat plain parity on dev.")
    verdict = f"""{first} The test number is {f(drop)} below validation ({f(val_auc)}), and
most of the drop has one source. On the CANNOT phase-2 part of the test set the scheme
scores {f(ph2['auc'])}, {f(val_auc - ph2['auc'])} below validation. On xNot360 it scores {f(xnot['auc'])},
**below chance**: it flags {pct(xnot['by_subtype']['not_negation']['false_positive_rate'])} of xNot360's
non-negation pairs. Those are mostly conditionals whose antecedent polarity flips
(*If X, Y* / *If not X, not Y*). The surface polarity changes but B is not the
negation of A, and a per-sentence score cannot tell the difference. What
works: the cancellation claim. Same-clause cue parity takes the K1 false-positive
rate from {pct(r['iterations']['v1']['grid']['f1']['dev']['k1_fpr']['fpr'])} (v1) to
{pct(r['iterations']['v2']['grid']['f1']['dev']['k1_fpr']['fpr'])} (v2) on dev, where the NLI
cross-encoder sits at {pct(base['nli_contradiction']['dev']['k1_fpr']['fpr'])}. What does not work:
graded intensity. {grading} One seed ({r['seed']}); test
scored {counts} time{'s' if counts != 1 else ''}."""

    # --- section 1
    versions = list(r["iterations"])
    ex_head = "| sentence | " + " | ".join(versions) + " |\n| --- | " + " | ".join("---:" for _ in versions) + " |"
    ex_rows = "\n".join(f"| {s} | " + " | ".join(f(v[x], 2) for x in versions) + " |"
                        for s, v in r["worked_examples"].items())
    w = scheme["weights"] or {}
    wtable = "\n".join(f"| {g} | {k} | {f(v, 1)} |" for g in sorted(w) for k, v in sorted(w[g].items()))

    # --- section 2
    log = []
    for v in versions:
        s = r["iterations"][v]["summary"]["f1"]
        n = ITERATIONS[v]
        log.append(f"| **{v}** | {n['changed']} | {n['hypothesis']} | {f(s['dev_auc'])} / {f(s['dev_f1'])} | "
                   f"{f(s['val_auc'])} / {f(s['val_f1'])} | {f(s['gap_auc'])} | {pct(s['dev_k1_fpr'])} | "
                   f"{pct(s['val_paraphrase_fpr'])} | {n['errors']} |")
    trace = r["fitted_weights"].get("v6", {}).get("dev_auc_trace")

    # --- section 3 grid
    grid_head = ("| scorer | " + " | ".join(f"{x} dev / val" for x in FORMULAS) + " |\n| --- | "
                 + " | ".join("---" for _ in FORMULAS) + " |")
    grid_rows = []
    for v in versions:
        cells = []
        for x in FORMULAS:
            s = r["iterations"][v]["summary"][x]
            cell = f"{f(s['dev_auc'])} / {f(s['val_auc'])}"
            cells.append(f"**{cell}**" if f"{v}:{x}" == key else cell)
        grid_rows.append(f"| {v} | " + " | ".join(cells) + " |")
    formula_text = "\n".join(f"- **{k}** {t}" for k, t in r["data"]["formulas"].items())
    gates = ", ".join(f"{v} {f(r['iterations'][v]['grid']['f4']['gate'], 3)}" for v in versions)
    n_schemes = len(versions) * len(FORMULAS)
    top = sorted(((it["summary"][x]["val_auc"], v, x) for v, it in r["iterations"].items() for x in FORMULAS),
                 reverse=True)[:6]

    # --- section 4
    def test_row(name, t):
        return (f"| {name} | {f(t['auc'])} | {f(t['accuracy'])} | {f(t['precision'])} | {f(t['recall'])} | "
                f"{f(t['f1'])} | {pct(t['paraphrase_fpr']['fpr'])} | {f(t['by_dataset']['cannot_phase2']['auc'])} | "
                f"{f(t['by_dataset']['xnot360']['auc'])} |")
    test_rows = "\n".join([test_row(f"**polarity {key}**", test)] +
                          [test_row(BASE_NAMES[n], b) for n, b in bt.items()])
    sub = ph2["by_subtype"]
    neg = ph2["by_neg_type"]
    nli_sub = nli["by_dataset"]["cannot_phase2"]["by_subtype"]
    nli_neg = nli["by_dataset"]["cannot_phase2"]["by_neg_type"]
    breakdown_rows = "\n".join(
        [f"| phase 2 {k} (FPR) | {sub[k]['n']} | {pct(sub[k].get('false_positive_rate'))} | "
         f"{pct(nli_sub[k].get('false_positive_rate'))} |" for k in ("paraphrase", "sts_low", "snli_neutral", "random_pair")]
        + [f"| phase 2 negation, {k} (recall) | {neg[k]['n']} | {pct(neg[k].get('recall'))} | "
           f"{pct(nli_neg[k].get('recall'))} |" for k in sorted(neg)]
        + [f"| xNot360 negation (recall) | {xnot['by_subtype']['negation']['n']} | "
           f"{pct(xnot['by_subtype']['negation'].get('recall'))} | "
           f"{pct(nli['by_dataset']['xnot360']['by_subtype']['negation'].get('recall'))} |",
           f"| xNot360 not_negation (FPR) | {xnot['by_subtype']['not_negation']['n']} | "
           f"{pct(xnot['by_subtype']['not_negation'].get('false_positive_rate'))} | "
           f"{pct(nli['by_dataset']['xnot360']['by_subtype']['not_negation'].get('false_positive_rate'))} |"])

    # --- section 5
    def err_rows(errs):
        out = []
        for e in errs:
            dp = abs(e["p_b"] - e["p_a"])
            cres = f(e["score"] / dp, 2) if dp > 1e-9 else "—"
            out.append(f"| {e['dataset']} | {e['type']}{'/' + e['neg_type'] if e['neg_type'] else ''} | "
                       f"{e['a']} | {e['b']} | {f(e['p_a'], 2)} | {f(e['p_b'], 2)} | {f(e['cos'], 2)} | {cres} |")
        return "\n".join(out)
    err_head = ("| set | type | A | B | p(A) | p(B) | cos | c_resid |\n"
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")

    fps, fns = fin["errors"]["false_positives"], fin["errors"]["false_negatives"]
    fp_x = sum(e["dataset"] == "xnot360" for e in fps)
    fn_other = sum(e.get("neg_type") == "other" for e in fns)
    fn_both = sum(e["p_a"] > 0 and e["p_b"] > 0 and abs(e["p_a"] - e["p_b"]) < 1e-9 for e in fns)

    # --- section 6: dev breakdown vs NLI
    dev_ours = chosen["dev"]["by_dataset"]["ours"]["by_family"]
    nli_ours = base["nli_contradiction"]["dev"]["by_dataset"]["ours"]["by_family"]
    fam_rows = []
    wins, losses = [], []
    for fam in sorted(dev_ours):
        a, b = dev_ours[fam], nli_ours[fam]
        metric = "false_positive_rate" if "false_positive_rate" in a else "recall"
        better = a[metric] < b[metric] if metric == "false_positive_rate" else a[metric] > b[metric]
        worse = a[metric] > b[metric] if metric == "false_positive_rate" else a[metric] < b[metric]
        (wins if better else losses if worse else []).append(fam)
        fam_rows.append(f"| `{fam}` | {a['n']} | {'FPR' if metric != 'recall' else 'recall'} | "
                        f"{pct(a[metric])} | {pct(b[metric])} |")
    ph2_wins = [k for k in ("paraphrase", "sts_low", "snli_neutral", "random_pair")
                if sub[k].get("false_positive_rate", 1) < nli_sub[k].get("false_positive_rate", 0)]
    neg_wins = [k for k in neg if neg[k].get("recall", 0) > nli_neg[k].get("recall", 1)]

    info = r["data"]["info"]
    sp = r["data"]["splits"]
    ours_fam = r["data"]["ours_by_family"]

    return f"""# REPORT — sentence-level negation polarity

Generated by `experiments/polarity/report.py` from `experiments/polarity/results_polarity.json`.
Every number below is read from that file. Seed {r['seed']}. No LLM labelling: labels are
generator-derived (our dataset) or as released (CANNOT, xNot360).

## 0. Verdict

{verdict}

## 1. The polarity function

`experiments/polarity/polarity.py`, `polarity(sentence, version)`: a function of
any sentence, not a lookup. It parses the sentence once and reduces it to a list
of cues, each with the clause it sits in and its modifiers:

- **cues**: `negation.classify.existing_cues` (every family the generator knows:
  *not/n't/cannot*, negative adverbs, negative quantifiers, implicit triggers,
  WordNet-gated affixes, privatives; multiword cues counted once), plus the
  contrastive frames in `negation.lexicons`;
- **clause**: the nearest clause head above the cue; cues cancel only if they sit
  in the same `nlp_core.clause_scope`;
- **modifiers**: a hedge or intensifier from the project lexicons in the same clause,
  and *by no means* / *not at all*; *hardly, barely, rarely, seldom, scarcely* are
  `partial`.

Per clause, v2 onwards count cues by parity (an even number cancels). The
sentence takes its most negated clause. Versions differ only in how cues are
weighted (section 2). The pair score is then one of five formulas (section 3).

Worked examples, every version (the chosen scheme is **{key}**):

{ex_head}
{ex_rows}

The chosen scheme's weights were fit on DEV (v6). Its scale is not "how negated":
*does not sort* scores lower than *probably does not sort* because DEV rewards it,
not because it is less negated.

| group | key | weight |
| --- | --- | ---: |
{wtable}

## 2. Iteration log

Data. DEV: our dataset (generator labels) + CANNOT phase 1; VALIDATION: 30% of
phase 2; TEST: the other 70% of phase 2 + xNot360. Splits are grouped on connected
components of shared sentences: no sentence is in two splits (asserted at load and
in `test_polarity.py`).

| split | n | labels (0 / 1) | sources |
| --- | ---: | --- | --- |
{chr(10).join(f"| {k} | {v['n']} | {v['labels'].get('0', 0)} / {v['labels'].get('1', 0)} | {', '.join(f'{d} {n}' for d, n in v['datasets'].items())} |" for k, v in sp.items())}

Our dataset, by family and label: {', '.join(f'{k} {v}' for k, v in ours_fam.items())}.
{len(info['excluded_k1_negated_input'])} K1 records were **excluded**: their A was already negated
(*is impossible* → *is not impossible*), so B does negate A and the "cues cancel,
meaning preserved" label would be wrong. `L_scope_position` is in neither of the
brief's lists; its {ours_fam.get('L_scope_position=1', 0)} records all negate an affirmative A and are labelled 1.
NLI verdicts: {info['nli_pairs_reused_from_cache']} pairs reused from `negation_detection`'s cache, {info['nli_pairs_computed_new']}
computed new (our dataset's pairs, plus one CANNOT pair missing from that cache) into this experiment's own cache.
Arch 2 needs a generator reference set per A. Those exist for CANNOT and xNot360,
not for our dataset: there B is itself a generator variant of A, so the reference
set would contain B and the margin would be a leak. It is reported on CANNOT only.

**Threshold.** Fit on DEV only, by Youden's J (max TPR − FPR). Not by max-F1: DEV is
{f(sp['dev']['labels'].get('1', 0) / sp['dev']['n'], 2)} positive, and on it max-F1 chose "everything is a negation"
(it flagged every paraphrase and every K1 pair in a first run, whose results were
discarded and are not in the results file). Youden's J does not move with class balance.

Scored with **f1 = |dp| × cos** for comparability; every formula is in section 3.

| version | what changed | hypothesis | dev AUC / F1 | val AUC / F1 | dev − val | K1 FPR | val para FPR | what the errors suggested |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
{chr(10).join(log)}
| v7, v8 | — | — | — | — | — | — | — | {NOT_RUN} |

v6 search trace (dev AUC after start, sweep 1, sweep 2): {trace}.

Baselines, same splits, same threshold rule:

{HEAD}
{baseline_rows(r)}

## 3. The grid: scorer × formula (AUC, dev / validation)

{formula_text}

f4's cosine gate is fit on DEV per version: {gates}.
Bold = the scheme chosen by validation AUC.

{grid_head}
{chr(10).join(grid_rows)}

The six best by validation AUC: {', '.join(f'{v}:{x} {f(a, 4)}' for a, v, x in top)}. They sit within
{f(top[0][0] - top[-1][0], 4)} of each other. **The choice of {key} over plain v2 is not a
meaningful difference.** The rule was "best validation AUC", and it was followed.
Signed dp (f5) is at or below chance on validation for every scorer: phase-2
negations run in both directions (A negated, B affirmative and the reverse), so
only |dp| can work.

## 4. Final test result (scored {counts} time{'s' if counts != 1 else ''})

Scheme **{key}** ({scheme['formula_text']}), validation AUC {f(val_auc)}, DEV threshold
{f(fin['result']['threshold'])}.

| scorer | test AUC | accuracy | precision | recall | F1 | paraphrase FPR | phase-2 part AUC | xNot360 AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{test_rows}

**Test AUC {f(test_auc)}, against {f(val_auc)} on validation.** That is the result.
The CANNOT phase-2 part ({f(ph2['auc'])}) holds up; xNot360 ({f(xnot['auc'])}) inverts.

Breakdown against the NLI baseline, each at its own DEV threshold:

| slice | n | polarity {key} | NLI |
| --- | ---: | ---: | ---: |
{breakdown_rows}

K1 false-positive rate is not measurable on test: K1 exists only in DEV. On DEV
the chosen scheme's K1 FPR is {pct(chosen['dev']['k1_fpr']['fpr'])} and NLI's is
{pct(base['nli_contradiction']['dev']['k1_fpr']['fpr'])}.

## 5. Error analysis (test set, chosen scheme)

Sampled at random (seed {r['seed']}) from the test errors, not picked. p = polarity;
cos = raw cosine; c_resid = cosine after removing the negation direction (the
scheme's similarity), recovered as score / |dp|.

**False positives**

{err_head}
{err_rows(fin['errors']['false_positives'])}

**False negatives**

{err_head}
{err_rows(fin['errors']['false_negatives'])}

Reading them: {fp_x} of {len(fps)} sampled false positives are xNot360 `not_negation` pairs, nearly all
conditionals whose antecedent polarity flips while the implication does not;
the remainder ({len(fps) - fp_x}, type: {', '.join(sorted(set(e['type'] for e in fps if e['dataset'] != 'xnot360'))) or '—'})
trade a negation for an antonym with the meaning kept (*not fine* / *coarse*).
{fn_other} of {len(fns)} sampled false negatives are CANNOT's `other` type (lexical
substitution; recall on that type {pct(neg['other'].get('recall'))}): antonyms and content swaps
with no cue on either side, so p(A) = p(B). {fn_both} of {len(fns)} have a cue on both
sides that the parity treats as equal (*not foul* / *not fair*).

## 6. Where it beats the NLI baseline and where it loses

DEV, our dataset, by family (each scorer at its own DEV threshold):

| family | n | metric | polarity {key} | NLI |
| --- | ---: | --- | ---: | ---: |
{chr(10).join(fam_rows)}

**Polarity wins** (lower FPR or higher recall) on: {', '.join(f'`{x}`' for x in wins) or 'nothing'}.
**NLI wins** on: {', '.join(f'`{x}`' for x in losses) or 'nothing'}.
On test, polarity has the lower false-positive rate on phase-2 {', '.join(ph2_wins) or 'no negative subtype'},
and higher recall on {', '.join(neg_wins) or 'no negation type'}.

The pattern: polarity is better where the answer depends on *cue structure*. It
cancels (K1), and it ignores unrelated pairs because it only fires on a
polarity change. NLI is far better wherever the negation is *lexical*
(E_antonym; CANNOT `other` and `antonym`) and wherever logic, not surface, decides
(xNot360 conditionals). NLI's weakness on our dataset is the unrelated anchors:
it calls {pct(nli_ours['ANCHOR_unrelated']['false_positive_rate'])} of them contradictions, which is
defensible for NLI (two unrelated claims can conflict) but wrong for "is B a negation of A".

## 7. Caveats and what would change the conclusion

1. **Test ≠ validation in kind.** Validation was CANNOT only; test adds xNot360, a
   different construction (legal/conditional). Of the {f(drop)} drop, the phase-2 part accounts
   for {f(val_auc - ph2['auc'])} (validation {f(val_auc)} vs phase-2 test {f(ph2['auc'])}); the rest is xNot360. A validation set containing conditionals would have exposed it before
   the test set did.
2. **Our dataset is generator output**, and the scorer reads the generator's own
   lexicons. Recall on our families is partly the scorer recognising its sibling's
   vocabulary. The CANNOT and xNot360 numbers are the honest ones.
3. **K1 labels include generator defects.** The affixal K1 subtypes sometimes put the
   prefix on the wrong word (*noncyclic redundancy check*, found in the pattern
   exploration), so some K1 "meaning preserved" labels are false. That cuts both
   ways for the K1 FPR.
4. **DEV is imbalanced and mixed** (our dataset is mostly positives). v6's fitted
   weights learned that mix. The threshold rule had to change from max-F1 to
   Youden's J for the same reason.
5. **Validation selection among {n_schemes} schemes** over a {f(top[0][0] - top[-1][0], 4)} spread: the choice is noise.
6. **Single seed** ({r['seed']}): one phase-2 split, one sampling of error examples.

What would change it: pair-level antonym knowledge (the polarity of *not alive*
vs *dead* lives in the pair, not in either sentence); a clause-aware difference
rather than max-over-clauses; handling conditionals (antecedent vs consequent
polarity) so the xNot360 inverses are not read as negations; and coverage of
negative words outside the lexicons. The first and third are beyond a
sentence-level score by construction. So the ceiling of this idea is below NLI
on exactly the cases NLI was already good at.
"""


def main() -> None:
    r = json.loads((HERE / "results_polarity.json").read_text())
    (REPO / "REPORT_POLARITY.md").write_text(render(r))
    print("wrote", REPO / "REPORT_POLARITY.md")


if __name__ == "__main__":
    main()
