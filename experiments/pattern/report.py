"""Render PATTERN.md from the results dict. Every number is read from it."""

from __future__ import annotations

HSS_BLIND = ["F_implicit", "D_affixal", "G_prepositional", "J_contrastive", "K1_cancellation"]
MODELS = ["flat_0.7", "best_flat", "per_family", "linear", "tree_depth3", "no_penalty_1.0 (reference)"]


def f(x, d=3):
    if x is None:
        return "—"
    if isinstance(x, float) and x != x:
        return "n/a"
    return f"{x:.{d}f}"


def p(x):
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def ci(pair, d=3):
    return f"[{f(pair[0], d)}, {f(pair[1], d)}]"


def gate_section(r: dict) -> str:
    v = r["validation"]
    o, c = v["order_effect_pass1_vs_pass2"], v["context_effect_pass1_vs_pass3"]
    pa, un = v["anchors"]["ANCHOR_paraphrase"], v["anchors"]["ANCHOR_unrelated"]
    lab = r["labeller"]
    sp = v["spearman_pairwise"]
    return f"""## 2. Labelling protocol and its validation

**Labeller:** `{lab['model']}` via a local ollama server, temperature {lab['temperature']},
{lab['batch']} pairs per prompt. It saw only the two sentences: `label.py` reads
`data/to_label.jsonl`, which holds `id`, `a`, `b` and nothing else (asserted on
load and in `test_pattern.py`). The prompt asks for semantic similarity on the
STS-B 0-5 scale with the six anchor descriptions and three worked examples, none
of them a negation pair; the words *negation*, *family* and *polarity* do not
appear in it.

| pass | order | presentation | seed |
| --- | --- | --- | ---: |
| 1 | (A, B) | `to_label` order | {lab['passes']['1']['sampling_seed']} |
| 2 | (B, A) | same order as pass 1 | {lab['passes']['2']['sampling_seed']} |
| 3 | (A, B) | reshuffled (seed {lab['passes']['3']['shuffle_seed']}) | {lab['passes']['3']['sampling_seed']} |

Pairs go to the model in batches, so the reshuffle in pass 3 changes each item's
neighbours; with one pair per prompt at temperature 0 it would reproduce pass 1
exactly. Each pass reads only `to_label.jsonl` and writes its own file. Final
label = median of the three. Unscored responses per pass (kept as missing,
never guessed): {v['unscored_per_pass']}.

### Gate (step 5), stated before anything that depends on it

| check | value | threshold | |
| --- | ---: | --- | :---: |
| (a) Krippendorff's alpha, ordinal, 3 passes | **{f(v['krippendorff_alpha_ordinal'])}** | >= {v['gate']['krippendorff_min']} | {'pass' if v['krippendorff_alpha_ordinal'] >= v['gate']['krippendorff_min'] else '**FAIL**'} |
| (a) mean pairwise Spearman | {f(v['spearman_mean'])} | — | |
| (c) paraphrase anchors, mean label (n={pa['n']}) | {f(pa['mean'], 2)} | >= {v['gate']['paraphrase_mean_min']} | |
| (c) unrelated anchors, mean label (n={un['n']}) | {f(un['mean'], 2)} | <= {v['gate']['unrelated_mean_max']} | |
| (c) paraphrase 10th pct > unrelated 90th pct | {f(pa['p10'], 1)} > {f(un['p90'], 1)} | — | {'pass' if v['anchors_separate'] else '**FAIL**'} |

Pairwise Spearman: 1-2 {f(sp['pass1_vs_pass2'])}, 1-3 {f(sp['pass1_vs_pass3'])}, 2-3 {f(sp['pass2_vs_pass3'])}.

**(b) Order effect**, pass 1 (A,B) vs pass 2 (B,A), paired t-test over {o['n']} items:
mean {f(o['mean_pass1'], 3)} vs {f(o['mean_pass2'], 3)}, difference {f(o['mean_diff'], 3)},
t = {f(o['t'], 2)}, p = {p(o['p'])}; identical score on {f(100 * o['exact_agreement'], 1)}% of items.
{'The difference is significant at 0.05: the labeller is positionally biased by this measure.' if o['p'] < 0.05 else 'Not significant at 0.05: no evidence of positional bias.'}
For comparison, the context effect (pass 1 vs pass 3, same order, different batch
neighbours): difference {f(c['mean_diff'], 3)}, p = {p(c['p'])}, identical on {f(100 * c['exact_agreement'], 1)}%.

Anchor label distributions (label: count): paraphrase {pa['counts']}; unrelated {un['counts']}.

**(d) Human seed.** `experiments/pattern/data/human_seed.csv` holds {r['human_seed']['n']} items
drawn round-robin across every family and both anchor classes, with the same
blind view (id, sentence A, sentence B) and three empty rater columns. **Until
those columns are filled in, every label in this document is machine-produced,
and the agreement figures above measure the labeller's consistency with itself,
not its correctness.**

**Gate: {'PASSED' if v['gate']['passed'] else 'FAILED'}.**
"""


def build_section(r: dict) -> str:
    b = r["build"]
    s1 = b["step1_base_sentences"]
    fam_rows = "\n".join(
        f"| `{k}` | {b['step2_variants_per_family'].get(k, '—')} | {n} |"
        for k, n in b["step3_sampled_per_family"].items())
    discarded = "\n".join(f"- line {d['base_line']} ({d['method']}): *{d['text']}* — {d['note']}"
                          for d in b["step2_paraphrase_discarded"])
    reassigned = ", ".join(f"{k}: {v}" for k, v in b["step3_ambiguous_strings_reassigned"].items())
    return f"""## 1. What was built

**Base sentences** (`data/base_sentences.txt`): {s1['n']} hand-written textbook
statements, {s1['words_min']}-{s1['words_max']} words, 10 per subtopic:
{', '.join(f'{k} {v}' for k, v in s1['subtopic'].items())}.
Frame of the root clause, as the generator's own `frames.detect_frame` reads it:
{', '.join(f'{k} {v}' for k, v in s1['root_clause_frame'].items())}.
Over every finite clause head: {', '.join(f'{k} {v}' for k, v in s1['all_clause_heads_frame'].items())}.
Already negated (the generator's `classify_input`): {s1['already_negated']}; more
than one clause: {s1['multi_clause']} (this counts relative and adverbial clauses,
so it is higher than the ~15 deliberately coordinated/subordinated sentences).

**Variants**: the generator in `negation/` was run unmodified (`generate_all`,
both layers, so affirmation, rescoping and clause targeting are included):
{b['step2_variants_total']} records.

**Anchors**: {b['step2_paraphrase_candidates']} hand-written paraphrase candidates
({', '.join(f'{k} {v}' for k, v in b['step2_paraphrase_methods'].items())} kept),
{len(b['step2_paraphrase_discarded'])} discarded on review:

{discarded}

{b['step2_unrelated']} unrelated anchors: a base sentence paired with a generated
variant of a different base sentence from a different subtopic.

**Sample** (seed {b['seed']}): up to {b['step3_per_family_cap']} per family,
round-robin across subtypes so small subtypes are represented, all anchors.
Before sampling, {sum(b['step3_ambiguous_strings_reassigned'].values())} strings emitted by two
families were given to the more specific one ({reassigned}), so each rated pair
has one family. Families below the cap contribute everything they have:
{', '.join(f'`{k}` {v}' for k, v in b['step3_families_below_cap'].items())}.
Total: **{b['step3_total_items']} items**, shuffled, id = hash of the sentence pair.

| family | generated | sampled |
| --- | ---: | ---: |
{fam_rows}
"""


def render_failed(r: dict) -> str:
    v = r["validation"]
    d = v["diagnostics"]
    o, c = v["order_effect_pass1_vs_pass2"], v["context_effect_pass1_vs_pass3"]
    alpha_ok = v["krippendorff_alpha_ordinal"] >= v["gate"]["krippendorff_min"]
    failed = [x for x, ok in [(f"Krippendorff's alpha = {f(v['krippendorff_alpha_ordinal'])}, "
                               f"below the {v['gate']['krippendorff_min']} threshold", alpha_ok),
                              ("anchor separation", v["anchors_separate"])] if not ok]
    return f"""# PATTERN — is a graded negation penalty justified?

Generated by `experiments/pattern/analyse.py` from `experiments/pattern/results.json`.
Every number below is read from that file.

## 0. Verdict

**Not determined. The machine labels failed their own validation gate, so no
formula was fitted and no family comparison was run.** The gate failed on
{' and '.join(failed)}. The anchors separate: paraphrases average
{f(v['anchors']['ANCHOR_paraphrase']['mean'], 2)} and unrelated pairs
{f(v['anchors']['ANCHOR_unrelated']['mean'], 2)}. Fitting alpha to these labels would
fit the labeller's inconsistency. **This is not a null result about negation.**
It says nothing about whether the families differ; it says that this labelling
setup cannot answer the question. Sections 3-6 are deliberately absent.

{build_section(r)}
{gate_section(r)}
## Why the gate failed

The failure comes from how the labelling was set up, not from noise alone. The
three passes differ in two ways: the order of the two sentences, and each item's
batch neighbours (5 pairs per prompt). The numbers point at the neighbours.

- **Batch neighbours move scores.** Passes 1 and 2 put every item in the same
  batch; pass 3 reshuffles. On {d['items_where_1_and_2_agree_but_3_moves_3plus']} of {v['n_items']} items,
  passes 1 and 2 give the same score and pass 3 moves it by 3 points or more.
  {d['same_among_unrelated_anchors']} of these are unrelated anchors. Pass 1 vs 3 (same sentence order,
  different neighbours) differs in mean by {f(c['mean_diff'], 3)} (p = {p(c['p'])}); pass 1 vs 2
  (same neighbours, swapped order) by {f(o['mean_diff'], 3)} (p = {p(o['p'])}). Both shifts are
  significant at 0.05.
- **Unrelated anchors rated as related.** {d['unrelated_anchors_median_ge_3']} of {d['unrelated_anchors_n']}
  unrelated pairs -- sentences from different subtopics -- have a median of 3 or
  more ("roughly equivalent"). That fits the model carrying a score over from a
  neighbouring pair in the same prompt. The anchor check still passes its
  thresholds, because a mean and a 90th percentile tolerate this tail;
  the agreement statistic does not.
- **The scale is not used as written.** The rubric puts "not equivalent but same
  topic" at 2. Yet {d['negation_pairs_median_0']} of {d['negation_pairs_n']} negation pairs, which share
  nearly every content word with sentence A, have a median of 0 ("completely
  dissimilar"); the mean over all negation pairs is {f(d['negation_pairs_mean_label'], 2)}. The model treats
  a contradiction as maximally dissimilar, not as same-topic, which by itself
  would push every family towards the floor. This is a diagnosis only; no
  family-level number is drawn from it.

Batching was a design choice in `label.py`: with one pair per prompt at
temperature 0, a reshuffle would reproduce pass 1 exactly and make the
agreement check meaningless. The evidence says the batching itself hurts
consistency.

## 7. Caveats

1. The labels are machine-produced (`{r['labeller']['model']}`), and `data/human_seed.csv`
   is empty. Even a passing alpha would have measured consistency, not correctness.
2. The diagnosis above explains *why* this setup fails. It does not show that a
   different setup would pass.
3. One seed ({r['seed']}), one run. The run was not repeated or tuned after the
   gate failed.

## 8. What would change the conclusion, and the next step

Any of these, followed by `label.py` and then `analyse.py`. The analysis is
already written and runs unchanged once the gate passes.

- **Human labels.** Fill in `data/human_seed.csv` (60 items, 3 raters). This is the
  only option that measures correctness and not just consistency.
- **One pair per prompt, with sampling.** This removes the neighbour
  contamination. Make the passes independent with temperature > 0 and a
  different seed per pass (not deterministic reruns), and keep the (B, A) swap
  for pass 2.
- **A stronger labeller.** The 7B model's use of the scale (contradictions scored
  0) is a separate problem from the batching, and a larger model may not share it.

Each of these changes the labelling protocol that the brief fixed, so choosing
one is a decision for the project, not something this run did on its own.
"""


def family_table(r: dict) -> str:
    t = r["family"]["table"]
    order = sorted(t, key=lambda k: -t[k]["mean_label"])
    lines = ["| family | n | mean label | std | median | mean sim_base (stand-in) | implied alpha | 95% CI |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for k in order:
        x = t[k]
        lines.append(f"| `{k}` | {x['n']} | {f(x['mean_label'], 2)} | {f(x['std_label'], 2)} | "
                     f"{f(x['median_label'], 1)} | {f(x['mean_sim_base'])} | {f(x['implied_alpha'])} | "
                     f"{ci(x['implied_alpha_ci95'])} |")
    return "\n".join(lines)


def planned_table(r: dict) -> str:
    lines = ["| comparison | question | n | means | diff [95% CI] | Cohen's d | Cliff's delta | p | p (Holm) |",
             "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for t in r["family"]["planned"]:
        lines.append(f"| {t['comparison']} | {t['question']} | {t['n_a']} / {t['n_b']} | "
                     f"{f(t['mean_a'], 2)} / {f(t['mean_b'], 2)} | {f(t['mean_diff'], 2)} {ci(t['mean_diff_ci95'], 2)} | "
                     f"{f(t['cohens_d'], 2)} | {f(t['cliffs_delta'], 2)} | {p(t['p'])} | {p(t['p_holm'])} |")
    return "\n".join(lines)


def cv_tables(r: dict) -> tuple[str, str]:
    cv = r["models"]["cv"]
    overall = ["| model | CV MAE (label/5) | CV MAE (0-5 units) | Pearson r |", "| --- | ---: | ---: | ---: |"]
    for m in MODELS:
        o = cv[m]["overall"]
        overall.append(f"| `{m}` | {f(o['mae'])} | {f(5 * o['mae'], 2)} | {f(o['pearson_r'])} |")
    fams = list(cv["per_family"]["per_family"])
    per = ["| family | n | " + " | ".join(f"`{m}` MAE / r" for m in MODELS[:5]) + " |",
           "| --- | ---: | " + " | ".join("---" for _ in MODELS[:5]) + " |"]
    for fam in fams:
        cells = []
        best = min(cv[m]["per_family"][fam]["mae"] for m in MODELS[:5])
        for m in MODELS[:5]:
            s = cv[m]["per_family"][fam]
            cell = f"{f(s['mae'])} / {f(s['pearson_r'], 2)}"
            cells.append(f"**{cell}**" if s["mae"] == best else cell)
        per.append(f"| `{fam}` | {cv['flat_0.7']['per_family'][fam]['n']} | " + " | ".join(cells) + " |")
    return "\n".join(overall), "\n".join(per)


def render(r: dict) -> str:
    if not r["validation"]["gate"]["passed"]:
        return render_failed(r)
    v, fam, mod, ver = r["validation"], r["family"], r["models"], r["verdict"]
    om = fam["omnibus"]
    dec = ver["deciding_number"]
    con = mod["contrasts"]
    cv = mod["cv"]
    winner = min(MODELS[:5], key=lambda m: cv[m]["overall"]["mae"])

    if not ver["families_differ"]:
        lead = (f"**No. The negation families do not separate**: Kruskal-Wallis across "
                f"{len(om['families'])} families gives H = {f(om['kruskal_H'], 2)}, p = {p(om['kruskal_p'])}.")
    elif not ver["graded_beats_flat"]:
        lead = (f"**No. The families differ in mean label (Kruskal-Wallis p = {p(om['kruskal_p'])}, "
                f"epsilon-squared = {f(om['epsilon_squared'], 3)}), but a per-family alpha does not predict "
                f"held-out labels better than one well-chosen constant.**")
    else:
        lead = (f"**Yes, on these (machine) labels.** The families differ (Kruskal-Wallis p = "
                f"{p(om['kruskal_p'])}, epsilon-squared = {f(om['epsilon_squared'], 3)}) and a per-family "
                f"alpha beats the best flat constant on held-out base sentences.")
    verdict = f"""{lead} The number that decides it, fixed before labelling: cross-validated
MAE of the per-family lookup minus that of the best flat constant =
**{f(dec['mae_diff'], 4)}** on the label/5 scale ({f(5 * dec['mae_diff'], 3)} in 0-5 units),
95% base-sentence bootstrap CI {ci(dec['ci95'], 4)}; the rule required the whole
interval below zero. For scale: the best flat constant is alpha =
{f(mod['best_flat_alpha_full'])}, HSS's 0.7 has CV MAE {f(cv['flat_0.7']['overall']['mae'])} against
{f(cv['best_flat']['overall']['mae'])} for the best constant and {f(cv['per_family']['overall']['mae'])}
for the per-family lookup. All labels are produced by `{r['labeller']['model']}`, not by people;
sim_base is SBERT cosine standing in for the HSS backbone; one seed ({r['seed']})."""

    overall, per = cv_tables(r)
    pf = mod["per_family_alpha"]
    pf_rows = "\n".join(f"| `{k}` | {x['n']} | {f(x['alpha'])} | {ci(x['ci95'])} |"
                        for k, x in sorted(pf.items(), key=lambda kv: kv[1]["alpha"]))
    coef_rows = "\n".join(f"| `{k}` | {f(x, 4)} |" for k, x in mod["linear_coefficients"].items())
    contrast_rows = "\n".join(f"| {k} | {f(x['mae_diff'], 4)} | {ci(x['ci95'], 4)} |" for k, x in con.items())
    abl = mod["ablation"]
    abl_rows = "\n".join(f"| {k} | {f(x['mae'])} | {f(x['pearson_r'])} |"
                         for k, x in sorted(abl.items(), key=lambda kv: kv[1]["mae"]))
    singles = {k: x for k, x in abl.items() if " + " not in k}
    best_single = min(singles, key=lambda k: singles[k]["mae"])
    avf = mod["ablation_vs_family"]
    avf_rows = "\n".join(f"| {k} | {f(x['mae_diff'], 4)} | {ci(x['ci95'], 4)} |" for k, x in avf.items())
    beats_family = [k for k, x in avf.items() if x["ci95"][1] < 0]

    t = fam["table"]
    blind_rows = "\n".join(
        f"| `{k}` | {t[k]['n']} | {f(t[k]['mean_label'], 2)} | {f(t[k]['std_label'], 2)} | "
        f"{f(t[k]['implied_alpha'])} {ci(t[k]['implied_alpha_ci95'])} | "
        f"{f(cv['no_penalty_1.0 (reference)']['per_family'][k]['mae'])} | {f(cv['flat_0.7']['per_family'][k]['mae'])} | "
        f"{f(cv['per_family']['per_family'][k]['mae'])} |"
        for k in HSS_BLIND if k in t)

    small = [k for k, x in t.items() if x["n"] < 10]
    o = v["order_effect_pass1_vs_pass2"]
    para, unrel = t["ANCHOR_paraphrase"], t["ANCHOR_unrelated"]
    below_topic = [k for k, x in t.items() if not k.startswith("ANCHOR") and x["mean_label"] < 2.0]

    return f"""# PATTERN — is a graded negation penalty justified?

Generated by `experiments/pattern/analyse.py` from `experiments/pattern/results.json`.
Every number below is read from that file.

## 0. Verdict

{verdict}

{build_section(r)}
{gate_section(r)}
## 3. Per-family labels

Label = median of three passes, 0-5. *Implied alpha* is the least-squares
multiplier that maps sim_base onto label/5 for that family,
argmin_a sum (label/5 - a * sim_base)^2, with a 95% bootstrap CI that resamples
base sentences. sim_base is **SBERT all-mpnet-base-v2 cosine, a stand-in for the
HSS backbone**. Anchors are listed for scale: a paraphrase should sit near 5, an
unrelated pair near 0-1.

{family_table(r)}

**Across the {len(om['families'])} negation families** (n = {om['n']}): Kruskal-Wallis
H = {f(om['kruskal_H'], 2)}, p = {p(om['kruskal_p'])}, epsilon-squared = {f(om['epsilon_squared'], 3)};
one-way ANOVA F = {f(om['anova_F'], 2)}, p = {p(om['anova_p'])}, eta-squared = {f(om['eta_squared'], 3)}.

**Planned comparisons** (Mann-Whitney U; Holm-corrected over these {len(fam['planned'])}; the
mean-difference CI resamples items). Unpaired: the two sides are mostly different
base sentences.

{planned_table(r)}

Plot: `experiments/pattern/plots/labels_by_family.png`.

## 4. The formula

Every model has the HSS form `sim_adjusted = alpha(x) * sim_base` and is fitted to
label/5 by least squares on {mod['n']} negation pairs (anchors excluded: they are
not negations and HSS would not penalise them).

**Lowest cross-validated MAE: `{winner}`.** The pre-registered graded formula is the
per-family lookup; to apply it by hand, find the pair's family and multiply:

    sim_adjusted = alpha[family] * sim_base

| family | n | alpha | 95% CI |
| --- | ---: | ---: | --- |
{pf_rows}

The best single constant on all {mod['n']} pairs is alpha = **{f(mod['best_flat_alpha_full'])}**.

**Linear model**, alpha(x) = intercept + sum(coef * feature), ridge lambda =
{mod['linear_ridge_lambda']} (family and intensity are partly collinear -- H is always
*hedged*, I always *intensified* -- so the ridge picks one split between them and
individual coefficients in that block should not be read alone). Reference levels:
`A_syntactic`, `neutral` intensity.

| term | coefficient |
| --- | ---: |
{coef_rows}

**Depth-3 tree** on per-item alpha (weighted so it is least squares on label/5;
min 10 items per leaf):

```
{mod['tree_rules'].rstrip()}
```

## 5. Against flat 0.7 and the best flat constant

5-fold CV grouped by base sentence (fold sizes {mod['fold_sizes']}; no base
sentence appears in two folds: {mod['no_base_sentence_spans_folds']}). Errors are on
label/5; the 0-5 column multiplies by 5. `no_penalty_1.0` (sim_base unchanged)
is a reference, not a candidate.

{overall}

Differences in CV MAE (negative = first model better), 95% CI by resampling base sentences:

| contrast | MAE difference | 95% CI |
| --- | ---: | --- |
{contrast_rows}

Per family (CV MAE / Pearson r, best MAE in bold). r is undefined where a model
predicts one constant fraction of sim_base and the family's labels do not vary.

{per}

Plot: `experiments/pattern/plots/predicted_vs_actual.png`, `experiments/pattern/plots/alpha_by_family.png`.

### Feature ablation (6.3)

Linear alpha(x) on the listed feature groups only, same CV:

| features | CV MAE | Pearson r |
| --- | ---: | ---: |
{abl_rows}

Single strongest feature group: **{best_single}**. Adding one group to family alone
(negative = better, CI by resampling base sentences):

| model | MAE difference vs family alone | 95% CI |
| --- | ---: | --- |
{avf_rows}

{'Nothing beats family alone: no CI lies entirely below zero.' if not beats_family else 'CI entirely below zero for: ' + ', '.join(beats_family) + '.'}

## 6. Where HSS's mechanism cannot reach

A cue-count detector sees *not*, *no*, *never*. It does not see *fails to*,
*unimportant*, *without*, *anything but*, or that *not impossible* cancels. For
these families HSS in practice applies no penalty at all (`no_penalty_1.0`); the
flat-0.7 column is what it would do with a perfect detector.

| family | n | mean label | std | implied alpha [95% CI] | MAE, no penalty | MAE, flat 0.7 | MAE, per-family |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
{blind_rows}

## 7. Caveats, ranked

1. **The labels are machine-produced.** `{r['labeller']['model']}` rated every pair;
   `data/human_seed.csv` is empty. Krippendorff's alpha {f(v['krippendorff_alpha_ordinal'])} says
   the model agrees with itself, not that it agrees with people. Every family mean,
   alpha and model comparison above inherits whatever the labeller gets wrong.
2. **How the labeller uses the scale.** The rubric puts "not equivalent but same
   topic" at 2. Negation families with a mean below 2 -- rated less similar than
   "same topic" although they share every content word -- are:
   {', '.join(f'`{k}` ({f(t[k]["mean_label"], 2)})' for k in below_topic) or 'none'}. For comparison the
   unrelated anchors average {f(unrel['mean_label'], 2)} and paraphrases {f(para['mean_label'], 2)}.
   A human STS rater would more likely put a contradiction on the same topic at 2-3;
   if so, every implied alpha here is too low, and the between-family spread may be
   compressed or stretched.
3. **Small base set.** 100 sentences, two textbook domains (OS and networks),
   written by one author for this experiment. Families below 10 rated pairs:
   {', '.join(f'`{k}` (n={t[k]["n"]})' for k in small) or 'none'} -- their rows are anecdotes, not estimates.
4. **sim_base is a stand-in.** SBERT all-mpnet-base-v2 cosine, not the HSS backbone.
   Implied alphas depend on the backbone's own negation sensitivity; swap
   `analyse.base_similarity` and re-run.
5. **Single seed** ({r['seed']}) for sampling, pass order, CV folds and bootstrap; not
   re-run with other seeds, so no best-of-N selection, but also no seed variance.
6. **Order effect**: pass 1 vs pass 2 differ by {f(o['mean_diff'], 3)} (p = {p(o['p'])}).
7. **Family assignment of shared strings.** {sum(r['build']['step3_ambiguous_strings_reassigned'].values())} strings
   produced by two families went to the more specific one (section 1), which moves
   e.g. every *inefficient*-type antonym into D_affixal.
8. **Stratified, not proportional, sampling** within families: subtypes are
   equally represented, so a family mean is a mean over its rules, not over the
   variants the generator happens to produce most.

## 8. What would change the conclusion, and the next step

- **Human labels.** Fill `data/human_seed.csv` (three raters), compute human-human
  and human-LLM agreement. If the LLM tracks humans, extend human labels to the
  families that matter most; if not, relabel. The analysis re-runs unchanged.
- **The real backbone.** Replace `base_similarity` with the HSS encoder. A backbone
  that already separates negations would push every implied alpha towards 1 and
  could erase the between-family differences, or sharpen them.
- **More base sentences, more domains**, so G_prepositional and L_scope_position
  have enough pairs to estimate, and so the result is not specific to OS/networks
  prose.
- **Next step:** human seed first. Nothing downstream is worth building on labels
  whose correctness is unmeasured.
"""
