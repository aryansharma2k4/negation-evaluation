"""Negation detector dashboard (reports/08-guardrail-ensemble.md Phase 6). Research artefact, not a product.

    ./venv/bin/streamlit run dashboard/app.py          # http://localhost:8501, CPU only

Every number on screen comes from a stage of the adopted system (scripts/f1/detector.py) or a
committed results file (reports/results/f1/*.json). The "probability" shown is the GBM's raw
output: it has never been calibrated, and is labelled as such.
"""

from __future__ import annotations

import html
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
F1 = REPO / "scripts" / "f1"
sys.path.insert(0, str(F1))
RESULTS = REPO / "reports" / "results" / "f1"

st.set_page_config(page_title="Negation detector", layout="wide")


@st.cache_resource
def detector():
    from detector import Detector
    return Detector.adopted()


@st.cache_data(show_spinner=False)
def predict(pairs: tuple, explain: bool) -> list[dict]:
    return detector().predict(list(pairs), explain=explain)


@st.cache_data(show_spinner=False)
def scopes(sentence: str):
    from detector import scopes as _s
    return _s(sentence)


@st.cache_data(show_spinner=False)
def conditional(sentence: str):
    from detector import conditional as _c
    return _c(sentence)


@st.cache_data(show_spinner=False)
def severity_of(t: str, s: str):
    sys.path.insert(0, str(REPO / "scripts" / "f2"))
    from severity import label_pairs
    return label_pairs([(t, s)])[0]


@st.cache_data(show_spinner=False)
def affirmations(sentence: str):
    from negation.generators.n_affirmation import affirm
    return [a.__dict__ for a in affirm(sentence)]


def highlight(sc: dict) -> str:
    """The sentence with the clause scope underlined and the cue in bold red."""
    out = []
    for i, tok in enumerate(sc["tokens"]):
        t = html.escape(tok)
        if i == sc["cue_index"]:
            t = f"<b style='color:#c0392b'>{t}</b>"
        if i in sc["scope"]:
            t = f"<span style='background:#fdebd0'>{t}</span>"
        out.append(t)
    return "".join(out)


def verdict_box(o: dict) -> None:
    color = "#c0392b" if o["negation"] else "#1e8449"
    word = "NEGATION" if o["negation"] else "NOT A NEGATION"
    st.markdown(f"<div style='font-size:2.2em;font-weight:700;color:{color}'>{word}</div>"
                f"<div>combiner raw output <b>{o['prob']:.3f}</b> (threshold {o['threshold']:.3f}; uncalibrated, not a "
                f"probability you can read as a confidence) · decided by: <b>{o['stage']}</b></div>",
                unsafe_allow_html=True)


def tab_pair() -> None:
    c1, c2 = st.columns(2)
    q = st.query_params
    t = c1.text_area("Teacher sentence", q.get("t", "TCP is a connection-oriented protocol."), height=90)
    s = c2.text_area("Student sentence", q.get("s", "TCP is not a connection-oriented protocol."), height=90)
    if not (st.button("Check", type="primary") or "t" in q):
        return
    t0 = time.perf_counter()
    o = predict(((t.strip(), s.strip()),), True)[0]
    st.caption(f"computed in {1000 * (time.perf_counter() - t0):.0f} ms")
    verdict_box(o)
    st.markdown("**Which stage decided it.** Stage order: 1 cancellation override (can only say *not a negation*) → "
                "2 combiner (GBM over the 17 features + the ensemble's NLI probabilities) → 3 low-overlap NLI override "
                "(can only say *negation*: overlap < θ, P(contradiction) > ρ and cosine ≥ κ; report_f2 step 1.3). "
                "Conditionals are combiner features, not an override.")
    cfg = detector().cfg
    ov = o["features"].get("token_overlap", float("nan"))
    rule = cfg.rule or {}
    zone = ("**in the shortcut-prone range** (< 0.6): on TEST the report_f1 combiner caught 80.7% of negations here vs 95.4% "
            "for NLI alone (report_f2 step 0.1)") if ov < 0.6 else "outside the shortcut-prone range (≥ 0.6)"
    st.markdown(f"**Word overlap (Jaccard)** = **{ov:.3f}** — {zone}. "
                + (f"Override condition: overlap < {rule.get('theta')}, P(contradiction) > {rule.get('rho')}, cosine ≥ {rule.get('kappa')} "
                   f"→ {'**fired**' if o['stage'] == 'low-overlap NLI override' else 'not fired'} "
                   f"(cosine {o['cos']:.3f})." if rule else ""))
    if o["negation"]:
        sev, cues = severity_of(t, s)
        st.markdown(f"<div style='font-size:1.3em'>Severity: <b>{sev}</b></div>"
                    f"<div>differing cue(s): {html.escape(', '.join(cues) or 'none (lexical: antonym or rewording)')}. "
                    f"Descriptive only — severity changes no verdict (report_f2 Phase 6).</div>", unsafe_allow_html=True)

    st.subheader("Features (combiner input)")
    top = {name for name, _ in o["contributions"]}
    df = pd.DataFrame([{"feature": k, "value": round(v, 4), "top contributor": "◀" if k in top else ""}
                       for k, v in o["features"].items()])
    st.dataframe(df.style.apply(lambda r: ["background:#fdebd0" if r["top contributor"] else "" for _ in r], axis=1),
                 width="stretch", hide_index=True, height=36 * (len(df) + 1) + 4)
    st.caption("Top contributors: change in the raw GBM output when the feature is replaced by its clean-DEV median "
               "(a local, approximate attribution): " + ", ".join(f"{n} {d:+.3f}" for n, d in o["contributions"]))

    st.subheader("Each NLI model's probabilities (premise = teacher)")
    st.dataframe(pd.DataFrame(o["nli"]).T.round(4), width="stretch")

    st.subheader("Negation cues and their clause scope")
    for who, sent in (("Teacher", t), ("Student", s)):
        sc = scopes(sent)
        if not sc:
            st.markdown(f"*{who}:* no negation cue found")
        for c in sc:
            st.markdown(f"*{who}* — cue **{c['cue']}** ({c['family']}, {'main' if c['main_clause'] else 'subordinate'} clause): "
                        + highlight(c), unsafe_allow_html=True)
        cond = conditional(sent)
        if cond:
            st.markdown(f"*{who}* is a conditional (**{cond['marker']}**): antecedent “{cond['antecedent']}” "
                        f"({'negated' if cond['antecedent_negated'] else 'positive'}) · consequent “{cond['consequent']}” "
                        f"({'negated' if cond['consequent_negated'] else 'positive'})")

    g = o.get("guardrail")
    st.subheader("Guardrail: generated negations of the teacher sentence")
    if g is None:
        why = "reports/08-guardrail-ensemble.md rejected the generate-and-match guardrail (step 3.5) and reports/09-overlap-shortcut.md did not reinstate it"
        st.info(f"Not part of the decision: {why}. The generated negations are shown for inspection only.")
        from variants import negations, embed
        neg = negations([t])[t]
        emb = embed([t, s] + [v for v, _ in neg])
        rows = sorted(({"variant": v, "family": f, "sim(S, variant)": float(emb[s] @ emb[v])} for v, f in neg),
                      key=lambda r: -r["sim(S, variant)"])
        s_orig = float(emb[s] @ emb[t])
        st.markdown(f"sim(S, T) = **{s_orig:.3f}** · max sim(S, n) = **{rows[0]['sim(S, variant)'] if rows else float('nan'):.3f}** · "
                    f"margin = **{(rows[0]['sim(S, variant)'] - s_orig) if rows else float('nan'):+.3f}**")
        st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)
    else:
        st.markdown(f"sim(S, T) = **{g.get('s_orig', float('nan')):.3f}** · aggregated sim(S, N) = **{g.get('s_neg', float('nan')):.3f}** "
                    f"· margin = **{(g['margin'] if g['margin'] is not None else float('nan')):+.3f}** (keep if > δ = {g.get('delta', 0):.3f})")
        st.dataframe(pd.DataFrame(g.get("variants", [])).round(3), width="stretch", hide_index=True)

    aff = affirmations(t)
    if aff:
        st.subheader("Affirmation of the teacher sentence (Phase 1)")
        st.dataframe(pd.DataFrame(aff)[["text", "kind", "clause", "primary", "approximate", "polarity_flip", "rule"]],
                     width="stretch", hide_index=True)


def tab_paragraph() -> None:
    from paragraphs import assign, cell, verdicts, fit_tau_cached
    c1, c2 = st.columns(2)
    T = c1.text_area("Teacher paragraph (one sentence per line)", "\n".join([
        "Round-robin scheduling gives each process a fixed time quantum.", "Round-robin scheduling is preemptive.",
        "A very large time quantum makes round-robin behave like first-come first-served.",
        "Round-robin scheduling can cause frequent context switches."]), height=180)
    S = c2.text_area("Student paragraph (one sentence per line)", "\n".join([
        "With a huge quantum, round robin degenerates into FCFS.", "Round robin is a non-preemptive scheduling algorithm.",
        "Every process receives the same fixed slice of CPU time under round robin.", "SJF minimizes the average waiting time."]),
        height=180)
    if not (st.button("Compare", type="primary") or st.query_params.get("view") == "paragraph"):
        return
    T = [x.strip() for x in T.splitlines() if x.strip()]
    S = [x.strip() for x in S.splitlines() if x.strip()]
    from variants import embed
    emb = embed(T + S)
    tau = fit_tau_cached("cos")
    M = np.array([[cell("cos", t, s, emb) for s in S] for t in T])
    preds = predict(tuple((t, s) for t in T for s in S), False)
    NEG = np.array([[p["negation"] for p in preds[i * len(S):(i + 1) * len(S)]] for i in range(len(T))])
    match = assign(M, "optimal", tau)
    tv, extra = verdicts(M, NEG, match, tau)
    st.markdown(f"Similarity: mpnet cosine. Coverage threshold τ = **{tau:.3f}** (fit on clean DEV). Assignment: Hungarian "
                f"(`scipy.optimize.linear_sum_assignment`) after removing cells below τ.")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.2 + 1.1 * len(S), 0.9 + 0.6 * len(T)))
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(T)):
        for j in range(len(S)):
            mark = ("■ " if match.get(i) == j else "") + ("¬" if NEG[i, j] else "")
            ax.text(j, i, f"{mark}{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(S)), [f"S{j + 1}" for j in range(len(S))])
    ax.set_yticks(range(len(T)), [f"T{i + 1}" for i in range(len(T))])
    ax.set_title("cosine similarity  (■ = assigned, ¬ = detector says negation)", fontsize=9)
    st.pyplot(fig, width="content")
    color = {"covered": "#1e8449", "contradicted": "#c0392b", "missing": "#7f8c8d"}
    for i, v in enumerate(tv):
        j = match.get(i)
        st.markdown(f"<b style='color:{color[v]}'>{v.upper()}</b> — T{i + 1}: {html.escape(T[i])}"
                    + (f"<br>&nbsp;&nbsp;&nbsp;↳ S{j + 1}: {html.escape(S[j])}" if j is not None and v != "missing" else ""),
                    unsafe_allow_html=True)
    for j in extra:
        st.markdown(f"<b style='color:#2471a3'>EXTRA</b> — S{j + 1}: {html.escape(S[j])}", unsafe_allow_html=True)
    covered = sum(v == "covered" for v in tv)
    st.markdown(f"**Coverage {covered} / {len(T)}** ({100 * covered / len(T):.0f}%) · **contradictions {tv.count('contradicted')}** · "
                f"missing {tv.count('missing')} · extra {len(extra)}  \n*Reported separately on purpose: a contradiction is not an omission.*")


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def load2(name):
    p = REPO / "reports" / "results" / "f2" / name
    return json.loads(p.read_text()) if p.exists() else None


def f2_line(name: str, ev: dict) -> dict:
    pc = lambda x: None if x is None else round(100 * x, 1)
    return {"configuration": name, "phase-2 acc": pc(ev["cannot_phase2"]["accuracy"]), "all-TEST": pc(ev["all"]["accuracy"]),
            "xNot360": pc(ev["xnot360"]["accuracy"]), "precision": pc(ev["all"]["precision"]), "recall": pc(ev["all"]["recall"]),
            "low-ov neg recall (n=129)": pc(ev["lowov_neg_recall"]["recall"]), "low-ov non-neg FPR (n=396)": pc(ev["lowov_non_fpr"]["fpr"]),
            "VAL": pc(ev["validation"]["accuracy"]), "val_ood (select)": pc(ev["val_ood_select"]["accuracy"])}


def tab_results() -> None:
    st.markdown("Tables rendered from the committed results files in `reports/results/f1/` and `reports/results/f2/`; "
                "nothing here is hard-coded. All values in %.")
    ov = load2("0_1_overlap.json")
    if ov:
        st.subheader("report_f2 0.1 — combiner vs NLI alone by word overlap, TEST negations (human-written)")
        st.dataframe(pd.DataFrame([{k: r[k] for k in ("bin", "n", "combiner", "nli_alone", "combiner_minus_nli", "p_permutation")}
                                   for r in ov["test"]["negations"]["fixed_bins"] if r["n"]]), width="stretch", hide_index=True)
    rows = []
    for fname in ("1_3_override.json", "2_2_deberta.json", "3_3_rebalance.json"):
        d = load2(fname)
        if d:
            rows += [f2_line(n, ev) for n, ev in d["rows"].items()]
    if rows:
        st.subheader("report_f2 ablation (TEST unless noted)")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.markdown("---")
    fin = load("final_f1.json")
    if fin:
        st.subheader("Ablation (TEST unless noted)")
        df = pd.DataFrame(fin["ablation_table"])
        df["configuration"] = df["configuration"].str.replace("**", "", regex=False)
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption(fin.get("selection_note", ""))
    sa = load("2_2_standalone.json")
    if sa:
        st.subheader("NLI models standalone (Phase 2.2)")
        st.dataframe(pd.DataFrame([{"model": k, "params": v["parameters"], "threshold": round(v["threshold"], 4),
                                    "phase-2 acc": v["row"]["cannot_phase2"]["accuracy"], "all-TEST acc": v["row"]["all"]["accuracy"],
                                    "AUC": v["row"]["all"]["auc"], "VAL acc": v["row"]["validation"]["accuracy"],
                                    "ms/pair": v.get("ms_per_pair")} for k, v in sa["models"].items()]), width="stretch", hide_index=True)
    dis = load("2_3_disagreement.json")
    if dis:
        st.subheader("Pairwise disagreement on TEST (share of pairs with different binary decisions)")
        keys = sorted({k for p in dis["test"]["disagreement"] for k in p.split("|")})
        mat = pd.DataFrame(index=keys, columns=keys, dtype=float)
        for p, v in dis["test"]["disagreement"].items():
            a, b = p.split("|")
            mat.loc[a, b] = mat.loc[b, a] = v
        st.dataframe(mat.round(3), width="stretch")
    cov = load("3_3_coverage.json")
    if cov:
        c = cov["by_split"]["dev_clean"]
        st.subheader("Generator coverage on clean DEV gold negations (Phase 3.3, the guardrail's precondition)")
        st.dataframe(pd.DataFrame([{"type": "all", "n": c["gold_negations"], **c["coverage"]}] +
                                  [{"type": k, **v} for k, v in c["by_gold_type"].items()]), width="stretch", hide_index=True)
    inv = load("1_3_inventory.json")
    if inv:
        st.subheader("Affirmation inventory (Phase 1.3, clean DEV)")
        st.dataframe(pd.DataFrame(inv["rows"]), width="stretch", hide_index=True)
    par = load("4_5_paragraphs.json")
    if par:
        st.subheader("Paragraph case study (Phase 4; machine-authored items)")
        st.json(par.get("summary", {}))
    dom = load("5_3_domain.json")
    if dom:
        st.subheader("Domain case study (Phase 5; machine-generated answers — not a benchmark)")
        st.dataframe(pd.DataFrame(dom["machine"]["by_category"]).T, width="stretch")


st.title("Negation detector — explainable view")
st.caption("reports/08-guardrail-ensemble.md, reports/09-overlap-shortcut.md · research artefact: every number is traceable to a feature, a stage, or a committed results file.")
VIEWS = {"pair": ("Single pair", tab_pair), "paragraph": ("Paragraph", tab_paragraph), "results": ("Results", tab_results)}
view = st.query_params.get("view")          # ?view=pair|paragraph|results renders one view directly (used for screenshots)
if view in VIEWS:
    st.header(VIEWS[view][0])
    VIEWS[view][1]()
else:
    for tab, (_, fn) in zip(st.tabs([v[0] for v in VIEWS.values()]), VIEWS.values()):
        with tab:
            fn()
