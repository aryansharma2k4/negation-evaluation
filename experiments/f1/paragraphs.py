"""Phase 4: paragraph-level evaluation (teacher paragraph vs student paragraph, sentences unordered).

    ./venv/bin/python experiments/f1/paragraphs.py --step 4.2   # matching: greedy vs optimal, cell variants
    ./venv/bin/python experiments/f1/paragraphs.py --step 4.5   # per-sentence verdicts of the adopted config

Matrix: m teacher x n student sentences; each cell holds a similarity and the adopted detector's
negation verdict for (teacher, student). Assignment then verdicts:

    matched, similarity >= tau, no negation   -> covered
    matched, negation detected                -> contradicted   (also requires similarity >= tau)
    no match with similarity >= tau           -> missing
    student sentence matched to nothing       -> extra

tau (the coverage threshold) is fit on clean DEV, never on the paragraph set: Youden's J of mpnet
cosine separating "same claim or its negation" pairs (CANNOT negations, paraphrases, SemAntoNeg) from
"different claim" pairs (random, STS-low, SNLI neutral, hard negatives). For the Jaccard and mixed
cells, tau is fit the same way on the same pairs with that similarity.

Choices compared (4.2): greedy vs optimal assignment; cosine / Jaccard / mixed cells; thresholding
the matrix before vs after assignment. There is no DEV paragraph set, so no alternative can be
*selected* on held-out data: the default (optimal assignment, cosine cells, threshold before
assignment) was fixed before the run as the reported configuration, and the alternatives are
reported beside it, not chosen from.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from core import C, N, REPO, load, save

import build_dev_clean as B  # noqa: E402  (nli2: Jaccard)
import variants as V

RELATED = {"paraphrase", "SemAntoNeg", "negation"}
UNRELATED = {"random_pair", "sts_low", "snli_neutral", "hard_negative"}
VERDICTS = ("covered", "contradicted", "missing")


def fit_tau(sim: str) -> float:
    """Coverage threshold on clean DEV (see module docstring)."""
    dev = [r for r in N.read_dev_clean() if r["subtype"] in RELATED or r["kind"] in RELATED | UNRELATED]
    rows = [dict(r, label=int(r["kind"] not in UNRELATED and r["subtype"] not in UNRELATED)) for r in dev]
    emb = V.embed([s for r in rows for s in (r["a"], r["b"])])
    return N.youden_threshold(rows, [cell(sim, r["a"], r["b"], emb) for r in rows])


def fit_tau_cached(sim: str) -> float:
    """tau as committed by step 4.2 (results/4_2_tau.json), fitting it only if absent."""
    from core import RESULTS
    path = RESULTS / "4_2_tau.json"
    taus = json.loads(path.read_text()) if path.exists() else {}
    if sim not in taus:
        taus[sim] = fit_tau(sim)
        path.write_text(json.dumps(taus, indent=1))
    return taus[sim]


def cell(sim: str, t: str, s: str, emb) -> float:
    cos = float(emb[t] @ emb[s])
    if sim == "cos":
        return cos
    jac = B.overlap(t, s)
    return jac if sim == "jaccard" else (cos + jac) / 2


def items(path=REPO / "data" / "paragraph_eval.jsonl") -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def assign(S: np.ndarray, method: str, tau: float | None) -> dict[int, int]:
    """teacher idx -> student idx. tau given: cells below it are not allowed to match."""
    W = S.copy()
    if tau is not None:
        W = np.where(W >= tau, W, -1e6)
    if method == "optimal":
        r, c = linear_sum_assignment(-W)
        pairs = list(zip(r, c))
    else:
        pairs, used_t, used_s = [], set(), set()
        for flat in np.argsort(-W, axis=None):
            i, j = np.unravel_index(flat, W.shape)
            if i not in used_t and j not in used_s:
                pairs.append((i, j)); used_t.add(i); used_s.add(j)
    return {int(i): int(j) for i, j in pairs if W[i, j] > -1e5}


def verdicts(S, NEG, match, tau) -> tuple[list[str], list[int]]:
    m, n = S.shape
    tv = []
    for i in range(m):
        j = match.get(i)
        if j is None or S[i, j] < tau:
            tv.append("missing")
        else:
            tv.append("contradicted" if NEG[i, j] else "covered")
    matched = {j for i, j in match.items() if S[i, j] >= tau}
    return tv, [j for j in range(n) if j not in matched]


def run(its, det, sim="cos", method="optimal", tau=None, threshold_first=True) -> list[dict]:
    sents = [s for it in its for s in it["teacher"] + it["student"]]
    emb = V.embed(sents)
    pairs = [(t, s) for it in its for t in it["teacher"] for s in it["student"]]
    pred = {(o["teacher"], o["student"]): o for o in det.predict(list(dict.fromkeys(pairs)))}
    out = []
    for it in its:
        T, S_ = it["teacher"], it["student"]
        S = np.array([[cell(sim, t, s, emb) for s in S_] for t in T])
        NEG = np.array([[pred[(t, s)]["negation"] for s in S_] for t in T])
        match = assign(S, method, tau if threshold_first else None)
        tv, extra = verdicts(S, NEG, match, tau)
        out.append({"id": it["id"], "pred_teacher": tv, "pred_extra": extra, "match": match,
                    "S": S.round(3).tolist(), "NEG": NEG.astype(int).tolist(), "probs": [[round(pred[(t, s)]["prob"], 3) for s in S_] for t in T],
                    "stages": [[pred[(t, s)]["stage"] for s in S_] for t in T]})
    return out


def score(its, res) -> dict:
    conf = Counter()
    extra_tp = extra_fp = extra_fn = 0
    match_right = match_total = 0
    units_right = units = 0
    for it, r in zip(its, res):
        for i, g in enumerate(it["gold_teacher"]):
            conf[(g["verdict"], r["pred_teacher"][i])] += 1
            units += 1; units_right += g["verdict"] == r["pred_teacher"][i]
            if g["student"] is not None:
                match_total += 1; match_right += r["match"].get(i) == g["student"]
        ge, pe = set(it["gold_extra"]), set(r["pred_extra"])
        extra_tp += len(ge & pe); extra_fp += len(pe - ge); extra_fn += len(ge - pe)
        units += len(it["student"])
        units_right += sum((j in ge) == (j in pe) for j in range(len(it["student"])))
    t_total = sum(conf.values())
    return {"teacher_sentences": t_total, "teacher_verdict_accuracy": round(sum(v for (g, p), v in conf.items() if g == p) / t_total, 4),
            "confusion_teacher": {f"{g}->{p}": v for (g, p), v in sorted(conf.items())},
            "extra": {"tp": extra_tp, "fp": extra_fp, "fn": extra_fn},
            "all_units": units, "all_unit_accuracy": round(units_right / units, 4),
            "assignment_accuracy": round(match_right / max(1, match_total), 4), "assignment_n": match_total,
            "contradiction": {"gold": sum(v for (g, p), v in conf.items() if g == "contradicted"),
                              "caught": conf[("contradicted", "contradicted")],
                              "false_alarms": sum(v for (g, p), v in conf.items() if p == "contradicted" and g != "contradicted")}}


def step_4_2() -> None:
    from detector import Detector
    det = Detector.adopted()
    its = items()
    taus = {sim: fit_tau_cached(sim) for sim in ("cos", "jaccard", "mix")}
    res = {"tau_fit_on": "clean DEV (see module docstring)", "taus": taus, "n_items": len(its),
           "n_teacher_sentences": sum(len(i["teacher"]) for i in its), "variants": {}}
    runs = {}
    for sim in ("cos", "jaccard", "mix"):
        for method in ("optimal", "greedy"):
            for first in (True, False):
                name = f"{method}, {sim} cells, threshold {'before' if first else 'after'} assignment"
                runs[name] = run(its, det, sim, method, taus[sim], first)
                res["variants"][name] = score(its, runs[name])
    # where greedy and optimal disagree (cosine, threshold before), item by item
    a, b = runs["optimal, cos cells, threshold before assignment"], runs["greedy, cos cells, threshold before assignment"]
    res["greedy_vs_optimal"] = {"items_where_assignment_differs": [x["id"] for x, y in zip(a, b) if x["match"] != y["match"]],
                                "teacher_sentences_assigned_differently": sum(x["match"].get(i) != y["match"].get(i)
                                                                               for x, y in zip(a, b) for i in range(len(x["pred_teacher"])))}
    res["default"] = "optimal, cos cells, threshold before assignment"
    save("4_2_matching.json", res)
    for k, v in res["variants"].items():
        print(f"{k:55s} teacher-verdict acc {v['teacher_verdict_accuracy']:.3f}  units {v['all_unit_accuracy']:.3f}  "
              f"assign {v['assignment_accuracy']:.3f}  contra {v['contradiction']}")
    print(res["greedy_vs_optimal"])


class NLIOnly:
    """Side-by-side alternative for 4.5 (reported, not selected: there is no paragraph DEV set): the
    reference NLI model alone, P(contradiction) >= its clean-DEV Youden threshold (Report_NLI2 1.3)."""

    def __init__(self):
        from nli_models import Scorer
        self.s = Scorer("deberta_base")
        dev = N.read_dev_clean()
        got = self.s.score([(r["a"], r["b"]) for r in dev])
        self.thr = N.youden_threshold(dev, [got[(r["a"], r["b"])]["contradiction"] for r in dev])

    def predict(self, pairs, explain=False):
        got = self.s.score(pairs)
        return [{"teacher": a, "student": b, "negation": got[(a, b)]["contradiction"] >= self.thr,
                 "prob": got[(a, b)]["contradiction"], "stage": "NLI alone"} for a, b in pairs]


def step_4_5() -> None:
    from detector import Detector
    det = Detector.adopted()
    tau = fit_tau_cached("cos")
    out = {"config": "optimal assignment, cosine cells, threshold before assignment", "tau": tau}
    for subset, path in (("machine", REPO / "data" / "paragraph_eval.jsonl"), ("human", REPO / "data" / "paragraph_eval_human.jsonl")):
        its = items(path)
        if not its:
            out[subset] = {"n_items": 0, "note": "no items yet (stub awaiting human-written paragraphs)"}
            continue
        res = run(its, det, "cos", "optimal", tau, True)
        sc = score(its, res)
        tags = Counter(t for it in its for t in it["tags"])
        per_tag = {}
        for tag in tags:
            sub = [(it, r) for it, r in zip(its, res) if tag in it["tags"]]
            per_tag[tag] = {"items": len(sub), **{k: v for k, v in score([x for x, _ in sub], [y for _, y in sub]).items()
                                                 if k in ("teacher_sentences", "teacher_verdict_accuracy", "contradiction")}}
        wrong = []
        for it, r in zip(its, res):
            gold_c = [i for i, g in enumerate(it["gold_teacher"]) if g["verdict"] == "contradicted"]
            pred_c = [i for i, v in enumerate(r["pred_teacher"]) if v == "contradicted"]
            if gold_c != pred_c:
                wrong.append({"id": it["id"], "tags": it["tags"], "teacher": it["teacher"], "student": it["student"],
                              "gold": [g["verdict"] for g in it["gold_teacher"]], "pred": r["pred_teacher"],
                              "match": r["match"], "S": r["S"], "probs": r["probs"], "stages": r["stages"]})
        scores = [{"id": it["id"], "coverage": round(r["pred_teacher"].count("covered") / len(it["teacher"]), 3),
                   "gold_coverage": round(sum(g["verdict"] == "covered" for g in it["gold_teacher"]) / len(it["teacher"]), 3),
                   "contradictions": r["pred_teacher"].count("contradicted"),
                   "gold_contradictions": sum(g["verdict"] == "contradicted" for g in it["gold_teacher"])} for it, r in zip(its, res)]
        out[subset] = {"n_items": len(its), "tags": dict(tags), **sc, "per_tag": per_tag,
                       "paragraph_scores": scores, "contradiction_errors": wrong, "items": res}
    alt = NLIOnly()
    its = items()
    res_alt = run(its, alt, "cos", "optimal", tau, True)
    out["alternative_nli_alone"] = {"threshold": alt.thr, **score(its, res_alt),
                                    "note": "side by side, not selected: no paragraph DEV set exists"}
    out["summary"] = {k: {kk: out[k].get(kk) for kk in ("n_items", "teacher_sentences", "teacher_verdict_accuracy", "all_unit_accuracy",
                                                        "assignment_accuracy", "contradiction", "extra")} for k in ("machine", "human")}
    save("4_5_paragraphs.json", out)
    print(json.dumps(out["summary"], indent=1))
    print(json.dumps(out["machine"]["confusion_teacher"], indent=1))
    print("NLI alone:", {k: out["alternative_nli_alone"][k] for k in ("teacher_verdict_accuracy", "contradiction", "confusion_teacher")})
    for w in out["machine"]["contradiction_errors"]:
        print(w["id"], w["gold"], w["pred"], w["stages"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["4.2", "4.5"])
    {"4.2": step_4_2, "4.5": step_4_5}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
