"""Phase 3: put the negation families to work against the overlap shortcut.

    ./venv/bin/python scripts/f2/rebalance.py --step 3.1   # token_overlap of (A, generated variant) by family
    ./venv/bin/python scripts/f2/rebalance.py --step 3.2   # augmented training sets
    ./venv/bin/python scripts/f2/rebalance.py --step 3.3   # retrain + measure

Bases are clean-DEV A sentences only (asserted disjoint from VALIDATION, TEST and val_ood).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from core2 import C, HERE, REPO, SEED, STOPPED, baseline, load, log_test_read, save

import build_dev_clean as B  # noqa: E402  (nli2: overlap)
import variants as V  # noqa: E402  (f1)


def dev_bases() -> list[str]:
    return list(dict.fromkeys(r["a"] for r in baseline()["sets"]["dev_clean"]))


def step_3_1() -> None:
    neg = V.negations(dev_bases())
    by = defaultdict(list)
    for a, vs in neg.items():
        for v, fam in vs:
            by[fam].append(B.overlap(a, v))
    res = {}
    for fam, ov in sorted(by.items(), key=lambda kv: np.median(kv[1])):
        ov = np.array(ov)
        res[fam] = {"n": len(ov), "mean": round(float(ov.mean()), 3), "median": round(float(np.median(ov)), 3),
                    "share_below_0.4": round(float((ov < 0.4).mean()), 4), "share_below_0.6": round(float((ov < 0.6).mean()), 4)}
    allv = np.concatenate([np.array(v) for v in by.values()])
    res["_all"] = {"n": len(allv), "share_below_0.4": round(float((allv < 0.4).mean()), 4), "share_below_0.6": round(float((allv < 0.6).mean()), 4),
                   "n_below_0.6": int((allv < 0.6).sum()), "n_below_0.4": int((allv < 0.4).sum())}
    save("3_1_family_overlap.json", {"bases": len(neg), "families": res})
    for k, v in res.items():
        print(k, v)


N_PER_CLASS = 600
AUG_PATH = REPO / ".cache" / "f2" / "augmentation.json"     # rebuilt deterministically (seed 0); composition in results


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", s.lower()))


def _forbidden() -> set:
    import val_ood as VO
    sets = baseline()["sets"]
    rows = sets["validation"] + sets["test"] + VO.read()
    return {_norm(s) for r in rows for s in (r["a"], r["b"])}


def natural(rng) -> list[dict]:
    """MNLI train + SNLI train: one sentence each, 5-40 words, overlap < 0.6, cosine >= 0.6. Positives:
    contradictions (300 per corpus). Negatives: entailment + neutral, overlap-matched in 0.1 bins (300 per
    corpus). Same construction as val_ood, but from the *train* splits."""
    import val_ood as VO
    out = []
    lab = {"mnli": {0: "entailment", 1: "neutral", 2: "contradiction"}, "snli": {0: "entailment", 1: "neutral", 2: "contradiction"}}
    for corpus, f in (("mnli", "mnli_train.parquet"), ("snli", "snli_train.parquet")):
        df = pd.read_parquet(REPO / "datasets" / "raw" / "nli" / f)
        df = df.sample(frac=1.0, random_state=SEED)
        cand = []
        for p_, h, l in zip(df["premise"], df["hypothesis"], df["label"]):
            if l in lab[corpus] and VO.one_sentence(p_) and VO.one_sentence(h):
                ov = B.overlap(p_, h)
                if ov < 0.6:
                    cand.append({"a": p_.strip(), "b": h.strip(), "nli_label": lab[corpus][l], "overlap": ov})
            if len(cand) >= 9000:
                break
        emb = V.embed([s for r in cand for s in (r["a"], r["b"])])
        cand = [r for r in cand if float(emb[r["a"]] @ emb[r["b"]]) >= 0.6]
        pos = [r for r in cand if r["nli_label"] == "contradiction"][: N_PER_CLASS // 2]
        want = Counter(min(int(r["overlap"] * 10), 5) for r in pos)
        negs = []
        for kind in ("entailment", "neutral"):
            pool = [r for r in cand if r["nli_label"] == kind]
            for b, k in want.items():
                negs += [r for r in pool if min(int(r["overlap"] * 10), 5) == b][: int(round(k / 2))]
        out += [{"a": r["a"], "b": r["b"], "label": int(r["nli_label"] == "contradiction"), "dataset": f"aug_{corpus}",
                 "kind": f"aug_{corpus}_{r['nli_label']}", "split": "dev", "neg_type": None} for r in pos + negs]
    return out


def generated(rng) -> list[dict]:
    """The generated tail: negations of clean-DEV bases with overlap < 0.6 (label 1), plus random
    re-pairings of clean-DEV bases with other bases' variants (label 0). The generator has no paraphrase
    family and its cancellations sit at high overlap (step 3.1), so these are the only low-overlap
    generated non-negations available."""
    neg = V.negations(dev_bases())
    tail = [(a, v, f) for a, vs in neg.items() for v, f in vs if B.overlap(a, v) < 0.6]
    pick = [tail[i] for i in rng.choice(len(tail), min(N_PER_CLASS, len(tail)), replace=False)]
    out = [{"a": a, "b": v, "label": 1, "dataset": "aug_generated", "kind": f"aug_gen_{f}", "split": "dev", "neg_type": None} for a, v, f in pick]
    bases = list(neg)
    allv = [(a, v) for a, vs in neg.items() for v, _ in vs]
    for _ in range(N_PER_CLASS):
        a = bases[rng.integers(len(bases))]
        b_a, v = allv[rng.integers(len(allv))]
        if b_a != a:
            out.append({"a": a, "b": v, "label": 0, "dataset": "aug_generated", "kind": "aug_gen_random_pair", "split": "dev", "neg_type": None})
    return out


def build() -> dict:
    if AUG_PATH.exists():
        return json.loads(AUG_PATH.read_text())
    rng = np.random.default_rng(SEED)
    sets = {"natural": natural(rng), "generated": generated(rng)}
    bad = _forbidden()
    for name, rows in sets.items():
        clash = [r for r in rows if _norm(r["a"]) in bad or _norm(r["b"]) in bad]
        sets[name] = [r for r in rows if r not in clash]
        print(name, "dropped for touching VALIDATION / TEST / val_ood:", len(clash))
    sets["mixed"] = sets["natural"] + sets["generated"]
    AUG_PATH.parent.mkdir(exist_ok=True)
    AUG_PATH.write_text(json.dumps(sets))
    return sets


def step_3_2() -> None:
    sets = build()
    bad = _forbidden()
    for rows in sets.values():
        assert not any(_norm(r["a"]) in bad or _norm(r["b"]) in bad for r in rows), "augmentation touches a held-out sentence"
    dev = baseline()["sets"]["dev_clean"]
    ov_dev = [B.overlap(r["a"], r["b"]) for r in dev]

    def dist(rows, label):
        v = np.array([B.overlap(r["a"], r["b"]) for r in rows if r["label"] == label])
        return {"n": len(v), "median": round(float(np.median(v)), 3), "share_below_0.4": round(float((v < 0.4).mean()), 3),
                "share_below_0.6": round(float((v < 0.6).mean()), 3)}
    res = {"composition": {k: dict(Counter(r["kind"] for r in v)) for k, v in sets.items()},
           "overlap": {"clean DEV": {"negations": dist(dev, 1), "non-negations": dist(dev, 0)}}}
    for k, v in sets.items():
        res["overlap"][f"clean DEV + {k}"] = {"negations": dist(dev + v, 1), "non-negations": dist(dev + v, 0)}
        res["overlap"][f"{k} alone"] = {"negations": dist(v, 1), "non-negations": dist(v, 0)}
    res["asserted_disjoint_from"] = ["validation", "test", "val_ood"]
    save("3_2_augmentation.json", res)
    print(json.dumps(res, indent=1))
    for k, v in sets.items():
        rng = np.random.default_rng(3)
        for i in rng.choice(len(v), 4, replace=False):
            print(k, v[i]["kind"], "|", v[i]["a"][:90], "|", v[i]["b"][:90])


def step_3_3() -> None:
    import deberta as D
    import ensemble as E
    import eval2 as E2
    import val_ood as VO
    ctx = D.context()
    aug = build()
    X_c = {k: np.hstack([ctx["X"][k]] + [ctx["P"][m][k] for m in D.PRUNED if m != "deberta_base"]) for k in E2.SETS}
    log_test_read("3.3", "adopted architecture retrained on clean DEV + augmentation")
    g0 = E.fit_gbm(ctx["sets"], X_c["dev_clean"])
    sc0 = {k: np.where(ctx["masks"][k], STOPPED, g0["model"].predict_proba(X_c[k])[:, 1]) for k in E2.SETS}
    current = E2.row(ctx["sets"], D.with_rule(ctx, sc0, "deberta_base"), g0["threshold"], ctx["ood"])
    res = {"current_adopted (C + override)": current, "rows": {}}
    para = {}
    for name, rows in aug.items():
        Xa, extra = VO.featurise(rows, keys=D.PRUNED)
        Xa = np.hstack([Xa] + [extra["probs"][m] for m in D.PRUNED if m != "deberta_base"])
        dev2 = list(ctx["sets"]["dev_clean"]) + rows
        g = E.fit_gbm({"dev_clean": dev2}, np.vstack([X_c["dev_clean"], Xa]))
        sc = {k: np.where(ctx["masks"][k], STOPPED, g["model"].predict_proba(X_c[k])[:, 1]) for k in E2.SETS}
        for tag, s in (("", sc), (" + override", D.with_rule(ctx, sc, "deberta_base"))):
            ev = E2.row(ctx["sets"], s, g["threshold"], ctx["ood"])
            ev["oof_dev_accuracy_with_augmentation"] = g["oof_dev_accuracy"]
            ev["gain_vs_phase0"] = __import__("override").F1gain(ctx, s, g["threshold"])
            ev["passes_rule"] = E2.adopt(current, ev)
            ev["n_augmented"] = len(rows)
            ev["importance_token_overlap"] = round(float(g["model"].feature_importances_[16]), 4)
            res["rows"][f"C retrained on clean DEV + {name} ({len(rows)} pairs){tag}"] = ev
    passing = [n for n, ev in res["rows"].items() if ev["passes_rule"]]
    res["selected"] = max(passing, key=lambda n: E2.gain_score(current, res["rows"][n])) if passing else "keep current (C + override)"
    save("3_3_rebalance.json", res)
    print(E2.HEADER)
    print(E2.line("current: C + override", current))
    for n, ev in res["rows"].items():
        print(E2.line(n, ev), "pass" if ev["passes_rule"] else "", "ov-imp", ev["importance_token_overlap"])
    print("SELECTED:", res["selected"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["3.1", "3.2", "3.3"])
    {"3.1": step_3_1, "3.2": step_3_2, "3.3": step_3_3}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
