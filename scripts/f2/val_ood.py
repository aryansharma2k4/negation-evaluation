"""Phase 0.3: datasets/splits/val_ood.csv, an out-of-distribution validation slice, built without touching TEST.

    ./venv/bin/python scripts/f2/val_ood.py      -> datasets/splits/val_ood.csv, results/0_3_val_ood.json

Section "mnli_lowoverlap" (natural, human-written; MNLI validation matched + mismatched, GLUE revision
bcdcba79): pairs whose premise and hypothesis are each one sentence of 5-40 words, token_overlap < 0.6,
and mpnet cosine >= 0.6 (same topic). Labels: MNLI contradiction -> negation (1); entailment and
neutral -> not a negation (0). Negatives are sampled to match the positives' overlap distribution in
0.1-wide bins, so overlap alone cannot separate the classes. Seed 0.

Caveat, stated before use: MNLI's "contradiction" is broader than CANNOT's "negation" (it includes
event-level incompatibility). The cosine floor removes most unrelated-scene contradictions but not all.

Section "conditionals" (synthetic, machine-generated) is appended by Phase 5 (step 5.1).
The file is asserted disjoint from TEST, VALIDATION and clean DEV at the sentence level.
"""

from __future__ import annotations

import csv
import hashlib
import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd

from core2 import C, CACHE, HERE, REPO, SEED, N, save

import build_dev_clean as B  # noqa: E402  (nli2: overlap)

PATH = REPO / "datasets" / "splits" / "val_ood.csv"
FIELDS = ["a", "b", "label", "dataset", "kind", "neg_type", "source", "template"]
N_POS = 400


def one_sentence(s: str) -> bool:
    n = len(s.split())
    return 5 <= n <= 40 and len(re.findall(r"[.!?](\s|$)", s.strip())) <= 1


def build_mnli() -> list[dict]:
    import variants as V
    frames = [pd.read_parquet(REPO / "datasets" / "raw" / "nli" / f) for f in ("mnli_validation_matched.parquet", "mnli_validation_mismatched.parquet")]
    df = pd.concat(frames, ignore_index=True)
    lab = {0: "entailment", 1: "neutral", 2: "contradiction"}
    rows = []
    for p, h, l in zip(df["premise"], df["hypothesis"], df["label"]):
        p, h = p.strip(), h.strip()
        if l in lab and one_sentence(p) and one_sentence(h) and B.overlap(p, h) < 0.6:
            rows.append({"a": p, "b": h, "nli_label": lab[l], "overlap": B.overlap(p, h)})
    emb = V.embed([s for r in rows for s in (r["a"], r["b"])])
    rows = [dict(r, cos=float(emb[r["a"]] @ emb[r["b"]])) for r in rows]
    rows = [r for r in rows if r["cos"] >= 0.6]
    rng = np.random.default_rng(SEED)
    pos = [r for r in rows if r["nli_label"] == "contradiction"]
    pos = [pos[i] for i in rng.choice(len(pos), min(N_POS, len(pos)), replace=False)]
    want = Counter(min(int(r["overlap"] * 10), 5) for r in pos)
    out = [dict(r, label=1, kind="mnli_contradiction") for r in pos]
    for kind, share in (("entailment", 0.5), ("neutral", 0.5)):
        cand = [r for r in rows if r["nli_label"] == kind]
        for b, k in want.items():
            pool = [r for r in cand if min(int(r["overlap"] * 10), 5) == b]
            take = min(len(pool), int(round(k * share)))
            out += [dict(pool[i], label=0, kind=f"mnli_{kind}") for i in rng.choice(len(pool), take, replace=False)]
    return [{"a": r["a"], "b": r["b"], "label": r["label"], "dataset": "val_ood_mnli", "kind": r["kind"], "neg_type": "",
             "source": "MNLI validation (GLUE bcdcba79)", "template": ""} for r in out]


def read(sections=None) -> list[dict]:
    rows = []
    with PATH.open() as f:
        for r in csv.DictReader(f):
            r["label"] = int(r["label"])
            r["split"] = "val_ood"
            r["neg_type"] = r["neg_type"] or None
            # fixed half split: "fit" may be used to fit parameters (Phase 1 asks for it), "select" only
            # for the 0.3 selection rule, so no parameter is ever selected on the data it was fit to
            r["part"] = "fit" if int(hashlib.sha1((r["a"] + "\t" + r["b"]).encode()).hexdigest(), 16) % 2 == 0 else "select"
            if sections is None or r["dataset"] in sections:
                rows.append(r)
    return rows


def write(rows) -> None:
    with PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def assert_disjoint(rows) -> None:
    from core2 import baseline
    sets = baseline()["sets"]
    norm = lambda s: " ".join(re.findall(r"[a-z0-9']+", s.lower()))
    mine = {norm(s) for r in rows for s in (r["a"], r["b"])}
    for name in ("test", "validation", "dev_clean"):
        theirs = {norm(s) for r in sets[name] for s in (r["a"], r["b"])}
        clash = mine & theirs
        assert not clash, f"{len(clash)} val_ood sentences also in {name}: {list(clash)[:3]}"


def featurise(rows, keys=("deberta_base",)) -> tuple[np.ndarray, dict]:
    """17 combiner features + every requested NLI model's probabilities, for new rows (cached)."""
    import detector as D
    import phase4 as P4
    import variants as V
    path = CACHE / "featurised.pkl"
    cache = pickle.loads(path.read_bytes()) if path.exists() else {}
    key = lambda r: (r["a"], r["b"])
    todo = [r for r in rows if key(r) not in cache]
    if todo:
        sents = list(dict.fromkeys(s for r in todo for s in (r["a"], r["b"])))
        analyses = C.P.analyse_many(sents)
        emb = V.embed(sents)
        P = D._probs("deberta_base", todo)
        rr = []
        for r, p in zip(todo, P):
            rr.append({**r, "nli_c": float(p[0]), "nli_e": float(p[1]), "nli_n": float(p[2]), "cos": float(emb[r["a"]] @ emb[r["b"]])})
        X = P4.featurise(rr, analyses)
        import phase0 as P0
        mask = P0.override_mask(rr, analyses)
        for r, x, m in zip(rr, X, mask):
            cache[key(r)] = {"x": x, "override": bool(m), "nli_c": r["nli_c"], "nli_e": r["nli_e"], "nli_n": r["nli_n"]}
        CACHE.mkdir(exist_ok=True)
        path.write_bytes(pickle.dumps(cache))
    X = np.array([cache[key(r)]["x"] for r in rows])
    for r in rows:
        r.update({k: cache[key(r)][k] for k in ("nli_c", "nli_e", "nli_n")})
    probs = {k: D._probs(k, rows) for k in keys}
    return X, {"override": np.array([cache[key(r)]["override"] for r in rows]), "probs": probs}


def main() -> None:
    rows = build_mnli()
    assert_disjoint(rows)
    write(rows)
    comp = Counter((r["kind"], r["label"]) for r in rows)
    ov = {k: [B.overlap(r["a"], r["b"]) for r in rows if r["kind"] == k] for k in {r["kind"] for r in rows}}
    res = {"path": "data/val_ood.csv", "sections": {"mnli_lowoverlap": {f"{k}|{l}": v for (k, l), v in sorted(comp.items())}},
           "overlap_by_kind": {k: {"n": len(v), "mean": round(float(np.mean(v)), 3), "median": round(float(np.median(v)), 3)} for k, v in ov.items()},
           "filters": "one sentence each, 5-40 words, overlap < 0.6, mpnet cosine >= 0.6; negatives overlap-matched in 0.1 bins",
           "disjoint_from": ["test", "validation", "dev_clean"]}
    rr = read()
    res["halves"] = {part: dict(Counter(f"{r['kind']}|{r['label']}" for r in rr if r["part"] == part)) for part in ("fit", "select")}
    save("0_3_val_ood.json", res)
    print(res)
    rng = np.random.default_rng(1)
    for i in rng.choice(len(rows), 10, replace=False):
        print(rows[i]["kind"], "|", rows[i]["a"][:110], "|", rows[i]["b"][:110])


if __name__ == "__main__":
    main()
