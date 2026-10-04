"""Fit on DEV, check on VALIDATION, score TEST once.

    ./venv/bin/python experiments/composite/run_composite.py           # dev + validation
    ./venv/bin/python experiments/composite/run_composite.py --timing  # + per-stage ms/pair
    ./venv/bin/python experiments/composite/run_composite.py --final   # + test, once

Everything lands in results_composite.json. --final refuses to run if
test_scorings.json already records a test scoring.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

import composite as C
from composite import P, prun

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_composite.json"
TEST_LOG = HERE / "test_scorings.json"
ROWS = ["nli", "gate", "nli_gate", "full", "polarity_v2"]
ROW_TEXT = {
    "nli": "NLI alone",
    "gate": "content gate alone (passes -> negation)",
    "nli_gate": "NLI + content gate",
    "full": "NLI + content gate + cancellation veto",
    "polarity_v2": "polarity v2 alone (|dp| x c_resid)",
}
# the configurations validation chooses between, simplest first: a tie goes to the simpler one
CANDIDATES = ["nli", "nli_gate", "full"]
FPR_KINDS = ["paraphrase", "sts_low", "snli_neutral", "random_pair", "K1", "SemAntoNeg",
             "ours_paraphrase", "ours_unrelated", "xnot360_not_negation"]
RECALL_TYPES = ["verbal", "antonym", "other", "ours_negation", "xnot360_negation"]


# ---------------------------------------------------------------------------
# Configuration: everything fit on DEV
# ---------------------------------------------------------------------------

def configure(splits, analyses) -> dict:
    dev = splits["dev"]
    nli_dev = np.array([r["nli"] for r in dev])
    nli_thr = C.fit_threshold(dev, nli_dev)
    gates = {sim: C.fit_gate(dev, sim, nli_thr) for sim in C.SIMILARITIES}
    gate = max(gates.values(), key=lambda g: g["dev_j"])          # similarity picked on DEV
    sim, t1 = gate["sim"], gate["t1"]
    pol = {s: P.score(a, "v2") for s, a in analyses.items()}
    pol_dev = polarity_score(dev, pol)
    return {
        "nli_threshold": nli_thr,
        "gate_fits": gates, "sim": sim, "t1": t1,
        "nli_gate_threshold": C.fit_threshold(dev, C.cascade(dev, analyses, sim, t1)[0]),
        "full_threshold": C.fit_threshold(dev, C.cascade(dev, analyses, sim, t1, True)[0]),
        "polarity_threshold": C.fit_threshold(dev, pol_dev),
        "_pol": pol,
    }


def polarity_score(rows, pol) -> np.ndarray:
    """v2:f2, the formula polarity's validation chose, with plain v2 parity."""
    return np.array([abs(pol[r["b"]] - pol[r["a"]]) * r["c_resid"] for r in rows])


def scorers(cfg, analyses) -> dict:
    """name -> rows -> (score, threshold, stage)."""
    sim, t1 = cfg["sim"], cfg["t1"]
    allnli = lambda rows: np.full(len(rows), 3)
    return {
        "nli": lambda rows: (np.array([r["nli"] for r in rows]), cfg["nli_threshold"], allnli(rows)),
        "gate": lambda rows: (np.array([r[sim] for r in rows]), t1, np.full(len(rows), 0)),
        "nli_gate": lambda rows: (*_casc(rows, analyses, sim, t1, False, cfg["nli_gate_threshold"]),),
        "full": lambda rows: (*_casc(rows, analyses, sim, t1, True, cfg["full_threshold"]),),
        "polarity_v2": lambda rows: (polarity_score(rows, cfg["_pol"]), cfg["polarity_threshold"],
                                     np.full(len(rows), 0)),
    }


def _casc(rows, analyses, sim, t1, use_veto, thr):
    score, stage = C.cascade(rows, analyses, sim, t1, use_veto)
    return score, thr, stage


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(rows, score, threshold, stage) -> dict:
    y = C.labels(rows)
    pred = score >= threshold
    out = {"n": len(rows), **prun.detect.metrics(y, score, threshold)}
    out["fpr_by_kind"] = {}
    for k in FPR_KINDS:
        m = np.array([r["kind"] == k and r["label"] == 0 for r in rows])
        if m.any():
            out["fpr_by_kind"][k] = {"n": int(m.sum()), "fpr": round(float(pred[m].mean()), 4)}
    out["recall_by_type"] = {}
    for t in RECALL_TYPES:
        m = np.array([r["label"] == 1 and (r.get("neg_type") == t or r["kind"] == t) for r in rows])
        if m.any():
            out["recall_by_type"][t] = {"n": int(m.sum()), "recall": round(float(pred[m].mean()), 4)}
    out["stage_counts"] = {str(k): int(v) for k, v in sorted(Counter(stage.tolist()).items())}
    out["reached_nli"] = int((stage == 3).sum())
    out["by_dataset"] = {}
    for ds in sorted({r["dataset"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["dataset"] == ds]
        out["by_dataset"][ds] = {"n": len(idx), **prun.detect.metrics(y[idx], score[idx], threshold)}
    return out


def run_split(rows, cfg, analyses) -> dict:
    return {name: evaluate(rows, *fn(rows)) for name, fn in scorers(cfg, analyses).items()}


def examples(rows, analyses, cfg, kind: str, k: int, rng) -> list[dict]:
    """k verbatim errors of the full system, sampled (not picked)."""
    score, thr, stage = scorers(cfg, analyses)["full"](rows)
    y, pred = C.labels(rows), score >= thr
    want = (pred & (y == 0)) if kind == "fp" else (~pred & (y == 1))
    idx = rng.sample(list(np.flatnonzero(want)), min(k, int(want.sum())))
    return [{"dataset": rows[i]["dataset"], "kind": rows[i]["kind"], "neg_type": rows[i].get("neg_type"),
             "a": rows[i]["a"], "b": rows[i]["b"], "label": int(y[i]),
             "gate_score": round(rows[i][cfg["sim"]], 3),
             "cancellation": bool(C.cancellation_detected(analyses[rows[i]["a"]], analyses[rows[i]["b"]])),
             "nli": round(rows[i]["nli"], 4), "stage": int(stage[i])} for i in idx]


# ---------------------------------------------------------------------------
# Timing: fresh model runs (no cache) on a DEV sample, CPU
# ---------------------------------------------------------------------------

def timing(dev, n: int = 256) -> dict:
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    from experiment import MODELS  # negation_direction (on sys.path via polarity's run)

    pairs = [(r["a"], r["b"]) for r in random.Random(C.SEED).sample(dev, n)]
    warm = pairs[:16]

    def clock(fn, items, batch):
        fn(warm[:batch])
        start = time.perf_counter()
        for i in range(0, len(items), batch):
            fn(items[i:i + batch])
        return 1000 * (time.perf_counter() - start) / len(items)

    tok = AutoTokenizer.from_pretrained(MODELS["mpnet"])
    emb = AutoModel.from_pretrained(MODELS["mpnet"]).eval()

    def gate(chunk):   # embed both sides, mean-pool, cosine
        enc = tok([s for p in chunk for s in p], padding=True, truncation=True, max_length=256,
                  return_tensors="pt")
        with torch.no_grad():
            h = emb(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1)
        return (v[0::2] * v[1::2]).sum(1)

    def cancel(chunk):   # spaCy parse + cue extraction of both sides, then the parity check
        an = P.analyse_many([s for p in chunk for s in p])
        return [C.cancellation_detected(an[a], an[b]) for a, b in chunk]

    nli = prun.detect.NLI()
    nli._load()

    def stage3(chunk):
        enc = nli.tokenizer([a for a, _ in chunk], [b for _, b in chunk], padding=True,
                            truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            return nli.model(**enc).logits.softmax(1)

    return {"n_pairs": n, "split": "dev", "device": "cpu", "torch_threads": torch.get_num_threads(),
            "ms_per_pair": {"gate": round(clock(gate, pairs, 16), 2),
                            "veto": round(clock(cancel, pairs, 16), 2),
                            "nli": round(clock(stage3, pairs, 16), 2)}}


def wall_clock(block: dict, ms: dict, embeddings_cached: bool, use_veto: bool) -> dict:
    """Cascade cost vs NLI on every pair, from measured per-pair costs.

    embeddings_cached: the sentence embeddings already exist (e.g. a retrieval
    index), so the gate is a dot product and costs ~0. Otherwise the gate pays
    for embedding both sentences, which on CPU costs about as much as NLI.
    """
    n, s = block["n"], block["stage_counts"]
    passed_gate = n - s.get("1", 0)
    gate_ms = 0.0 if embeddings_cached else ms["gate"]
    cascade = n * gate_ms + use_veto * passed_gate * ms["veto"] + s.get("3", 0) * ms["nli"]
    nli_all = n * ms["nli"]
    return {"nli_everything_s": round(nli_all / 1000, 2), "cascade_s": round(cascade / 1000, 2),
            "saved_s": round((nli_all - cascade) / 1000, 2),
            "saved_fraction": round(1 - cascade / nli_all, 4),
            "nli_calls_avoided": n - s.get("3", 0)}


def category(r) -> str:
    """dataset|kind|detail|label for the datasets table; detail = neg_type, or family for ours."""
    detail = r["family"] if r["dataset"] == "ours" else r.get("neg_type")
    return f"{r['dataset']}|{r['kind']}|{detail}|{r['label']}"


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()

    splits, analyses = C.load()
    cfg = configure(splits, analyses)
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res["seed"] = C.SEED
    res["rows"] = ROW_TEXT
    res["config"] = {k: v for k, v in cfg.items() if not k.startswith("_")}
    res["data"] = {name: {"n": len(rows), "labels": dict(Counter(str(r["label"]) for r in rows)),
                          "kinds": dict(Counter(map(category, rows)))}
                   for name, rows in splits.items()}
    res["examples_by_kind"] = {}
    for name, rows in splits.items():
        seen = {}
        for r in rows:
            seen.setdefault(category(r), (r["a"], r["b"]))
        res["examples_by_kind"][name] = seen
    res["dev"] = run_split(splits["dev"], cfg, analyses)
    res["validation"] = run_split(splits["validation"], cfg, analyses)
    chosen = max(CANDIDATES, key=lambda c: res["validation"][c]["accuracy"])   # first max = simplest
    res["chosen_by_validation"] = {"config": chosen,
                                   "criterion": "validation accuracy; a tie goes to the simpler config",
                                   "validation_accuracy": {c: res["validation"][c]["accuracy"]
                                                           for c in CANDIDATES}}
    if args.timing:
        res["timing"] = timing(splits["dev"])

    if args.final:
        log = json.loads(TEST_LOG.read_text()) if TEST_LOG.exists() else {"count": 0, "runs": []}
        assert log["count"] == 0, f"test already scored {log['count']} time(s); refusing"
        res["test"] = run_split(splits["test"], cfg, analyses)
        rng = random.Random(C.SEED)
        res["errors"] = {"false_positives": examples(splits["test"], analyses, cfg, "fp", 10, rng),
                         "false_negatives": examples(splits["test"], analyses, cfg, "fn", 10, rng)}
        log["count"] += 1
        log["runs"].append({"config": chosen, "test_accuracy": res["test"][chosen]["accuracy"],
                            "frozen": res["config"]})
        TEST_LOG.write_text(json.dumps(log, indent=2, default=float))
        res["test_scorings"] = log["count"]

    if "timing" in res:
        res["wall_clock"] = {split: {f"{c}|{scenario}": wall_clock(res[split][c], res["timing"]["ms_per_pair"],
                                                                  scenario == "embeddings_cached", c == "full")
                                     for c in ("nli_gate", "full")
                                     for scenario in ("embeddings_fresh", "embeddings_cached")}
                             for split in ("dev", "validation", "test") if split in res}
    RESULTS.write_text(json.dumps(res, indent=2, default=float))
    for split in ("dev", "validation", "test"):
        if split in res:
            print(split, {r: (res[split][r]["accuracy"], res[split][r]["auc"]) for r in ROWS})
    print("config:", {k: v for k, v in res["config"].items() if k != "gate_fits"}, "chosen:", chosen)


if __name__ == "__main__":
    main()
