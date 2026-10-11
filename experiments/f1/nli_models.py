"""Phase 2.1: score every pair with several NLI cross-encoders, revision-pinned, disk-cached.

    ./venv/bin/python experiments/f1/nli_models.py --time            # ms/pair per model, 64 DEV pairs
    ./venv/bin/python experiments/f1/nli_models.py --score KEY ...   # score all 5,854 rows

Probabilities are softmax over each model's own label set, mapped by config.id2label, so label
order differences between models (bart: c/n/e, MoritzLaurer: e/n/c, cross-encoder: c/e/n) are
handled. Premise = A, hypothesis = B, as in every earlier NLI number. Cache: f1/cache/nli_<key>.pkl.
"""

from __future__ import annotations

import argparse
import pickle
import time

import numpy as np

from core import CACHE, save

MODELS = {   # key: (hub id, pinned revision, parameters)
    "deberta_base": ("cross-encoder/nli-deberta-v3-base", "6c749ce3425cd33b46d187e45b92bbf96ee12ec7", "184M"),
    "roberta_base": ("cross-encoder/nli-roberta-base", "1be0567456f0543475805e758725f151f283705a", "125M"),
    "distilroberta": ("cross-encoder/nli-distilroberta-base", "b14d131f9d32668a5e6a982729b57ff6ed5dfcbd", "82M"),
    "ml_deberta_base_anli": ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "6f5cf0a2b59cabb106aca4c287eed12e357e90eb", "184M"),
    "bart_large": ("facebook/bart-large-mnli", "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce", "407M"),
    "deberta_small": ("cross-encoder/nli-deberta-v3-small", "fa2804872c3b4bd748f38c0185cc85775361e735", "142M"),
    # additions beyond the brief's table: larger models trained on adversarial NLI
    "ml_deberta_large_wanli": ("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                               "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7", "435M"),
    "roberta_large_anli": ("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
                           "5b605abab9b75bc87ab66cfc049ef58d9d64b8ed", "355M"),
}


class Scorer:
    def __init__(self, key: str):
        self.key = key
        self.path = CACHE / f"nli_{key}.pkl"
        self.cache = pickle.loads(self.path.read_bytes()) if self.path.exists() else {}
        if key == "deberta_base":   # the reference: start from the exact scores the baseline features use
            from core import REPO
            for p in (REPO / "experiments/negation_detection/cache/nli.pkl",
                      REPO / "experiments/polarity/cache/nli_with_ours.pkl", REPO / "experiments/nli2/cache/nli.pkl"):
                if p.exists():
                    self.cache = {**pickle.loads(p.read_bytes()), **self.cache}
        self.model = None

    def _load(self):
        if self.model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            torch.set_num_threads(8)
            name, rev, _ = MODELS[self.key]
            self.torch = torch
            local = CACHE / "models" / self.key          # fetch_models.py: same revision, sha256-verified
            src, kw = (str(local), {}) if (local / "manifest.json").exists() else (name, {"revision": rev})
            self.tok = AutoTokenizer.from_pretrained(src, **kw)
            # float32 for every model: transformers 5 keeps a checkpoint's stored dtype, and the fp16
            # MoritzLaurer checkpoint ran 5x slower on CPU in fp16 (238 vs ~45 ms/pair)
            self.model = AutoModelForSequenceClassification.from_pretrained(src, dtype=torch.float32, **kw).eval()
            labels = [self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)]
            self.labels = [{"contradiction": "contradiction", "entailment": "entailment", "neutral": "neutral"}
                           .get(l, l) for l in labels]
            assert set(self.labels) == {"contradiction", "entailment", "neutral"}, labels

    def _run(self, chunk):
        enc = self.tok([p for p, _ in chunk], [h for _, h in chunk], padding=True, truncation=True,
                       max_length=256, return_tensors="pt")
        with self.torch.no_grad():
            return self.torch.softmax(self.model(**enc).logits, dim=1).numpy()

    def score(self, pairs, batch: int = 16, save_every: int = 50) -> dict:
        todo = [p for p in dict.fromkeys(pairs) if p not in self.cache]
        if todo:
            self._load()
            start = time.time()
            for n, i in enumerate(range(0, len(todo), batch)):
                chunk = todo[i:i + batch]
                for pair, row in zip(chunk, self._run(chunk)):
                    self.cache[pair] = dict(zip(self.labels, map(float, row)))
                if n % save_every == 0:
                    print(f"  {self.key} {i + len(chunk)}/{len(todo)} {(time.time() - start) / (i + len(chunk)) * 1000:.0f} ms/pair",
                          flush=True)
                    self._flush()
            self._flush()
        return {p: self.cache[p] for p in pairs}

    def _flush(self):
        CACHE.mkdir(exist_ok=True)
        self.path.write_bytes(pickle.dumps(self.cache))

    def time_per_pair(self, pairs, batch: int = 16) -> float:
        self._load()
        self._run(pairs[:batch])                                   # warm-up
        t = time.perf_counter()
        for i in range(0, len(pairs), batch):
            self._run(pairs[i:i + batch])
        return (time.perf_counter() - t) / len(pairs) * 1000


def all_pairs() -> dict[str, list[tuple[str, str]]]:
    import phase4 as P4
    import pipeline as PL
    splits, _, sl = PL.load()
    sets = {"dev_clean": sl["dev_clean"], "validation": sl["validation"], "test": sl["test"], "k1": P4.k1_rows()}
    return {k: [(r["a"], r["b"]) for r in v] for k, v in sets.items()}


def probs(key: str, pairs) -> np.ndarray:
    """(n, 3) array of P(contradiction), P(entailment), P(neutral)."""
    got = Scorer(key).score(pairs)
    return np.array([[got[p]["contradiction"], got[p]["entailment"], got[p]["neutral"]] for p in pairs])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", nargs="*", help="model keys to time in isolation (results merged into 2_1_timing.json)")
    ap.add_argument("--score", nargs="*")
    args = ap.parse_args()
    pairs = all_pairs()
    flat = list(dict.fromkeys(p for v in pairs.values() for p in v))
    if args.time:
        from core import RESULTS
        prev = RESULTS / "2_1_timing.json"
        out = __import__("json").loads(prev.read_text())["models"] if prev.exists() else {}
        for key in args.time:
            name, rev, size = MODELS[key]
            ms = Scorer(key).time_per_pair(pairs["dev_clean"][:64])
            out[key] = {"hub_id": name, "revision": rev, "parameters": size, "ms_per_pair": round(ms, 1),
                        "projected_minutes_all_rows": round(ms * len(flat) / 60000, 1), "rows": len(flat)}
            print(key, out[key], flush=True)
        save("2_1_timing.json", {"setup": "CPU (Ryzen 7 7730U), 8 torch threads, batch 16, max_length 256, 64 clean-DEV pairs "
                                          "after one warm-up batch, nothing else running", "models": out})
    for key in args.score or []:
        t = time.time()
        Scorer(key).score(flat)
        print(key, "done", round(time.time() - t), "s", flush=True)


if __name__ == "__main__":
    main()
