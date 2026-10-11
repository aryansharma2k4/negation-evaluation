"""Generated negations of a teacher sentence T, cached, with their mpnet embeddings (Phase 3).

N(T) = every stage-1 generator variant whose negation parity differs from T's own cue parity (the
refsets labelling rule of nli2 step 5.1: a K1 cancellation or a double negation is not a negation),
plus, when T is itself negative, the Phase 1 affirmations of T with polarity_flip = True.

    ./venv/bin/python scripts/f1/variants.py     # precompute for every A sentence in clean DEV,
                                                      # VALIDATION, TEST and K1
Caches: .cache/f1/variants.pkl  {T: [(variant, family), ...]},  .cache/f1/mpnet.pkl {sentence: vector}.
"""

from __future__ import annotations

import pickle
import sys
import time
from functools import lru_cache

import numpy as np

from core import CACHE, REPO

import common as N  # noqa: F401  (nli2: REPO on sys.path)

sys.path.insert(0, str(REPO / "scripts" / "negation_direction"))

from negation.classify import classify_input  # noqa: E402
from negation.driver import generate_all, parse_bases  # noqa: E402
from negation.generators.n_affirmation import affirm_doc  # noqa: E402

VPATH = CACHE / "variants.pkl"
EPATH = CACHE / "mpnet.pkl"
_v: dict | None = None


def _load() -> dict:
    global _v
    if _v is None:
        _v = pickle.loads(VPATH.read_bytes()) if VPATH.exists() else {}
    return _v


def negations_of(sentence: str) -> list[tuple[str, str]]:
    """[(variant text, family)] for one T, uncached."""
    doc = parse_bases([sentence])[0]
    parity = classify_input(doc).existing_count % 2
    out = []
    try:
        for v in generate_all([sentence], stages=(1,)):
            if v.net_negation % 2 != parity:
                out.append((v.variant, v.family))
    except AssertionError:          # a generator's depth contract fired on this base (counted in reports)
        out.append(("", "GENERATOR_ERROR"))
    if parity:
        out += [(a.text, "N_affirmation" + ("_approx" if a.approximate else "")) for a in affirm_doc(doc) if a.polarity_flip]
    seen, uniq = set(), []
    for t, f in out:
        if t not in seen and t != sentence:
            seen.add(t)
            uniq.append((t, f))
    return uniq


def negations(sentences, save_every: int = 200) -> dict[str, list[tuple[str, str]]]:
    cache = _load()
    todo = [s for s in dict.fromkeys(sentences) if s not in cache]
    t0 = time.time()
    for i, s in enumerate(todo):
        cache[s] = negations_of(s)
        if (i + 1) % save_every == 0:
            CACHE.mkdir(exist_ok=True)
            VPATH.write_bytes(pickle.dumps(cache))
            print(f"  variants {i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    if len(todo) >= BATCH:
        CACHE.mkdir(exist_ok=True)
        VPATH.write_bytes(pickle.dumps(cache))
    return {s: [x for x in cache[s] if x[1] != "GENERATOR_ERROR"] for s in sentences}


_emb: dict | None = None
BATCH = 32     # below this many new items, keep them in memory only (interactive use: no pickle rewrite)


@lru_cache(maxsize=1)
def _mpnet():
    import torch
    from transformers import AutoModel, AutoTokenizer
    from experiment import MODELS
    return AutoTokenizer.from_pretrained(MODELS["mpnet"]), AutoModel.from_pretrained(MODELS["mpnet"]).eval(), torch


def _encode(sentences: list[str]) -> dict[str, np.ndarray]:
    """negation_direction's embed(), with the model loaded once per process."""
    tok, model, torch = _mpnet()
    out = {}
    for i in range(0, len(sentences), 32):
        chunk = sentences[i:i + 32]
        enc = tok(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9), p=2, dim=1)
        out.update({s: v.numpy().astype(np.float32) for s, v in zip(chunk, pooled)})
    return out


def embed(sentences) -> dict[str, np.ndarray]:
    """mpnet (all-mpnet-base-v2, revision e8c3b32, as in reports/07-honest-baseline.md), L2-normalised. Reads the
    existing nli2 / negation_direction / polarity / f1 caches once per process."""
    global _emb
    if _emb is None:
        _emb = {}
        for path in (REPO / ".cache/negation_direction/mpnet.pkl", REPO / ".cache/polarity/mpnet_ours.pkl",
                     REPO / ".cache/nli2/mpnet.pkl", EPATH):
            if path.exists():
                _emb.update(pickle.loads(path.read_bytes()))
    sents = [s for s in dict.fromkeys(sentences) if s]
    missing = [s for s in sents if s not in _emb]
    if missing:
        new = _encode(missing)
        _emb.update(new)
        if len(missing) >= BATCH:
            own = pickle.loads(EPATH.read_bytes()) if EPATH.exists() else {}
            own.update(new)
            CACHE.mkdir(exist_ok=True)
            EPATH.write_bytes(pickle.dumps(own))
    return {s: _emb[s] for s in sents}


def teacher_sentences() -> list[str]:
    import phase4 as P4
    import pipeline as PL
    _, _, sl = PL.load()
    rows = sl["dev_clean"] + sl["validation"] + sl["test"] + P4.k1_rows()
    return list(dict.fromkeys(r["a"] for r in rows))


def main() -> None:
    ts = teacher_sentences()
    print(len(ts), "teacher sentences", flush=True)
    neg = negations(ts)
    flat = [v for vs in neg.values() for v, _ in vs]
    print(len(flat), "variants; embedding", flush=True)
    embed(ts + flat)
    print("done", flush=True)


if __name__ == "__main__":
    main()
