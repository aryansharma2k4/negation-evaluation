"""Generated negations of a teacher sentence T, cached, with their mpnet embeddings (Phase 3).

N(T) = every stage-1 generator variant whose negation parity differs from T's own cue parity (the
refsets labelling rule of nli2 step 5.1: a K1 cancellation or a double negation is not a negation),
plus, when T is itself negative, the Phase 1 affirmations of T with polarity_flip = True.

    ./venv/bin/python experiments/f1/variants.py     # precompute for every A sentence in clean DEV,
                                                      # VALIDATION, TEST and K1
Caches: f1/cache/variants.pkl  {T: [(variant, family), ...]},  f1/cache/mpnet.pkl {sentence: vector}.
"""

from __future__ import annotations

import pickle
import sys
import time

import numpy as np

from core import CACHE, REPO

import common as N  # noqa: F401  (nli2: REPO on sys.path)

sys.path.insert(0, str(REPO / "experiments" / "negation_direction"))

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
    if todo:
        CACHE.mkdir(exist_ok=True)
        VPATH.write_bytes(pickle.dumps(cache))
    return {s: [x for x in cache[s] if x[1] != "GENERATOR_ERROR"] for s in sentences}


def embed(sentences) -> dict[str, np.ndarray]:
    """mpnet (all-mpnet-base-v2, revision e8c3b32, as in Report_NLI2.md), L2-normalised; reads the
    existing nli2 / negation_direction / polarity caches first."""
    from experiment import embed as _embed
    shared = {}
    for path in (REPO / "experiments/negation_direction/cache/mpnet.pkl", REPO / "experiments/polarity/cache/mpnet_ours.pkl",
                 REPO / "experiments/nli2/cache/mpnet.pkl"):
        if path.exists():
            shared.update(pickle.loads(path.read_bytes()))
    sents = [s for s in dict.fromkeys(sentences) if s]
    missing = [s for s in sents if s not in shared]
    if missing:
        shared.update(_embed(missing, "mpnet", cache_path=EPATH))
    return {s: shared[s] for s in sents}


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
