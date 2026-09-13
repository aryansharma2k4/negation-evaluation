#!/usr/bin/env python
"""Run the antonym-vector comparison and print the tables the report quotes.

    ./venv/bin/python scripts/antonym_experiment.py
    ./venv/bin/python scripts/antonym_experiment.py --source contextual
    ./venv/bin/python scripts/antonym_experiment.py --json results.json

Every method is fitted on the same training pairs, scored against the same
candidate pool, and reported on the same slices. Ridge's regularisation strength
is chosen on a split of *training*; nothing is selected on test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from negation.antonym_vec.data import POS_TAGS, build_dataset  # noqa: E402
from negation.antonym_vec.embeddings import (  # noqa: E402
    EmbeddingsNotFetched,
    load_contextual,
    load_static,
)
from negation.antonym_vec.evaluate import (  # noqa: E402
    aggregate,
    format_table,
    gold_antonyms,
    score_pairs,
    slice_pairs,
    synonym_index,
    vocabulary_coverage,
)
from negation.antonym_vec.methods import (  # noqa: E402
    BaselineNegate,
    BaselineWordNet,
    Candidates,
    CounterFitting,
    LinearMap,
    MLPMap,
    Reflection,
    tune_ridge_alpha,
)


def build_candidates(source, dataset, pool_size: int) -> Candidates:
    """The shared candidate pool: frequent vocabulary plus every dataset lemma.

    Ranking against a large pool is the honest setting. Restricting candidates
    to the few hundred test lemmas would raise every precision several-fold and
    measure nothing a deployed module could rely on.
    """
    lemmas = dataset.train_lemmas() | dataset.test_lemmas()
    vocabulary = list(source.vocabulary())
    words: list[str] = []
    seen: set[str] = set()
    for word in vocabulary[:pool_size]:
        if word not in seen:
            seen.add(word)
            words.append(word)
    for word in sorted(lemmas):
        if word not in seen and source.vector(word, "a") is not None:
            seen.add(word)
            words.append(word)
    matrix = np.stack([source.vector(w, "a") for w in words])
    return Candidates(tuple(words), matrix)


def in_vocabulary(dataset, source) -> tuple[list, list]:
    def keep(pairs):
        return [
            p
            for p in pairs
            if source.vector(p.word, p.pos) is not None
            and source.vector(p.antonym, p.pos) is not None
        ]

    return keep(dataset.train), keep(dataset.test)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default="static", choices=["static", "contextual"])
    parser.add_argument("--vectors", default="glove.6B.300d.txt",
                        help="static vector filename under data/embeddings")
    parser.add_argument("--model", default="bert-base-uncased",
                        help="transformer for --source contextual")
    parser.add_argument("--pool-size", type=int, default=50_000,
                        help="frequent-vocabulary candidates (default 50k)")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--examples", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    print("building dataset from WordNet ...")
    dataset = build_dataset(POS_TAGS, test_fraction=0.2, seed=args.seed)
    lemmas = dataset.train_lemmas() | dataset.test_lemmas()
    print(f"  train {len(dataset.train)} pairs / {len(dataset.train_lemmas())} lemmas")
    print(f"  test  {len(dataset.test)} pairs / {len(dataset.test_lemmas())} lemmas")
    overlap = dataset.train_lemmas() & dataset.test_lemmas()
    assert not overlap, f"lemma leakage: {sorted(overlap)[:5]}"

    print(f"loading {args.source} embeddings ...")
    try:
        if args.source == "static":
            source = load_static(args.vectors, restrict_to=lemmas,
                                 max_vocab=args.pool_size)
        else:
            source = load_contextual(args.model)
            source.set_vocabulary(sorted(lemmas))
    except EmbeddingsNotFetched as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    print(f"  source={source.name} dim={source.dim} vocab={len(source.vocabulary()):,}")

    train, test = in_vocabulary(dataset, source)
    dropped_train = len(dataset.train) - len(train)
    dropped_test = len(dataset.test) - len(test)
    print(f"  in-vocabulary: train {len(train)} (-{dropped_train}), "
          f"test {len(test)} (-{dropped_test})")
    if not test:
        print("no test pairs survive the vocabulary filter", file=sys.stderr)
        return 2

    candidates = build_candidates(source, dataset, args.pool_size)
    print(f"  candidate pool: {len(candidates.words):,} words")

    alpha = tune_ridge_alpha(train, source, seed=args.seed)
    print(f"  ridge alpha chosen on a training split: {alpha}")

    methods = [
        BaselineNegate(),
        BaselineWordNet(),
        LinearMap(alpha=alpha),
        MLPMap(seed=args.seed),
        Reflection(seed=args.seed),
    ]
    for method in methods:
        method.fit(train, source)

    # Counter-fitting rebuilds the space itself, so it is set up separately.
    train_ant = [(p.word, p.antonym) for p in train]
    train_syn = [
        (p.word, p.antonym)
        for p in dataset.synonyms
        if p.word in dataset.train_lemmas() and p.antonym in dataset.train_lemmas()
    ]
    print(f"  counter-fitting on {len(train_ant)} antonym / {len(train_syn)} synonym "
          f"constraints (training lemmas only)")
    honest = CounterFitting(transductive=False)
    fitted = honest.fit_space(candidates.words, candidates.matrix, train_ant, train_syn)
    honest.install(candidates.words, fitted)
    methods.append(honest)

    transductive = CounterFitting(transductive=True)
    transductive.name = "counterfit_transductive"
    test_ant = [(p.word, p.antonym) for p in test]
    fitted_t = transductive.fit_space(
        candidates.words, candidates.matrix, train_ant + test_ant, train_syn
    )
    transductive.install(candidates.words, fitted_t)
    methods.append(transductive)

    all_gold = gold_antonyms(dataset.train + dataset.test)
    synonyms = synonym_index(dataset)
    slices = slice_pairs(test)

    print("\n" + "=" * 110)
    print(f"ANTONYM PREDICTION -- {source.name}, {len(candidates.words):,} candidates")
    print("=" * 110)

    # Retrieve once per method, then fold into every slice: each query costs two
    # 50k-row dot products and belongs to three slices, so scoring per slice
    # would triple the work for identical numbers.
    scored: dict[str, list] = {}
    for method in methods:
        print(f"  scoring {method.name} ...", flush=True)
        scored[method.name] = score_pairs(
            method, test, source, candidates, all_gold=all_gold, synonyms=synonyms
        )

    collected = []
    for slice_name, pairs in slices.items():
        keep = {(p.word, p.pos) for p in pairs}
        results = [
            aggregate(
                method.name,
                [s for s in scored[method.name] if (s.pair.word, s.pair.pos) in keep],
                slice_name,
            )
            for method in methods
        ]
        collected.extend(results)
        print()
        print(format_table(results))

    # Coverage on general vocabulary -- the question the integration turns on.
    print("\n" + "=" * 110)
    print("COVERAGE ON GENERAL VOCABULARY (not the WordNet-derived test set)")
    print("=" * 110)
    sample = [w for w in list(source.vocabulary())[:5000] if w.isalpha()][:1000]
    print(f"  sample: {len(sample)} frequent words, queried as adjectives")
    for method in methods:
        rate = vocabulary_coverage(method, sample, "a", source, candidates)
        print(f"    {method.name:<26} fires on {rate:6.1%}")

    if args.examples:
        print("\n" + "=" * 110)
        print("EXAMPLE PREDICTIONS (test set)")
        print("=" * 110)
        for result in collected:
            if result.slice_name != "all":
                continue
            print(f"\n  {result.method}")
            for word, gold, got in result.examples[:6]:
                mark = "OK " if got and got[0] == gold else "   "
                print(f"    {mark}{word:<16} gold={gold:<16} got={got[:4]}")

    if args.json:
        payload = {
            "source": source.name,
            "dim": source.dim,
            "candidates": len(candidates.words),
            "ridge_alpha": alpha,
            "train_pairs": len(train),
            "test_pairs": len(test),
            "results": [r.as_dict() for r in collected],
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
