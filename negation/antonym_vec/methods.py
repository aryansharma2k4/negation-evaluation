"""The six approaches, behind one interface.

Each method answers "given a word, rank candidate opposites". Five do it by
predicting a *vector* and taking its nearest neighbours; one (WordNet lookup)
returns words directly and exists to establish the ceiling the others have to
beat.

Everything is numpy. Ridge and reflection are closed-form or a few hundred
gradient steps; the MLP is a hand-written two-layer net with Adam. No deep
learning framework is needed for a 300-dimensional map fitted on ~4,000 pairs,
and that ratio -- 90,000 parameters in ``W`` from 3,976 examples -- is itself
the central difficulty, not an implementation detail.

One structural fact governs how :class:`CounterFitting` is read. Counter-fitting
moves the vectors of words that appear in its constraints. The split is
lemma-disjoint, so no test lemma appears in any training constraint, so their
vectors are moved only by the space-preservation term that pulls every vector
back toward where it started. It therefore *cannot* improve generalisation to
unseen words, by construction rather than by accident. Both the honest
(train-constraint) and the transductive (test-constraints-included) variants are
implemented so the difference between "cannot generalise" and "is broken" is
visible rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np

from .data import Pair


def l2_normalise(matrix: np.ndarray, axis: int = -1) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=axis, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


@dataclass
class Candidates:
    """The pool a predicted vector is matched against.

    Fixed once per experiment so every method is scored against the same
    alternatives. Vectors are stored L2-normalised, which makes ranking by
    cosine a single matrix product.
    """

    words: tuple[str, ...]
    matrix: np.ndarray
    index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.matrix = l2_normalise(self.matrix)
        self.index = {w: i for i, w in enumerate(self.words)}

    def rank(self, vector: np.ndarray, exclude: Sequence[str] = (), top: int = 10):
        """``(word, cosine)`` for the ``top`` nearest candidates."""
        scores = self.matrix @ (vector / max(float(np.linalg.norm(vector)), 1e-8))
        for word in exclude:
            position = self.index.get(word)
            if position is not None:
                scores[position] = -np.inf
        if top >= len(scores):
            order = np.argsort(-scores)
        else:
            partial = np.argpartition(-scores, top)[:top]
            order = partial[np.argsort(-scores[partial])]
        return [(self.words[i], float(scores[i])) for i in order[:top]]

    def rank_of(self, vector: np.ndarray, target: str, exclude: Sequence[str] = ()) -> Optional[int]:
        """1-based rank of ``target``, or ``None`` if it is not a candidate."""
        position = self.index.get(target)
        if position is None:
            return None
        scores = self.matrix @ (vector / max(float(np.linalg.norm(vector)), 1e-8))
        for word in exclude:
            other = self.index.get(word)
            if other is not None:
                scores[other] = -np.inf
        return int((scores > scores[position]).sum()) + 1


class Method(Protocol):
    """Rank candidate opposites for a word."""

    name: str

    def fit(self, pairs: Sequence[Pair], source) -> None: ...

    def predict(self, word: str, pos: str, source) -> Optional[np.ndarray]:
        """Predicted antonym vector, or ``None`` if the method cannot fire."""

    def suggest(self, word: str, pos: str, source, candidates: Candidates, top: int = 10):
        """``(word, score)`` ranked. Default: nearest neighbours of the prediction."""


class VectorMethod:
    """Shared machinery for the five methods that predict a vector."""

    name = ""

    def fit(self, pairs: Sequence[Pair], source) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def predict(self, word: str, pos: str, source) -> Optional[np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    def suggest(self, word: str, pos: str, source, candidates: Candidates, top: int = 10):
        vector = self.predict(word, pos, source)
        if vector is None:
            return []
        return candidates.rank(vector, exclude=(word,), top=top)


def training_matrices(
    pairs: Sequence[Pair], source, *, normalise: bool = True
) -> tuple[np.ndarray, np.ndarray, list[Pair]]:
    """``(X, Y, kept)`` for the pairs whose words are both in vocabulary."""
    xs, ys, kept = [], [], []
    for pair in pairs:
        a = source.vector(pair.word, pair.pos)
        b = source.vector(pair.antonym, pair.pos)
        if a is None or b is None:
            continue
        xs.append(a)
        ys.append(b)
        kept.append(pair)
    if not xs:
        return np.zeros((0, source.dim)), np.zeros((0, source.dim)), []
    X, Y = np.stack(xs), np.stack(ys)
    if normalise:
        X, Y = l2_normalise(X), l2_normalise(Y)
    return X.astype(np.float64), Y.astype(np.float64), kept


# ---------------------------------------------------------------------------
# 1. baseline_negate
# ---------------------------------------------------------------------------


class BaselineNegate(VectorMethod):
    """``f(v) = -v``. The baseline that is expected to fail.

    Included because it is the obvious thing to try and because its failure is
    the premise of the whole exercise: distributional vectors occupy a narrow
    cone, so ``-v`` points into a region of the space that is empty of related
    words rather than at the opposite meaning.
    """

    name = "baseline_negate"

    def fit(self, pairs, source) -> None:
        return None

    def predict(self, word, pos, source):
        vector = source.vector(word, pos)
        return None if vector is None else -np.asarray(vector, dtype=np.float64)


# ---------------------------------------------------------------------------
# 2. baseline_wordnet
# ---------------------------------------------------------------------------


class BaselineWordNet:
    """Direct WordNet lookup. The coverage ceiling, and the accuracy ceiling.

    On a test set *drawn from* WordNet its precision is trivially near 1.0, and
    that number means nothing on its own -- the informative measurement is its
    **coverage over words the generator actually meets**, which
    :mod:`.evaluate` measures separately against a general vocabulary sample.
    """

    name = "baseline_wordnet"

    def fit(self, pairs, source) -> None:
        return None

    def predict(self, word, pos, source):
        return None  # returns words, not a vector

    def suggest(self, word, pos, source, candidates: Candidates, top: int = 10):
        from nltk.corpus import wordnet as wn

        wn_pos = {"a": wn.ADJ, "v": wn.VERB, "r": wn.ADV, "n": wn.NOUN}.get(pos)
        found: list[tuple[str, float]] = []
        seen: set[str] = set()
        try:
            synsets = wn.synsets(word, pos=wn_pos)
        except Exception:  # pragma: no cover - corpus edge cases
            return []
        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.name().lower() != word.lower():
                    continue
                for antonym in lemma.antonyms():
                    name = antonym.name().lower()
                    if "_" in name or name in seen:
                        continue
                    seen.add(name)
                    found.append((name, 1.0))
        return found[:top]


# ---------------------------------------------------------------------------
# 3. linear_map
# ---------------------------------------------------------------------------


@dataclass
class LinearMap(VectorMethod):
    """Ridge regression: ``W = argmin ||XW - Y||^2 + lambda ||W||^2``.

    Closed form, so there is nothing to tune but ``alpha`` -- which is chosen on
    a split of the *training* pairs, never on test.
    """

    name: str = "linear_map"
    alpha: float = 1.0
    W: Optional[np.ndarray] = None

    def fit(self, pairs, source) -> None:
        X, Y, _ = training_matrices(pairs, source)
        if len(X) == 0:
            self.W = None
            return
        gram = X.T @ X + self.alpha * np.eye(X.shape[1])
        self.W = np.linalg.solve(gram, X.T @ Y)

    def predict(self, word, pos, source):
        if self.W is None:
            return None
        vector = source.vector(word, pos)
        if vector is None:
            return None
        normalised = np.asarray(vector, dtype=np.float64)
        normalised = normalised / max(float(np.linalg.norm(normalised)), 1e-8)
        return normalised @ self.W


def tune_ridge_alpha(
    pairs, source, alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0), seed: int = 0
) -> float:
    """Pick ``alpha`` by held-out cosine on a split of the training pairs.

    Kept strictly inside training: tuning on test is how a comparison like this
    quietly stops being a comparison.
    """
    X, Y, _ = training_matrices(pairs, source)
    if len(X) < 20:
        return 1.0
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    cut = int(len(X) * 0.8)
    tr, va = order[:cut], order[cut:]

    best, best_score = 1.0, -np.inf
    for alpha in alphas:
        gram = X[tr].T @ X[tr] + alpha * np.eye(X.shape[1])
        W = np.linalg.solve(gram, X[tr].T @ Y[tr])
        predicted = l2_normalise(X[va] @ W)
        score = float((predicted * l2_normalise(Y[va])).sum(axis=1).mean())
        if score > best_score:
            best, best_score = alpha, score
    return best


# ---------------------------------------------------------------------------
# 4. mlp_map
# ---------------------------------------------------------------------------


@dataclass
class MLPMap(VectorMethod):
    """Two-layer MLP on the same objective as :class:`LinearMap`.

    Hand-written with Adam because a 300->h->300 net on 4,000 examples does not
    justify a framework dependency. Early stopping on a split of training, so
    the capacity advantage over ridge is not spent on memorising.
    """

    name: str = "mlp_map"
    hidden: int = 512
    epochs: int = 400
    batch: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 0
    params: Optional[dict] = None

    def _forward(self, X: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
        hidden_pre = X @ params["W1"] + params["b1"]
        hidden = np.tanh(hidden_pre)
        return hidden, hidden @ params["W2"] + params["b2"]

    def fit(self, pairs, source) -> None:
        X, Y, _ = training_matrices(pairs, source)
        if len(X) < 20:
            self.params = None
            return

        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(X))
        cut = int(len(X) * 0.9)
        train_idx, val_idx = order[:cut], order[cut:]
        dim = X.shape[1]

        params = {
            "W1": rng.normal(0, (2.0 / dim) ** 0.5, (dim, self.hidden)),
            "b1": np.zeros(self.hidden),
            "W2": rng.normal(0, (2.0 / self.hidden) ** 0.5, (self.hidden, dim)),
            "b2": np.zeros(dim),
        }
        moment1 = {k: np.zeros_like(v) for k, v in params.items()}
        moment2 = {k: np.zeros_like(v) for k, v in params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        best_params, best_loss, patience, step = None, np.inf, 0, 0
        for epoch in range(self.epochs):
            shuffled = rng.permutation(train_idx)
            for start in range(0, len(shuffled), self.batch):
                batch = shuffled[start : start + self.batch]
                xb, yb = X[batch], Y[batch]
                hidden, output = self._forward(xb, params)

                d_out = 2.0 * (output - yb) / len(batch)
                grads = {
                    "W2": hidden.T @ d_out + self.weight_decay * params["W2"],
                    "b2": d_out.sum(axis=0),
                }
                d_hidden = (d_out @ params["W2"].T) * (1.0 - hidden**2)
                grads["W1"] = xb.T @ d_hidden + self.weight_decay * params["W1"]
                grads["b1"] = d_hidden.sum(axis=0)

                step += 1
                for key in params:
                    moment1[key] = beta1 * moment1[key] + (1 - beta1) * grads[key]
                    moment2[key] = beta2 * moment2[key] + (1 - beta2) * grads[key] ** 2
                    m_hat = moment1[key] / (1 - beta1**step)
                    v_hat = moment2[key] / (1 - beta2**step)
                    params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

            _, val_out = self._forward(X[val_idx], params)
            loss = float(((val_out - Y[val_idx]) ** 2).mean())
            if loss < best_loss - 1e-6:
                best_loss, best_params, patience = loss, {k: v.copy() for k, v in params.items()}, 0
            else:
                patience += 1
                if patience >= 30:
                    break
        self.params = best_params or params

    def predict(self, word, pos, source):
        if self.params is None:
            return None
        vector = source.vector(word, pos)
        if vector is None:
            return None
        normalised = np.asarray(vector, dtype=np.float64)
        normalised = normalised / max(float(np.linalg.norm(normalised)), 1e-8)
        _, output = self._forward(normalised[None, :], self.params)
        return output[0]


# ---------------------------------------------------------------------------
# 5. reflection
# ---------------------------------------------------------------------------


@dataclass
class Reflection(VectorMethod):
    """One hyperplane per POS; the antonym is the reflection across it.

    ``f(v) = v - 2 (v.n - b) n`` with ``||n|| = 1``. Two properties make this
    worth trying where an unconstrained map is not: it is an *involution*, so
    the antonym of the antonym is the original word, which is what antonymy
    actually does; and it has ``d + 1`` parameters per POS rather than ``d^2``,
    which matters a great deal at ~4,000 training pairs.
    """

    name: str = "reflection"
    steps: int = 2000
    lr: float = 0.05
    seed: int = 0
    planes: dict = field(default_factory=dict)

    def _fit_one(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
        rng = np.random.default_rng(self.seed)
        normal = rng.normal(size=X.shape[1])
        normal /= np.linalg.norm(normal)
        offset = 0.0

        m1n = np.zeros_like(normal)
        m2n = np.zeros_like(normal)
        m1b = m2b = 0.0
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for step in range(1, self.steps + 1):
            projection = X @ normal - offset                      # (N,)
            predicted = X - 2.0 * projection[:, None] * normal[None, :]
            residual = predicted - Y                              # (N,d)

            # d/dn of  X - 2 (Xn - b) n   contracted with the residual.
            grad_normal = (
                -2.0 * (residual * projection[:, None]).sum(axis=0)
                - 2.0 * ((residual @ normal)[:, None] * X).sum(axis=0)
            ) * (2.0 / len(X))
            grad_offset = (2.0 * 2.0 / len(X)) * float((residual @ normal).sum())

            m1n = beta1 * m1n + (1 - beta1) * grad_normal
            m2n = beta2 * m2n + (1 - beta2) * grad_normal**2
            normal -= self.lr * (m1n / (1 - beta1**step)) / (
                np.sqrt(m2n / (1 - beta2**step)) + eps
            )
            normal /= max(float(np.linalg.norm(normal)), 1e-8)

            m1b = beta1 * m1b + (1 - beta1) * grad_offset
            m2b = beta2 * m2b + (1 - beta2) * grad_offset**2
            offset -= self.lr * (m1b / (1 - beta1**step)) / (
                (m2b / (1 - beta2**step)) ** 0.5 + eps
            )
        return normal, float(offset)

    def fit(self, pairs, source) -> None:
        self.planes = {}
        by_pos: dict[str, list[Pair]] = {}
        for pair in pairs:
            by_pos.setdefault(pair.pos, []).append(pair)
        for pos, group in by_pos.items():
            X, Y, _ = training_matrices(group, source)
            if len(X) >= 20:
                self.planes[pos] = self._fit_one(X, Y)

    def predict(self, word, pos, source):
        plane = self.planes.get(pos)
        if plane is None:
            return None
        vector = source.vector(word, pos)
        if vector is None:
            return None
        normal, offset = plane
        v = np.asarray(vector, dtype=np.float64)
        v = v / max(float(np.linalg.norm(v)), 1e-8)
        return v - 2.0 * (float(v @ normal) - offset) * normal


# ---------------------------------------------------------------------------
# 6. counterfit_style
# ---------------------------------------------------------------------------


@dataclass
class CounterFitting(VectorMethod):
    """Mrkšić-style retrofitting, then nearest-neighbour lookup.

    Three terms, optimised by gradient descent over the vectors themselves:
    *antonym repel* pushes constrained antonyms at least ``delta`` apart,
    *synonym attract* pulls constrained synonyms within ``gamma``, and *vector
    space preservation* holds everything near where it started.

    ``transductive`` decides which constraints are used and is the whole point
    of the class. With ``False`` (the default and the only honest setting) only
    training pairs supply constraints, so test lemmas -- which by construction
    appear in none -- keep their original vectors and the method can only score
    whatever the underlying retrieval scores. With ``True`` the test antonym
    constraints are included as well, which is not a generalisation test but
    does show what the method achieves when it has seen the pair, separating
    "cannot generalise" from "does not work".
    """

    name: str = "counterfit_style"
    delta: float = 1.0
    gamma: float = 0.0
    rho: float = 0.2
    steps: int = 20
    lr: float = 0.1
    transductive: bool = False
    _vectors: Optional[np.ndarray] = None
    _index: Optional[dict] = None
    _space: Optional[Candidates] = None

    def fit_space(
        self,
        words: Sequence[str],
        matrix: np.ndarray,
        antonym_pairs: Sequence[tuple[str, str]],
        synonym_pairs: Sequence[tuple[str, str]],
    ) -> np.ndarray:
        """Return a counter-fitted copy of ``matrix``."""
        index = {w: i for i, w in enumerate(words)}
        original = l2_normalise(matrix.astype(np.float64))
        vectors = original.copy()

        ant = np.array(
            [(index[a], index[b]) for a, b in antonym_pairs if a in index and b in index],
            dtype=np.int64,
        ).reshape(-1, 2)
        syn = np.array(
            [(index[a], index[b]) for a, b in synonym_pairs if a in index and b in index],
            dtype=np.int64,
        ).reshape(-1, 2)

        for _ in range(self.steps):
            gradient = np.zeros_like(vectors)

            if len(ant):
                left, right = vectors[ant[:, 0]], vectors[ant[:, 1]]
                similarity = (left * right).sum(axis=1)
                violated = similarity > (1.0 - self.delta)
                if violated.any():
                    np.add.at(gradient, ant[violated, 0], right[violated])
                    np.add.at(gradient, ant[violated, 1], left[violated])

            if len(syn):
                left, right = vectors[syn[:, 0]], vectors[syn[:, 1]]
                similarity = (left * right).sum(axis=1)
                violated = similarity < (1.0 - self.gamma)
                if violated.any():
                    np.add.at(gradient, syn[violated, 0], -right[violated])
                    np.add.at(gradient, syn[violated, 1], -left[violated])

            gradient += self.rho * (vectors - original)
            vectors -= self.lr * gradient
            vectors = l2_normalise(vectors)
        return vectors

    def fit(self, pairs, source) -> None:  # pragma: no cover - driven by the runner
        raise NotImplementedError(
            "CounterFitting needs the candidate pool; use fit_space via the runner"
        )

    def install(self, words: Sequence[str], vectors: np.ndarray) -> None:
        """Adopt the counter-fitted space for *both* the query and the pool.

        Retrieval has to happen inside the modified space.  Ranking a
        counter-fitted query against the original vectors would compare points
        from two different geometries and measure neither -- which is the whole
        content of the method, since counter-fitting changes the space rather
        than the mapping.
        """
        self._vectors = vectors
        self._index = {w: i for i, w in enumerate(words)}
        self._space = Candidates(tuple(words), vectors.copy())

    def suggest(self, word, pos, source, candidates: Candidates, top: int = 10):
        vector = self.predict(word, pos, source)
        if vector is None or self._space is None:
            return []
        return self._space.rank(vector, exclude=(word,), top=top)

    def predict(self, word, pos, source):
        if self._vectors is None or self._index is None:
            return None
        position = self._index.get(word.lower())
        if position is None:
            return None
        # In a counter-fitted space the word's own neighbourhood has had its
        # antonyms pushed away, so the retrieval target is the *negated*
        # direction, as for the naive baseline -- the space, not the map, is
        # what this method changes.
        return -self._vectors[position]
