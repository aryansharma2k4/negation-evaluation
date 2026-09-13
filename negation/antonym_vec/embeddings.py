"""Embedding sources: static vectors, and contextual ones from a template sentence.

Both satisfy the same small protocol, so every method in :mod:`.methods` and
every metric in :mod:`.evaluate` runs unchanged against either. Which one is in
use is a configuration choice, not a code path.

The distinction matters for the question being asked. A static vector is a
single point per word type, so *hot* has one neighbourhood and the experiment is
clean. A contextual model gives a different vector per occurrence, so a word type
has to be *made* into a point somehow -- here by embedding it in a neutral
template sentence and averaging the last four layers, which is the standard
recipe and the one the brief asks for. That construction is itself a confound
worth stating: the template contributes to the vector, and two words in the same
template are already more similar than two words in different sentences.

Nothing is vendored. ``fetch.py`` pulls the vector files into
``data/embeddings``; a missing file raises with instructions rather than
silently returning zeros.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np

#: Where vector files are expected.  Not tracked by git.
EMBEDDING_ROOT = Path(__file__).resolve().parents[2] / "data" / "embeddings"

#: Sentence frames the contextual source uses.  Deliberately bland: the point is
#: to give the model a grammatical context, not to prime it toward a meaning.
TEMPLATES: dict[str, str] = {
    "a": "It is {word} .",
    "v": "They {word} it .",
    "r": "It happened {word} .",
    "n": "It is a {word} .",
}
DEFAULT_TEMPLATE = "The word is {word} ."


class EmbeddingsNotFetched(FileNotFoundError):
    """Raised when a vector file is missing."""


class EmbeddingSource(Protocol):
    """What the rest of the package needs from an embedding model."""

    name: str
    dim: int

    def vector(self, word: str, pos: str = "a", context: Optional[str] = None):
        """The embedding of ``word``, or ``None`` if it is out of vocabulary."""

    def vocabulary(self) -> Sequence[str]:
        """Words available as nearest-neighbour candidates."""


# ---------------------------------------------------------------------------
# Static vectors
# ---------------------------------------------------------------------------


@dataclass
class StaticEmbeddings:
    """GloVe / fastText style text-format vectors, loaded into one matrix.

    Only the words actually needed are kept when ``restrict_to`` is given, which
    turns a 400k-row file into a few thousand rows and makes the whole
    experiment fit comfortably in memory. The full vocabulary is still worth
    loading for nearest-neighbour search, so the two uses are separated:
    ``restrict_to`` is for fitting, ``max_vocab`` for the candidate pool.
    """

    name: str
    matrix: np.ndarray
    index: dict[str, int]

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def vector(self, word: str, pos: str = "a", context: Optional[str] = None):
        row = self.index.get(word.lower())
        return None if row is None else self.matrix[row]

    def vocabulary(self) -> Sequence[str]:
        return list(self.index)

    def matrix_for(self, words: Sequence[str]) -> np.ndarray:
        return np.stack([self.matrix[self.index[w]] for w in words])


def _iter_text_vectors(path: Path) -> Iterable[tuple[str, np.ndarray]]:
    """Yield ``(word, vector)`` from a GloVe/fastText text file, or a zip of one."""
    def parse(stream):
        for lineno, raw in enumerate(stream):
            line = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else raw
            parts = line.rstrip().split(" ")
            if lineno == 0 and len(parts) == 2:
                continue  # fastText header: "<count> <dim>"
            if len(parts) < 3:
                continue
            yield parts[0], np.asarray(parts[1:], dtype=np.float32)

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            inner = sorted(n for n in archive.namelist() if n.endswith((".txt", ".vec")))
            if not inner:
                raise EmbeddingsNotFetched(f"no vector file inside {path}")
            with archive.open(inner[0]) as handle:
                yield from parse(handle)
    else:
        with path.open("rb") as handle:
            yield from parse(handle)


def load_static(
    filename: str = "glove.6B.300d.txt",
    *,
    name: Optional[str] = None,
    restrict_to: Optional[Iterable[str]] = None,
    max_vocab: Optional[int] = 100_000,
) -> StaticEmbeddings:
    """Load a static vector file from :data:`EMBEDDING_ROOT`.

    ``restrict_to`` keeps only the listed words *in addition to* the first
    ``max_vocab`` rows, so the fitting words are guaranteed present even when
    they are rare, while the candidate pool stays the frequent head of the
    vocabulary. Frequency order is the file's own, which for GloVe and fastText
    is descending corpus frequency.
    """
    path = EMBEDDING_ROOT / filename
    if not path.exists():
        raise EmbeddingsNotFetched(
            f"{path} not found. Fetch vectors first:\n"
            f"    ./venv/bin/python -m negation.antonym_vec.fetch"
        )
    wanted = {w.lower() for w in restrict_to} if restrict_to is not None else None

    words: list[str] = []
    rows: list[np.ndarray] = []
    for position, (word, vector) in enumerate(_iter_text_vectors(path)):
        in_head = max_vocab is None or position < max_vocab
        if in_head or (wanted is not None and word in wanted):
            words.append(word)
            rows.append(vector)
    if not rows:
        raise EmbeddingsNotFetched(f"{path} yielded no vectors")

    matrix = np.stack(rows)
    index = {w: i for i, w in enumerate(words)}
    return StaticEmbeddings(name or path.stem, matrix, index)


# ---------------------------------------------------------------------------
# Contextual vectors
# ---------------------------------------------------------------------------


@dataclass
class ContextualEmbeddings:
    """Transformer vectors for a word placed in a template sentence.

    The word is embedded in a bland frame, its sub-tokens are averaged, and the
    last ``layers`` hidden layers are averaged together -- the usual recipe for
    turning a contextual model into type-level vectors.

    Two caveats travel with every number produced from this source. The template
    is part of the vector, so all words of a POS share a component that has
    nothing to do with their meaning; and a word split into several word-pieces
    is represented by their mean, which is not the same object as a whole-word
    embedding. Neither is a reason to avoid the source, but both are reasons not
    to read small differences against the static source as meaningful.
    """

    name: str
    _model: object
    _tokenizer: object
    dim: int
    layers: int = 4
    _cache: dict = None  # type: ignore[assignment]
    _vocab: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self._cache is None:
            self._cache = {}

    def vector(self, word: str, pos: str = "a", context: Optional[str] = None):
        import torch

        key = (word.lower(), pos, context)
        if key in self._cache:
            return self._cache[key]

        template = context or TEMPLATES.get(pos, DEFAULT_TEMPLATE)
        sentence = template.format(word=word) if "{word}" in template else template
        encoded = self._tokenizer(sentence, return_tensors="pt")
        word_ids = self._tokenizer(
            word, add_special_tokens=False
        )["input_ids"]
        if not word_ids:
            return None

        ids = encoded["input_ids"][0].tolist()
        start = _find_subsequence(ids, word_ids)
        if start is None:
            return None

        with torch.no_grad():
            output = self._model(**encoded, output_hidden_states=True)
        stacked = torch.stack(output.hidden_states[-self.layers :])  # L x 1 x T x H
        averaged = stacked.mean(dim=0)[0]  # T x H
        span = averaged[start : start + len(word_ids)].mean(dim=0)
        result = span.numpy().astype(np.float32)
        self._cache[key] = result
        return result

    def vocabulary(self) -> Sequence[str]:
        return list(self._vocab)

    def set_vocabulary(self, words: Iterable[str]) -> None:
        """Fix the candidate pool; a transformer has no natural word vocabulary."""
        self._vocab = tuple(dict.fromkeys(w.lower() for w in words))

    def matrix_for(self, words: Sequence[str], pos_of: dict[str, str]) -> np.ndarray:
        rows = []
        for word in words:
            vec = self.vector(word, pos_of.get(word, "a"))
            rows.append(np.zeros(self.dim, dtype=np.float32) if vec is None else vec)
        return np.stack(rows)


def _find_subsequence(haystack: list[int], needle: list[int]) -> Optional[int]:
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


def load_contextual(
    model_name: str = "bert-base-uncased", layers: int = 4
) -> ContextualEmbeddings:
    """Load a transformer as a type-level embedding source.

    Requires ``torch`` and ``transformers``; both are optional dependencies of
    this package and the import error names them if they are missing.
    """
    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise EmbeddingsNotFetched(
            "contextual embeddings need torch and transformers:\n"
            "    ./venv/bin/python -m pip install torch transformers"
        ) from exc

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    dim = int(model.config.hidden_size)
    return ContextualEmbeddings(f"{model_name}-last{layers}", model, tokenizer, dim, layers)
