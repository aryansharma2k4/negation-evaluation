"""Download word vectors into ``data/embeddings``.

    python -m negation.antonym_vec.fetch              # GloVe 6B (822 MB)
    python -m negation.antonym_vec.fetch fasttext     # fastText wiki-news
    python -m negation.antonym_vec.fetch --check

Vectors are not vendored: they are hundreds of megabytes and carry their own
licences (GloVe is PDDL/ODC-BY via Common Crawl and Wikipedia; fastText vectors
are CC-BY-SA-3.0). Both are fine for research use; neither belongs in git.

The contextual source needs ``torch`` and ``transformers`` instead, which are
optional dependencies and are not installed by this script.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Optional

from .embeddings import EMBEDDING_ROOT

_UA = "negation-antonym-vec/1.0 (research; stdlib urllib)"


@dataclass(frozen=True)
class VectorSet:
    key: str
    url: str
    archive: str
    #: File inside the archive that the loader will read.
    member: str
    licence: str
    approx_mb: int


VECTOR_SETS: dict[str, VectorSet] = {
    "glove": VectorSet(
        key="glove",
        url="https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
        archive="glove.6B.zip",
        member="glove.6B.300d.txt",
        licence="PDDL / ODC-BY (Wikipedia + Gigaword)",
        approx_mb=822,
    ),
    "fasttext": VectorSet(
        key="fasttext",
        url=(
            "https://dl.fbaipublicfiles.com/fasttext/vectors-english/"
            "wiki-news-300d-1M.vec.zip"
        ),
        archive="wiki-news-300d-1M.vec.zip",
        member="wiki-news-300d-1M.vec",
        licence="CC-BY-SA-3.0",
        approx_mb=681,
    ),
}


def fetch(vectors: VectorSet, *, force: bool = False) -> None:
    EMBEDDING_ROOT.mkdir(parents=True, exist_ok=True)
    target = EMBEDDING_ROOT / vectors.member
    archive = EMBEDDING_ROOT / vectors.archive

    print(f"\n{vectors.key}: {vectors.member}")
    print(f"  licence: {vectors.licence}")
    if target.exists() and not force:
        print(f"  have {target.name}")
        return

    if not archive.exists() or force:
        print(f"  downloading {vectors.url} (~{vectors.approx_mb} MB) ...")
        request = urllib.request.Request(vectors.url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(request, timeout=600) as response, \
                 archive.open("wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            return

    print(f"  extracting {vectors.member} ...")
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if n.endswith(vectors.member)]
        if not names:
            print(f"  {vectors.member} not found inside {archive.name}", file=sys.stderr)
            return
        with zf.open(names[0]) as src, target.open("wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    print(f"  wrote {target} ({target.stat().st_size:,} bytes)")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sets", nargs="*", default=["glove"], choices=list(VECTOR_SETS))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        for key in VECTOR_SETS:
            path = EMBEDDING_ROOT / VECTOR_SETS[key].member
            print(f"  {key:10s} {'present' if path.exists() else 'MISSING':8s} {path}")
        return 0

    for key in args.sets:
        fetch(VECTOR_SETS[key], force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
