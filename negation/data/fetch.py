"""Download the third-party corpora into ``data/external``.

Nothing is vendored into this repository.  Two of the corpora state no licence
at all, one is GPL-3.0 copyleft, and one is only "free for research", so
redistribution is not ours to make -- and the largest is 10 MB of text that does
not belong in git either.  This script pulls each corpus from the source
:mod:`negation.data.sources` records as verified, and prints the licence before
it does.

    python -m negation.data.fetch            # everything
    python -m negation.data.fetch sfu cdsco  # just these
    python -m negation.data.fetch --check    # report what is present, fetch nothing

Only the standard library is used, so this runs without adding a dependency.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from .sources import CORPORA, STATUS_OK, Corpus

_UA = "negation-dataset-eval/1.0 (research; stdlib urllib)"
_TIMEOUT = 120


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return response.read()


def _target_name(corpus: Corpus, url: str) -> str:
    """Local filename for a source URL.

    CD-SCO's files are prefixed so a directory listing shows what they are, and
    BioScope's are given the short names its loader expects; everything else
    keeps the name it is served under.
    """
    stem = url.rsplit("/", 1)[-1]
    if corpus.key == "cdsco":
        return f"cdsco_{stem}"
    if corpus.key == "bioscope":
        return {"abstracts_pmid.xml": "abstracts.xml"}.get(stem, stem)
    if corpus.key == "negnli":
        return f"negnli_{stem}"
    return stem


def _unpack(corpus: Corpus, payload: bytes, destination: Path) -> None:
    """Write a payload, expanding the archives whose loaders expect a tree.

    The SFU corpus is 400 XML files in a zip and BioScope's package carries the
    clinical annotations that are not downloadable on their own, so both are
    expanded; everything else is a single file.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if corpus.key == "sfu":
            archive.extractall(destination)
            return
        # BioScope: lift just the clinical annotations out of the package.
        for member in archive.namelist():
            if member.endswith("clinical_records_anon.xml"):
                (destination / "clinical.xml").write_bytes(archive.read(member))
                return


def fetch(corpus: Corpus, *, force: bool = False) -> int:
    """Download one corpus.  Returns the number of files written."""
    destination = corpus.path()
    destination.mkdir(parents=True, exist_ok=True)
    print(f"\n{corpus.name}")
    print(f"  licence: {corpus.licence or 'NONE STATED - check terms before use'}")
    if corpus.homepage:
        print(f"  homepage: {corpus.homepage}")

    written = 0
    for source in corpus.sources:
        if source.status != STATUS_OK:
            print(f"  skip (recorded as {source.status}): {source.url}")
            continue
        name = _target_name(corpus, source.url)
        path = destination / name
        if path.exists() and not force:
            print(f"  have {name}")
            continue
        try:
            payload = _get(source.url)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  FAILED {source.url}\n         {exc}", file=sys.stderr)
            continue
        if source.url.endswith(".zip"):
            _unpack(corpus, payload, destination)
            print(f"  expanded {name} ({len(payload):,} bytes)")
        else:
            path.write_bytes(payload)
            print(f"  wrote {name} ({len(payload):,} bytes)")
        written += 1
    return written


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("corpora", nargs="*", default=list(CORPORA),
                        choices=list(CORPORA) or None,
                        help="which corpora to fetch (default: all)")
    parser.add_argument("--check", action="store_true",
                        help="report what is already present and exit")
    parser.add_argument("--force", action="store_true",
                        help="re-download files that are already present")
    args = parser.parse_args(argv)

    if args.check:
        for key in args.corpora:
            corpus = CORPORA[key]
            state = "present" if corpus.present() else "MISSING"
            print(f"  {key:10s} {state:8s} {corpus.path()}")
        return 0

    for key in args.corpora:
        fetch(CORPORA[key], force=args.force)
    print("\nLicences differ and two corpora state none at all. Check "
          "negation/data/sources.py before redistributing anything derived "
          "from these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
