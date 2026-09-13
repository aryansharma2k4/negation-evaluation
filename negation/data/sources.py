"""Where each corpus was actually found, and under what terms.

Every URL here was fetched and checked on 2026-09-13; ``status`` records what
came back, including for the ones that did not work.  The dead entries are kept
deliberately -- the canonical *SEM 2012 page and the URL the HuggingFace
BioScope loader still points at are both gone, and a future reader needs to know
that was checked rather than assumed.

Nothing is vendored into the repository.  Licences differ (the SFU corpus is
GPL-3.0, which is copyleft), several corpora are only "free for research", and
two of them have no licence statement at all, so redistribution is not ours to
make.  ``fetch.py`` pulls them into ``data/external`` on request; the loaders
read from there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

#: Root the loaders read from.  Not tracked by git.
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "external"

#: The date every ``status`` below was observed.
VERIFIED_ON = "2026-09-13"

STATUS_OK = "verified-downloadable"
STATUS_DEAD = "dead"
STATUS_UNVERIFIED = "not-verified"


@dataclass(frozen=True)
class Source:
    """One file (or one archive) a corpus is distributed as."""

    url: str
    status: str
    #: Bytes actually received when checked; ``None`` for a dead URL.
    size_bytes: Optional[int] = None
    note: str = ""


@dataclass(frozen=True)
class Corpus:
    """A third-party corpus, its provenance and its terms."""

    key: str
    name: str
    #: SPDX identifier where one is stated, else a plain description.  ``None``
    #: means no licence statement was found anywhere -- which is not the same
    #: as permissive, and is called out in the report.
    licence: Optional[str]
    licence_source: str
    redistributable: bool
    #: Subdirectory of :data:`DATA_ROOT` the loader expects.
    subdir: str
    sources: tuple[Source, ...]
    homepage: str = ""
    citation: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def path(self) -> Path:
        return DATA_ROOT / self.subdir

    def present(self) -> bool:
        p = self.path()
        return p.exists() and any(p.rglob("*"))


_GITHUB_CSENGE = (
    "https://raw.githubusercontent.com/csenge-szabo/Negation_Scope_Detection"
    "/main/data/raw_data"
)

CD_SCO = Corpus(
    key="cdsco",
    name="ConanDoyle-neg / CD-SCO (*SEM 2012 Shared Task)",
    licence=None,
    licence_source=(
        "No licence found. The original CLiPS distribution page is gone and the "
        "mirror carrying the files states no terms. Morante & Daelemans (2012) "
        "describe the corpus as freely available for research."
    ),
    redistributable=False,
    subdir="cdsco",
    homepage="https://aclanthology.org/L12-1077/",
    citation="Morante & Daelemans (2012), LREC",
    sources=(
        Source(
            "https://www.clips.uantwerpen.be/sem2012-st-neg/data.html",
            STATUS_DEAD,
            None,
            "Canonical shared-task download page: HTTP 404. The whole "
            "sem2012-st-neg directory is gone.",
        ),
        Source(f"{_GITHUB_CSENGE}/SEM-2012-SharedTask-CD-SCO-training-09032012.txt",
               STATUS_OK, 2898575, "training, 3644 sentences"),
        Source(f"{_GITHUB_CSENGE}/SEM-2012-SharedTask-CD-SCO-dev-09032012.txt",
               STATUS_OK, 547140, "dev"),
        Source(f"{_GITHUB_CSENGE}/SEM-2012-SharedTask-CD-SCO-test-cardboard-GOLD.txt",
               STATUS_OK, 401244, "test, The Adventure of the Cardboard Box"),
        Source(f"{_GITHUB_CSENGE}/SEM-2012-SharedTask-CD-SCO-test-circle-GOLD.txt",
               STATUS_OK, 342076, "test, The Adventure of the Red Circle"),
    ),
    notes=(
        "Recovered from a third-party research mirror, not an official source.",
        "Negation only. No speculation or hedge layer.",
    ),
)

SFU = Corpus(
    key="sfu",
    name="SFU Review Corpus, annotated for negation and speculation",
    licence="GPL-3.0-or-later",
    licence_source=(
        "Stated verbatim in README.txt inside the distributed zip: "
        "'This corpus is free software: you can redistribute it and/or modify "
        "it under the terms of the GNU General Public License ... either "
        "version 3 of the License, or (at your option) any later version.'"
    ),
    redistributable=True,
    subdir="sfu",
    homepage="https://www.sfu.ca/~mtaboada/SFU_Review_Corpus.html",
    citation="Konstantinova et al. (2012), LREC",
    sources=(
        Source(
            "https://www.sfu.ca/~mtaboada/docs/research/"
            "SFU_Review_Corpus_Negation_Speculation.zip",
            STATUS_OK,
            1540440,
            "412 files: 400 annotated XML reviews in 8 categories, plus the "
            "annotation guidelines PDF and the LREC paper.",
        ),
    ),
    notes=(
        "Copyleft. Redistributing a derived corpus means releasing it under "
        "GPL-3.0 too; model weights trained on it are a murkier question.",
        "The similarly named SINAI/SFU-Review-SP-Neg on HuggingFace is the "
        "SPANISH corpus under CC-BY-NC-SA-4.0, a different resource.",
    ),
)

BIOSCOPE = Corpus(
    key="bioscope",
    name="BioScope",
    licence="CC-BY-2.0 (as asserted by the bigbio HuggingFace card)",
    licence_source=(
        "No licence text on the corpus homepage, which says only that it is "
        "publicly available for research. The CC-BY-2.0 claim comes from the "
        "bigbio/bioscope dataset card, not from the corpus authors."
    ),
    redistributable=False,
    subdir="bioscope",
    homepage="https://rgai.inf.u-szeged.hu/node/105",
    citation="Vincze et al. (2008), BMC Bioinformatics",
    sources=(
        Source(
            "https://rgai.sed.hu/sites/rgai.sed.hu/files/bioscope.zip",
            STATUS_DEAD,
            None,
            "The URL the HuggingFace bigbio/bioscope loader still uses. The "
            "rgai.sed.hu host does not resolve, so that loader cannot run.",
        ),
        Source("https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/"
               "abstracts_pmid.xml", STATUS_OK, 2955559, "GENIA abstracts"),
        Source("https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/"
               "full_papers.xml", STATUS_OK, 591638, "full articles"),
        Source("https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/"
               "bioscope.zip", STATUS_OK, 1078189,
               "complete package; contains clinical_records_anon.xml, which is "
               "the only route to the clinical subcorpus"),
    ),
    notes=(
        "Annotates negation and speculation as separate cue types.",
        "Biomedical register only: abstracts, full papers, clinical records.",
    ),
)

CANNOT = Corpus(
    key="cannot",
    name="CANNOT",
    licence="CC-BY-SA-4.0",
    licence_source="Declared on the tum-nlp/cannot-dataset HuggingFace card.",
    redistributable=True,
    subdir="cannot",
    homepage="https://github.com/dmlls/cannot-dataset",
    citation="Anschütz et al. (2023), arXiv:2307.13989",
    sources=(
        Source("https://huggingface.co/datasets/tum-nlp/cannot-dataset/resolve/"
               "main/cannot_dataset_v1.1.tsv", STATUS_OK, 10_000_000,
               "77,376 premise/hypothesis/label rows"),
    ),
    notes=(
        "Sentence-pair NLI-style only. No cue spans, no scope spans.",
        "Share-alike: a derived corpus must also be CC-BY-SA-4.0.",
    ),
)

NEGNLI = Corpus(
    key="negnli",
    name="NegNLI (new negation benchmarks for RTE / SNLI / MNLI)",
    licence=None,
    licence_source="No LICENSE file in the repository (checked, HTTP 404).",
    redistributable=False,
    subdir="negnli",
    homepage="https://github.com/mosharafhossain/negation-and-nli",
    citation="Hossain et al. (2020), EMNLP",
    sources=(
        Source("https://raw.githubusercontent.com/mosharafhossain/"
               "negation-and-nli/master/data/new_benchmarks/clean_data/RTE.txt",
               STATUS_OK, 308027, "1500 pairs"),
        Source("https://raw.githubusercontent.com/mosharafhossain/"
               "negation-and-nli/master/data/new_benchmarks/clean_data/SNLI.txt",
               STATUS_OK, 172716, "1500 pairs"),
        Source("https://raw.githubusercontent.com/mosharafhossain/"
               "negation-and-nli/master/data/new_benchmarks/clean_data/MNLI.txt",
               STATUS_OK, 239931, "1500 pairs"),
    ),
    notes=("Sentence-pair only. Evaluation suite, not training data.",),
)

NAN_NLI = Corpus(
    key="nannli",
    name="NaN-NLI (sub-clausal negation test suite)",
    licence="CC-BY-SA-4.0",
    licence_source="Declared on the joey234/nan-nli HuggingFace card.",
    redistributable=True,
    subdir="nannli",
    homepage="https://huggingface.co/datasets/joey234/nan-nli",
    citation="Truong et al. (2022), AACL-IJCNLP",
    sources=(
        Source("https://huggingface.co/datasets/joey234/nan-nli/resolve/main/"
               "nan.csv", STATUS_OK, 52317, "258 pairs with construction labels"),
    ),
    notes=(
        "Tiny, but the only resource found that labels negation constructions "
        "linguistically (verbal/non-verbal, analytic/synthetic, "
        "clausal/sub-clausal) rather than just marking cues.",
        "Its 'not all' / 'some not' pairs are the same alternation our "
        "L_scope_position and rescope generators produce.",
    ),
)

#: Everything the loaders know how to read.
CORPORA: dict[str, Corpus] = {
    c.key: c for c in (CD_SCO, SFU, BIOSCOPE, CANNOT, NEGNLI, NAN_NLI)
}

#: Checked and deliberately not loaded; see docs/dataset_evaluation.md.
REJECTED: dict[str, str] = {
    "SNLI": (
        "CC-BY-SA-4.0, 550k pairs, verified available (stanfordnlp/snli). No "
        "negation annotation of any kind; mining contradiction pairs whose "
        "hypothesis adds a cue would label them with our own detector, which "
        "is circular. Use as a source of raw sentences only."
    ),
    "MultiNLI": (
        "Verified available (nyu-mll/multi_nli, 392,702 train). Licence is "
        "per-genre: cc-by-3.0, cc-by-sa-3.0, mit and 'other' are all declared "
        "on one card, so blanket redistribution is not safe. No negation "
        "annotation. Same circularity as SNLI."
    ),
    "xNLI-neg": (
        "No dataset by that name found on HuggingFace or elsewhere. The "
        "closest real resource is Hartmann et al. (2021), CoNLL, "
        "'A Multilingual Benchmark for Probing Negation-Awareness with Minimal "
        "Pairs', released at github.com/mahartmann/negationminpairs (repository "
        "reachable, HTTP 200; no licence stated). XNLI-derived minimal pairs, "
        "sentence-pair level, no cue or scope spans, and multilingual where "
        "this project is English-only."
    ),
    "SFU-Review-SP-Neg": (
        "SINAI/SFU-Review-SP-Neg on HuggingFace is Spanish, CC-BY-NC-SA-4.0. "
        "Not the Konstantinova English corpus, and the NC clause would bar "
        "commercial use."
    ),
    "joey234/conandoyle_cue_scope": (
        "235 rows, no licence, no dataset card. A partial re-release of CD-SCO; "
        "the canonical CoNLL files carry the same content plus the event layer."
    ),
    "MisterStino/negation-scope-cue-given": (
        "Hound of the Baskervilles with scope tags but no cue column, and no "
        "licence. Strictly less information than CD-SCO itself."
    ),
    "dannashao/sem2012forNegbert": (
        "CC0-1.0, correct CD-SCO split sizes (3779/815/1116), but collapses "
        "cue and scope into one tag sequence. Useful as a cross-check on the "
        "CoNLL parse, not as the primary source."
    ),
}
