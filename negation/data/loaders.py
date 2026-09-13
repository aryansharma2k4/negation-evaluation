"""One loader per usable corpus, all producing :class:`AnnotatedSentence`.

Each loader reads whatever ``fetch.py`` put under ``data/external`` and answers
the same four questions: sentence, tokens, cue spans, scope spans.  Where a
corpus cannot answer -- the NLI pair sets have no spans at all -- the loader
says so with ``provenance="pair-level-only"`` and empty spans rather than
inventing them, so a downstream count of usable training material is honest.

Three annotation formats are involved and none of them resemble each other:

``CD-SCO``    CoNLL columns.  Seven fixed columns, then a *triple* of columns
              (cue, scope, event) for every negation in the sentence.  A
              sentence with three negations has sixteen columns.
``SFU``       inline XML: ``<cue type=... ID=n>`` and ``<xcope ID=m>`` with a
              ``<ref SRC=n>`` inside the scope pointing back at its cue.
``BioScope``  inline XML too, but the other way round: ``<xcope id="X1.2">``
              wraps the scope and the ``<cue ref="X1.2">`` inside it points
              *out* at the scope it belongs to.

The two XML dialects both nest scope and cue, so tokenisation and span
extraction are done in one walk that records character offsets as it goes.
"""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Optional

from .family_map import cue_family, scope_position_family
from .schema import (
    ANN_NEGATION,
    ANN_SPECULATION,
    PROV_GOLD,
    PROV_PAIR_ONLY,
    UNMAPPED,
    AnnotatedSentence,
)
from .sources import BIOSCOPE, CANNOT, CD_SCO, NAN_NLI, NEGNLI, SFU, Corpus


class CorpusNotFetched(FileNotFoundError):
    """Raised when a loader is asked for data that has not been downloaded."""


def _require(corpus: Corpus) -> Path:
    path = corpus.path()
    if not corpus.present():
        raise CorpusNotFetched(
            f"{corpus.name} is not in {path}. Fetch it first:\n"
            f"    ./venv/bin/python -m negation.data.fetch {corpus.key}\n"
            f"Licence: {corpus.licence or 'NONE STATED — check before use'}"
        )
    return path


def _spans(indices: list[int]) -> list[tuple[int, int]]:
    """Collapse sorted token indices into half-open contiguous spans.

    Both cues and scopes are routinely discontinuous -- *neither ... nor* is one
    cue in two pieces -- so runs are grouped rather than assumed contiguous.
    """
    out: list[tuple[int, int]] = []
    for i in sorted(set(indices)):
        if out and out[-1][1] == i:
            out[-1] = (out[-1][0], i + 1)
        else:
            out.append((i, i + 1))
    return out


# ---------------------------------------------------------------------------
# CD-SCO (ConanDoyle-neg)
# ---------------------------------------------------------------------------

#: Columns before the first negation triple: chapter, sent#, tok#, word, lemma,
#: POS, parse bit.
_CDSCO_FIXED = 7

_CDSCO_FILES = {
    "train": "cdsco_SEM-2012-SharedTask-CD-SCO-training-09032012.txt",
    "dev": "cdsco_SEM-2012-SharedTask-CD-SCO-dev-09032012.txt",
    "test-cardboard": "cdsco_SEM-2012-SharedTask-CD-SCO-test-cardboard-GOLD.txt",
    "test-circle": "cdsco_SEM-2012-SharedTask-CD-SCO-test-circle-GOLD.txt",
}


def _cdsco_sentences(path: Path) -> Iterator[list[list[str]]]:
    rows: list[list[str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                if rows:
                    yield rows
                    rows = []
                continue
            rows.append(line.split("\t"))
    if rows:
        yield rows


def load_cdsco(split: Optional[str] = None) -> list[AnnotatedSentence]:
    """ConanDoyle-neg in its *SEM 2012 CoNLL form.

    Emits one record per negation instance.  Sentences with no negation are
    marked ``***`` in column 7 and contribute nothing, which is why the record
    count is far below the sentence count.

    The cue column holds the *affix* for an affixal negation (``in`` on
    *infrequent*) and the whole token otherwise, so ``cue_text`` carries the
    corpus's own string and the family mapper uses it to spot affixal cues.
    """
    root = _require(CD_SCO)
    wanted = list(_CDSCO_FILES) if split is None else [split]
    out: list[AnnotatedSentence] = []

    for name in wanted:
        path = root / _CDSCO_FILES[name]
        if not path.exists():
            continue
        for sent_idx, rows in enumerate(_cdsco_sentences(path)):
            tokens = [r[3] for r in rows]
            text = " ".join(tokens)
            first = rows[0]
            if len(first) <= _CDSCO_FIXED or first[_CDSCO_FIXED] == "***":
                continue  # no negation annotated in this sentence
            n_instances = (len(first) - _CDSCO_FIXED) // 3
            sid = f"cdsco:{name}:{first[0]}:{first[1]}:{sent_idx}"

            for k in range(n_instances):
                base = _CDSCO_FIXED + 3 * k
                cue_idx, scope_idx, cue_strings = [], [], []
                for i, row in enumerate(rows):
                    if len(row) <= base + 1:
                        continue
                    if row[base] != "_":
                        cue_idx.append(i)
                        cue_strings.append(row[base])
                    if row[base + 1] != "_":
                        scope_idx.append(i)
                if not cue_idx:
                    continue
                cue_text = " ".join(cue_strings)
                family = cue_family(cue_text, tokens[cue_idx[0]])
                family = scope_position_family(family, tokens, cue_idx)
                out.append(
                    AnnotatedSentence(
                        sentence=text,
                        tokens=tokens,
                        cue_spans=_spans(cue_idx),
                        scope_spans=_spans(scope_idx),
                        cue_family=family,
                        source_dataset="CD-SCO",
                        split=name,
                        sentence_id=sid,
                        annotation_type=ANN_NEGATION,
                        cue_text=cue_text,
                        provenance=PROV_GOLD,
                        extra={"instance": k, "chapter": first[0]},
                    )
                )
    return out


# ---------------------------------------------------------------------------
# Shared inline-XML walker for SFU and BioScope
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


class _InlineXmlSentence:
    """Collects tokens while remembering which element each one came from."""

    def __init__(self) -> None:
        self.tokens: list[str] = []
        #: token index -> set of enclosing element ids
        self.owners: list[set[str]] = []

    def add(self, text: str, owners: set[str]) -> None:
        for word in _WS.sub(" ", text).strip().split(" "):
            if word:
                self.tokens.append(word)
                self.owners.append(set(owners))

    def indices_owned_by(self, key: str) -> list[int]:
        return [i for i, own in enumerate(self.owners) if key in own]


def _walk(node: ET.Element, sent: _InlineXmlSentence, owners: set[str], keyer) -> None:
    """Depth-first walk accumulating tokens under their enclosing annotations.

    ``keyer(node)`` returns an owner key for an annotation element, or ``None``
    for structural elements.  Text before and after each child is attributed to
    whatever is currently open, which is what makes a scope interrupted by its
    own cue come out as two spans.
    """
    key = keyer(node)
    here = owners | {key} if key else owners
    if node.text:
        sent.add(node.text, here)
    for child in node:
        _walk(child, sent, here, keyer)
        if child.tail:
            sent.add(child.tail, owners | ({key} if key else set()))


# ---------------------------------------------------------------------------
# SFU Review Corpus
# ---------------------------------------------------------------------------

_SFU_CATEGORIES = (
    "BOOKS", "CARS", "COMPUTERS", "COOKWARE",
    "HOTELS", "MOVIES", "MUSIC", "PHONES",
)


def _sfu_keyer(node: ET.Element) -> Optional[str]:
    if node.tag == "cue":
        return f"cue:{node.get('ID')}"
    if node.tag == "xcope":
        return f"xcope:{node.get('ID')}"
    return None


def load_sfu(
    annotation_types: tuple[str, ...] = (ANN_NEGATION, ANN_SPECULATION),
    split: str = "all",
) -> list[AnnotatedSentence]:
    """The SFU Review Corpus, negation **and** speculation.

    This is the corpus that separates the two: every ``<cue>`` carries
    ``type="negation"`` or ``type="speculation"``, so the hedging layer the
    intensity regressor needs can be loaded on its own.

    Scope is linked to cue by a ``<ref SRC=...>`` marker *inside* the
    ``<xcope>``, so the scope belonging to a cue is found by matching that
    reference rather than by nesting or adjacency.

    The corpus ships no splits; ``split`` is recorded as given so a caller can
    impose its own.
    """
    root = _require(SFU)
    base = root / "SFU_Review_Corpus_Negation_Speculation"
    out: list[AnnotatedSentence] = []

    for category in _SFU_CATEGORIES:
        folder = base / category
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.xml")):
            try:
                tree = ET.parse(path)
            except ET.ParseError:
                continue  # a handful of files carry malformed entities
            for s_idx, sentence in enumerate(tree.iter("SENTENCE")):
                sent = _InlineXmlSentence()
                _walk(sentence, sent, set(), _sfu_keyer)
                if not sent.tokens:
                    continue
                text = " ".join(sent.tokens)
                sid = f"sfu:{category}:{path.stem}:{s_idx}"

                # cue ID -> the xcope whose <ref SRC> names it
                scope_of: dict[str, str] = {}
                for xcope in sentence.iter("xcope"):
                    for ref in xcope.iter("ref"):
                        src = ref.get("SRC")
                        if src is not None:
                            scope_of[src] = f"xcope:{xcope.get('ID')}"

                for cue in sentence.iter("cue"):
                    ctype = (cue.get("type") or "").lower()
                    if ctype not in annotation_types:
                        continue
                    cue_id = cue.get("ID")
                    cue_idx = sent.indices_owned_by(f"cue:{cue_id}")
                    if not cue_idx:
                        continue
                    scope_key = scope_of.get(cue_id or "")
                    scope_idx = (
                        sent.indices_owned_by(scope_key) if scope_key else []
                    )
                    cue_text = " ".join(sent.tokens[i] for i in cue_idx)
                    family = cue_family(
                        cue_text, sent.tokens[cue_idx[0]], annotation_type=ctype
                    )
                    family = scope_position_family(family, sent.tokens, cue_idx)
                    out.append(
                        AnnotatedSentence(
                            sentence=text,
                            tokens=sent.tokens,
                            cue_spans=_spans(cue_idx),
                            scope_spans=_spans(scope_idx),
                            cue_family=family,
                            source_dataset="SFU",
                            split=split,
                            sentence_id=sid,
                            annotation_type=ctype,
                            cue_text=cue_text,
                            provenance=PROV_GOLD,
                            extra={"category": category, "document": path.stem},
                        )
                    )
    return out


# ---------------------------------------------------------------------------
# BioScope
# ---------------------------------------------------------------------------

_BIOSCOPE_FILES = {
    "abstracts": "abstracts.xml",
    "full_papers": "full_papers.xml",
    "clinical": "clinical.xml",
}

#: Subcorpora loaded when none is named.  The clinical one is excluded because
#: what BioScope distributes for it is an annotation skeleton with every word
#: replaced by ``*`` -- see :func:`bioscope_is_redacted`.
_BIOSCOPE_DEFAULT = ("abstracts", "full_papers")


class RedactedSubcorpus(ValueError):
    """Raised when a subcorpus ships without the text it annotates."""


def bioscope_is_redacted(path: Path, sample: int = 200) -> bool:
    """True when a BioScope file carries annotations but no words.

    The clinical subcorpus is distributed as ``clinical_records_anon.xml``, in
    which every token is ``*``; the real text has to be merged in from the 2007
    CMC Medical NLP Challenge release, which the corpus README points at
    ``computationalmedicine.org/catalog``.  That host no longer resolves, so the
    clinical subcorpus cannot currently be reconstructed from any documented
    route -- and a loader that silently returned thousands of sentences reading
    ``* * *`` would be worse than one that refuses.
    """
    tree = ET.parse(path)
    seen = redacted = 0
    for sentence in tree.iter("sentence"):
        text = "".join(sentence.itertext()).strip()
        if not text:
            continue
        seen += 1
        # A redacted sentence keeps its punctuation ("*.", "* * *."), so the
        # test is for the absence of any word character at all, not for the
        # presence of asterisks.
        if not any(ch.isalnum() for ch in text):
            redacted += 1
        if seen >= sample:
            break
    return seen > 0 and redacted == seen


def _bioscope_keyer(node: ET.Element) -> Optional[str]:
    if node.tag == "xcope":
        return f"xcope:{node.get('id')}"
    if node.tag == "cue":
        return f"cue:{node.get('ref')}:{id(node)}"
    return None


def load_bioscope(subcorpus: Optional[str] = None) -> list[AnnotatedSentence]:
    """BioScope, negation **and** speculation, biomedical register.

    BioScope inverts SFU's linking: the ``<xcope id="X1.2">`` wraps the scope
    and the ``<cue ref="X1.2">`` sits *inside* it, pointing out at the scope it
    belongs to.  So a cue's scope is its own ``ref``, and the cue tokens are a
    subset of the scope tokens -- which is why the cue is subtracted from the
    scope below, to match CD-SCO's and SFU's convention that scope excludes cue.

    Only the abstracts and full papers load by default; the clinical subcorpus
    is distributed with its text redacted and is refused rather than returned
    empty.  Pass ``subcorpus="clinical"`` to get the explaining exception.
    """
    root = _require(BIOSCOPE)
    wanted = list(_BIOSCOPE_DEFAULT) if subcorpus is None else [subcorpus]
    out: list[AnnotatedSentence] = []

    for name in wanted:
        path = root / _BIOSCOPE_FILES[name]
        if not path.exists():
            continue
        if bioscope_is_redacted(path):
            if subcorpus is None:
                continue
            raise RedactedSubcorpus(
                f"BioScope '{name}' at {path} has annotations but no text: every "
                f"token is '*'. The words come from the 2007 CMC Medical NLP "
                f"Challenge data, whose distribution host "
                f"(computationalmedicine.org) no longer resolves."
            )
        tree = ET.parse(path)
        for s_idx, sentence in enumerate(tree.iter("sentence")):
            sent = _InlineXmlSentence()
            _walk(sentence, sent, set(), _bioscope_keyer)
            if not sent.tokens:
                continue
            text = " ".join(sent.tokens)
            sid = f"bioscope:{name}:{sentence.get('id') or s_idx}"

            for cue in sentence.iter("cue"):
                ctype = (cue.get("type") or "").lower()
                if ctype not in (ANN_NEGATION, ANN_SPECULATION):
                    continue
                ref = cue.get("ref")
                cue_key = f"cue:{ref}:{id(cue)}"
                cue_idx = sent.indices_owned_by(cue_key)
                if not cue_idx:
                    continue
                scope_idx = [
                    i
                    for i in sent.indices_owned_by(f"xcope:{ref}")
                    if i not in set(cue_idx)
                ]
                cue_text = " ".join(sent.tokens[i] for i in cue_idx)
                family = cue_family(
                    cue_text, sent.tokens[cue_idx[0]], annotation_type=ctype
                )
                family = scope_position_family(family, sent.tokens, cue_idx)
                out.append(
                    AnnotatedSentence(
                        sentence=text,
                        tokens=sent.tokens,
                        cue_spans=_spans(cue_idx),
                        scope_spans=_spans(scope_idx),
                        cue_family=family,
                        source_dataset="BioScope",
                        split=name,
                        sentence_id=sid,
                        annotation_type=ctype,
                        cue_text=cue_text,
                        provenance=PROV_GOLD,
                        extra={"subcorpus": name},
                    )
                )
    return out


# ---------------------------------------------------------------------------
# Sentence-pair corpora.  No spans; loaded so their absence is countable.
# ---------------------------------------------------------------------------


def _pair_record(
    hypothesis: str, source: str, split: str, sid: str, extra: dict
) -> AnnotatedSentence:
    tokens = hypothesis.split()
    return AnnotatedSentence(
        sentence=hypothesis,
        tokens=tokens,
        cue_spans=[],
        scope_spans=[],
        cue_family=UNMAPPED,
        source_dataset=source,
        split=split,
        sentence_id=sid,
        annotation_type=ANN_NEGATION,
        cue_text="",
        provenance=PROV_PAIR_ONLY,
        extra=extra,
    )


def load_cannot(split: str = "all") -> list[AnnotatedSentence]:
    """CANNOT negated text pairs.

    Pair-level only: the corpus labels whether the hypothesis contradicts the
    premise, never which token did it.  Records therefore carry empty spans and
    ``provenance="pair-level-only"``; the premise and label live in ``extra``.
    """
    root = _require(CANNOT)
    path = root / "cannot_dataset_v1.1.tsv"
    out: list[AnnotatedSentence] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            hyp = (row.get("hypothesis") or "").strip()
            if not hyp:
                continue
            out.append(
                _pair_record(
                    hyp,
                    "CANNOT",
                    split,
                    f"cannot:{i}",
                    {"premise": (row.get("premise") or "").strip(),
                     "label": row.get("label")},
                )
            )
    return out


_NEGNLI_FILES = {"RTE": "negnli_RTE.txt", "SNLI": "negnli_SNLI.txt",
                 "MNLI": "negnli_MNLI.txt"}


#: The NegNLI release is Latin-1, not UTF-8 (it carries bytes such as 0xe9 that
#: are invalid UTF-8 continuations), so it is decoded explicitly rather than
#: left to the platform default.
_NEGNLI_ENCODING = "latin-1"


def load_negnli(subset: Optional[str] = None) -> list[AnnotatedSentence]:
    """Hossain et al.'s negation-focused RTE / SNLI / MNLI benchmarks.

    Pair-level only, and an evaluation suite rather than training data: the
    negation was added by hand to existing NLI pairs, so the sentences are not
    naturally occurring.
    """
    root = _require(NEGNLI)
    wanted = list(_NEGNLI_FILES) if subset is None else [subset]
    out: list[AnnotatedSentence] = []
    for name in wanted:
        path = root / _NEGNLI_FILES[name]
        if not path.exists():
            continue
        with path.open(encoding=_NEGNLI_ENCODING, newline="") as handle:
            for i, row in enumerate(csv.DictReader(handle, delimiter="\t")):
                hyp = (row.get("Hypothesis") or "").strip()
                if not hyp:
                    continue
                out.append(
                    _pair_record(
                        hyp, "NegNLI", name, f"negnli:{name}:{i}",
                        {"premise": (row.get("Text") or "").strip(),
                         "label": row.get("gold_label")},
                    )
                )
    return out


def load_nan_nli(split: str = "test") -> list[AnnotatedSentence]:
    """NaN-NLI, the sub-clausal negation test suite.

    Pair-level, and tiny, but it is the only resource found that labels each
    pair with the *construction* of its negation -- verbal vs non-verbal,
    analytic vs synthetic, clausal vs sub-clausal, ordinary vs meta-linguistic.
    Those labels are carried through in ``extra`` because they are the closest
    external analogue to our own family taxonomy.
    """
    root = _require(NAN_NLI)
    path = root / "nan.csv"
    out: list[AnnotatedSentence] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            hyp = (row.get("hypothesis") or "").strip()
            if not hyp:
                continue
            extra = {
                "premise": (row.get("premise") or "").strip(),
                "label": row.get("label"),
                "construction": row.get("Construction"),
                "construction_subtype": row.get("Construction Subtype"),
            }
            out.append(_pair_record(hyp, "NaN-NLI", split, f"nannli:{i}", extra))
    return out


#: Loader per corpus key, so the statistics script can iterate.
LOADERS = {
    "cdsco": load_cdsco,
    "sfu": load_sfu,
    "bioscope": load_bioscope,
    "cannot": load_cannot,
    "negnli": load_negnli,
    "nannli": load_nan_nli,
}
