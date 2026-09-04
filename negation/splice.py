"""Character-offset splicing.

Every variant string in this package is built by applying a list of
:class:`Edit` objects to the *base* sentence text at spaCy token character
offsets.  Nothing here ever regexes the raw text: an edit names a half-open
``[start, end)`` character range of the base and the pieces that replace it.

Pieces are ``(text, is_cue)`` pairs so that a single edit can mix scaffolding
with the negation cue itself -- e.g. do-support rewrites the whole ROOT span as
``["does ", ("not", cue), " sort"]`` -- and the splicer can report exactly where
the cue landed in the *output* string.
"""

from __future__ import annotations

from dataclasses import dataclass

Piece = tuple[str, bool]

#: A realised cue in the output string: ``(token_text, start, end)``.
Cue = tuple[str, int, int]


@dataclass(frozen=True)
class Edit:
    """Replace ``base[start:end)`` with the concatenation of ``pieces``.

    ``start == end`` is a pure insertion at that offset.
    """

    start: int
    end: int
    pieces: tuple[Piece, ...]

    @staticmethod
    def insert(at: int, text: str, *, cue: bool = False) -> "Edit":
        return Edit(at, at, ((text, cue),))

    @staticmethod
    def replace(start: int, end: int, text: str, *, cue: bool = False) -> "Edit":
        return Edit(start, end, ((text, cue),))

    @staticmethod
    def compound(start: int, end: int, pieces: list[Piece]) -> "Edit":
        return Edit(start, end, tuple(pieces))

    @property
    def cue_text(self) -> str:
        return "".join(t for t, is_cue in self.pieces if is_cue)


class OverlappingEditError(ValueError):
    """Raised when two edits claim overlapping regions of the base string."""


def apply_edits(text: str, edits: list[Edit]) -> tuple[str, list[Cue]]:
    """Apply ``edits`` to ``text``; return ``(new_text, cues)``.

    Cues come back in output order, one per whitespace-delimited token of each
    cue piece, each carrying the exact substring and its offsets in the
    *returned* string.  Deriving both the tokens and the spans here -- rather
    than having generators hand-maintain a parallel token list -- is what keeps
    ``cue_tokens`` and ``cue_char_spans`` aligned no matter how many edits a
    variant splices together or what order they were built in.
    """
    ordered = sorted(edits, key=lambda e: (e.start, e.end))
    for prev, nxt in zip(ordered, ordered[1:]):
        if nxt.start < prev.end:
            raise OverlappingEditError(f"{prev} overlaps {nxt}")

    out: list[str] = []
    written = 0  # length of the output produced so far
    pos = 0  # cursor into the base string
    cues: list[Cue] = []

    for edit in ordered:
        chunk = text[pos:edit.start]
        out.append(chunk)
        written += len(chunk)
        for piece_text, is_cue in edit.pieces:
            if is_cue and piece_text.strip():
                # One span per whitespace-delimited cue token, so cue_tokens and
                # cue_char_spans stay index-aligned for multiword cues such as
                # "by no means".
                offset = 0
                for word in piece_text.split():
                    start = piece_text.index(word, offset)
                    cues.append((word, written + start, written + start + len(word)))
                    offset = start + len(word)
            out.append(piece_text)
            written += len(piece_text)
        pos = edit.end

    out.append(text[pos:])
    return "".join(out), sorted(cues, key=lambda c: c[1])


def normalize_spacing(text: str, cues: list[Cue]) -> tuple[str, list[Cue]]:
    """Collapse runs of whitespace and drop space before closing punctuation.

    Cue spans are remapped through the same transformation so they keep pointing
    at the cue.  Done with an explicit character walk rather than ``re.sub`` so
    the offset mapping stays exact.
    """
    mapping: list[int] = []  # old index -> new index
    out: list[str] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space or not out:
                mapping.append(len(out))
                continue
            prev_space = True
            mapping.append(len(out))
            out.append(" ")
            continue
        if ch in ",.;:!?" and out and out[-1] == " ":
            # After the pop, ``len(out)`` is exactly the index the removed
            # space occupied, so the punctuation slots into that position and
            # every previously recorded mapping stays valid.
            out.pop()
        prev_space = False
        mapping.append(len(out))
        out.append(ch)
    mapping.append(len(out))

    new_text = "".join(out).strip()
    # ``strip`` only removes leading space when the text started with one, which
    # the loop above already suppressed; a trailing strip never affects spans.
    remapped = [
        (word, min(mapping[s], len(new_text)), min(mapping[e], len(new_text)))
        for word, s, e in cues
    ]
    return new_text, remapped


def recase_like(base: str, variant: str) -> str:
    """Match the variant's sentence-initial casing to the base's.

    Splicing at the front of a sentence (``"Everyone" -> "Not everyone"``) would
    otherwise leave a stray capital mid-string, so we lower the old initial and
    raise the new one.  Only the first alphabetic character is touched, and only
    when the base itself was capitalised.
    """
    if not base or not variant:
        return variant
    first = next((c for c in base if c.isalpha()), "")
    if not first.isupper():
        return variant
    chars = list(variant)
    for i, c in enumerate(chars):
        if c.isalpha():
            chars[i] = c.upper()
            break
    return "".join(chars)


def lower_initial(word: str) -> str:
    """Lowercase a word's first letter unless it looks like a proper noun."""
    if not word or len(word) > 1 and word[1:].lower() != word[1:]:
        return word  # e.g. "NASA", "McCarthy" -- leave alone
    return word[:1].lower() + word[1:]
