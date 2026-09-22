"""Generate paraphrase-control candidates for the negation-direction experiment.

A control pair is (base_sentence, paraphrase) where the paraphrase carries no
negation, so it measures how far an embedding moves when the meaning is
*preserved*.  Two mechanical sources, both cheap and both noisy:

  1. WordNet synonym swap on one content word (never a negation-bearing one),
     re-inflected with the generator's own morphology helpers.
  2. Clause reorder for coordinated / subordinated sentences.

Output is *candidates*: WordNet's first synsets produce plenty of nonsense
("array" -> "raiment"), so the candidate list is hand-filtered once into
``data/control_pairs.jsonl``, on English acceptability alone and before any
embedding was computed.  Rerun with:

    ./venv/bin/python experiments/negation_direction/make_controls.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nltk.corpus import wordnet as wn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from negation.lexicons import LEXICALLY_NEGATIVE, NEGATIVE_ADVERBS, ANTONYM_VERB_STOPLIST
from negation.nlp_core import match_inflection, matches_case, parse_batch, wn_pos

HERE = Path(__file__).resolve().parent
BASES = HERE.parents[1] / "tests" / "data" / "base_sentences_v1.txt"

SWAP_POS = frozenset({"NOUN", "VERB", "ADJ", "ADV"})
SKIP = LEXICALLY_NEGATIVE | frozenset(NEGATIVE_ADVERBS) | ANTONYM_VERB_STOPLIST


def synonym_swaps(doc, limit: int = 3) -> list[tuple[str, str]]:
    """(paraphrase, note) for single-word WordNet synonym substitutions."""
    out = []
    for token in doc:
        if token.pos_ not in SWAP_POS or token.lower_ in SKIP or token.lemma_ in SKIP:
            continue
        pos = wn_pos(token.pos_)
        if pos is None:
            continue
        for synset in wn.synsets(token.lemma_, pos=pos)[:2]:
            for lemma in synset.lemma_names():
                if "_" in lemma or lemma.lower() == token.lemma_.lower():
                    continue
                word = matches_case(match_inflection(lemma.lower(), token), token)
                text = doc.text[: token.idx] + word + doc.text[token.idx + len(token.text) :]
                out.append((text, f"synonym:{token.text}->{word}"))
                break
            if len(out) >= limit:
                return out
    return out


def clause_reorders(doc) -> list[tuple[str, str]]:
    """(paraphrase, note) for coordination / subordination reorderings."""
    text = doc.text.rstrip(".")
    out = []
    if " and " in text:
        left, _, right = text.partition(" and ")
        out.append((f"{right[:1].upper()}{right[1:]} and {left[:1].lower()}{left[1:]}.",
                    "reorder:coordination"))
    if text.lower().startswith("although "):
        clause, _, main = text.partition(", ")
        out.append((f"{main[:1].upper()}{main[1:]} {clause[:1].lower()}{clause[1:]}.",
                    "reorder:subordination"))
    return out


def main() -> None:
    sentences = [s.strip() for s in BASES.read_text().splitlines() if s.strip()]
    rows = []
    for sentence, doc in zip(sentences, parse_batch(sentences)):
        for text, note in clause_reorders(doc) + synonym_swaps(doc):
            if text != sentence:
                rows.append({"base_sentence": sentence, "variant": text,
                             "family": "CONTROL_paraphrase", "subtype": note.split(":")[0],
                             "net_negation": 0, "intensity_hint": "neutral",
                             "operation": "paraphrase", "note": note})
    out = HERE / "data" / "control_candidates.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"{len(rows)} candidates -> {out}")
    for r in rows:
        print(f"  {r['note']:<34} {r['variant']}")


if __name__ == "__main__":
    main()
