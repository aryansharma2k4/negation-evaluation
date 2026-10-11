"""Reference sets for architecture 2: run the stage-1 generator on every A.

    ./venv/bin/python scripts/negation_detection/refsets.py

For each sentence A the generator produces variants; each variant is sorted by
*net polarity relative to A*:

    NEG  : parity of the variant's net_negation differs from A's own cue parity
           -- the variant says the opposite of A
    SAME : everything else, plus A itself. K1 cancellations (net_negation = 0)
           and K2 compounds (net_negation = 2) land here by construction, which
           is the point of the test.

Writes data/refsets.jsonl, one record per unique A, and logs coverage failures
(an A the generator could not negate at all).
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from negation import generate_all  # noqa: E402

BATCH = 200


def unique_a() -> list[str]:
    seen: dict[str, None] = {}
    for name in ("phase1.jsonl", "phase2.jsonl", "xnot360.jsonl"):
        for line in (REPO / "datasets" / "splits" / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.setdefault(json.loads(line)["a"], None)
    return list(seen)


def main() -> None:
    sentences = unique_a()
    out = REPO / "datasets" / "generated" / "negation_detection" / "refsets.jsonl"
    stats = Counter()
    start = time.time()
    with out.open("w", encoding="utf-8") as handle:
        for i in range(0, len(sentences), BATCH):
            chunk = sentences[i : i + BATCH]
            try:
                records = generate_all(chunk)
            except Exception as exc:  # one bad sentence must not kill the run
                print(f"batch {i} failed ({exc}); falling back to one at a time")
                records = []
                for sentence in chunk:
                    try:
                        records.extend(generate_all([sentence]))
                    except Exception:
                        stats["generator_error"] += 1
            by_base: dict[str, list] = {s: [] for s in chunk}
            for record in records:
                by_base.setdefault(record.base_sentence, []).append(record)
            for sentence in chunk:
                rows = by_base.get(sentence, [])
                neg, same = [], []
                a_parity = rows[0].input_cue_count % 2 if rows else 0
                for r in rows:
                    flipped = (r.net_negation % 2) != (r.input_cue_count % 2)
                    (neg if flipped else same).append(
                        {"text": r.variant, "family": r.family, "subtype": r.subtype,
                         "net_negation": r.net_negation})
                stats["total"] += 1
                stats["no_neg"] += not neg
                stats["no_variants"] += not rows
                handle.write(json.dumps({
                    "a": sentence, "n_variants": len(rows), "a_cue_parity": a_parity,
                    "neg": neg, "same": same, "coverage_fail": not neg}) + "\n")
            done = min(i + BATCH, len(sentences))
            print(f"  {done}/{len(sentences)}  {time.time() - start:.0f}s", flush=True)
    print(f"wrote {out}")
    print(dict(stats))


if __name__ == "__main__":
    main()
