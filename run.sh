#!/usr/bin/env bash
# Generate negation variants for a file of sentences.
#
#   ./run.sh                           # data/sample_sentences.txt -> variants.jsonl
#   ./run.sh my_sentences.txt          # your file -> variants.jsonl
#   ./run.sh my_sentences.txt out.jsonl
#   ./run.sh --quiet                   # JSONL only, no summary or preview
#
# Writes one JSON record per variant, and prints a readable base -> variant
# preview so you can see what happened without opening the JSONL.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=./venv/bin/python
[ -x "$PYTHON" ] || PYTHON=python3

QUIET=0
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--quiet" ] || [ "$arg" = "-q" ]; then QUIET=1; else ARGS+=("$arg"); fi
done

INPUT="${ARGS[0]:-data/sample_sentences.txt}"
OUTPUT="${ARGS[1]:-variants.jsonl}"

if [ ! -f "$INPUT" ]; then
    echo "no such input file: $INPUT" >&2
    echo "usage: ./run.sh [sentences.txt] [out.jsonl]" >&2
    exit 2
fi

if [ "$QUIET" -eq 1 ]; then
    "$PYTHON" -m negation.cli "$INPUT" -o "$OUTPUT" --quiet
    exit 0
fi

"$PYTHON" -m negation.cli "$INPUT" -o "$OUTPUT"

# The summary above counts families; this shows the sentences themselves, which
# is what you actually want to look at when checking a new input file.
PREVIEW_PER_SENTENCE="${PREVIEW:-6}" "$PYTHON" - "$OUTPUT" <<'PY'
import json, os, sys
from collections import defaultdict

limit = int(os.environ.get("PREVIEW_PER_SENTENCE", "6"))
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]

grouped = defaultdict(list)
for row in rows:
    grouped[row["base_sentence"]].append(row)

print("\n" + "=" * 78)
print(f"VARIANTS  (showing up to {limit} per sentence; all {len(rows)} are in {sys.argv[1]})")
print("=" * 78)
for base, variants in grouped.items():
    print(f"\n  {base}    [{len(variants)} variants]")
    for row in variants[:limit]:
        tag = f"{row['family']}/{row['subtype']}"
        print(f"      {tag:<34} {row['variant']}")
    if len(variants) > limit:
        print(f"      {'...':<34} and {len(variants) - limit} more")
print()
PY
