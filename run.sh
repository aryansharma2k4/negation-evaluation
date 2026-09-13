#!/usr/bin/env bash
# Generate negation variants for a file of sentences.
#
#   ./run.sh                           # data/sample_sentences.txt -> variants.jsonl
#   ./run.sh my_sentences.txt          # your file -> variants.jsonl
#   ./run.sh my_sentences.txt out.jsonl
#   ./run.sh --quiet                   # JSONL only, no summary or preview
#   ./run.sh --verify                  # also run stage 2 (slow, opt-in)
#
# Generation does NOT use the LLM. Every record comes out with verified = null.
# Stage 2 is a separate pass, opt-in via --verify, for two reasons: it costs
# roughly 15 seconds per record on CPU, and as of the last validation run the
# verifier does not clear its own quality gate -- see
# reports/verifier_validation.md before trusting any verdict it produces.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=./venv/bin/python
[ -x "$PYTHON" ] || PYTHON=python3

QUIET=0
VERIFY=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --quiet|-q) QUIET=1 ;;
        --verify)   VERIFY=1 ;;
        *)          ARGS+=("$arg") ;;
    esac
done

INPUT="${ARGS[0]:-data/sample_sentences.txt}"
OUTPUT="${ARGS[1]:-variants.jsonl}"

if [ ! -f "$INPUT" ]; then
    echo "no such input file: $INPUT" >&2
    echo "usage: ./run.sh [sentences.txt] [out.jsonl] [--quiet] [--verify]" >&2
    exit 2
fi

run_verify() {
    local count minutes verified
    count=$(wc -l < "$OUTPUT")
    minutes=$(( count * 15 / 60 ))
    verified="${OUTPUT%.jsonl}_verified.jsonl"
    {
        echo
        echo "------------------------------------------------------------------"
        echo "Stage 2: verifying $count records with a local LLM."
        echo "  Expect roughly ${minutes} minutes on CPU. Ctrl-C is safe: verdicts"
        echo "  are cached per batch and a re-run resumes where it stopped."
        echo
        echo "  WARNING: the verifier has not passed its own validation gate."
        echo "  See reports/verifier_validation.md before trusting a verdict."
        echo "------------------------------------------------------------------"
        echo
    } >&2
    "$PYTHON" -m negation.verification.cli \
        --input "$OUTPUT" --output "$verified" --resume
}

if [ "$QUIET" -eq 1 ]; then
    "$PYTHON" -m negation.cli "$INPUT" -o "$OUTPUT" --quiet
    [ "$VERIFY" -eq 1 ] && run_verify
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

if [ "$VERIFY" -eq 1 ]; then
    run_verify
else
    echo "Not verified: every record has verified = null."
    echo "To run stage 2 as well:  ./run.sh $INPUT $OUTPUT --verify"
    echo
fi
