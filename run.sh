#!/bin/sh
# Generate negation variants. With no arguments, runs on data/sample_sentences.txt.
cd "$(dirname "$0")" || exit 1
exec ./venv/bin/python -m negation.cli "$@"
