#!/usr/bin/env bash
# Run the test suite.
#
#   ./run_test.sh              # everything (~1 min)
#   ./run_test.sh fast         # skip anything needing downloaded models/corpora
#   ./run_test.sh verify       # stage 2 only, no model needed (~1 s)
#   ./run_test.sh -v           # verbose; reads as a list of what is guaranteed
#   ./run_test.sh -k classify  # any other pytest arguments pass straight through
#
# Tests that need a resource which is not present (word vectors, the third-party
# corpora) skip with a message rather than failing, so a clean checkout is green.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=./venv/bin/python
[ -x "$PYTHON" ] || PYTHON=python3

case "${1:-all}" in
    fast)
        shift
        echo "running: logic only (no vectors, no corpora, no LLM)"
        exec "$PYTHON" -m pytest tests -q \
            --ignore=tests/test_antonym_vec.py \
            --ignore=tests/test_data_loaders.py "$@"
        ;;
    verify)
        shift
        echo "running: stage 2 verification logic (fake backend, no model)"
        exec "$PYTHON" -m pytest tests/test_verification.py -q "$@"
        ;;
    all)
        shift || true
        exec "$PYTHON" -m pytest tests -q "$@"
        ;;
    *)
        # Anything else is pytest's business, e.g. -v, -k, a path.
        exec "$PYTHON" -m pytest tests -q "$@"
        ;;
esac
