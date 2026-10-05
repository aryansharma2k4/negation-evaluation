"""Make the experiment modules importable from the test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
