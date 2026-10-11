"""Put the experiment module on the path so the test can import it directly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
