"""On-disk verdict cache, so a re-run costs nothing and an interruption costs one batch.

SQLite rather than JSONL. A JSONL cache has to be rewritten or compacted to
update a row, and a run killed mid-write leaves a truncated final line that the
next run has to detect and discard. SQLite gives durable per-row upserts and a
crash-safe file for free, which is exactly the property ``--resume`` needs.

Verdicts are keyed by ``sha256(base_sentence + variant + family)`` and stamped
with the model that produced them. The model is part of the *value*, not the
key, and is checked on read: a cached verdict from a different model is a miss,
because the whole point of recording ``verify_model`` is that verdicts are not
interchangeable across models.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional

from .schema import Verdict, record_key

DEFAULT_CACHE = Path(".verification_cache.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verdicts (
    key         TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    verdict     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS verdicts_model ON verdicts (model);
"""


class VerdictCache:
    """Keyed store of LLM verdicts. Safe to open concurrently and to interrupt."""

    def __init__(self, path: Path | str = DEFAULT_CACHE, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._conn: Optional[sqlite3.Connection] = None
        if enabled:
            self._conn = sqlite3.connect(str(self.path))
            # WAL keeps a reader working while a writer commits, and survives a
            # kill -9 without corrupting the file.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def get(self, record: dict, model: str) -> Optional[Verdict]:
        if not self.enabled or self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT verdict FROM verdicts WHERE key = ? AND model = ?",
            (record_key(record), model),
        ).fetchone()
        if row is None:
            return None
        try:
            return Verdict.from_json(json.loads(row[0]))
        except (json.JSONDecodeError, TypeError):  # pragma: no cover - corrupt row
            return None

    def put(self, record: dict, verdict: Verdict) -> None:
        """Store one verdict. Unjudged verdicts are never cached.

        Caching a parse failure would make it permanent: the next run would read
        it back and never retry, turning a transient malformed response into a
        record that can never be verified.
        """
        if not self.enabled or self._conn is None or verdict.verified is None:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO verdicts (key, model, verdict, created_at) "
            "VALUES (?, ?, ?, ?)",
            (record_key(record), verdict.model, json.dumps(verdict.to_json()), time.time()),
        )
        self._conn.commit()

    def put_many(self, pairs: Iterable[tuple[dict, Verdict]]) -> int:
        """Store a batch in one transaction. Returns how many were written."""
        if not self.enabled or self._conn is None:
            return 0
        rows = [
            (record_key(r), v.model, json.dumps(v.to_json()), time.time())
            for r, v in pairs
            if v.verified is not None
        ]
        if rows:
            self._conn.executemany(
                "INSERT OR REPLACE INTO verdicts (key, model, verdict, created_at) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def count(self, model: Optional[str] = None) -> int:
        if not self.enabled or self._conn is None:
            return 0
        if model is None:
            return int(self._conn.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0])
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM verdicts WHERE model = ?", (model,)
            ).fetchone()[0]
        )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "VerdictCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
