"""The verification record: what tier 1 and tier 2 add to a generated variant.

Stage 1 emits records with ``verified = null``. This layer fills that in, plus
the evidence behind it, and writes the result back out as JSONL.

**Generator code is not touched.** Records are read as plain dicts and the
verification fields are added to those dicts, rather than extending
:class:`negation.schema.NegationVariant`. The generator's schema describes what
a generator produced; verification is a separate, later, and revisable judgement
about it, and keeping the two apart means a re-run of verification cannot
invalidate a stage-1 artefact.

``verified`` is deliberately three-valued:

``True``   all three checks passed.
``False``  at least one check failed, or the grammaticality filter rejected it.
``None``   the verifier could not reach a judgement -- a parse failure, a model
           that never answered. This is *not* the same as a rejection, and
           collapsing it into ``False`` would silently turn infrastructure
           flakiness into training signal.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# -- reasons -----------------------------------------------------------------

#: Tier 1 dropped it on acceptability, before any LLM saw it.
REASON_COLA_REJECT = "cola_reject"
#: All three LLM checks passed.
REASON_PASS = "pass"
#: The LLM answered, and at least one check was false. Suffixed with which.
REASON_LLM_REJECT = "llm_reject"
#: The LLM never returned parseable JSON for this record, even alone.
REASON_PARSE_FAILURE = "llm_parse_failure"
#: The model or server could not be reached.
REASON_UNAVAILABLE = "llm_unavailable"
#: Present in the input already carrying a verdict, and --resume was set.
REASON_CACHED = "cached"

#: Fields this layer adds. Kept as a constant so the writer, the reader and the
#: tests agree on exactly what verification owns.
VERIFICATION_FIELDS: tuple[str, ...] = (
    "verified",
    "verify_grammatical",
    "verify_semantic",
    "verify_category",
    "suggested_family",
    "verify_reason",
    "verify_model",
    "verify_confidence",
    "cola_score",
)


def record_key(record: dict) -> str:
    """Cache key: ``sha256(base_sentence + variant + family)``.

    Not the whole record: two records that differ only in, say, ``generator``
    ask the LLM exactly the same question, and re-asking it would be waste. The
    three fields that are hashed are precisely the ones the prompt varies on
    that change the answer.
    """
    payload = "\x1f".join(
        (
            record.get("base_sentence", ""),
            record.get("variant", ""),
            record.get("family", ""),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class Verdict:
    """One LLM judgement, or the absence of one."""

    grammatical: Optional[bool] = None
    semantic: Optional[bool] = None
    category: Optional[bool] = None
    suggested_family: Optional[str] = None
    reason: str = REASON_PARSE_FAILURE
    model: str = ""
    confidence: Optional[float] = None

    @property
    def verified(self) -> Optional[bool]:
        """``True`` only if all three checks passed; ``None`` if unjudged."""
        if self.grammatical is None or self.semantic is None or self.category is None:
            return None
        return bool(self.grammatical and self.semantic and self.category)

    def failures(self) -> list[str]:
        named = (
            ("grammatical", self.grammatical),
            ("semantic", self.semantic),
            ("category", self.category),
        )
        return [name for name, value in named if value is False]

    def as_fields(self, cola_score: Optional[float] = None) -> dict[str, Any]:
        """The fields this verdict contributes to an output record."""
        verified = self.verified
        reason = self.reason
        if reason == REASON_PARSE_FAILURE and verified is not None:
            failed = self.failures()
            reason = REASON_PASS if not failed else f"{REASON_LLM_REJECT}:{'+'.join(failed)}"
        return {
            "verified": verified,
            "verify_grammatical": self.grammatical,
            "verify_semantic": self.semantic,
            "verify_category": self.category,
            "suggested_family": self.suggested_family,
            "verify_reason": reason,
            "verify_model": self.model,
            "verify_confidence": self.confidence,
            "cola_score": cola_score,
        }

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> "Verdict":
        return cls(**{k: blob.get(k) for k in cls.__dataclass_fields__})


def rejected_by_cola(score: float, model: str) -> dict[str, Any]:
    """Output fields for a record tier 1 rejected.

    The LLM checks are ``None`` rather than ``False``: the record was never
    shown to the LLM, so claiming it failed a semantic check would be inventing
    evidence. Only ``verified`` is committed to.
    """
    return {
        "verified": False,
        "verify_grammatical": False,
        "verify_semantic": None,
        "verify_category": None,
        "suggested_family": None,
        "verify_reason": REASON_COLA_REJECT,
        "verify_model": model,
        "verify_confidence": None,
        "cola_score": score,
    }


@dataclass
class RunStats:
    """Counters the CLI summary is built from."""

    total: int = 0
    cola_scored: int = 0
    cola_rejected: int = 0
    llm_sent: int = 0
    cache_hits: int = 0
    passed: int = 0
    failed: int = 0
    unjudged: int = 0
    parse_failures: int = 0
    batches: int = 0
    retries: int = 0
    single_fallbacks: int = 0
    seconds: float = 0.0
    per_family: dict[str, list[int]] = field(default_factory=dict)

    def note_family(self, family: str, passed: bool) -> None:
        counts = self.per_family.setdefault(family, [0, 0])
        counts[0] += 1
        if passed:
            counts[1] += 1

    @property
    def cache_hit_rate(self) -> float:
        seen = self.cache_hits + self.llm_sent
        return self.cache_hits / seen if seen else 0.0
