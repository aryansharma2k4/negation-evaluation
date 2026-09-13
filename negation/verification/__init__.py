"""Stage 2: local-LLM verification of generated variants.

Reads stage 1's JSONL (every record carrying ``verified = null``), writes it
back with a verdict and the evidence behind it. Two tiers -- a cheap
acceptability filter, then a local LLM on whatever survives -- and no hosted API
at any point.

Stage 1's generator code is not modified. Records are handled as dicts, and the
verification fields are added to those dicts.
"""

from .schema import Verdict, VERIFICATION_FIELDS, record_key  # noqa: F401

__all__ = ["Verdict", "VERIFICATION_FIELDS", "record_key"]
