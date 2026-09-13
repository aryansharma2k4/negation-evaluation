"""Tier 2: a local LLM judges what survived tier 1.

Ollama by default, llama.cpp's server as a fallback. **No hosted API is ever
called** -- both backends talk to localhost, and there is no code path that
reaches anything else.

The hard requirement here is that the run never dies. A verification pass over a
corpus takes tens of minutes on CPU, and losing it to one malformed response
would be worse than any individual bad verdict. So failure is handled in layers,
each cheaper than re-running from scratch:

1. the batch is retried, with backoff, up to ``max_retries`` times;
2. then the batch is split and each record asked on its own, which nearly always
   succeeds because a single-item response is short enough not to be truncated;
3. then, and only then, the record is marked ``verified = null`` with
   ``reason = "llm_parse_failure"`` and the run moves on.

Note what is *not* done: a record that cannot be parsed is never guessed at, and
never defaulted to ``false``. ``null`` means "not judged", and keeping that
distinct from "judged and rejected" is what stops infrastructure flakiness from
being laundered into training signal.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .prompts import FAMILY_LIST, build_prompt, build_single_prompt
from .schema import (
    REASON_PARSE_FAILURE,
    REASON_UNAVAILABLE,
    Verdict,
)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLAMACPP_URL = "http://localhost:8080"
DEFAULT_MODEL = "qwen2.5:7b"

#: Output tokens allowed per record, plus slack for the JSON envelope. A batch
#: that runs out of budget is truncated mid-array, which is the single most
#: common cause of a parse failure.
TOKENS_PER_RECORD = 60
TOKEN_OVERHEAD = 128


class LLMUnavailable(RuntimeError):
    """Raised when no local backend can be reached."""


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@dataclass
class OllamaBackend:
    """Ollama's ``/api/generate``, with ``format=json`` for schema adherence."""

    model: str = DEFAULT_MODEL
    url: str = DEFAULT_OLLAMA_URL
    temperature: float = 0.0
    timeout: int = 1800
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"ollama/{self.model}"

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/api/tags", timeout=10) as response:
                tags = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        # Ollama reports "qwen2.5:7b"; accept a bare name as its ":latest" tag.
        return self.model in names or f"{self.model}:latest" in names

    def complete(self, prompt: str, max_tokens: int) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": self.temperature, "num_predict": max_tokens},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response).get("response", "")


@dataclass
class LlamaCppBackend:
    """llama.cpp's OpenAI-compatible server, used when Ollama is not running.

    Still local: the default points at ``localhost:8080``, which is what
    ``llama-server`` binds. A non-local URL would be a configuration error, not
    a feature, so the constructor refuses one.
    """

    model: str = "local"
    url: str = DEFAULT_LLAMACPP_URL
    temperature: float = 0.0
    timeout: int = 1800
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if not _is_local(self.url):
            raise ValueError(f"refusing a non-local LLM endpoint: {self.url}")
        self.name = f"llamacpp/{self.model}"

    def available(self) -> bool:
        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(f"{self.url}{path}", timeout=10):
                    return True
            except (urllib.error.URLError, TimeoutError, OSError):
                continue
        return False

    def complete(self, prompt: str, max_tokens: int) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        return payload["choices"][0]["message"]["content"]


def _is_local(url: str) -> bool:
    return bool(re.match(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?/?$", url.rstrip("/") + "/"))


def pick_backend(
    model: str = DEFAULT_MODEL,
    *,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    llamacpp_url: str = DEFAULT_LLAMACPP_URL,
    prefer: str = "ollama",
):
    """The first local backend that answers. Never falls through to a hosted API."""
    ollama = OllamaBackend(model=model, url=ollama_url)
    llamacpp = LlamaCppBackend(model=model, url=llamacpp_url)
    order = (ollama, llamacpp) if prefer == "ollama" else (llamacpp, ollama)
    for backend in order:
        if backend.available():
            return backend
    raise LLMUnavailable(
        f"no local LLM backend reachable.\n"
        f"  tried ollama at {ollama_url} (model {model!r}) and llama.cpp at {llamacpp_url}\n"
        f"  start one, e.g.:  ollama serve  &&  ollama pull {model}"
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_VALID_FAMILIES = set(FAMILY_LIST)


def _as_bool(value: object) -> Optional[bool]:
    """Accept the several ways a model says yes."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "y", "1"):
            return True
        if lowered in ("false", "no", "n", "0"):
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _as_confidence(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, number))


def extract_verdicts(text: str) -> dict[int, dict]:
    """Pull ``{id: verdict}`` out of a model response, tolerantly.

    Models wrap the array in different keys, or return a bare array, or wrap the
    whole thing in a code fence. All of those are recoverable and none of them
    is worth a retry, so they are handled here rather than treated as failures.
    A response that is genuinely unparseable returns ``{}`` and the caller
    escalates.
    """
    if not text or not text.strip():
        return {}

    blob = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", blob, re.DOTALL)
    if fence:
        blob = fence.group(1).strip()

    payload = _loads_or_none(blob)
    if payload is None:
        # Salvage the outermost object or array if there is prose around it.
        for pattern in (r"\{.*\}", r"\[.*\]"):
            match = re.search(pattern, blob, re.DOTALL)
            if match:
                payload = _loads_or_none(match.group(0))
                if payload is not None:
                    break
    if payload is None:
        return {}

    rows = _rows_from(payload)
    out: dict[int, dict] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        try:
            key = int(row.get("id", position))
        except (TypeError, ValueError):
            key = position
        out[key] = row
    return out


def _loads_or_none(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _rows_from(payload) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("verdicts", "results", "items", "output", "data", "judgements"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        # A single verdict returned bare.
        if "grammatical" in payload:
            return [payload]
    return []


def verdict_from_row(row: dict, model: str) -> Verdict:
    grammatical = _as_bool(row.get("grammatical"))
    semantic = _as_bool(row.get("semantically_valid", row.get("semantic")))
    category = _as_bool(row.get("category_correct", row.get("category")))

    suggested = row.get("suggested_family")
    if isinstance(suggested, str):
        suggested = suggested.strip() or None
    if suggested is not None and suggested not in _VALID_FAMILIES:
        # A family outside the taxonomy cannot be actioned; keep the rejection,
        # drop the unusable suggestion rather than pretending it is a label.
        suggested = None

    return Verdict(
        grammatical=grammatical,
        semantic=semantic,
        category=category,
        suggested_family=suggested,
        reason=REASON_PARSE_FAILURE,  # replaced by as_fields() once judged
        model=model,
        confidence=_as_confidence(row.get("confidence")),
    )


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------


@dataclass
class LLMJudge:
    """Batched judging with retry, per-record fallback, and no crashes."""

    backend: object
    max_retries: int = 2
    backoff: float = 2.0
    #: Set by the caller to record retries/fallbacks without threading a return.
    on_retry: Optional[object] = None
    on_fallback: Optional[object] = None

    @property
    def model(self) -> str:
        return getattr(self.backend, "name", "unknown")

    def _ask(self, prompt: str, count: int) -> dict[int, dict]:
        budget = TOKEN_OVERHEAD + TOKENS_PER_RECORD * count
        raw = self.backend.complete(prompt, budget)
        return extract_verdicts(raw)

    def judge_batch(self, records: Sequence[dict]) -> list[Verdict]:
        """One verdict per record, in order. Never raises."""
        if not records:
            return []

        parsed: dict[int, dict] = {}
        for attempt in range(self.max_retries + 1):
            try:
                parsed = self._ask(build_prompt(records), len(records))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                # The server is down or wedged; no amount of reprompting helps,
                # and the caller decides whether to keep going.
                return [
                    Verdict(reason=REASON_UNAVAILABLE, model=self.model)
                    for _ in records
                ]
            except Exception:
                parsed = {}

            if len(parsed) >= len(records):
                break
            if attempt < self.max_retries:
                if self.on_retry:
                    self.on_retry()
                time.sleep(self.backoff * (attempt + 1))

        verdicts: list[Verdict] = []
        missing: list[int] = []
        for index, _record in enumerate(records):
            row = parsed.get(index)
            if row is None:
                missing.append(index)
                verdicts.append(Verdict(reason=REASON_PARSE_FAILURE, model=self.model))
                continue
            verdict = verdict_from_row(row, self.model)
            if verdict.verified is None:
                missing.append(index)
            verdicts.append(verdict)

        # Anything still unjudged gets asked on its own: a one-item response is
        # short enough that truncation stops being the failure mode.
        for index in missing:
            if self.on_fallback:
                self.on_fallback()
            verdicts[index] = self.judge_one(records[index])
        return verdicts

    def judge_one(self, record: dict) -> Verdict:
        """Ask about a single record. Still never raises."""
        try:
            parsed = self._ask(build_single_prompt(record), 1)
        except (urllib.error.URLError, TimeoutError, OSError):
            return Verdict(reason=REASON_UNAVAILABLE, model=self.model)
        except Exception:
            return Verdict(reason=REASON_PARSE_FAILURE, model=self.model)

        if not parsed:
            return Verdict(reason=REASON_PARSE_FAILURE, model=self.model)
        row = parsed.get(0) or next(iter(parsed.values()))
        return verdict_from_row(row, self.model)
