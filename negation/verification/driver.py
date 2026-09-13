"""Orchestration: read JSONL, filter, judge, write, count.

The order is the whole design. Tier 1 scores everything once (cheap, batched,
seconds for the full corpus), and only what survives is shown to the LLM at
roughly nine seconds a record. Cache lookups happen before *either*, so a re-run
touches neither model.

Output is written incrementally. A run that dies at record 400 of 455 leaves 400
verified records on disk plus a cache that makes the next run skip straight to
401 -- which is the behaviour ``--resume`` promises and the reason verdicts are
committed per batch rather than at the end.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from .cache import VerdictCache
from .cola import ColaScorer, ColaUnavailable, DEFAULT_THRESHOLD, load_scorer, threshold_sweep
from .llm import LLMJudge, LLMUnavailable, pick_backend
from .schema import (
    REASON_CACHED,
    RunStats,
    Verdict,
    rejected_by_cola,
)


def read_records(path: Path) -> list[dict]:
    """Load stage-1 JSONL. Malformed lines are skipped, not fatal."""
    out: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def write_records(records: Iterable[dict], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def already_verified(record: dict) -> bool:
    """True when a previous run left a usable verdict on this record."""
    return record.get("verified") is not None or record.get("verify_reason") is not None


def run_tier1(
    records: Sequence[dict],
    scorer: Optional[ColaScorer],
    threshold: float,
    stats: RunStats,
    log: Callable[[str], None] = print,
) -> tuple[list[dict], list[float]]:
    """Score every record, reject below ``threshold``, return the survivors.

    The sweep is printed whatever the threshold, because the point of the filter
    being configurable is that the operator can see what a different choice
    would have cost before making it.
    """
    if scorer is None:
        log("  tier 1: skipped (no acceptability model); everything goes to the LLM")
        return list(records), []

    started = time.time()
    scores = scorer.score([r.get("variant", "") for r in records])
    stats.cola_scored = len(scores)
    log(f"  tier 1: scored {len(scores):,} variants in {time.time() - started:.1f}s "
        f"on {scorer.device}")

    survivors: list[dict] = []
    for record, score in zip(records, scores):
        if score < threshold:
            record.update(rejected_by_cola(score, scorer.name))
            stats.cola_rejected += 1
        else:
            record["cola_score"] = score
            survivors.append(record)
    log(f"  tier 1: rejected {stats.cola_rejected:,} below {threshold:.2f}, "
        f"{len(survivors):,} go to the LLM")
    return survivors, scores


def run_tier2(
    records: Sequence[dict],
    judge: LLMJudge,
    cache: VerdictCache,
    batch_size: int,
    stats: RunStats,
    log: Callable[[str], None] = print,
    checkpoint: Optional[Callable[[], None]] = None,
) -> None:
    """Judge the survivors, in batches, writing each batch's verdicts through."""
    pending: list[dict] = []
    for record in records:
        cached = cache.get(record, judge.model)
        if cached is not None:
            record.update(cached.as_fields(record.get("cola_score")))
            record["verify_reason"] = f"{record['verify_reason']}|{REASON_CACHED}"
            stats.cache_hits += 1
        else:
            pending.append(record)

    if stats.cache_hits:
        log(f"  tier 2: {stats.cache_hits:,} verdicts served from cache")
    if not pending:
        return

    total_batches = (len(pending) + batch_size - 1) // batch_size
    log(f"  tier 2: {len(pending):,} records in {total_batches} batches of {batch_size}")

    for number, start in enumerate(range(0, len(pending), batch_size), start=1):
        chunk = pending[start : start + batch_size]
        began = time.time()
        verdicts = judge.judge_batch(chunk)

        for record, verdict in zip(chunk, verdicts):
            record.update(verdict.as_fields(record.get("cola_score")))
        cache.put_many(zip(chunk, verdicts))

        stats.batches += 1
        stats.llm_sent += len(chunk)
        elapsed = time.time() - began
        judged = sum(1 for v in verdicts if v.verified is not None)
        log(f"    batch {number}/{total_batches}: {judged}/{len(chunk)} judged "
            f"in {elapsed:.0f}s ({elapsed / max(len(chunk), 1):.1f}s/record)")
        if checkpoint:
            checkpoint()


def summarise(records: Sequence[dict], stats: RunStats) -> RunStats:
    for record in records:
        verified = record.get("verified")
        family = record.get("family", "?")
        if verified is True:
            stats.passed += 1
            stats.note_family(family, True)
        elif verified is False:
            stats.failed += 1
            stats.note_family(family, False)
        else:
            stats.unjudged += 1
            if str(record.get("verify_reason", "")).startswith("llm_parse_failure"):
                stats.parse_failures += 1
    return stats


def verify(
    input_path: Path,
    output_path: Path,
    *,
    model: str,
    cola_threshold: float = DEFAULT_THRESHOLD,
    cola_model: Optional[str] = None,
    batch_size: int = 32,
    resume: bool = True,
    cache_path: Optional[Path] = None,
    limit: Optional[int] = None,
    log: Callable[[str], None] = print,
) -> tuple[list[dict], RunStats]:
    """Run both tiers over ``input_path`` and write ``output_path``."""
    started = time.time()
    records = read_records(input_path)
    if limit:
        records = records[:limit]
    stats = RunStats(total=len(records))
    log(f"read {len(records):,} records from {input_path}")

    # Records that already carry a verdict are left exactly as they are.
    if resume:
        done = [r for r in records if already_verified(r)]
        todo = [r for r in records if not already_verified(r)]
        if done:
            log(f"  resume: {len(done):,} records already carry a verdict, left alone")
    else:
        done, todo = [], list(records)

    scorer = None
    if cola_threshold > 0:
        try:
            scorer = load_scorer(cola_model or "textattack/roberta-base-CoLA")
        except ColaUnavailable as exc:
            log(f"  tier 1 unavailable ({exc}); continuing without it")

    survivors, scores = run_tier1(todo, scorer, cola_threshold, stats, log)
    if scores:
        from .cola import format_sweep

        log("\n  what other thresholds would have cut:")
        log(format_sweep(threshold_sweep(scores)))
        log("")

    cache = VerdictCache(cache_path or ".verification_cache.sqlite", enabled=True)
    try:
        backend = pick_backend(model)
        judge = LLMJudge(backend)
        judge.on_retry = lambda: setattr(stats, "retries", stats.retries + 1)
        judge.on_fallback = lambda: setattr(
            stats, "single_fallbacks", stats.single_fallbacks + 1
        )
        log(f"  tier 2: {judge.model}")
        run_tier2(
            survivors, judge, cache, batch_size, stats, log,
            checkpoint=lambda: write_records(records, output_path),
        )
    except LLMUnavailable as exc:
        log(f"  tier 2 unavailable: {exc}")
    finally:
        cache.close()

    write_records(records, output_path)
    stats.seconds = time.time() - started
    return records, summarise(records, stats)
