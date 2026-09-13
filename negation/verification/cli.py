"""Command-line entry point for stage 2.

    python -m negation.verification.cli \\
        --input variants.jsonl --output variants_verified.jsonl \\
        --model qwen2.5:7b --cola-threshold 0.3 --batch-size 32 --resume

Prints the pass rate overall and per family, the tier-1 versus tier-2 rejection
split, the cache hit rate and wall time -- the numbers needed to decide whether
the run is worth repeating and where its cost went.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .cola import DEFAULT_MODEL as DEFAULT_COLA, DEFAULT_THRESHOLD
from .driver import verify
from .llm import DEFAULT_MODEL as DEFAULT_LLM
from .report import write_disagreement_report
from .schema import RunStats

DEFAULT_REPORT = Path("reports/verification_disagreements.md")


def print_summary(records: Sequence[dict], stats: RunStats, stream=sys.stdout) -> None:
    say = lambda text="": print(text, file=stream)  # noqa: E731

    say("\n" + "=" * 72)
    say("VERIFICATION SUMMARY")
    say("=" * 72)

    judged = stats.passed + stats.failed
    say(f"  records            : {stats.total:,}")
    say(f"  wall time          : {stats.seconds / 60:.1f} min")
    say("")
    say(f"  verified = true    : {stats.passed:,}"
        + (f"  ({stats.passed / judged:.1%} of judged)" if judged else ""))
    say(f"  verified = false   : {stats.failed:,}")
    say(f"  verified = null    : {stats.unjudged:,}"
        + (f"  ({stats.parse_failures:,} parse failures)" if stats.parse_failures else ""))
    say("")
    say("  rejections by tier:")
    say(f"    tier 1 (acceptability) : {stats.cola_rejected:,}")
    say(f"    tier 2 (LLM)           : {max(stats.failed - stats.cola_rejected, 0):,}")
    say("")
    say(f"  LLM calls          : {stats.llm_sent:,} records in {stats.batches:,} batches")
    say(f"  cache hits         : {stats.cache_hits:,} ({stats.cache_hit_rate:.1%})")
    if stats.retries or stats.single_fallbacks:
        say(f"  retries            : {stats.retries:,}")
        say(f"  single fallbacks   : {stats.single_fallbacks:,}")

    if stats.per_family:
        say("")
        say("  pass rate by family:")
        say(f"    {'family':<20} {'judged':>8} {'passed':>8} {'rate':>8}")
        say(f"    {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8}")
        for family, (total, passed) in sorted(
            stats.per_family.items(), key=lambda kv: -kv[1][0]
        ):
            rate = passed / total if total else 0.0
            say(f"    {family:<20} {total:>8,} {passed:>8,} {rate:>7.1%}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify",
        description="Stage 2: verify generated negation variants with a local LLM.",
    )
    parser.add_argument("--input", type=Path, default=Path("variants.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("variants_verified.jsonl"))
    parser.add_argument("--model", default=DEFAULT_LLM,
                        help=f"local LLM tag (default: {DEFAULT_LLM})")
    parser.add_argument("--cola-model", default=DEFAULT_COLA)
    parser.add_argument("--cola-threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"reject below this (default: {DEFAULT_THRESHOLD}; "
                             f"0 disables tier 1)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true",
                        help="leave records that already carry a verdict alone")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--cache", type=Path, default=Path(".verification_cache.sqlite"))
    parser.add_argument("--limit", type=int, help="verify only the first N records")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2

    log = (lambda *_: None) if args.quiet else print
    records, stats = verify(
        args.input,
        args.output,
        model=args.model,
        cola_threshold=args.cola_threshold,
        cola_model=args.cola_model,
        batch_size=args.batch_size,
        resume=args.resume,
        cache_path=args.cache,
        limit=args.limit,
        log=log,
    )

    if not args.quiet:
        print_summary(records, stats)

    summary = write_disagreement_report(
        records, args.report,
        model=f"ollama/{args.model}", corpus_size=len(records),
    )
    if not args.quiet:
        print(f"\nwrote {args.output}")
        print(f"wrote {args.report} "
              f"({summary['total']} miscategorised, {summary['clusters']} clusters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
