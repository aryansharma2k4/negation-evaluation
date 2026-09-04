"""Command-line entry point.

    python -m negation.cli                       # runs on data/sample_sentences.txt
    python -m negation.cli sentences.txt -o out.jsonl

Reads newline-delimited base sentences, writes one JSON record per line, and
prints a per-family / per-subtype summary plus the filter drop counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .driver import generate_all, summarize
from .filters import FilterReport
from .schema import NegationVariant

#: Used when no input file is given, so the CLI runs out of the box.
DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "data" / "sample_sentences.txt"


def read_sentences(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def write_jsonl(records: Sequence[NegationVariant], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")


def print_summary(
    records: Sequence[NegationVariant], report: FilterReport, stream=sys.stdout
) -> None:
    counts = summarize(records)
    print(f"\nvariants kept: {len(records)}", file=stream)

    print("\nby family:", file=stream)
    for family, n in sorted(counts["family"].items()):
        print(f"  {family:<18} {n:>6}", file=stream)

    print("\nby family/subtype:", file=stream)
    for key, n in sorted(counts["subtype"].items()):
        print(f"  {key:<44} {n:>6}", file=stream)

    print("\nby net_negation:", file=stream)
    for net, n in sorted(counts["net_negation"].items()):
        print(f"  {net:>2} {n:>6}", file=stream)

    if report.dropped:
        print("\nfiltered out:", file=stream)
        for reason, n in sorted(report.dropped.items()):
            print(f"  {reason:<28} {n:>6}", file=stream)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="negation",
        description="Generate negated variants of affirmative base sentences.",
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"file of newline-delimited sentences (default: {DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("variants.jsonl"),
        help="JSONL output path (default: variants.jsonl)",
    )
    parser.add_argument(
        "--global-dedup",
        action="store_true",
        help=(
            "deduplicate on surface form alone; by default two families that "
            "derive the same string each keep their own record"
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress the summary report"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"input file not found: {args.input}", file=sys.stderr)
        return 2

    sentences = read_sentences(args.input)
    if not sentences:
        print(f"no sentences in {args.input}", file=sys.stderr)
        return 2

    report = FilterReport()
    records = generate_all(
        sentences, per_family_dedup=not args.global_dedup, report=report
    )
    write_jsonl(records, args.output)

    if not args.quiet:
        print(f"read {len(sentences)} base sentences from {args.input}")
        print(f"wrote {len(records)} variants to {args.output}")
        print_summary(records, report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
