#!/usr/bin/env python
"""Validate the verifier before its output is used as training signal.

    ./venv/bin/python scripts/verify_validate.py --sample 100 --constructed 60

Two reference sets, because they fail in different directions.

**Constructed** records are perturbed so the right answer follows from how they
were made -- two adjacent words swapped is ungrammatical whatever a reader
thinks; a family label replaced with an incompatible one is miscategorised by
definition. No annotator judgement is involved, so no annotator bias can leak
in. This is the stronger evidence.

**Labelled** records are annotated against the three criteria. Closer to the
real target, weaker guarantee: whoever produced the labels may share blind spots
with the verifier. The provenance of these labels is recorded in the report
rather than left to be assumed.

Run this first. The full pass is ~45 minutes of CPU inference, and there is no
point spending it on a verifier that turns out to agree with nothing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from negation.verification.cache import VerdictCache  # noqa: E402
from negation.verification.driver import read_records  # noqa: E402
from negation.verification.llm import DEFAULT_MODEL, LLMJudge, LLMUnavailable, pick_backend  # noqa: E402
from negation.verification.validate import (  # noqa: E402
    build_constructed_set,
    compare,
    index_by_key,
    load_labels,
    records_to_verify,
    save_labels,
)
from negation.verification.validate_report import write_validation_report  # noqa: E402

DEFAULT_REPORT = Path("reports/verifier_validation.md")
LABELS_PATH = Path("reports/verifier_labels.jsonl")
SAMPLE_PATH = Path("reports/verifier_label_sample.jsonl")


def judge_all(records, judge, cache, batch_size, log) -> list[dict]:
    """Run the verifier over ``records``, returning them with verdicts attached."""
    out = []
    pending = []
    for record in records:
        cached = cache.get(record, judge.model)
        if cached is not None:
            record.update(cached.as_fields(None))
            out.append(record)
        else:
            pending.append(record)

    log(f"  {len(out)} from cache, {len(pending)} to judge")
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        began = time.time()
        verdicts = judge.judge_batch(chunk)
        for record, verdict in zip(chunk, verdicts):
            record.update(verdict.as_fields(None))
        cache.put_many(zip(chunk, verdicts))
        out.extend(chunk)
        log(f"    {start + len(chunk)}/{len(pending)} "
            f"({time.time() - began:.0f}s for {len(chunk)})")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=Path("variants.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample", type=int, default=100,
                        help="records to draw for the labelled set")
    parser.add_argument("--constructed", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cache", type=Path, default=Path(".verification_cache.sqlite"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--emit-sample", action="store_true",
                        help="write the sample to be labelled, then stop")
    args = parser.parse_args(argv)

    records = read_records(args.input)
    if not records:
        print(f"no records in {args.input}", file=sys.stderr)
        return 2
    print(f"read {len(records):,} records from {args.input}")

    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.sample, len(records)))

    if args.emit_sample:
        SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SAMPLE_PATH.open("w", encoding="utf-8") as handle:
            for record in sample:
                handle.write(json.dumps({
                    "base_sentence": record["base_sentence"],
                    "variant": record["variant"],
                    "family": record["family"],
                    "subtype": record.get("subtype"),
                    "operation": record.get("operation"),
                }, ensure_ascii=False) + "\n")
        print(f"wrote {len(sample)} records to {SAMPLE_PATH} for labelling")
        return 0

    constructed = build_constructed_set(records, n=args.constructed, seed=args.seed)
    breakdown = Counter(row["kind"] for row in constructed)
    print(f"constructed set: {len(constructed)} records {dict(breakdown)}")

    try:
        backend = pick_backend(args.model)
    except LLMUnavailable as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    judge = LLMJudge(backend)
    print(f"verifier: {judge.model}")

    cache = VerdictCache(args.cache)
    try:
        print("\njudging the constructed set ...")
        judged_constructed = judge_all(
            records_to_verify(constructed), judge, cache, args.batch_size, print
        )
        constructed_agreements, constructed_matched = compare(
            constructed, index_by_key(judged_constructed)
        )

        labelled_agreements, labelled_matched, provenance = None, 0, ""
        if args.labels.exists():
            labels = load_labels(args.labels)
            print(f"\njudging the {len(labels)} labelled records ...")
            wanted = {row["key"] for row in labels}
            by_key = index_by_key(records)
            to_judge = [record for key, record in by_key.items() if key in wanted]
            judged_labelled = judge_all(to_judge, judge, cache, args.batch_size, print)
            labelled_agreements, labelled_matched = compare(
                labels, index_by_key(judged_labelled)
            )
            provenance = (labels[0].get("_provenance") or "") if labels else ""
        else:
            print(f"\nno labels at {args.labels}; skipping the labelled set")
    finally:
        cache.close()

    print("\n" + "=" * 72)
    print("VALIDATION")
    print("=" * 72)
    for group, name in ((constructed_agreements, "constructed"),
                        (labelled_agreements, "labelled")):
        if not group:
            continue
        print(f"  {name}:")
        for item in group:
            if item.n:
                gate = "pass" if item.passes_gate else "FAIL"
                print(f"    {item.question:<14} n={item.n:<4} acc={item.accuracy:.3f} "
                      f"kappa={item.kappa:+.3f}  {gate}")

    passed = write_validation_report(
        args.report,
        model=judge.model,
        labelled=labelled_agreements,
        labelled_n=labelled_matched,
        labelled_provenance=provenance,
        constructed=constructed_agreements,
        constructed_n=constructed_matched,
        constructed_breakdown=dict(breakdown),
        notes=NOTES,
    )
    print(f"\nwrote {args.report}")
    print("gate: " + ("PASSED" if passed else "NOT PASSED — see the report"))
    return 0


NOTES = (
    "The constructed set can only assert the labels its perturbations determine. "
    "A clean record is asserted grammatical but not semantically valid or "
    "correctly categorised, because asserting those would be assuming the "
    "generator is right, which is the thing under test.",
    "Cohen's kappa on a skewed binary label is unstable at small n. Treat a "
    "kappa computed over fewer than ~30 records as indicative only.",
    "Agreement is not correctness. Both reference sets could be wrong in the "
    "same direction as the verifier, and a high kappa would not reveal it.",
)


if __name__ == "__main__":
    raise SystemExit(main())
