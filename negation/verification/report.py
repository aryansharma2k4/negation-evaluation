"""The disagreement report: generator bugs, with a return address.

A record the LLM marks ``category_correct = false`` while supplying a
``suggested_family`` is not a bad row to drop. It is a claim that a specific
generator put a specific kind of output under the wrong label, and when many
records make the same claim it stops being a judgement call and becomes a
reproducible defect.

So the report is grouped by ``(labelled_family -> suggested_family)`` and sorted
by volume. A cluster of twenty ``D_affixal -> E_antonym`` rows is one bug in the
affixal generator, not twenty bad sentences, and the size of the cluster is the
evidence that it is worth fixing at source.

Volume is reported alongside the share of the labelled family it represents,
because those answer different questions. Twenty records is a large cluster out
of a family of thirty and a rounding error out of a family of two thousand.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

#: Clusters shown in full.
TOP_CLUSTERS = 10
#: Example rows printed per cluster.
EXAMPLES_PER_CLUSTER = 4


def collect_disagreements(records: Sequence[dict]) -> list[dict]:
    """Every record the LLM judged miscategorised."""
    return [r for r in records if r.get("verify_category") is False]


def cluster(records: Sequence[dict]) -> list[dict]:
    """Group miscategorisations by ``(labelled -> suggested)``, largest first."""
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        labelled = record.get("family", "?")
        suggested = record.get("suggested_family") or "(none offered)"
        buckets[(labelled, suggested)].append(record)

    family_totals: Counter[str] = Counter()
    for record in records:
        family_totals[record.get("family", "?")] += 1

    out = []
    for (labelled, suggested), rows in buckets.items():
        out.append(
            {
                "labelled_family": labelled,
                "suggested_family": suggested,
                "count": len(rows),
                "generators": Counter(r.get("generator", "?") for r in rows),
                "subtypes": Counter(r.get("subtype", "?") for r in rows),
                "records": rows,
            }
        )
    out.sort(key=lambda c: -c["count"])
    return out


def write_disagreement_report(
    records: Sequence[dict], path: Path, *, model: str = "", corpus_size: int = 0
) -> dict:
    """Write ``reports/verification_disagreements.md``. Returns a small summary."""
    disagreements = collect_disagreements(records)
    clusters = cluster(disagreements)

    judged = [r for r in records if r.get("verify_category") is not None]
    family_totals: Counter[str] = Counter(r.get("family", "?") for r in judged)

    lines: list[str] = []
    lines.append("# Verification disagreements")
    lines.append("")
    lines.append(
        "Records the verifier judged to be in the **wrong family**. These are not "
        "bad sentences to discard -- they are claims that a generator is "
        "mislabelling its output, and a cluster of them is a bug with a return "
        "address."
    )
    lines.append("")
    lines.append(f"- verifier: `{model}`")
    lines.append(f"- records judged on category: {len(judged):,} of {corpus_size:,}")
    lines.append(
        f"- judged miscategorised: **{len(disagreements):,}** "
        f"({len(disagreements) / len(judged):.1%} of judged)" if judged else
        "- judged miscategorised: 0"
    )
    lines.append(f"- distinct (labelled -> suggested) clusters: {len(clusters)}")
    lines.append("")
    lines.append(
        "> The verifier is a 7B local model and is itself imperfect -- see "
        "`verifier_validation.md` for how far it should be trusted. Treat a "
        "cluster as a lead to investigate, not a proven defect."
    )
    lines.append("")

    if not clusters:
        lines.append("No miscategorisations were reported.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"total": 0, "clusters": 0}

    lines.append(f"## Top {min(TOP_CLUSTERS, len(clusters))} clusters by volume")
    lines.append("")
    lines.append("| # | labelled | suggested | count | share of labelled family | main generator |")
    lines.append("| ---: | --- | --- | ---: | ---: | --- |")
    for index, group in enumerate(clusters[:TOP_CLUSTERS], start=1):
        total = family_totals.get(group["labelled_family"], 0)
        share = f"{group['count'] / total:.1%}" if total else "—"
        top_generator = group["generators"].most_common(1)[0][0]
        lines.append(
            f"| {index} | `{group['labelled_family']}` | `{group['suggested_family']}` | "
            f"{group['count']} | {share} | `{top_generator}` |"
        )
    lines.append("")

    lines.append("## Clusters in detail")
    lines.append("")
    for index, group in enumerate(clusters[:TOP_CLUSTERS], start=1):
        total = family_totals.get(group["labelled_family"], 0)
        share = f"{group['count'] / total:.1%}" if total else "—"
        lines.append(
            f"### {index}. `{group['labelled_family']}` -> "
            f"`{group['suggested_family']}` ({group['count']} records, {share} of "
            f"`{group['labelled_family']}`)"
        )
        lines.append("")
        generators = ", ".join(f"`{g}` ({n})" for g, n in group["generators"].most_common(3))
        subtypes = ", ".join(f"`{s}` ({n})" for s, n in group["subtypes"].most_common(3))
        lines.append(f"- generators: {generators}")
        lines.append(f"- subtypes: {subtypes}")
        lines.append("")
        lines.append("| base | variant | subtype |")
        lines.append("| --- | --- | --- |")
        for record in group["records"][:EXAMPLES_PER_CLUSTER]:
            base = str(record.get("base_sentence", "")).replace("|", "\\|")
            variant = str(record.get("variant", "")).replace("|", "\\|")
            lines.append(f"| {base} | {variant} | `{record.get('subtype', '')}` |")
        if len(group["records"]) > EXAMPLES_PER_CLUSTER:
            lines.append(
                f"| … | *{len(group['records']) - EXAMPLES_PER_CLUSTER} more* | |"
            )
        lines.append("")

    if len(clusters) > TOP_CLUSTERS:
        lines.append(f"## Remaining {len(clusters) - TOP_CLUSTERS} clusters")
        lines.append("")
        lines.append("| labelled | suggested | count |")
        lines.append("| --- | --- | ---: |")
        for group in clusters[TOP_CLUSTERS:]:
            lines.append(
                f"| `{group['labelled_family']}` | `{group['suggested_family']}` | "
                f"{group['count']} |"
            )
        lines.append("")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "total": len(disagreements),
        "clusters": len(clusters),
        "top": [
            (c["labelled_family"], c["suggested_family"], c["count"])
            for c in clusters[:TOP_CLUSTERS]
        ],
    }
