"""Render ``reports/verifier_validation.md``.

Kept apart from :mod:`.validate` so the measurement and its presentation can be
changed independently -- and so the gate logic has exactly one home.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .validate import KAPPA_GATE, Agreement


def _interpret(kappa: float) -> str:
    """Landis & Koch's conventional bands, named rather than implied."""
    if kappa < 0.0:
        return "worse than chance"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"


def _table(agreements: Sequence[Agreement]) -> list[str]:
    lines = [
        "| question | n | accuracy | Cohen's kappa | strength | caught known-bad | gate (k >= 0.6) |",
        "| --- | ---: | ---: | ---: | --- | ---: | :---: |",
    ]
    for item in agreements:
        if item.n == 0:
            lines.append(
                f"| `{item.question}` | 0 | — | — | not measured | — | n/a |"
            )
            continue
        caught = item.detection_rate
        caught_text = "—" if caught is None else f"{caught:.0%}"
        if not item.measurable:
            # Reference used one class only; kappa is undefined, not failed.
            lines.append(
                f"| `{item.question}` | {item.n} | {item.accuracy:.3f} | undefined | "
                f"reference has no variance | {caught_text} | n/a |"
            )
            continue
        mark = "pass" if item.passes_gate else "**FAIL**"
        lines.append(
            f"| `{item.question}` | {item.n} | {item.accuracy:.3f} | {item.kappa:.3f} | "
            f"{_interpret(item.kappa)} | {caught_text} | {mark} |"
        )
    return lines


def _confusion(item: Agreement) -> list[str]:
    lines = [
        f"**`{item.question}`** — reference true: {item.gold_true}/{item.n}, "
        f"verifier true: {item.pred_true}/{item.n}",
        "",
        "| | verifier says true | verifier says false |",
        "| --- | ---: | ---: |",
    ]
    for gold_value, label in ((True, "reference true"), (False, "reference false")):
        yes = item.confusion.get((gold_value, True), 0)
        no = item.confusion.get((gold_value, False), 0)
        lines.append(f"| **{label}** | {yes} | {no} |")
    lines.append("")
    return lines


def write_validation_report(
    path: Path,
    *,
    model: str,
    labelled: Optional[Sequence[Agreement]] = None,
    labelled_n: int = 0,
    labelled_provenance: str = "",
    constructed: Optional[Sequence[Agreement]] = None,
    constructed_n: int = 0,
    constructed_breakdown: Optional[dict] = None,
    notes: Sequence[str] = (),
) -> bool:
    """Write the report. Returns whether every measured question cleared the gate."""
    lines: list[str] = []
    lines.append("# Verifier validation")
    lines.append("")
    lines.append(
        "The verifier's output is only worth using as training signal if the "
        "verifier itself has been checked. This measures agreement between it "
        "and two reference sets, per question."
    )
    lines.append("")
    lines.append(f"- verifier under test: `{model}`")
    lines.append(f"- gate: Cohen's kappa >= {KAPPA_GATE} on every question")
    lines.append("")
    lines.append(
        "**Kappa, not accuracy, is the number to read.** These labels are heavily "
        "skewed -- most generated variants really are grammatical -- and on a set "
        "that is 90% one class a verifier answering \"true\" unconditionally scores "
        "0.90 accuracy while being worthless. Kappa corrects for that."
    )
    lines.append("")

    all_pass = True

    if constructed:
        lines.append("## 1. Constructed reference set (ground truth by construction)")
        lines.append("")
        lines.append(
            "The stronger of the two. Records were perturbed so that the correct "
            "answer follows from *how they were made* rather than from anyone's "
            "judgement: a variant with two adjacent words swapped is ungrammatical "
            "whatever a reader thinks, and a record whose family label has been "
            "replaced with an incompatible one is miscategorised by definition. No "
            "annotator is involved, so no annotator bias can leak in."
        )
        lines.append("")
        if constructed_breakdown:
            lines.append("Composition:")
            lines.append("")
            for kind, count in sorted(constructed_breakdown.items()):
                lines.append(f"- `{kind}`: {count}")
            lines.append("")
            lines.append(
                "The untouched group matters as much as the perturbed ones: without "
                "it, a verifier that rejects everything would score perfectly."
            )
            lines.append("")
        lines.append(f"Records matched: {constructed_n}")
        lines.append("")
        lines.extend(_table(constructed))
        lines.append("")
        for item in constructed:
            if item.n:
                lines.extend(_confusion(item))
        all_pass &= all(i.passes_gate for i in constructed if i.n)

    if labelled:
        lines.append("## 2. Hand-labelled reference set")
        lines.append("")
        if labelled_provenance:
            lines.append(labelled_provenance)
            lines.append("")
        lines.append(f"Records matched: {labelled_n}")
        lines.append("")
        lines.extend(_table(labelled))
        lines.append("")
        for item in labelled:
            if item.n:
                lines.extend(_confusion(item))
        all_pass &= all(i.passes_gate for i in labelled if i.n)

    lines.append("## Verdict")
    lines.append("")
    measured = [
        i for group in (constructed or [], labelled or []) for i in group if i.measurable
    ]
    unmeasured = [
        i
        for group in (constructed or [], labelled or [])
        for i in group
        if not i.measurable
    ]
    failing = [i for i in measured if not i.passes_gate]
    if unmeasured:
        lines.append(
            "Not every question could be measured. Where the reference set used a "
            "single class throughout, kappa is undefined rather than zero, and the "
            "table says so instead of recording a failure: "
            + ", ".join(f"`{i.question}` (n={i.n})" for i in unmeasured)
            + ". For those, read the *caught known-bad* column instead."
        )
        lines.append("")
    if not measured:
        lines.append("No questions could be measured; the verifier is **unvalidated**.")
    elif failing:
        lines.append(
            "**The verifier does not clear the gate.** Questions below kappa "
            f"{KAPPA_GATE}: "
            + ", ".join(f"`{i.question}` (k = {i.kappa:.3f})" for i in failing)
            + "."
        )
        lines.append("")
        lines.append(
            "Its output should not be used as training signal on those questions "
            "until the prompt is improved and this is re-run."
        )
    else:
        lines.append(
            "**The verifier clears the gate on every measured question.** Its "
            "output can be used as training signal, with the usual caveat that "
            "agreement is not correctness."
        )

    if notes:
        lines.append("")
        lines.append("## Notes and limitations")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return all_pass
