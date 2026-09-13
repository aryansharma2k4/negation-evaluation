"""Validating the verifier itself, before its output is used as training signal.

A verifier that has not been checked is just a second opinion with extra steps.
This module measures agreement between the verifier and a reference set, per
question, reporting accuracy and Cohen's kappa.

**Kappa, not accuracy, is the number that matters.** These labels are heavily
skewed -- most generated variants really are grammatical -- and on a set that is
90% one class, a verifier that answers "true" unconditionally scores 90%
accuracy and is worth nothing. Kappa corrects for exactly that agreement-by-
chance, which is why the < 0.6 gate is stated on kappa.

Two reference sets are supported, and they answer different questions.

``labelled``     records annotated by hand against the three criteria. Closest
                 to the real target, and the weakest guarantee: whoever produced
                 them may share blind spots with the verifier.
``constructed``  records perturbed so the correct answer is known *by
                 construction* -- a variant with a deliberately corrupted word
                 order is ungrammatical whatever anyone thinks, and a record
                 whose family label has been swapped for an incompatible one is
                 miscategorised by definition. No annotator judgement is
                 involved, so no annotator bias can leak in.

The constructed set is the stronger evidence and the labelled set is the more
realistic one. Reporting only one of them would be reporting half the picture.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .schema import record_key

#: Below this, the brief says the prompt needs work before a full run.
KAPPA_GATE = 0.6

#: The three questions, and the record field each verifier answer lands in.
QUESTIONS: tuple[tuple[str, str], ...] = (
    ("grammatical", "verify_grammatical"),
    ("semantic", "verify_semantic"),
    ("category", "verify_category"),
)


@dataclass
class Agreement:
    """Agreement on one question."""

    question: str
    n: int
    agreed: int
    kappa: float
    gold_true: int
    pred_true: int
    #: (gold, predicted) -> count, for the 2x2 confusion.
    confusion: dict

    @property
    def accuracy(self) -> float:
        return self.agreed / self.n if self.n else 0.0

    @property
    def measurable(self) -> bool:
        """Whether kappa means anything here.

        Kappa is undefined when the reference set used only one class: expected
        agreement is 1.0 and the statistic collapses. Reporting that as a kappa
        of 0.0 and a failed gate conflates "we measured this and it was bad"
        with "we could not measure this", which are different problems with
        different fixes -- the first needs a better verifier, the second needs a
        better reference set.
        """
        return self.n > 0 and 0 < self.gold_true < self.n

    @property
    def detection_rate(self) -> Optional[float]:
        """Share of genuinely-false items the verifier caught.

        The right statistic when the reference is all-false by construction, as
        the family-swapped group is: kappa cannot be computed, but "how many of
        the 20 deliberate errors did it spot" is still exactly what we want to
        know.
        """
        negatives = self.n - self.gold_true
        if negatives == 0:
            return None
        return self.confusion.get((False, False), 0) / negatives

    @property
    def passes_gate(self) -> bool:
        return self.measurable and self.kappa >= KAPPA_GATE


def cohens_kappa(gold: Sequence[bool], predicted: Sequence[bool]) -> float:
    """Cohen's kappa for two binary label sequences.

    Returns 0.0 when one rater used a single class throughout: expected
    agreement is then 1.0 and kappa is undefined. Reporting 0.0 rather than
    raising is deliberate -- a verifier that answers the same way every time is
    exactly the failure kappa exists to catch, and it should show up as "no
    agreement beyond chance", not as a crash.
    """
    n = len(gold)
    if n == 0 or n != len(predicted):
        return 0.0

    observed = sum(1 for g, p in zip(gold, predicted) if g == p) / n
    gold_true = sum(1 for g in gold if g) / n
    pred_true = sum(1 for p in predicted if p) / n
    expected = gold_true * pred_true + (1 - gold_true) * (1 - pred_true)
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def compare(
    gold_records: Sequence[dict], verified_by_key: dict[str, dict]
) -> tuple[list[Agreement], int]:
    """Agreement per question, plus how many reference records were matched."""
    pairs: dict[str, list[tuple[bool, bool]]] = {name: [] for name, _ in QUESTIONS}
    matched = 0

    for gold in gold_records:
        key = gold.get("key") or record_key(gold)
        predicted = verified_by_key.get(key)
        if predicted is None:
            continue
        matched += 1
        for name, field in QUESTIONS:
            gold_value = gold.get(name)
            pred_value = predicted.get(field)
            if isinstance(gold_value, bool) and isinstance(pred_value, bool):
                pairs[name].append((gold_value, pred_value))

    out: list[Agreement] = []
    for name, _field in QUESTIONS:
        rows = pairs[name]
        golds = [g for g, _ in rows]
        preds = [p for _, p in rows]
        confusion: dict = {}
        for g, p in rows:
            confusion[(g, p)] = confusion.get((g, p), 0) + 1
        out.append(
            Agreement(
                question=name,
                n=len(rows),
                agreed=sum(1 for g, p in rows if g == p),
                kappa=cohens_kappa(golds, preds),
                gold_true=sum(1 for g in golds if g),
                pred_true=sum(1 for p in preds if p),
                confusion=confusion,
            )
        )
    return out, matched


# ---------------------------------------------------------------------------
# Constructed reference set
# ---------------------------------------------------------------------------

#: Family swaps that are unambiguously wrong: the surface form of the variant
#: cannot belong to the substituted family, so "category_correct = false" is
#: true by construction rather than by opinion.
_INCOMPATIBLE_FAMILY = {
    "A_syntactic": "G_prepositional",
    "B_quantifier": "D_affixal",
    "C_neg_adverb": "G_prepositional",
    "D_affixal": "B_quantifier",
    "E_antonym": "G_prepositional",
    "F_implicit": "B_quantifier",
    "G_prepositional": "C_neg_adverb",
    "H_hedged": "G_prepositional",
    "I_intensified": "G_prepositional",
    "J_contrastive": "B_quantifier",
    "K1_cancellation": "G_prepositional",
    "K2_compound": "G_prepositional",
    "L_scope_position": "G_prepositional",
    "M_modal": "G_prepositional",
}


def corrupt_grammar(sentence: str, rng: random.Random) -> Optional[str]:
    """Make a sentence ungrammatical in a way no reading rescues.

    Swapping two adjacent mid-sentence words is used rather than deleting or
    duplicating one: deletion often leaves a shorter grammatical sentence, and a
    duplicated function word can read as a stutter rather than an error.
    """
    words = sentence.split()
    if len(words) < 5:
        return None
    index = rng.randrange(1, len(words) - 2)
    words[index], words[index + 1] = words[index + 1], words[index]
    swapped = " ".join(words)
    return None if swapped == sentence else swapped


def build_constructed_set(
    records: Sequence[dict], n: int = 60, seed: int = 0
) -> list[dict]:
    """Reference records whose correct answers follow from how they were made.

    Three equal groups: untouched records (expected to pass on all three
    questions), grammar-corrupted records (``grammatical`` must be false), and
    family-swapped records (``category`` must be false while ``grammatical``
    stays true). The untouched group is what keeps the set from rewarding a
    verifier that simply rejects everything.
    """
    rng = random.Random(seed)
    pool = [r for r in records if r.get("variant") and r.get("family")]
    rng.shuffle(pool)

    per_group = max(1, n // 3)
    out: list[dict] = []

    for record in pool[:per_group]:
        out.append(
            {
                "key": record_key(record),
                "kind": "clean",
                "base_sentence": record["base_sentence"],
                "variant": record["variant"],
                "family": record["family"],
                # A clean record should pass; grammaticality is the only one we
                # can assert confidently without judging the generator, so the
                # other two are left unlabelled rather than assumed.
                "grammatical": True,
                "semantic": None,
                "category": None,
                # The untouched record itself, so the clean group reaches the
                # verifier with the same context (subtype, operation) a real
                # record would. Without it the clean group is quietly given a
                # thinner prompt than the perturbed groups, and the comparison
                # stops being like-for-like.
                "_record": dict(record),
            }
        )

    for record in pool[per_group : per_group * 2]:
        broken = corrupt_grammar(record["variant"], rng)
        if broken is None:
            continue
        perturbed = dict(record)
        perturbed["variant"] = broken
        out.append(
            {
                "key": record_key(perturbed),
                "kind": "grammar_corrupted",
                "base_sentence": record["base_sentence"],
                "variant": broken,
                "family": record["family"],
                "grammatical": False,
                "semantic": None,
                "category": None,
                "_record": perturbed,
            }
        )

    for record in pool[per_group * 2 : per_group * 3]:
        wrong = _INCOMPATIBLE_FAMILY.get(record["family"])
        if wrong is None or wrong == record["family"]:
            continue
        perturbed = dict(record)
        perturbed["family"] = wrong
        out.append(
            {
                "key": record_key(perturbed),
                "kind": "family_swapped",
                "base_sentence": record["base_sentence"],
                "variant": record["variant"],
                "family": wrong,
                "original_family": record["family"],
                "grammatical": True,
                "semantic": None,
                "category": False,
                "_record": perturbed,
            }
        )
    return out


def records_to_verify(constructed: Sequence[dict]) -> list[dict]:
    """The perturbed records themselves, ready to be run through the verifier."""
    return [dict(item.get("_record") or item) for item in constructed]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_labels(path: Path) -> list[dict]:
    out = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_labels(labels: Sequence[dict], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in labels:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")


def index_by_key(records: Sequence[dict]) -> dict[str, dict]:
    return {record_key(r): r for r in records}
