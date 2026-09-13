"""Prompt construction for tier 2.

Three narrow questions, never one vague one. "Is this a good variant?" invites a
single global impression, and a model asked for one will produce a number that
correlates with fluency and nothing else. Asked separately, the three questions
fail in different places and the failures are diagnostic:

``grammatical``        is the *variant alone* well-formed English?
``semantically_valid`` does it stand in the claimed relation to the base -- and
                       would a person plausibly write it?
``category_correct``   does it really belong to the labelled family/subtype?

The third is the one that pays for the exercise. A wrong ``category_correct``
with a ``suggested_family`` is not a bad row to discard; it is a generator bug
with a return address, which is what ``reports/verification_disagreements.md``
is built from.

Two instructions matter more than they look. **Judge only what is asked** --
without it, instruct models rewrite the sentence and then grade their own
rewrite. And **the base sentence is not under judgement** -- several families
take an already-negated base, and a model left to itself will mark a variant
wrong because it dislikes the input it was derived from.

The few-shot examples are chosen to span the three outcomes the model has to
keep apart: a clean pass, a fluent sentence in the wrong family, and an
ungrammatical one. All three are hand-written rather than drawn from the corpus,
so no evaluated record appears in its own prompt.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from ..schema import FAMILIES

#: The families the model may suggest. Passed explicitly: a model asked to name
#: a category without a closed list invents plausible-sounding ones, and a
#: suggestion outside the taxonomy cannot be actioned.
FAMILY_LIST: tuple[str, ...] = tuple(sorted(FAMILIES))

FAMILY_GLOSSES: dict[str, str] = {
    "A_syntactic": "negation by 'not'/'n't' in the verb complex (do-support, auxiliary, copula)",
    "B_quantifier": "negation carried by a quantifier word (no, none, neither, nobody, nothing)",
    "C_neg_adverb": "negation carried by an adverb (never, rarely, hardly, barely, seldom, scarcely)",
    "D_affixal": "negation carried by an affix on a word (unimportant, impossible, useless)",
    "E_antonym": "a content word replaced by its opposite (increased -> decreased)",
    "F_implicit": "negation carried by a verb's meaning (fails to, refuses to, is unable to, lacks)",
    "G_prepositional": "negation carried by a privative preposition (without, devoid of, free of)",
    "H_hedged": "a negation weakened by a hedge (probably not, might not, maybe not)",
    "I_intensified": "a negation strengthened (definitely not, by no means, not at all)",
    "J_contrastive": "contrastive framing of a complement (anything but, far from, the opposite of)",
    "K1_cancellation": "two negations where one scopes over the other, so they cancel (is not impossible)",
    "K2_compound": "two negations in separate clauses, so both stand (does not X and does not Y)",
    "L_scope_position": "negation positioned to change scope over a quantifier (Not everyone passed)",
    "M_modal": "negation of a modal, with its reading (must not = prohibition, need not = no obligation)",
}

OPERATION_GLOSSES: dict[str, str] = {
    "negate": "the variant should NEGATE the base sentence",
    "affirm": "the variant should AFFIRM the base sentence (remove a negation it had)",
    "rescope": "the variant should keep the same number of negations but change what they scope over",
}

_SYSTEM = """You are a linguistic annotator checking generated sentence variants.
You answer only the three questions asked. You never rewrite or improve a sentence.
You output only JSON."""

_FEW_SHOT: tuple[dict[str, Any], ...] = (
    {
        "item": {
            "id": 0,
            "base_sentence": "The compiler accepts the program.",
            "variant": "The compiler does not accept the program.",
            "family": "A_syntactic",
            "subtype": "do_support",
            "operation": "negate",
            "net_negation": 1,
        },
        "verdict": {
            "id": 0,
            "grammatical": True,
            "semantically_valid": True,
            "category_correct": True,
            "suggested_family": None,
            "confidence": 0.95,
        },
        "why": "well-formed, it does negate the base, and 'does not' is exactly A_syntactic",
    },
    {
        "item": {
            "id": 1,
            "base_sentence": "The measurement was precise.",
            "variant": "The measurement was imprecise.",
            "family": "E_antonym",
            "subtype": "antonym_adj",
            "operation": "negate",
            "net_negation": 1,
        },
        "verdict": {
            "id": 1,
            "grammatical": True,
            "semantically_valid": True,
            "category_correct": False,
            "suggested_family": "D_affixal",
            "confidence": 0.85,
        },
        "why": (
            "fluent and it does negate the base, but 'imprecise' is 'precise' plus the "
            "prefix im-, which is D_affixal; E_antonym is for a different word entirely"
        ),
    },
    {
        "item": {
            "id": 2,
            "base_sentence": "She has finished the report.",
            "variant": "She has not finished not the report.",
            "family": "A_syntactic",
            "subtype": "existing_aux",
            "operation": "negate",
            "net_negation": 1,
        },
        "verdict": {
            "id": 2,
            "grammatical": False,
            "semantically_valid": False,
            "category_correct": False,
            "suggested_family": None,
            "confidence": 0.98,
        },
        "why": "the second 'not' is stray; the sentence is not English, so nothing else can hold",
    },
)


def _family_reference() -> str:
    return "\n".join(f"  {name}: {FAMILY_GLOSSES.get(name, '')}" for name in FAMILY_LIST)


def _few_shot_block() -> str:
    lines = ["EXAMPLES (these are illustrations, not items to judge):", ""]
    for shot in _FEW_SHOT:
        lines.append("ITEM: " + json.dumps(shot["item"], ensure_ascii=False))
        lines.append("VERDICT: " + json.dumps(shot["verdict"], ensure_ascii=False))
        lines.append(f"REASONING (do not output this): {shot['why']}")
        lines.append("")
    return "\n".join(lines)


def item_for(record: dict, index: int) -> dict[str, Any]:
    """The fields the model is shown, and only those.

    Deliberately excludes ``generator``, ``cue_char_spans`` and the rest: they
    are provenance, not evidence, and a model shown a generator name will start
    grading the generator's reputation instead of the sentence.
    """
    return {
        "id": index,
        "base_sentence": record.get("base_sentence", ""),
        "variant": record.get("variant", ""),
        "family": record.get("family", ""),
        "subtype": record.get("subtype", ""),
        "operation": record.get("operation", "negate"),
        "net_negation": record.get("net_negation", 1),
    }


def build_prompt(records: Sequence[dict]) -> str:
    """The full prompt for one batch."""
    items = [item_for(record, index) for index, record in enumerate(records)]
    operations = sorted({item["operation"] for item in items})
    operation_help = "\n".join(
        f"  {op}: {OPERATION_GLOSSES.get(op, '')}" for op in operations
    )

    return f"""{_SYSTEM}

You will be given {len(items)} items. Each is a BASE SENTENCE, a VARIANT derived
from it, and the labels the generator assigned.

For each item answer exactly three questions:

1. "grammatical": Is the VARIANT, on its own, well-formed English?
   Judge only form. A sentence can be grammatical and still be wrong elsewhere.
   Marked or old-fashioned phrasing is still grammatical.

2. "semantically_valid": Does the VARIANT stand in the claimed relation to the
   BASE SENTENCE, given "operation"?
{operation_help}
   And would a competent speaker plausibly write the VARIANT? A sentence that is
   grammatical but that nobody would say is not semantically valid.

3. "category_correct": Does the VARIANT genuinely belong to the labelled
   "family"? If not, set "suggested_family" to the family it actually belongs to,
   chosen from the list below. If it belongs to no family, use null.

FAMILY REFERENCE:
{_family_reference()}

RULES:
- Judge ONLY these three questions. Do NOT rewrite, correct or improve any sentence.
- The BASE SENTENCE is NOT under judgement. It may itself already contain a
  negation; that is expected and is not a fault in the VARIANT.
- If "grammatical" is false, the other two are almost always false too.
- "confidence" is your confidence in the whole verdict, from 0.0 to 1.0.
- Output one verdict per item, with the same "id". Output every id from 0 to
  {len(items) - 1}.

{_few_shot_block()}
Output JSON with exactly this shape and nothing else:
{{"verdicts": [{{"id": 0, "grammatical": true, "semantically_valid": true, "category_correct": true, "suggested_family": null, "confidence": 0.9}}]}}

ITEMS TO JUDGE:
{json.dumps(items, ensure_ascii=False, indent=None)}
"""


def build_single_prompt(record: dict) -> str:
    """One-record prompt, used as the fallback after batch parsing fails twice."""
    return build_prompt([record])
