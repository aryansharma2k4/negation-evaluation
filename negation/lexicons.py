"""Every lexicon the generators consult, precompiled at import time.

All containers here are ``frozenset`` / immutable ``dict`` built once when the
module is imported, so generator hot loops do only hash lookups.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# ---------------------------------------------------------------------------
# B_quantifier
# ---------------------------------------------------------------------------


class QuantSwap(NamedTuple):
    """A quantifier substitution and how it is spelled out."""

    replacement: str
    subtype: str
    #: When set, the swap consumes a following determiner (``all the`` -> ``none of the``
    #: must not produce ``none of the the``).
    absorbs_following_det: bool = False


#: Determiner / pronoun / adverb level positive->negative quantifier swaps.
QUANTIFIER_SWAPS: dict[str, QuantSwap] = {
    "some": QuantSwap("no", "some_to_no"),
    "all": QuantSwap("none of the", "all_to_none_of", absorbs_following_det=True),
    "every": QuantSwap("no", "every_to_no"),
    "each": QuantSwap("no", "each_to_no"),
    "both": QuantSwap("neither", "both_to_neither"),
    "always": QuantSwap("never", "always_to_never"),
    "someone": QuantSwap("nobody", "someone_to_nobody"),
    "somebody": QuantSwap("nobody", "somebody_to_nobody"),
    "everyone": QuantSwap("no one", "everyone_to_no_one"),
    "everybody": QuantSwap("nobody", "everybody_to_nobody"),
    "something": QuantSwap("nothing", "something_to_nothing"),
    "everything": QuantSwap("nothing", "everything_to_nothing"),
    "somewhere": QuantSwap("nowhere", "somewhere_to_nowhere"),
    "everywhere": QuantSwap("nowhere", "everywhere_to_nowhere"),
    "sometimes": QuantSwap("never", "sometimes_to_never"),
    "often": QuantSwap("never", "often_to_never"),
    "usually": QuantSwap("never", "usually_to_never"),
}

QUANTIFIER_KEYS: frozenset[str] = frozenset(QUANTIFIER_SWAPS)

#: Quantified subjects that license the L_scope_position contrast.
SCOPE_BEARING_QUANTIFIERS: frozenset[str] = frozenset(
    {"all", "every", "each", "everyone", "everybody", "everything", "both", "many", "most"}
)

# ---------------------------------------------------------------------------
# C_neg_adverb
# ---------------------------------------------------------------------------

#: adverb -> (subtype, intensity_hint)
NEGATIVE_ADVERBS: dict[str, tuple[str, str]] = {
    "never": ("absolute", "neutral"),
    "hardly": ("partial", "partial"),
    "barely": ("partial", "partial"),
    "rarely": ("partial", "partial"),
    "seldom": ("partial", "partial"),
    "scarcely": ("partial", "partial"),
}

# ---------------------------------------------------------------------------
# D_affixal
# ---------------------------------------------------------------------------

#: Negative prefixes, ordered by how commonly they attach.  Candidacy is only a
#: hypothesis: every derived form is checked against WordNet before emission.
NEGATIVE_PREFIXES: tuple[str, ...] = ("un", "in", "im", "ir", "il", "non", "dis")

#: Cheap phonotactic gate applied before the WordNet check, so we do not spend
#: lookups on forms English never builds (``im-`` only before labials, ``ir-``
#: only before ``r``, ``il-`` only before ``l``).
PREFIX_ONSET_CONSTRAINTS: dict[str, frozenset[str]] = {
    "im": frozenset("bmp"),
    "ir": frozenset("r"),
    "il": frozenset("l"),
}

#: Words that already carry negative polarity; prefixing them again is either
#: ungrammatical or produces an unintended cancellation, so D skips them and
#: K1 picks them up instead.
LEXICALLY_NEGATIVE: frozenset[str] = frozenset(
    {
        "impossible", "unlikely", "unable", "incapable", "unfit", "unclear",
        "invalid", "incorrect", "inaccurate", "inconsistent", "incomplete",
        "unavailable", "unsafe", "unstable", "unusual", "unhappy", "unfair",
        "irrelevant", "illegal", "illegible", "improbable", "imperfect",
        "insufficient", "unsuccessful", "unwilling", "unreliable", "unknown",
        "nonexistent", "dishonest", "useless", "hopeless", "harmless",
        "careless", "meaningless", "powerless", "worthless", "fruitless",
        "absent", "lacking", "devoid", "void", "empty", "missing",
        "fail", "fails", "failed", "lack", "lacks", "lacked", "refuse",
        "refuses", "refused", "deny", "denies", "denied", "neglect",
        "neglects", "neglected", "cease", "ceases", "ceased",
    }
)

# ---------------------------------------------------------------------------
# E_antonym
# ---------------------------------------------------------------------------

#: Content POS eligible for antonym substitution.
ANTONYM_POS: frozenset[str] = frozenset({"ADJ", "VERB", "ADV"})

#: Verbs whose antonyms are semantically noisy or which are structural rather
#: than contentful; substituting them yields garbage records.
ANTONYM_VERB_STOPLIST: frozenset[str] = frozenset(
    {"be", "have", "do", "get", "make", "take", "use", "go", "say", "see"}
)

#: Adverbs/adjectives whose WordNet antonyms are degenerate.
ANTONYM_WORD_STOPLIST: frozenset[str] = frozenset({"other", "same", "such", "own", "certain"})

# ---------------------------------------------------------------------------
# F_implicit  (NegEx-style verbal triggers)
# ---------------------------------------------------------------------------


class ImplicitTrigger(NamedTuple):
    """A verbal trigger that negates the complement it embeds.

    ``frame`` says how the original predicate is rebuilt:

    ``to_inf``   ``<subj> <trigger:finite> to <lemma> <rest>``
    ``gerund``   ``<subj> <trigger:finite> <lemma:-ing> <rest>``
    ``copular``  ``<subj> <be:finite> <adjective> to <lemma> <rest>``
    ``np``       ``<subj> <trigger:finite> <object NP>`` (replaces ``have``)
    """

    lemma: str
    frame: str
    subtype: str
    #: Only for the ``copular`` frame, e.g. ``"unable"``.
    predicate: str = ""


IMPLICIT_TRIGGERS: tuple[ImplicitTrigger, ...] = (
    ImplicitTrigger("fail", "to_inf", "fails_to"),
    ImplicitTrigger("refuse", "to_inf", "refuses_to"),
    ImplicitTrigger("neglect", "to_inf", "neglects_to"),
    ImplicitTrigger("cease", "to_inf", "ceases_to"),
    ImplicitTrigger("be", "copular", "is_unable_to", predicate="unable"),
    ImplicitTrigger("deny", "gerund", "denies"),
)

#: Trigger for the ``have`` -> ``lack`` rewrite, which takes an NP not a clause.
LACK_TRIGGER = ImplicitTrigger("lack", "np", "lacks")

#: Verbs whose direct object can be re-expressed as a lacked possession.
POSSESSION_VERBS: frozenset[str] = frozenset({"have", "possess", "contain", "include"})

# ---------------------------------------------------------------------------
# H_hedged / I_intensified
# ---------------------------------------------------------------------------


class Modifier(NamedTuple):
    """A hedge or intensifier and the syntactic slot it actually occupies.

    ``mode`` values:

    ``adverb``     a sentence adverb placed relative to the finite auxiliary
                   (*"probably does not sort"*, *"is certainly not impossible"*).
    ``modal``      a modal that *replaces* the finite auxiliary slot entirely
                   (*"might not sort"*); stacking it on ``does`` would be
                   ungrammatical.
    ``post_aux``   a negative phrase that substitutes for ``not`` after an
                   existing auxiliary (*"is by no means impossible"*); it needs
                   a real auxiliary, so do-support frames are skipped.
    """

    text: str
    mode: str
    subtype: str
    #: Clause-frame subtypes this modifier may attach to; empty means "any".
    frames: tuple[str, ...] = ()


HEDGES: tuple[Modifier, ...] = (
    Modifier("maybe", "adverb", "maybe_not"),
    Modifier("possibly", "adverb", "possibly_not"),
    Modifier("probably", "adverb", "probably_not"),
    Modifier("might", "modal", "might_not"),
    Modifier("may", "modal", "may_not"),
)

INTENSIFIERS: tuple[Modifier, ...] = (
    Modifier("surely", "adverb", "surely_not"),
    Modifier("definitely", "adverb", "definitely_not"),
    Modifier("certainly", "adverb", "certainly_not"),
    Modifier("absolutely", "adverb", "absolutely_not"),
    Modifier("not at all", "post_aux", "not_at_all", frames=("copula",)),
    Modifier("by no means", "post_aux", "by_no_means", frames=("copula", "existing_aux")),
)

# ---------------------------------------------------------------------------
# J_contrastive
# ---------------------------------------------------------------------------

#: Contrastive frames, applied to a copular complement.
CONTRASTIVE_FRAMES: tuple[tuple[str, str], ...] = (
    ("anything but", "anything_but"),
    ("far from", "far_from"),
    ("the opposite of", "the_opposite_of"),
)

# ---------------------------------------------------------------------------
# Auxiliary handling (A_syntactic)
# ---------------------------------------------------------------------------

#: Auxiliaries with a lexicalised negative form.
CONTRACTED_NEGATIVE_AUX: dict[str, str] = {"can": "cannot"}

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

#: Known-bad surface patterns.  Any match rejects the variant outright.
BLACKLIST_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("repeated_not", r"\bnot\s+not\b"),
        ("repeated_never", r"\bnever\s+never\b"),
        ("no_no", r"\bno\s+no\b"),
        # "does not have" and "did not do" are fine -- only a *finite* form
        # after do-support is malformed ("*does not has", "*did not was").
        ("do_plus_aux", r"\b(?:do|does|did)\s+(?:not\s+)?(?:am|is|are|was|were|been|being|has|had|does|did|can|could|will|would|shall|should|may|might|must)\b"),
        ("stacked_modals", r"\b(?:can|could|will|would|shall|should|may|might|must)\s+(?:not\s+)?(?:can|could|will|would|shall|should|may|might|must)\b"),
        ("stacked_copula", r"\b(?:is|are|was|were)\s+(?:not\s+)?(?:is|are|was|were)\b"),
        ("doubled_determiner", r"\b(?:the|a|an|this|that|these|those)\s+(?:the|a|an|this|that|these|those)\b"),
        ("of_of", r"\bof\s+of\b"),
        ("to_to", r"\bto\s+to\b"),
        ("dangling_to", r"\bto\s*[.,;:!?]"),
        ("neg_stack_never_not", r"\bnever\s+not\b|\bnot\s+never\b"),
        ("hardly_not", r"\b(?:hardly|barely|rarely|seldom|scarcely)\s+not\b"),
        ("empty_predicate", r"\b(?:does|do|did)\s+not\s*[.,;:!?]?\s*$"),
        ("double_space_cue", r"\bnot\s+to\s*$"),
    )
)
