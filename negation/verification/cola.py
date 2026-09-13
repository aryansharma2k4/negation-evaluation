"""Tier 1: grammatical acceptability, cheap enough to run on everything.

A CoLA-finetuned classifier scores every variant before any LLM is called.
Records below the threshold are marked ``verified = false`` with
``reason = "cola_reject"`` and never reach tier 2. On a corpus where a sizeable
fraction of candidates are malformed, this is most of the saving: at ~9 seconds
per record on CPU, every record the filter removes is nine seconds not spent.

The threshold is **reported, not guessed**. :func:`threshold_sweep` scores the
whole corpus once and shows what each candidate threshold would cut, so it can
be chosen against the actual distribution rather than by picking 0.5 and hoping.
The filter is also the one irreversible step in the pipeline -- a rejected
record never gets a second opinion -- so the default is deliberately permissive.

The classifier is a *filter*, not an oracle. CoLA is trained on linguists'
acceptability judgements over ordinary prose, and negated variants are not
ordinary prose: double negation, scope alternation and archaic-sounding but
grammatical forms (*"Is the task not impossible?"*) are exactly where it is
least reliable. That is why its verdict maps to ``verify_grammatical`` only, and
why the LLM re-asks the grammaticality question independently for everything
that survives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

#: Default acceptability model. Any sequence-classification checkpoint with an
#: acceptable/unacceptable head works; the label index is resolved at load time.
DEFAULT_MODEL = "textattack/roberta-base-CoLA"

#: Deliberately permissive. Tier 1 is irreversible, tier 2 is the real judge,
#: and a false reject here costs a record that no later stage can recover.
DEFAULT_THRESHOLD = 0.30

#: Thresholds reported by the sweep.
SWEEP_POINTS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class ColaUnavailable(RuntimeError):
    """Raised when the acceptability model cannot be loaded."""


@dataclass
class ColaScorer:
    """Batched acceptability scoring. GPU when there is one, CPU otherwise."""

    name: str
    _tokenizer: object
    _model: object
    _acceptable_index: int
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 128

    def score(self, sentences: Sequence[str]) -> list[float]:
        """P(acceptable) for each sentence, in order."""
        import torch

        out: list[float] = []
        for start in range(0, len(sentences), self.batch_size):
            chunk = list(sentences[start : start + self.batch_size])
            encoded = self._tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, self._acceptable_index]
            out.extend(probs.detach().cpu().tolist())
        return out


def load_scorer(
    model_name: str = DEFAULT_MODEL, *, batch_size: int = 32, device: Optional[str] = None
) -> ColaScorer:
    """Load the acceptability classifier, or raise :class:`ColaUnavailable`.

    The index of the "acceptable" class is read from the checkpoint's own
    ``id2label`` rather than assumed to be 1. Published CoLA heads disagree about
    the ordering, and guessing it inverts the filter -- which would silently
    reject every *well-formed* sentence, a failure that looks like a very
    aggressive threshold rather than a bug.
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ColaUnavailable(
            "tier 1 needs torch and transformers:\n"
            "    ./venv/bin/python -m pip install torch transformers"
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as exc:
        raise ColaUnavailable(f"could not load {model_name}: {exc}") from exc

    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(resolved)
    model.eval()

    index = _acceptable_index(model.config)
    return ColaScorer(model_name, tokenizer, model, index, resolved, batch_size)


def _acceptable_index(config) -> int:
    """Which logit means "acceptable", read off the checkpoint."""
    id2label = getattr(config, "id2label", None) or {}
    for index, label in id2label.items():
        text = str(label).lower()
        if text in ("label_1", "acceptable", "1") or "accept" in text:
            return int(index)
    # Binary head with uninformative labels: CoLA's own convention is 1 = ok.
    return 1 if int(getattr(config, "num_labels", 2)) > 1 else 0


def threshold_sweep(
    scores: Iterable[float], points: Sequence[float] = SWEEP_POINTS
) -> list[dict]:
    """What each threshold would cut, so the choice can be made on evidence."""
    values = sorted(scores)
    total = len(values)
    if not total:
        return []
    out = []
    for point in points:
        cut = sum(1 for v in values if v < point)
        out.append(
            {
                "threshold": point,
                "rejected": cut,
                "kept": total - cut,
                "reject_rate": cut / total,
            }
        )
    return out


def format_sweep(sweep: Sequence[dict]) -> str:
    lines = [
        "  threshold   rejected     kept   reject rate",
        "  ---------   --------   ------   -----------",
    ]
    for row in sweep:
        lines.append(
            f"  {row['threshold']:>9.2f}   {row['rejected']:>8,}   {row['kept']:>6,}   "
            f"{row['reject_rate']:>10.1%}"
        )
    return "\n".join(lines)
