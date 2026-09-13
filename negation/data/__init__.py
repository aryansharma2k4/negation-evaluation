"""Third-party negation corpora: verified sources, loaders, common schema.

Nothing here vendors data.  :mod:`negation.data.sources` records where each
corpus was actually found and under what licence; :mod:`negation.data.loaders`
reads whatever has been fetched into ``data/external`` and normalises it to the
single record type in :mod:`negation.data.schema`.

See ``docs/dataset_evaluation.md`` for what was verified, what was rejected and
why.
"""

from .schema import AnnotatedSentence  # noqa: F401

__all__ = ["AnnotatedSentence"]
