"""Stage 1 of the graded negation-sensitivity project: variant generation."""

from .schema import NegationVariant  # noqa: F401

__all__ = ["NegationVariant", "generate_all"]


def __getattr__(name: str):  # lazy so `import negation` stays cheap
    if name == "generate_all":
        from .driver import generate_all

        return generate_all
    raise AttributeError(name)
