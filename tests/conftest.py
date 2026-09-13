"""Shared fixtures.

The spaCy model is loaded once at import of :mod:`negation.nlp_core`; these
helpers make sure tests parse through the same batched entry point the pipeline
uses, so a test never exercises a code path the driver does not.
"""

from __future__ import annotations

import pytest
from spacy.tokens import Doc

from negation.base import registry
from negation.driver import parse_bases


@pytest.fixture(scope="session")
def parse():
    """``parse("...")`` -> a Doc stamped with a base_id, as the driver produces."""

    def _parse(sentence: str) -> Doc:
        return parse_bases([sentence])[0]

    return _parse


@pytest.fixture(scope="session")
def run_family():
    """``run_family(doc, "A_syntactic")`` -> records that family licenses.

    Gated by ``licensed_for`` as well as ``applies``, exactly as the driver is,
    so a test never sees output the pipeline would not produce.
    """

    def _run(doc: Doc, family: str) -> list:
        out = []
        for generator in registry():
            if generator.family != family:
                continue
            if generator.licensed_for(doc) and generator.applies(doc):
                out.extend(generator.generate(doc))
        return out

    return _run


@pytest.fixture(scope="session")
def run_generator():
    """``run_generator(doc, "syn_do_support_v1")`` -> that one generator's records."""

    def _run(doc: Doc, name: str) -> list:
        for generator in registry():
            if generator.name == name:
                licensed = generator.licensed_for(doc) and generator.applies(doc)
                return generator.generate(doc) if licensed else []
        raise AssertionError(f"no generator named {name!r}")

    return _run


def variants(records) -> set[str]:
    return {r.variant for r in records}
