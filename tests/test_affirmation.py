"""N_affirmation (reports/08-guardrail-ensemble.md Phase 1): one test per row of the brief's table, plus the three
warnings pinned as regressions (never != always, no != all, affixal exceptions) and litotes.

    ./venv/bin/python -m pytest tests/test_affirmation.py -q
"""

import pytest

from negation.generators.n_affirmation import affirm


def texts(sentence, *, primary=None, flip=None):
    return {a.text for a in affirm(sentence)
            if (primary is None or a.primary == primary) and (flip is None or a.polarity_flip == flip)}


@pytest.mark.parametrize("sentence,expected", [
    ("Delhi is not the capital of India.", "Delhi is the capital of India."),           # is not X
    ("The pages were not loaded.", "The pages were loaded."),                           # were not
    ("The scheduler does not run the process.", "The scheduler runs the process."),     # reverse do-support
    ("The kernel did not run the job.", "The kernel ran the job."),                     # past tense
    ("The kernel did not work.", "The kernel worked."),                                 # not "wrought" (pyinflect)
    ("The threads do not run in parallel.", "The threads run in parallel."),            # no inflection
    ("The process cannot access the memory.", "The process can access the memory."),    # cannot
    ("The process can't access the memory.", "The process can access the memory."),     # can't (was "ca")
    ("The server will not respond.", "The server will respond."),
    ("The server won't respond.", "The server will respond."),                          # won't (was "wo")
    ("The transaction has not committed.", "The transaction has committed."),           # perfect kept
    ("No students passed.", "Some students passed."),                                    # no -> some
    ("The protocol is unreliable.", "The protocol is reliable."),                        # affixal un-
    ("The data is inconsistent.", "The data is consistent."),                            # affixal in-
    ("The approach is useless.", "The approach is useful."),                             # -less -> -ful
    ("The system lacks redundancy.", "The system has redundancy."),                      # lack -> have
])
def test_primary_affirmation(sentence, expected):
    assert expected in texts(sentence, primary=True)


@pytest.mark.parametrize("sentence,expected", [
    ("The server refuses to accept the connection.", "The server agrees to accept the connection."),
    ("He denies taking the money.", "He admits taking the money."),
    ("The process fails to acquire the lock.", "The process manages to acquire the lock."),
    ("The client avoids sending duplicates.", "The client sends duplicates."),
])
def test_implicit_negatives_use_the_curated_lexicon(sentence, expected):
    assert expected in texts(sentence, primary=True)


def test_never_affirms_to_plain_removal_and_sometimes_but_never_always():
    got = {a.text: a for a in affirm("He never calls.")}
    assert got["He calls."].primary
    assert not got["He sometimes calls."].primary            # secondary reading, labelled
    assert "He always calls." not in got                    # warning 1: the contrary, not the negation


def test_no_affirms_to_some_never_all():
    got = texts("No students passed.")
    assert "Some students passed." in got
    assert "All students passed." not in got                # warning 2


@pytest.mark.parametrize("sentence,wrong", [
    ("The record is invaluable.", "The record is valuable."),
    ("The gas is inflammable.", "The gas is flammable."),
    ("The employee is disgruntled.", "The employee is gruntled."),
])
def test_affixal_exceptions_are_not_stripped(sentence, wrong):
    assert wrong not in texts(sentence)                     # warning 3


def test_affixal_prefix_is_not_stripped_inside_a_term():
    # step-0.1 rule: attributive adjectives in technical terms stay put
    assert not any("reliable protocol" in t for t in texts("UDP is an unreliable protocol."))


def test_litotes_is_approximate_and_not_a_polarity_flip():
    got = {a.text: a for a in affirm("It is not uncommon for a process to block.")}
    lit = got["It is common for a process to block."]
    assert lit.approximate and not lit.polarity_flip and lit.primary     # warning 4
    # the one-cue removals are negations of the litotes, kept as secondary
    assert all(not a.primary for t, a in got.items() if t != lit.text)


def test_only_the_main_clause_is_affirmed():
    assert texts("A process that is not running is blocked.") == set()


def test_affirmative_sentence_has_no_affirmation():
    assert texts("The scheduler runs the process.") == set()


def test_sentence_adverb_is_not_a_litotes():
    got = affirm("Unfortunately it will not recharge my phone.")
    assert all(a.polarity_flip for a in got) and "Fortunately" not in " ".join(a.text for a in got)


def test_a_complementiser_in_the_base_does_not_block_the_variant():
    # "that the" matched the doubled_determiner blacklist and vetoed every variant (step 1.2)
    assert "I thought that the instructions were helpful." in texts("I didn't think that the instructions were helpful.")


# --- report_f2 step 4.1: cues outside the single main clause -------------------------------------------

def test_coordinated_main_clause_is_affirmed():
    got = {a.text: a for a in affirm("The function sorts the array and does not print the result.")}
    a = got["The function sorts the array and prints the result."]
    assert a.clause == "coordinated" and a.polarity_flip


def test_coordinated_clause_after_but_is_affirmed():
    got = {a.text: a for a in affirm("The kernel schedules the thread, but it cannot preempt the driver.")}
    assert got["The kernel schedules the thread, but it can preempt the driver."].clause == "coordinated"


def test_adverbial_clause_is_affirmed_but_not_as_a_negation_of_the_sentence():
    got = {a.text: a for a in affirm("If the process does not hold the lock, it waits.")}
    a = got["If the process holds the lock, it waits."]
    assert a.clause == "advcl" and not a.polarity_flip      # an antecedent flip, not a negation


def test_complement_clause_is_affirmed_but_not_as_a_negation_of_the_sentence():
    got = {a.text: a for a in affirm("He said that the server is not running.")}
    a = got["He said that the server is running."]
    assert a.clause == "ccomp" and not a.polarity_flip


def test_relative_clause_is_still_not_affirmed():
    assert texts("A process that is not running is blocked.") == set()      # step-1.1 restriction, no regression


def test_headline_case_still_works():
    got = {a.text: a for a in affirm("Delhi is not the capital of India.")}
    assert got["Delhi is the capital of India."].clause == "main" and got["Delhi is the capital of India."].primary


def test_headline_case_in_a_coordinated_clause():
    got = {a.text for a in affirm("Mumbai is a large city, but it is not the capital of India.")}
    assert "Mumbai is a large city, but it is the capital of India." in got
