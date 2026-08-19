import numpy as np

from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.structures import SymmetricBernoulli

us = UniformSampler(40)

# assertDFA tolerance — slightly looser than the synthesis target so we don't
# flake when synthesis converges near the threshold.  See GitHub issue on
# tightening synthesis output.
assertion_allowed_error = 0.03


def sample_with_exclusion(exclude_pattern, *, symbols, count):
    rng = np.random.default_rng(0x1234)
    results = []
    while len(results) < count:
        s = us.sample(rng, symbols)
        if exclude_pattern is None or not exclude_pattern(s):
            results.append(s)
    return results


def compute_dfa_accuracy(
    dfa, oracle_creator, exclude_pattern=None, symbols=2, count=10_000
):
    """Evaluate dfa against a noiseless oracle. Returns (accuracy, false_positives, false_negatives)."""
    oracle = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
    false_positives, false_negatives = [], []
    for s in sample_with_exclusion(exclude_pattern, symbols=symbols, count=count):
        expected = oracle.membership_query(s)
        actual = dfa.accepts_input(s)
        if expected and not actual:
            false_negatives.append(s)
        elif not expected and actual:
            false_positives.append(s)
    accuracy = 1 - (len(false_positives) + len(false_negatives)) / count
    return accuracy, false_positives, false_negatives


def evaluate_accuracy(
    dfa, oracle_creator, exclude_pattern=None, symbols=2, count=10_000
):
    """Return accuracy of dfa against a noiseless oracle."""
    accuracy, _, _ = compute_dfa_accuracy(
        dfa, oracle_creator, exclude_pattern, symbols, count
    )
    return accuracy


def assertDFA(
    testcase, dfa, oracle_creator, exclude_pattern=None, symbols=2, *, count=10_000
):
    accuracy, false_positives, false_negatives = compute_dfa_accuracy(
        dfa, oracle_creator, exclude_pattern, symbols, count
    )
    if accuracy < 1 - assertion_allowed_error:
        print("DFA is incorrect!")
        print(dfa)
        print(f"False positives: {false_positives}")
        print(f"False negatives: {false_negatives}")
        testcase.fail(
            f"DFA incorrect. False positives: {len(false_positives)}, False negatives: {len(false_negatives)}"
        )


def assertDoesNotMeetProperty(
    testcase, oracle_creator, counterexample_generator, count=10_000
):
    rng = np.random.default_rng(0)
    oracle = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
    valid = []
    for _ in range(count):
        suffix = us.sample(rng, 2)
        prefix = counterexample_generator(suffix)
        s = prefix + suffix
        if oracle.membership_query(s) == oracle.membership_query(prefix):
            valid.append((suffix, prefix))
    if len(valid) / count < 0.001:
        return
    for suffix, prefix in valid[:10]:
        print(f"Counterexample: prefix={prefix}, suffix={suffix}")
    testcase.fail(
        f"Oracle meets property; found {len(valid)} / {count} counterexamples."
    )
