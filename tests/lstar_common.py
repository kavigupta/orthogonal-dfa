import contextlib

import numpy as np

import orthogonal_dfa.l_star.cluster as _cluster
import orthogonal_dfa.l_star.counterexample_synthesis as _counterexample_synthesis
import orthogonal_dfa.l_star.transition_resolver as _transition_resolver
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.statistics import binomial_side_of_boundary
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


# Every family the synthesis builds is seeded at the empty suffix, so its decisive
# classifications should realise the accept-preserving split: the noiseless
# membership 1[x in L]. verify_rounds checks this after each family build, over the
# prefixes the family decides (indecisive ones are boundary strings, excluded).
#
# A round fails only when its misclassification count is, by a binomial test,
# significantly above the calibration's own per-decision error budget -- not against
# a hand-picked fraction. This is sample-size-aware: a handful of near-boundary
# misses on a few-hundred-prefix pool is within the fpr floor, while a family that
# computes the wrong cut misclassifies far more than the floor predicts.
round_verify_fpr = 0.01  # matches acceptable_fpr in learn.build_pst
round_verify_alpha = 1e-4  # binomial significance for flagging a round


def _round_accept_preserving_counts(pst, vs, truth_oracle):
    """``(decisive, misclassified)`` counts over the representative prefixes this
    family decides, or None when it decides none of them."""
    mask = pst.table.representative
    rep = np.flatnonzero(mask)
    if len(rep) == 0:
        return None
    rate = pst.table.observed_masks(vs, mask).mean(0)
    accept = rate >= pst.decision_boundary + pst.evidence_margin
    reject = rate < pst.decision_boundary - pst.evidence_margin
    decisive = accept | reject
    if not decisive.any():
        return None
    truth = np.array(
        [bool(truth_oracle.membership_query(list(pst.table.prefixes[i]))) for i in rep]
    )
    return int(decisive.sum()), int(np.sum(accept[decisive] != truth[decisive]))


@contextlib.contextmanager
def verify_rounds(oracle_creator, fpr=round_verify_fpr, alpha=round_verify_alpha):
    """Assert the accept-preserving invariant after every family build inside."""
    truth_oracle = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
    violations = []
    original = _cluster.sample_suffix_family

    def wrapped(pst, v):
        vs, boundary = original(pst, v)
        counts = _round_accept_preserving_counts(pst, vs, truth_oracle)
        if counts is not None:
            decisive, wrong = counts
            if binomial_side_of_boundary(wrong, decisive, fpr, failure_prob=alpha):
                violations.append((wrong, decisive))
        return vs, boundary

    # transition_resolver / counterexample_synthesis import the name directly.
    _cluster.sample_suffix_family = wrapped
    _transition_resolver.sample_suffix_family = wrapped
    _counterexample_synthesis.sample_suffix_family = wrapped
    try:
        yield
    finally:
        _cluster.sample_suffix_family = original
        _transition_resolver.sample_suffix_family = original
        _counterexample_synthesis.sample_suffix_family = original
    if violations:
        wrong, decisive = max(violations, key=lambda c: c[0] / c[1])
        raise AssertionError(
            f"a synthesis round misclassified {wrong}/{decisive} of its decisive "
            f"prefixes against the noiseless accept-preserving split -- significantly "
            f"above the fpr budget {fpr} (binomial test, alpha={alpha})"
        )


def learn_dfa_verified(oracle_creator, **kwargs):
    """``learn_dfa`` with the per-round accept-preserving check active."""
    with verify_rounds(oracle_creator):
        return learn_dfa(oracle_creator, **kwargs)
