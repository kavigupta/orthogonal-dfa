import unittest

import numpy as np

from orthogonal_dfa.l_star.decision_tree_to_dfa import (
    PrefixSuffixTracker,
    do_counterexample_driven_synthesis,
    population_size_and_evidence_thresh,
)
from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.sampler import UniformSampler

us = UniformSampler(40)


def sample_with_exclusion(exclude_pattern):
    rng = np.random.default_rng(0x1234)
    results = []
    while len(results) < 10000:
        s = us.sample(rng, 2)
        if exclude_pattern is None or not exclude_pattern(s):
            results.append(s)
    return results


def assertDFA(testcase, dfa, oracle_creator, exclude_pattern=None):
    oracle = oracle_creator(1.0, 0)
    false_positives, false_negatives = [], []
    for s in sample_with_exclusion(exclude_pattern):
        expected = oracle.membership_query(s)
        actual = dfa.accepts_input(s)
        if expected and not actual:
            false_negatives.append(s)
        elif not expected and actual:
            false_positives.append(s)
    if false_positives or false_negatives:
        print("DFA is incorrect!")
        print(dfa)
        print(f"False positives: {false_positives}")
        print(f"False negatives: {false_negatives}")
        testcase.fail(
            f"DFA incorrect. False positives: {len(false_positives)}, False negatives: {len(false_negatives)}"
        )


def compute_dfa_for_oracle(oracle_creator, *, accuracy, seed):
    oracle = oracle_creator(accuracy, seed)
    n, eps = population_size_and_evidence_thresh(
        p_acc=accuracy, acceptable_fpr=0.01, acceptable_fnr=0.01, relative_eps=1
    )
    print(f"Using suffix population size {n} and eps {eps}")
    pst = PrefixSuffixTracker.create(
        us,
        np.random.default_rng(0),
        oracle,
        alphabet_size=2,
        num_prefixes=2_000,
        suffix_family_size=n,
        chi_squared_p_min=0.005,
        evidence_thresh=0.50 + eps,
        suffix_prevalence=0.05,
    )
    dfa, dt = do_counterexample_driven_synthesis(
        pst, min_state_size=0.01, additional_counterexamples=200, acc_threshold=0.98
    )
    return pst, dfa, dt


def assertDoesNotMeetProperty(
    testcase, oracle_creator, counterexample_generator, count=10_000
):
    rng = np.random.default_rng(0)
    oracle = oracle_creator(1.0, 0)
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


class TestLStar(unittest.TestCase):
    def test_modulo(self):
        oracle_creator = lambda accuracy, seed: BernoulliParityOracle(
            accuracy, seed, modulo=9, allowed_moduluses=(3, 6)
        )
        _, dfa, _ = compute_dfa_for_oracle(oracle_creator, accuracy=0.8, seed=0)
        assertDFA(self, dfa, oracle_creator)

    def test_specific_subsequence(self):
        oracle_creator = lambda accuracy, seed: BernoulliRegex(
            accuracy, seed, regex=r".*1010101.*"
        )
        _, dfa, _ = compute_dfa_for_oracle(oracle_creator, accuracy=0.8, seed=0)
        assertDFA(self, dfa, oracle_creator)

    def test_two_subsequences(self):
        oracle_creator = lambda accuracy, seed: BernoulliRegex(
            accuracy, seed, regex=r".*1111.*1111.*"
        )
        _, dfa, _ = compute_dfa_for_oracle(oracle_creator, accuracy=0.8, seed=0)
        assertDFA(self, dfa, oracle_creator)

    def test_specific_alternation(self):
        oracle_creator = lambda accuracy, seed: BernoulliRegex(
            accuracy, seed, regex=r".*(1111|0000)11.*"
        )
        _, dfa, _ = compute_dfa_for_oracle(oracle_creator, accuracy=0.8, seed=0)
        assertDFA(self, dfa, oracle_creator, exclude_pattern=lambda s: s[:5] == [1] * 5)

    def test_specific_alternation_with_nothing_at_end_does_not_meet_property(self):
        oracle_creator = lambda accuracy, seed: BernoulliRegex(
            accuracy, seed, regex=r".*(11111|00000).*"
        )

        def counterexample_generator(suffix):
            if suffix[0] == 1:
                return [1, 1, 1, 1]
            return [0, 0, 0, 0]

        assertDoesNotMeetProperty(self, oracle_creator, counterexample_generator)

    def test_specific_alternation_with_only_one_at_end_does_not_meet_property(self):
        oracle_creator = lambda accuracy, seed: BernoulliRegex(
            accuracy, seed, regex=r".*(11111|00000)1.*"
        )

        def counterexample_generator(suffix):
            if suffix[0] == 1:
                return [1, 1, 1, 1, 1]
            return [0, 0, 0, 0]

        assertDoesNotMeetProperty(self, oracle_creator, counterexample_generator)
