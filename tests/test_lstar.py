import unittest

import numpy as np
from parameterized import parameterized

from orthogonal_dfa.l_star.decision_tree_to_dfa import (
    PrefixSuffixTracker,
    compute_prefix_set_size,
    do_counterexample_driven_synthesis,
    population_size_and_evidence_thresh,
)
from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    AllFramesClosedOracle,
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.sampler import UniformSampler

us = UniformSampler(40)

allowed_error = 0.02


def sample_with_exclusion(exclude_pattern, *, symbols, count):
    rng = np.random.default_rng(0x1234)
    results = []
    while len(results) < count:
        s = us.sample(rng, symbols)
        if exclude_pattern is None or not exclude_pattern(s):
            results.append(s)
    return results


def assertDFA(
    testcase, dfa, oracle_creator, exclude_pattern=None, symbols=2, *, count=10_000
):
    oracle = oracle_creator(1.0, 0)
    false_positives, false_negatives = [], []
    for s in sample_with_exclusion(exclude_pattern, symbols=symbols, count=count):
        expected = oracle.membership_query(s)
        actual = dfa.accepts_input(s)
        if expected and not actual:
            false_negatives.append(s)
        elif not expected and actual:
            false_positives.append(s)
    if len(false_positives) + len(false_negatives) > allowed_error * count:
        print("DFA is incorrect!")
        print(dfa)
        print(f"False positives: {false_positives}")
        print(f"False negatives: {false_negatives}")
        testcase.fail(
            f"DFA incorrect. False positives: {len(false_positives)}, False negatives: {len(false_negatives)}"
        )


def compute_dfa_for_oracle(oracle_creator, *, accuracy, seed, symbols=2):
    pst = compute_pst(oracle_creator, accuracy, seed, symbols=symbols)
    dfa, dt = do_counterexample_driven_synthesis(
        pst, additional_counterexamples=200, acc_threshold=1 - allowed_error
    )
    return pst, dfa, dt


def compute_pst(oracle_creator, accuracy, seed, *, symbols, use_dynamic=True):
    oracle = oracle_creator(accuracy, seed)
    n, eps = population_size_and_evidence_thresh(
        p_acc=accuracy, acceptable_fpr=0.01, acceptable_fnr=0.01, relative_eps=1
    )
    k = compute_prefix_set_size(0.05, accuracy, 0.05)
    kwargs = (
        dict(num_prefixes=200, num_addtl_prefixes=200)
        if use_dynamic
        else dict(num_prefixes=k)
    )
    print(f"Using suffix population size {n}, eps {eps}, and {k} prefixes.")
    pst = PrefixSuffixTracker.create(
        us,
        np.random.default_rng(0),
        oracle,
        alphabet_size=symbols,
        **kwargs,
        suffix_family_size=n,
        chi_squared_p_min=None,
        evidence_thresh=0.50 + eps,
        suffix_prevalence=0.05,
        decision_rule_fpr=0.01,
    )

    return pst


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

    def test_modulo_harder(self):
        oracle_creator = lambda accuracy, seed: BernoulliParityOracle(
            accuracy, seed, modulo=9, allowed_moduluses=(3, 6)
        )
        _, dfa, _ = compute_dfa_for_oracle(oracle_creator, accuracy=0.7, seed=0)
        assertDFA(self, dfa, oracle_creator)

    def test_modulo_even_harder(self):
        oracle_creator = lambda accuracy, seed: BernoulliParityOracle(
            accuracy, seed, modulo=9, allowed_moduluses=(3, 6)
        )
        _, dfa, _ = compute_dfa_for_oracle(oracle_creator, accuracy=0.6, seed=0)
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

    def test_two_subsequences_with_alternation(self):
        oracle_creator = lambda accuracy, seed: BernoulliRegex(
            accuracy, seed, regex=r".*1111.*(1111|0000)11.*"
        )
        _, dfa, _ = compute_dfa_for_oracle(oracle_creator, accuracy=0.8, seed=0)
        assertDFA(self, dfa, oracle_creator)

    def test_specific_alternation(self):
        oracle_creator = lambda accuracy, seed: BernoulliRegex(
            accuracy, seed, regex=r".*(1111|0000)11.*"
        )
        _, dfa, _ = compute_dfa_for_oracle(oracle_creator, accuracy=0.8, seed=0)
        assertDFA(self, dfa, oracle_creator, exclude_pattern=lambda s: s[:5] == [1] * 5)

    def test_specific_alternation_with_nothing_at_end_3_syms(self):
        oracle_creator = lambda accuracy, seed: BernoulliRegex(
            accuracy, seed, regex=r".*(111|000).*"
        )
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, accuracy=0.8, seed=0, symbols=3
        )
        assertDFA(self, dfa, oracle_creator, symbols=3)

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


class TestLStarORF(unittest.TestCase):
    @parameterized.expand([(accuracy,) for accuracy in (0.8, 0.7)])
    def test_no_orf(self, accuracy):
        oracle_creator = AllFramesClosedOracle
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, accuracy=accuracy, seed=0, symbols=4
        )
        assertDFA(self, dfa, oracle_creator, symbols=4)
