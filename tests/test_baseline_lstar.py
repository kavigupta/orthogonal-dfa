"""
Tests for standard L* baseline against the same test cases
used for the orthonormal L* implementation.

Compares standard L* (via aalpy) at different noise levels:
- Noiseless: standard L* should learn the correct DFA
- Noisy: standard L* is expected to struggle
"""

import unittest

import numpy as np

from orthogonal_dfa.baseline_lstar.baseline_lstar import run_baseline_lstar
from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    AllFramesClosedOracle,
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.structures import SymmetricBernoulli
from tests.test_lstar import assertDFA, evaluate_accuracy

us = UniformSampler(40)


def base_rate_accuracy(oracle_creator, symbols=2, count=10_000):
    """Compute the base rate accuracy (always predict majority class)."""
    oracle = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
    rng = np.random.default_rng(0x1234)
    acc_frac = np.mean(
        [oracle.membership_query(us.sample(rng, symbols)) for _ in range(count)]
    )
    return max(acc_frac, 1 - acc_frac)


class TestBaselineLStarNoiseless(unittest.TestCase):
    """Standard L* with a noiseless oracle. Should succeed on all cases."""

    def _run(self, oracle_creator, symbols=2):
        oracle = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
        dfa = run_baseline_lstar(oracle)
        num_states = len(dfa.states)
        print(f"  Learned DFA with {num_states} states")
        assertDFA(self, dfa, oracle_creator, symbols=symbols)

    def test_modulo(self):
        self._run(
            lambda nm, s: BernoulliParityOracle(
                nm, s, modulo=9, allowed_moduluses=(3, 6)
            )
        )

    def test_specific_subsequence(self):
        self._run(lambda nm, s: BernoulliRegex(nm, s, regex=r".*1010101.*"))

    def test_two_subsequences(self):
        self._run(lambda nm, s: BernoulliRegex(nm, s, regex=r".*1111.*1111.*"))

    def test_specific_alternation(self):
        self._run(lambda nm, s: BernoulliRegex(nm, s, regex=r".*(1111|0000)11.*"))

    def test_alternation_3_syms(self):
        self._run(
            lambda nm, s: BernoulliRegex(
                nm, s, regex=r".*(111|000).*", alphabet_size=3
            ),
            symbols=3,
        )

    def test_no_orf(self):
        self._run(AllFramesClosedOracle, symbols=4)


class TestBaselineLStarNoisy(unittest.TestCase):
    """Standard L* with a noisy oracle, capped by max_states."""

    def _run_noisy(self, oracle_creator, p_correct, symbols=2, max_states=50):
        oracle = oracle_creator(SymmetricBernoulli(p_correct=p_correct), 0)
        dfa = run_baseline_lstar(oracle, max_states=max_states)

        num_states = len(dfa.states)
        accuracy = evaluate_accuracy(dfa, oracle_creator, symbols=symbols)
        expected = base_rate_accuracy(oracle_creator, symbols=symbols)
        threshold = (1 + expected) / 2
        print(
            f"  Learned DFA with {num_states} states, "
            f"accuracy={accuracy:.4f}, base_rate={expected:.4f}, "
            f"threshold={threshold:.4f} (noise p={p_correct})"
        )
        self.assertLess(
            accuracy,
            threshold,
            f"Standard L* with noise shouldn't exceed midpoint of base rate "
            f"({expected:.4f}) and perfect accuracy; got {accuracy:.4f}",
        )

    def test_modulo_noisy_0_8(self):
        self._run_noisy(
            lambda nm, s: BernoulliParityOracle(
                nm, s, modulo=9, allowed_moduluses=(3, 6)
            ),
            p_correct=0.8,
        )

    def test_modulo_noisy_0_6(self):
        self._run_noisy(
            lambda nm, s: BernoulliParityOracle(
                nm, s, modulo=9, allowed_moduluses=(3, 6)
            ),
            p_correct=0.6,
        )

    def test_regex_noisy_0_8(self):
        self._run_noisy(
            lambda nm, s: BernoulliRegex(nm, s, regex=r".*1010101.*"),
            p_correct=0.8,
        )

    def test_regex_noisy_0_6(self):
        self._run_noisy(
            lambda nm, s: BernoulliRegex(nm, s, regex=r".*1010101.*"),
            p_correct=0.6,
        )

    def test_modulo_noisy_0_9(self):
        self._run_noisy(
            lambda nm, s: BernoulliParityOracle(
                nm, s, modulo=9, allowed_moduluses=(3, 6)
            ),
            p_correct=0.9,
        )

    def test_modulo_noisy_0_95(self):
        self._run_noisy(
            lambda nm, s: BernoulliParityOracle(
                nm, s, modulo=9, allowed_moduluses=(3, 6)
            ),
            p_correct=0.95,
        )
