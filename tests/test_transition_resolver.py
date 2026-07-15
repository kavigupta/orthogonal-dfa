"""Correctness tests for the incremental transition resolver.

Mirrors a handful of the ``test_lstar`` benchmarks but builds the DFA with
``do_resolver_driven_synthesis`` instead of the discovery + optimal_dfa path.
Reuses the ``test_lstar`` harness (prefix pool construction and the accuracy
assertion) so the two paths are compared under identical settings.
"""

import unittest

from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.transition_resolver import do_resolver_driven_synthesis
from tests.test_lstar import allowed_error, assertDFA, compute_pst


def _resolve(oracle_creator, min_signal_strength, seed=0):
    pst = compute_pst(oracle_creator, min_signal_strength, seed)
    dfa, _ = do_resolver_driven_synthesis(
        pst, additional_counterexamples=200, acc_threshold=1 - allowed_error
    )
    return dfa


class TestTransitionResolver(unittest.TestCase):
    def test_modulo(self):
        oracle_creator = lambda noise_model, seed: BernoulliParityOracle(
            noise_model, seed, modulo=9, allowed_moduluses=(3, 6)
        )
        assertDFA(self, _resolve(oracle_creator, 0.3), oracle_creator)

    def test_specific_subsequence(self):
        oracle_creator = lambda noise_model, seed: BernoulliRegex(
            noise_model, seed, regex=r".*1010101.*"
        )
        assertDFA(self, _resolve(oracle_creator, 0.3), oracle_creator)

    def test_two_subsequences(self):
        oracle_creator = lambda noise_model, seed: BernoulliRegex(
            noise_model, seed, regex=r".*1111.*1111.*"
        )
        assertDFA(self, _resolve(oracle_creator, 0.3), oracle_creator)


if __name__ == "__main__":
    unittest.main()
