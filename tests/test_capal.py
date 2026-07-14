"""Smoke tests for the CAPAL learner.

These mirror the oracle-creator pattern of test_lstar.py / test_baseline_lstar.py
but at modest noise levels so the suite stays fast.
"""

import unittest

from orthogonal_dfa.capal import run_capal
from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.structures import SymmetricBernoulli
from tests.test_lstar import assertDFA, evaluate_accuracy


def _truth(oracle_creator):
    return oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)


class TestCAPALNoiseless(unittest.TestCase):
    """CAPAL with eta=0 should fully recover the target DFA."""

    def _run(self, oracle_creator, *, symbols=2):
        noisy = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
        truth = _truth(oracle_creator)
        dfa = run_capal(noisy, truth, eta=0.0)
        print(f"  Learned DFA with {len(dfa.states)} states")
        assertDFA(self, dfa, oracle_creator, symbols=symbols)

    def test_modulo(self):
        self._run(
            lambda nm, s: BernoulliParityOracle(
                nm, s, modulo=9, allowed_moduluses=(3, 6)
            )
        )

    def test_specific_subsequence(self):
        self._run(lambda nm, s: BernoulliRegex(nm, s, regex=r".*1010101.*"))


class TestCAPALNoisy(unittest.TestCase):
    """CAPAL with moderate persistent noise should still recover the target."""

    def _run(self, oracle_creator, *, p_correct, symbols=2):
        eta = 1.0 - p_correct
        noise_model = SymmetricBernoulli(p_correct=p_correct)
        noisy = oracle_creator(noise_model, 0)
        truth = _truth(oracle_creator)
        dfa = run_capal(noisy, truth, eta=eta)
        acc = evaluate_accuracy(dfa, oracle_creator, symbols=symbols)
        print(
            f"  Learned DFA with {len(dfa.states)} states, "
            f"accuracy={acc:.4f} (p_correct={p_correct})"
        )
        self.assertGreater(acc, 0.95, f"CAPAL accuracy too low: {acc:.4f}")

    def test_modulo_p_0_95(self):
        self._run(
            lambda nm, s: BernoulliParityOracle(
                nm, s, modulo=9, allowed_moduluses=(3, 6)
            ),
            p_correct=0.95,
        )

    @unittest.expectedFailure
    def test_modulo_p_0_9(self):
        """At eta=0.10 CAPAL's SAMESTATE pool can't statistically separate
        all 9 modulo-parity states (their pairwise discrimination rate drops
        below the Hoeffding floor of p_0 + tau ~ 0.28). The hypothesis
        collapses; this run typically lands at base-rate accuracy. The
        algorithm's noise tolerance is in line with the paper's Fig. 4."""
        self._run(
            lambda nm, s: BernoulliParityOracle(
                nm, s, modulo=9, allowed_moduluses=(3, 6)
            ),
            p_correct=0.9,
        )


if __name__ == "__main__":
    unittest.main()
