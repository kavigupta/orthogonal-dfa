"""The certificate that ends synthesis: exact bounds on a DFA's advantage over
the oracle, and its verdicts on the target and on a DFA that merges a state."""

import unittest

import numpy as np
import scipy.stats
from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star.certificate import advantage_bounds, certifies
from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle

LEVEL = 0.05
DRAWS = 2000


def _ones_mod_3(accepting):
    return DFA(
        states={0, 1, 2},
        input_symbols={0, 1},
        transitions={q: {0: q, 1: (q + 1) % 3} for q in range(3)},
        initial_state=0,
        final_states=set(accepting),
    )


TARGET = _ones_mod_3({0})
#: Residue 1, a third of the strings, read as accepting.
MERGED = _ones_mod_3({0, 1})


class TestAdvantageBounds(unittest.TestCase):
    @parameterized.expand([(0.6, 0.2, 40, 400), (0.9, 0.85, 500, 30)])
    def test_each_bound_misses_at_most_at_its_level(self, one, zero, drawn, drawn_0):
        rng = np.random.default_rng(0)
        bounds = [
            advantage_bounds(
                (
                    (rng.binomial(drawn, one), drawn),
                    (rng.binomial(drawn_0, zero), drawn_0),
                ),
                LEVEL,
            )
            for _ in range(DRAWS)
        ]
        for missed in (
            sum(lo > one - zero for lo, _ in bounds),
            sum(hi < one - zero for _, hi in bounds),
        ):
            self.assertGreater(scipy.stats.binom.sf(missed - 1, DRAWS, LEVEL), 1e-3)


def _pst(signal):
    return build_pst(
        lambda nm, s: NoisyOracle(DFAOracle(TARGET), nm, s),
        min_signal_strength=signal,
        seed=0,
        sampler=UniformSampler(20),
        noise_model=AsymmetricBernoulli(p_0=0.2, p_1=0.8),
    )


class TestVerdicts(unittest.TestCase):
    # The band's signal is 0.3: stated exactly, and understated.
    @parameterized.expand([(0.3,), (0.2,)])
    def test_the_target_is_certified(self, signal):
        self.assertTrue(certifies(_pst(signal), TARGET, alpha=LEVEL))

    @parameterized.expand([(0.3,), (0.2,)])
    def test_a_merge_past_the_slack_is_not(self, signal):
        self.assertFalse(certifies(_pst(signal), MERGED, alpha=LEVEL))


if __name__ == "__main__":
    unittest.main()
