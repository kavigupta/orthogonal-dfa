"""The certificate that ends synthesis: its test on a DFA's advantage over the
oracle, and its verdicts."""

import unittest

import numpy as np
import scipy.stats
from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star.certificate import certifies, clears
from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle

LEVEL = 0.05


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


class TestClears(unittest.TestCase):
    @parameterized.expand([(5, 0.2), (20, 0.38), (60, 0.5), (60, 0.05)])
    def test_clears_at_most_at_its_level_whatever_the_rates(self, drawn, gap):
        counts = np.arange(drawn + 1)
        passes = np.array(
            [[clears((k1, k0), drawn, gap, LEVEL) for k0 in counts] for k1 in counts]
        )
        for low in np.linspace(0, 1 - gap, 41):
            chance = np.outer(
                scipy.stats.binom.pmf(counts, drawn, low + gap),
                scipy.stats.binom.pmf(counts, drawn, low),
            )
            self.assertLessEqual(chance[passes].sum(), LEVEL)


def _pst(signal, length):
    return build_pst(
        lambda nm, s: NoisyOracle(DFAOracle(TARGET), nm, s),
        min_signal_strength=signal,
        seed=0,
        sampler=UniformSampler(length),
        noise_model=AsymmetricBernoulli(p_0=0.2, p_1=0.8),
    )


class TestVerdicts(unittest.TestCase):
    # The band's signal is 0.3: stated exactly, and understated.
    @parameterized.expand([(0.3,), (0.2,)])
    def test_the_target_is_certified(self, signal):
        self.assertTrue(certifies(_pst(signal, 20), TARGET, alpha=LEVEL))

    @parameterized.expand([(0.3,), (0.2,)])
    def test_a_merge_past_the_slack_is_not(self, signal):
        self.assertFalse(certifies(_pst(signal, 20), MERGED, alpha=LEVEL))

    def test_a_dfa_with_one_label_is_not(self):
        self.assertFalse(certifies(_pst(0.3, 20), _ones_mod_3(set()), alpha=LEVEL))


if __name__ == "__main__":
    unittest.main()
