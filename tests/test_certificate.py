"""The certificate that ends synthesis: its bounds on a DFA's advantage over the
oracle, and its verdicts."""

import itertools
import unittest

import scipy.stats
from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star.certificate import TooMuchRead, advantage_bounds, certifies
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


def _outcomes(drawn, read_mass, unread_rate):
    """(k, o, P) over n = drawn draws: k read before, o of the rest reading 1."""
    for read, ones in itertools.product(range(drawn + 1), repeat=2):
        if ones > drawn - read:
            continue
        chance = scipy.stats.binom.pmf(read, drawn, read_mass) * scipy.stats.binom.pmf(
            ones, drawn - read, unread_rate
        )
        if chance:
            yield read, ones, chance


def _miss_probabilities(sides, read_rates):
    """(P(lo > A), P(hi < A)) exactly, the strings read before reading 1 at
    read_rates[s] on side s."""
    (drawn_1, mass_1, rate_1), (drawn_0, mass_0, rate_0) = sides
    advantage = (
        (1 - mass_1) * rate_1
        + mass_1 * read_rates[0]
        - (1 - mass_0) * rate_0
        - mass_0 * read_rates[1]
    )
    over = under = 0.0
    for (k1, o1, p1), (k0, o0, p0) in itertools.product(
        list(_outcomes(drawn_1, mass_1, rate_1)),
        list(_outcomes(drawn_0, mass_0, rate_0)),
    ):
        lo, hi = advantage_bounds(((drawn_1, k1, o1), (drawn_0, k0, o0)), LEVEL)
        over += p1 * p0 * (lo > advantage)
        under += p1 * p0 * (hi < advantage)
    return over, under


class TestAdvantageBounds(unittest.TestCase):
    @parameterized.expand(
        [
            ((40, 0.0, 0.6), (30, 0.0, 0.2)),
            ((25, 0.0, 0.9), (35, 0.0, 0.85)),
            ((20, 0.1, 0.7), (20, 0.1, 0.3)),
        ]
    )
    def test_each_bound_misses_at_most_at_its_level(self, side_1, side_0):
        # Read strings at each extreme, the one that lowers A and the one that
        # raises it.
        for read_rates in ((0.0, 1.0), (1.0, 0.0)):
            for missed in _miss_probabilities((side_1, side_0), read_rates):
                self.assertLessEqual(missed, LEVEL)


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

    def test_a_space_read_through_raises_rather_than_drawing_forever(self):
        pst = _pst(0.3, 8)
        pst.table.memo.membership_queries(
            [bytes(bits) for bits in itertools.product((0, 1), repeat=8)]
        )
        with self.assertRaises(TooMuchRead):
            certifies(pst, TARGET, alpha=LEVEL)


if __name__ == "__main__":
    unittest.main()
