"""The prefix pool is sized from the signal, like the suffix family already was."""

import unittest

import numpy as np
import scipy.stats

from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliParityOracle
from orthogonal_dfa.l_star.learn import MIN_PREFIXES, build_pst
from orthogonal_dfa.l_star.statistics import (
    common_in_prefixes_threshold,
    compute_prefix_pool_size,
)


def _oracle(noise_model, seed):
    return BernoulliParityOracle(noise_model, seed)


class TestCommonInPrefixesThreshold(unittest.TestCase):
    def test_a_state_given_the_threshold_votes_its_way(self):
        # The threshold is a normal approximation, so the exact binomial it
        # stands in for is allowed to miss the rate it was asked for, but only
        # by a small factor and only towards the small-population end.
        for signal in (0.3, 0.2, 0.15, 0.1, 0.05):
            for fpr in (0.01, 0.001):
                votes = int(np.ceil(common_in_prefixes_threshold(signal, fpr)))
                wrong = scipy.stats.binom.cdf(votes // 2, votes, 0.5 + signal)
                self.assertLess(wrong, 4 * fpr, f"signal {signal}, fpr {fpr}")

    def test_half_the_threshold_is_not_enough(self):
        for signal in (0.2, 0.1):
            votes = int(np.ceil(common_in_prefixes_threshold(signal, 0.001))) // 2
            wrong = scipy.stats.binom.cdf(votes // 2, votes, 0.5 + signal)
            self.assertGreater(wrong, 0.001, f"signal {signal}")

    def test_weaker_signal_needs_more(self):
        sizes = [compute_prefix_pool_size(s, 5, 0.01) for s in (0.3, 0.2, 0.15, 0.1)]
        self.assertEqual(sizes, sorted(sizes))

    def test_scales_as_one_over_signal_squared(self):
        # (1/4 - s^2)/s^2 -> 1/(4 s^2) as s shrinks, so halving a small signal
        # asks for four times the pool.
        small = compute_prefix_pool_size(0.02, 5, 0.01)
        smaller = compute_prefix_pool_size(0.01, 5, 0.01)
        self.assertAlmostEqual(smaller / small, 4, delta=0.05)


class TestPoolIsSizedForTheSignal(unittest.TestCase):
    def test_strong_signal_keeps_the_floor(self):
        pst = build_pst(_oracle, min_signal_strength=0.3, seed=0)
        self.assertLess(compute_prefix_pool_size(0.3, 8, 0.01), MIN_PREFIXES)
        self.assertEqual(pst.config.num_addtl_prefixes, MIN_PREFIXES)

    def test_weak_signal_raises_the_pool(self):
        pst = build_pst(_oracle, min_signal_strength=0.1, seed=0)
        self.assertEqual(
            pst.config.num_addtl_prefixes, compute_prefix_pool_size(0.1, 8, 0.01)
        )
        self.assertGreater(pst.num_prefixes, MIN_PREFIXES)

    def test_max_states_is_the_knob(self):
        few = build_pst(_oracle, min_signal_strength=0.1, seed=0, max_states=4)
        many = build_pst(_oracle, min_signal_strength=0.1, seed=0, max_states=16)
        self.assertGreater(
            many.config.num_addtl_prefixes, few.config.num_addtl_prefixes
        )
