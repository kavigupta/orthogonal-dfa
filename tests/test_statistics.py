"""Sizing a decision population from the signal it has to separate."""

import unittest

import numpy as np
import scipy

from orthogonal_dfa.l_star.statistics import (
    evidence_margin_for_population_size,
    population_size_and_evidence_margin,
)

SIGNALS = [0.45, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]


def error_rates(signal, N, eps, center=0.5):
    """The (fpr, fnr) a population of N decides at with margin ``eps``."""
    k_low = int(np.floor(N * (center - eps)))
    k_high = int(np.ceil(N * (center + eps)))
    fpr = scipy.stats.binom.cdf(k_low, N, center) + (
        1 - scipy.stats.binom.cdf(k_high - 1, N, center)
    )
    fnr = scipy.stats.binom.cdf(k_high - 1, N, signal + center) - scipy.stats.binom.cdf(
        k_low, N, signal + center
    )
    return fpr, fnr


class TestPopulationSizing(unittest.TestCase):
    def test_low_signal_is_sized_rather_than_diverging(self):
        # Regression: the margin grid ran from a fixed 0.01 up to the signal, so
        # at a signal of 0.01 or below no grid point was usable at any N and the
        # search doubled N until it overflowed -- and just above it, the only
        # usable margins sat against the signal, sizing the population orders of
        # magnitude too large to hold in memory.
        for signal in SIGNALS:
            N, eps = population_size_and_evidence_margin(signal, 0.01, 0.01, center=0.5)
            self.assertLess(0, eps)
            self.assertLess(eps, signal, signal)
            fpr, fnr = error_rates(signal, N, eps)
            self.assertLessEqual(fpr, 0.01, signal)
            self.assertLessEqual(fnr, 0.01, signal)

    def test_size_scales_as_the_inverse_square_of_the_signal(self):
        # N ~ 1/signal^2, so halving the signal must not cost much more than 4x.
        # Only asymptotically: near a signal of 0.5 the population is tiny and
        # rounding to whole counts dominates.
        for signal in [s for s in SIGNALS if s <= 0.2]:
            N, _ = population_size_and_evidence_margin(signal, 0.01, 0.01, center=0.5)
            N_half, _ = population_size_and_evidence_margin(
                signal / 2, 0.01, 0.01, center=0.5
            )
            self.assertLess(N_half, 4.5 * N, signal)

    def test_size_is_minimal(self):
        for signal in [0.3, 0.05, 0.01]:
            N, _ = population_size_and_evidence_margin(signal, 0.01, 0.01, center=0.5)
            self.assertIsNone(
                evidence_margin_for_population_size(
                    signal, 0.01, 0.01, N - 1, center=0.5
                ),
                signal,
            )


if __name__ == "__main__":
    unittest.main()
