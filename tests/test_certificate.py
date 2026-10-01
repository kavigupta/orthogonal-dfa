"""The certificate's bound on a DFA's error from each state's rate."""

import unittest

import numpy as np

from orthogonal_dfa.l_star.certificate import error_bound


def _rates(low_rate, gap, contaminated):
    """r_S = p_0 + gap c_S for c_S the share of accepting strings in each state."""
    return low_rate + gap * np.asarray(contaminated, dtype=float)


class TestErrorBound(unittest.TestCase):
    def test_never_below_the_error_while_the_intervals_hold(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            states = int(rng.integers(1, 6))
            masses = rng.dirichlet(np.ones(states))
            accepting = rng.random(states) < 0.5
            gap = float(rng.uniform(0.1, 0.9))
            low_rate = float(rng.uniform(0, 1 - gap))
            shares = rng.random(states) * (rng.random(states) < 0.5)
            accepted_share = np.where(accepting, 1 - shares, shares)
            rates = _rates(low_rate, gap, accepted_share)
            low = np.clip(rates - rng.random(states) * 0.1, 0, 1)
            high = np.clip(rates + rng.random(states) * 0.1, 0, 1)
            self.assertGreaterEqual(
                error_bound(masses, accepting, low, high, gap),
                float(masses @ shares) - 1e-12,
            )

    def test_exact_with_a_clean_state_of_each_label(self):
        # A light rejecting state holding a fifth accepting strings, next to clean
        # heavy states: the share of signal it costs is large, the error small.
        masses = [0.88, 0.06, 0.06]
        accepting = [True, False, False]
        rates = _rates(0.3, 0.4, [1.0, 0.0, 0.2])
        self.assertAlmostEqual(
            error_bound(masses, accepting, rates, rates, 0.4), 0.06 * 0.2
        )


if __name__ == "__main__":
    unittest.main()
