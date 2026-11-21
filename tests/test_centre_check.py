"""The centre check on synthetic two-binomial votes: a family of FAMILY reads
per prefix, the seed's one read, each at its class's rate."""

import unittest

import numpy as np
from parameterized import parameterized

from orthogonal_dfa.l_star.cluster import centre_check
from orthogonal_dfa.l_star.statistics import centres

FAMILY = 200
PREFIXES = 20000
ACCEPTING = 0.3
CLASS_FLOOR = 0.01

BANDS = [(0.60, 0.90), (0.10, 0.40), (0.35, 0.95), (0.05, 0.65), (0.20, 0.80)]


def _reads(p_0, p_1, seed):
    rng = np.random.default_rng(seed)
    rate = np.where(rng.random(PREFIXES) < ACCEPTING, p_1, p_0)
    return rng.binomial(FAMILY, rate) / FAMILY, (rng.random(PREFIXES) < rate) * 1.0


def _check(p_0, p_1, centre, seed=0):
    votes, reads = _reads(p_0, p_1, seed)
    return centre_check(
        votes,
        reads,
        centre,
        signal=(p_1 - p_0) / 2,
        class_floor=CLASS_FLOOR,
    )


class TestCentres(unittest.TestCase):
    @parameterized.expand([(0.1,), (0.15,), (0.3,), (0.45,)])
    def test_half_a_signal_apart_within_the_band(self, signal):
        grid = centres(signal)
        self.assertIn(0.5, grid)
        self.assertTrue(all(signal <= c <= 1 - signal for c in grid))
        np.testing.assert_allclose(np.diff(grid), signal / 2, atol=1e-9)
        # No centre in [signal, 1 - signal] is more than a quarter signal off one.
        for c in np.linspace(signal, 1 - signal, 101):
            self.assertLessEqual(min(abs(c - g) for g in grid), signal / 2 + 1e-9)


class TestCentreCheck(unittest.TestCase):
    @parameterized.expand(BANDS)
    def test_the_nearest_centre_is_plausible_and_refines_to_the_band(self, p_0, p_1):
        truth, signal = (p_0 + p_1) / 2, (p_1 - p_0) / 2
        nearest = min(centres(signal), key=lambda c: abs(c - truth))
        check = _check(p_0, p_1, nearest)
        self.assertTrue(check.plausible)
        self.assertAlmostEqual(check.refined, truth, delta=0.02)
        self.assertTrue(_check(p_0, p_1, check.refined).plausible)

    def test_the_clamped_boundary_of_issue_319_is_refused(self):
        self.assertFalse(_check(0.60, 0.90, 0.85).plausible)

    def test_a_cut_with_everything_on_one_side_is_refused(self):
        # (0.60, 0.90) read at a half: both classes vote above it.
        self.assertFalse(_check(0.60, 0.90, 0.5).plausible)


if __name__ == "__main__":
    unittest.main()
