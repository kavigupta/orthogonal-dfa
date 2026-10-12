"""The binomial rate test keeps the promises ``RateTestSound`` makes, and is ``rateSide``."""

import itertools
import math
import unittest
from fractions import Fraction

import numpy as np

from orthogonal_dfa.l_star.statistics import binomial_side_of_boundary

#: Float slack on a probability the test is compared against.
_SLACK = 1e-9


def side(h, n, theta, a):
    return binomial_side_of_boundary(h, n, theta, failure_prob=a)


def pmf(k, n, p):
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def chance_of(verdict, n, theta, a, p):
    """``P(side(X, n, θ, a) = verdict)`` for ``X ~ Bin(n, p)``, summed exactly."""
    return sum(pmf(k, n, p) for k in range(n + 1) if side(k, n, theta, a) is verdict)


def lean_rate_side(theta, a, n0, n, h):
    """``rateSide θ a n₀ n h`` of ``OrthoDFA/RateTest.lean``, in exact arithmetic."""

    def sf_ge(j):
        return sum(
            math.comb(n, i) * theta**i * (1 - theta) ** (n - i) for i in range(j, n + 1)
        )

    if n0 > n:
        return None
    if sf_ge(h) < a:
        return True
    if 1 - sf_ge(h + 1) < a:
        return False
    return None


class TestSoundExactly(unittest.TestCase):
    def test_small_counts(self):
        grid = itertools.product(
            range(0, 31),
            np.linspace(0, 1, 11),
            (1e-5, 1e-3, 0.05, 0.3, 1.0),
        )
        for n, theta, a in grid:
            for p in np.linspace(0, 1, 21):
                if p <= theta:
                    self.assertLessEqual(chance_of(True, n, theta, a, p), a + _SLACK)
                if p >= theta:
                    self.assertLessEqual(chance_of(False, n, theta, a, p), a + _SLACK)


class TestSoundBySimulation(unittest.TestCase):
    def test_large_counts_at_the_boundary(self):
        rng = np.random.default_rng(0)
        trials = 20000
        for n in (100, 1000, 5000):
            for theta in (0.05, 0.3, 0.5, 0.9):
                a = 0.05
                counts = rng.binomial(n, theta, size=trials)
                verdicts = [side(int(k), n, theta, a) for k in counts]
                margin = 5 * math.sqrt(a * (1 - a) / trials)
                self.assertLessEqual(verdicts.count(True) / trials, a + margin)
                self.assertLessEqual(verdicts.count(False) / trials, a + margin)

    def test_large_counts_off_the_boundary(self):
        rng = np.random.default_rng(1)
        trials = 20000
        a = 0.01
        for n in (200, 2000):
            for theta, p in ((0.5, 0.45), (0.2, 0.1)):
                counts = rng.binomial(n, p, size=trials)
                above = [side(int(k), n, theta, a) is True for k in counts]
                self.assertLessEqual(sum(above) / trials, a)
            for theta, p in ((0.5, 0.55), (0.2, 0.3)):
                counts = rng.binomial(n, p, size=trials)
                below = [side(int(k), n, theta, a) is False for k in counts]
                self.assertLessEqual(sum(below) / trials, a)


class TestEdges(unittest.TestCase):
    def test_no_draws_says_nothing(self):
        for theta in (0.0, 0.3, 1.0):
            self.assertIsNone(side(0, 0, theta, 0.05))

    def test_extreme_counts(self):
        self.assertIs(side(0, 100, 0.5, 1e-5), False)
        self.assertIs(side(100, 100, 0.5, 1e-5), True)
        self.assertIsNone(side(50, 100, 0.5, 1e-5))

    def test_a_count_of_zero_is_never_above(self):
        for n in range(0, 20):
            for theta in (0.0, 0.5, 1.0):
                self.assertIsNot(side(0, n, theta, 1.0), True)

    def test_a_full_count_is_never_below(self):
        for n in range(0, 20):
            for theta in (0.0, 0.5, 1.0):
                self.assertIsNot(side(n, n, theta, 1.0), False)


class TestIsRateSide(unittest.TestCase):
    def test_agrees_with_the_lean_definition(self):
        thetas = [Fraction(i, 20) for i in range(21)]
        levels = [
            Fraction(1, 100000),
            Fraction(1, 1000),
            Fraction(1, 20),
            Fraction(1, 2),
        ]
        checked = 0
        grid = itertools.product(range(0, 26), thetas, levels)
        for n, theta, a in grid:
            for h in range(0, n + 1):
                tails = (
                    sum(pmf(k, n, float(theta)) for k in range(h, n + 1)),
                    sum(pmf(k, n, float(theta)) for k in range(0, h + 1)),
                )
                if any(abs(t - float(a)) <= _SLACK for t in tails):
                    continue
                got = side(h, n, float(theta), float(a))
                self.assertEqual(
                    lean_rate_side(theta, a, 0, n, h), got, (n, theta, a, h)
                )
                checked += 1
        self.assertGreater(checked, 25000)


if __name__ == "__main__":
    unittest.main()
