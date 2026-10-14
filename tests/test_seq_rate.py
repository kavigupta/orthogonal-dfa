"""``seq_rate`` keeps the promise ``OrthoDFA/SeqTest.lean`` makes: under its null rate, or any
rate beyond it from the alternative, it ever fires at most ``level`` of the time."""

import itertools
import math
import unittest

import numpy as np

from orthogonal_dfa.l_star.statistics import seq_rate

#: (null, alternative): two tests for a rate above the null, two for one below.
PAIRS = ((0.05, 0.1), (0.4, 0.5), (0.1, 0.075), (0.02, 0.005))


def fire_chance(p0, p1, level, rate, looks):
    """The exact chance the test fires at some look up to ``looks``, over a stream of draws at
    ``rate``: the paths not yet fired, by hits so far, carried forward one draw at a time.
    """
    alive = np.zeros(looks + 1)
    alive[0] = 1.0
    fired = 0.0
    for n in range(1, looks + 1):
        step = np.zeros(looks + 1)
        step[: n + 1] += alive[: n + 1] * (1 - rate)
        step[1 : n + 1] += alive[:n] * rate
        fires = np.array([seq_rate(p0, p1, level, n, h) for h in range(n + 1)])
        fired += step[: n + 1][fires].sum()
        step[: n + 1][fires] = 0.0
        alive = step
    return fired


def null_side(p0, p1):
    """The null rate, and rates beyond it from the alternative."""
    away = -1 if p1 > p0 else 1
    return [p0, p0 + away * 0.5 * p0 * min(1, abs(p1 - p0) / p0)]


class TestTheRatio(unittest.TestCase):
    def test_it_compares_the_likelihood_ratio_with_one_over_the_level(self):
        for (p0, p1), level, n in itertools.product(PAIRS, (0.01, 0.1), (5, 40)):
            for h in range(n + 1):
                ratio = (p1 / p0) ** h * ((1 - p1) / (1 - p0)) ** (n - h)
                if abs(math.log(ratio) - math.log(1 / level)) > 1e-9:
                    self.assertEqual(ratio >= 1 / level, seq_rate(p0, p1, level, n, h))

    def test_a_certain_alternative_fires_only_on_hits(self):
        self.assertTrue(seq_rate(0.5, 1.0, 0.01, 7, 7))
        self.assertFalse(seq_rate(0.5, 1.0, 0.01, 7, 6))


class TestSoundness(unittest.TestCase):
    def test_exactly_it_ever_fires_at_most_level_of_the_time(self):
        for (p0, p1), level in itertools.product(PAIRS, (0.01, 0.1)):
            for rate in null_side(p0, p1):
                with self.subTest(p0=p0, p1=p1, level=level, rate=rate):
                    self.assertLessEqual(fire_chance(p0, p1, level, rate, 300), level)

    def test_by_simulation_over_a_long_stream(self):
        rng = np.random.default_rng(0)
        reps, looks = 4000, 3000
        for p0, p1 in PAIRS:
            hits = np.cumsum(rng.random((reps, looks)) < p0, axis=1)
            n = np.arange(1, looks + 1)
            llr = hits * math.log(p1 / p0) + (n - hits) * math.log((1 - p1) / (1 - p0))
            for level in (0.01, 0.1):
                fired = (llr >= math.log(1 / level)).any(axis=1).mean()
                slack = 3 * math.sqrt(level * (1 - level) / reps)
                with self.subTest(p0=p0, p1=p1, level=level):
                    self.assertLessEqual(fired, level + slack)


if __name__ == "__main__":
    unittest.main()
