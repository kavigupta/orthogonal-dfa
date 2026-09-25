"""How many samples it takes to catch a rate below the null."""

import unittest

import scipy.stats

from orthogonal_dfa.l_star.statistics import low_tail_detection_size

#: ``(null, alternative, level, miss_rate)``, spanning a wide gap and a narrow one.
CASES = [
    (0.65, 0.35, 0.05, 0.05),
    (0.65, 0.35, 0.00625, 0.05),
    (0.70, 0.30, 0.025, 0.05),
    (0.58, 0.42, 0.025, 0.05),
    (0.65, 0.50, 0.025, 0.10),
    (0.90, 0.10, 0.01, 0.01),
]


def _power(n, null, alternative, level):
    """Chance ``n`` samples drawn at ``alternative`` land where a test at
    ``level`` under ``null`` calls them low.

    Walks every count rather than calling the search under test, which would
    agree with it whatever either of them did.
    """
    region = [k for k in range(n + 1) if scipy.stats.binom.cdf(k, n, null) <= level]
    if not region:
        return 0.0
    return float(sum(scipy.stats.binom.pmf(k, n, alternative) for k in region))


class TestTheSizeHoldsTheMissRate(unittest.TestCase):
    def test_the_alternative_is_caught_all_but_the_miss_rate(self):
        for null, alt, level, miss in CASES:
            with self.subTest(null=null, alt=alt, level=level):
                n = low_tail_detection_size(null, alt, level, miss)
                self.assertGreaterEqual(_power(n, null, alt, level), 1 - miss)

    def test_one_sample_fewer_does_not(self):
        for null, alt, level, miss in CASES:
            with self.subTest(null=null, alt=alt, level=level):
                n = low_tail_detection_size(null, alt, level, miss)
                self.assertLess(_power(n - 1, null, alt, level), 1 - miss)


class TestWhatTheSizeRespondsTo(unittest.TestCase):
    def test_a_tighter_level_never_costs_fewer(self):
        wider = low_tail_detection_size(0.65, 0.35, 0.05, 0.05)
        tighter = low_tail_detection_size(0.65, 0.35, 0.00625, 0.05)

        self.assertGreaterEqual(tighter, wider)

    def test_a_narrower_gap_costs_more(self):
        wide = low_tail_detection_size(0.70, 0.30, 0.025, 0.05)
        narrow = low_tail_detection_size(0.58, 0.42, 0.025, 0.05)

        self.assertGreater(narrow, wide)

    def test_sizing_on_the_level_alone_is_not_enough(self):
        null, alt, level, miss = 0.65, 0.35, 0.025, 0.05
        # The size the level alone buys: the first at which a reading of nothing
        # but zeroes is significant.

        firing = next(
            n for n in range(1, 500) if scipy.stats.binom.cdf(0, n, null) <= level
        )

        self.assertLess(_power(firing, null, alt, level), 1 - miss)
        self.assertGreater(low_tail_detection_size(null, alt, level, miss), firing)

    def test_an_alternative_at_or_above_the_null_is_refused(self):
        with self.assertRaises(AssertionError):
            low_tail_detection_size(0.5, 0.5, 0.05, 0.05)

    def test_a_level_or_miss_rate_no_size_can_reach_is_refused(self):
        # Left alone, a level of zero is cleared only when the tail underflows,
        # and a zero miss rate doubles until the counts overflow.
        with self.assertRaises(AssertionError):
            low_tail_detection_size(0.65, 0.35, 0.0, 0.05)
        with self.assertRaises(AssertionError):
            low_tail_detection_size(0.65, 0.35, 0.05, 0.0)


if __name__ == "__main__":
    unittest.main()
