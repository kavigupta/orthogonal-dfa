"""Drawing a pool of distinct prefixes from a sampler that may not have one."""

import math
import unittest

import numpy as np

from orthogonal_dfa.l_star.prefix_suffix_tracker import (
    DRAWS_PER_PREFIX,
    _distinct_prefixes,
)


class _Sampler:
    """Uniform strings of ``length`` bytes; support is ``alphabet_size**length``."""

    def __init__(self, length):
        self.length = length

    def sample(self, rng, *, alphabet_size):
        return bytes(rng.integers(0, alphabet_size, self.length).tolist())


def _draw(length, count, *, held=(), seed=0):
    return _distinct_prefixes(
        _Sampler(length),
        np.random.default_rng(seed),
        alphabet_size=2,
        count=count,
        held=held,
    )


class TestDistinctPrefixes(unittest.TestCase):
    def test_it_draws_the_count_asked_for_all_distinct(self):
        drawn = _draw(40, 200)
        self.assertEqual(len(drawn), 200)
        self.assertEqual(len(set(drawn)), 200)

    def test_it_skips_what_is_already_held(self):
        held = set(_draw(12, 50))
        self.assertFalse(held & set(_draw(12, 50, held=held, seed=1)))

    def test_a_sampler_too_narrow_to_fill_the_pool_says_so(self):
        # 2^6 = 64 distinct strings, 200 wanted.  Looping until it has them
        # never returns, so the budget is what turns a hang into a message.
        with self.assertRaises(AssertionError) as caught:
            _draw(6, 200)
        self.assertIn("64 distinct prefixes of 200", str(caught.exception))

    def test_the_budget_is_generous_where_the_support_is_ample(self):
        # Falling short is a chance event, so this is over seeds, not one.
        for seed in range(20):
            self.assertEqual(len(_draw(40, 200, seed=seed)), 200)

    def test_the_budget_dominates_the_barely_enough_case(self):
        # The multiplier has to beat ln(count): that is the cost per prefix when
        # the pool nearly exhausts the sampler, which is the worst case that can
        # still succeed.
        self.assertGreater(DRAWS_PER_PREFIX, math.log(10**8))


if __name__ == "__main__":
    unittest.main()
