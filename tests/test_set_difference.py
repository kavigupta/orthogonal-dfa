import unittest

import numpy as np

from orthogonal_dfa.l_star.examples.set_difference import SetDifferenceOracle
from orthogonal_dfa.l_star.structures import Oracle


class PredicateOracle(Oracle):
    """Accepts a string iff ``predicate(string)``; records what it was asked."""

    def __init__(self, predicate, *, alphabet_size=4, string_length=8):
        self._predicate = predicate
        self._alphabet_size = alphabet_size
        self._string_length = string_length
        self.seen = []

    @property
    def alphabet_size(self):
        return self._alphabet_size

    @property
    def string_length(self):
        return self._string_length

    def membership_queries(self, strings):
        self.seen.append([list(s) for s in strings])
        return np.array([bool(self._predicate(s)) for s in strings], dtype=bool)

    def membership_query(self, string):
        return bool(self._predicate(string))


class TestSetDifferenceOracle(unittest.TestCase):
    def setUp(self):
        # a accepts strings starting with 0; b accepts strings ending with 0.
        self.a = PredicateOracle(lambda s: s[0] == 0)
        self.b = PredicateOracle(lambda s: s[-1] == 0)
        self.oracle = SetDifferenceOracle(self.a, self.b)

    def test_accepts_exactly_a_and_not_b(self):
        strings = [[0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]]
        result = self.oracle.membership_queries(strings)
        # a: [T, T, F, F], b: [T, F, T, F] -> a & ~b: [F, T, F, F]
        np.testing.assert_array_equal(result, [False, True, False, False])
        self.assertEqual(result.dtype, bool)

    def test_membership_query_singular(self):
        self.assertTrue(self.oracle.membership_query([0, 1, 1]))  # a yes, b no
        self.assertFalse(self.oracle.membership_query([0, 1, 0]))  # a yes, b yes
        self.assertFalse(self.oracle.membership_query([1, 1, 1]))  # a no

    def test_both_sub_oracles_see_every_string(self):
        strings = [[0, 0], [1, 0]]
        self.oracle.membership_queries(strings)
        self.assertEqual(self.a.seen[-1], strings)
        self.assertEqual(self.b.seen[-1], strings)

    def test_dimensions_come_from_the_first_oracle(self):
        oracle = SetDifferenceOracle(
            PredicateOracle(lambda s: True, alphabet_size=4, string_length=12),
            PredicateOracle(lambda s: False, alphabet_size=4, string_length=12),
        )
        self.assertEqual(oracle.alphabet_size, 4)
        self.assertEqual(oracle.string_length, 12)

    def test_rejects_mismatched_alphabets(self):
        with self.assertRaises(AssertionError):
            SetDifferenceOracle(
                PredicateOracle(lambda s: True, alphabet_size=4),
                PredicateOracle(lambda s: True, alphabet_size=2),
            )

    def test_empty_batch(self):
        result = self.oracle.membership_queries([])
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, bool)


if __name__ == "__main__":
    unittest.main()
